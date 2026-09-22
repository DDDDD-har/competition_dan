#!/usr/bin/env python3
"""Track the ordered inspection waypoints with Navfn and Pure Pursuit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from path_safety_filter import densify_path, predict_straight_path, predict_unicycle_path, swept_clear
from path_speed_profile import (
    ackermann_feasible_command,
    build_path_speed_profile,
    local_curvatures,
    hybrid_execution_segments,
    orient_path_for_ackermann,
    refine_planned_path,
    max_lateral_from_chord,
    segment_peak_curvature,
    split_path_by_curvature,
    primary_c_turn_start_arc,
    c_leg_turn_via_point,
    cumulative_arc_lengths,
    estimated_profile_time,
    point_to_segment_distance,
    required_speed_change_distance,
    _reachable_speed,
    trim_path_to_score_stop,
    profile_summary,
    terminal_path_is_trackable,
    terminal_path_metrics,
    widen_last_c_turn,
)
from task1_scene_contract import (
    CHECKPOINT_ZONE_RADIUS_M,
    OFFICIAL_REGION_RADIUS_M,
    BODY_LINK_FORWARD_OFFSET_M,
    VALIDATION_MAX_CORRIDOR_LATERAL_M,
    VALIDATION_MAX_PATH_CURVATURE_1PM,
    VALIDATION_MIN_PATH_CLEARANCE_M,
    VALIDATION_MISSION_TIMEOUT_S,
    VALIDATION_MISSION_DEADLINE_ENABLED,
    VALIDATION_MOTION_STALL_DISPLACEMENT_M,
    VALIDATION_MOTION_STALL_SPEED_MPS,
    VALIDATION_MOTION_STALL_STARTUP_GRACE_S,
    VALIDATION_MOTION_STALL_TIMEOUT_S,
    apply_startup_ramp,
    checkpoint_settle_duration_s,
    scaled_profile_linear_accel,
    scorer_aligned_root_target,
    scoring_speed_policy,
    startup_ramp_scale,
    start_to_a_motion_elapsed_ok,
    validation_command_slew_limits,
    validation_profile_lookahead_m,
    OFFICIAL_CHECKPOINT_DEEP_RADIUS_M,
    OFFICIAL_CHECKPOINT_PASS_RADIUS_SCALE,
    OFFICIAL_CIRCLE_CAPTURE_RADIUS_SCALE,
    VALIDATION_C_CHECKPOINT_PASS_M,
    VALIDATION_C_LEG_COMMITTED_TURN_LOOKAHEAD_M,
    VALIDATION_C_LEG_PRE_BEND_CURVATURE_1PM,
    VALIDATION_C_LEG_PRE_BEND_LOOKAHEAD_M,
    VALIDATION_C_LEG_PRIMARY_TURN_CURVATURE_1PM,
    VALIDATION_C_STALL_GRACE_M,
    VALIDATION_C_TURN_MAX_PUSH_M,
    VALIDATION_C_TURN_TARGET_CLEARANCE_M,
    VALIDATION_C_LEG_TURN_HEADING_CHANGE_RAD,
    VALIDATION_C_LEG_TURN_MIN_ARC_M,
    VALIDATION_C_LEG_TURN_MIN_STRAIGHT_M,
    VALIDATION_C_LEG_TURN_ZONE_WINDOW_M,
    VALIDATION_LOCAL_REPLAN_SETTLE_S,
    VALIDATION_ROUTE_ORDER,
    OFFICIAL_SCORER_START_MOVE_HOLD_S,
    OFFICIAL_SCORER_START_MOVE_MIN_MPS,
    VALIDATION_TERMINAL_ZONE_M,
    VALIDATION_START_TO_A_MIN_ELAPSED_S,
    VALIDATION_STARTUP_PLATEAU_S,
    VALIDATION_C_TERMINAL_CAPTURE_CAP_MPS,
    VALIDATION_AB_CHECKPOINT_DWELL_S,
    VALIDATION_POST_ROBOT_START_DWELL_S,
    ROBOT_STARTED_MARKER_NAME,
    VALIDATION_WAYPOINT_PASS_TOLERANCE_M,
)
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from nav2_msgs.msg import SpeedLimit
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener


class MotionStalled(RuntimeError):
    """The chassis stopped making progress while a leg was still unfinished."""


class ABCNavigator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("agibot_abc_demo")
        self.args = args
        self.pose = None
        self.samples = []
        self.leg_directions = {}
        self.navigation_status = None
        self.occupied_world_xy = None
        self.path_safety_reports = {}
        self.planned_paths = {}
        self.profile_leg_arc_bounds: dict[str, tuple[float, float]] = {}
        # In plan-only mode there is no live motion to advance the TF pose.
        # Keep a separate virtual planning pose so each leg starts where the
        # previous leg ends and the preview represents the actual A→B→C
        # mission sequence.
        self.plan_pose = None
        self.coordinate_reach = {}
        self.checkpoint_intervals = []
        self.path_speed_profile = []
        self.path_speed_profile_summary = {}
        self.speed_limit_history = []
        self.checkpoint_pass_speeds = {}
        self.checkpoint_pass_speeds_detail: dict[str, dict[str, float]] = {}
        self.route_geometry_audits: list[dict] = []
        self.execution_trace: list[dict] = []
        self.current_control_mode = ""
        self.current_leg_name = ""
        self.last_speed_cap = 0.0
        self.last_limiting_factor = ""
        self.last_cross_track_m = 0.0
        self.mission_started_monotonic = None
        self.cruise_ramp_started_monotonic = None
        self.leg_timestamps: dict[str, float] = {}
        self._motion_stall_suspended = False
        self._mission_has_moved = False
        self._stall_reference_time: float | None = None
        self._stall_reference_pose: tuple[float, float, float] | None = None
        self._nav2_monotonic_index = 0
        self._nav2_path_arc_lengths: list[float] = []
        self._planner_produced_path = False
        self.current_odom_speed = 0.0
        self.task_points = {}
        self.safety_footprint = (
            (0.35, 0.37), (0.35, -0.34),
            (-0.41, -0.34), (-0.41, 0.37),
        )
        self.command_safety_checks = 0
        self.last_command_safety_report = None
        self.last_visual_pose_time = None
        self.last_raw_pose = None
        # Keep latency below one camera frame while suppressing small TF noise.
        self.pose_filter_alpha = 0.65
        self.motion_started = False
        self.pose_jump_rejections = 0
        self.pose_update_count = 0
        self.last_command_time = time.monotonic()
        self.audit_path_yaw: float | None = None
        self.last_linear_command = 0.0
        self.last_angular_command = 0.0
        self.last_global_pose_time = None
        self.have_global_pose = False
        self.truth_body_link_map_xy: tuple[float, float] | None = None
        self.last_truth_body_link_time: float | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.speed_limit_pub = self.create_publisher(
            SpeedLimit, args.speed_limit_topic, 10
        )
        self.route_path_pub = self.create_publisher(NavPath, "/task1/planned_route", 1)
        self.active_path_pub = self.create_publisher(NavPath, "/task1/active_global_path", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/task1/route_markers", 1)
        self.create_subscription(NavPath, "/plan", self._on_nav2_plan, 10)
        self.route_visualization_timer = self.create_timer(1.0, self.publish_route_visualization)
        self.create_subscription(Odometry, args.odom_topic, self._on_odom, 20)
        if args.truth_pose_topic:
            self.create_subscription(
                PoseStamped, args.truth_pose_topic, self._on_truth_pose, 20
            )
        if args.truth_body_link_topic:
            self.create_subscription(
                PoseStamped,
                args.truth_body_link_topic,
                self._on_truth_body_link_pose,
                20,
            )
        self.create_subscription(String, args.status_topic, self._on_status, 10)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, args.map_topic, self._on_map, map_qos)
        self.planner = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.controller = ActionClient(self, FollowPath, "/follow_path")
        self.navigator = ActionClient(self, NavigateToPose, "/navigate_to_pose")

    @property
    def route_terminal_name(self) -> str:
        return str(getattr(self.args, "route_stop_at", "C") or "C").upper()

    @staticmethod
    def _nav_path(points, stamp=None) -> NavPath:
        from geometry_msgs.msg import PoseStamped
        msg = NavPath()
        msg.header.frame_id = "map"
        if stamp is not None:
            msg.header.stamp = stamp
        for x, y, yaw in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.06
            pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
            pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
            msg.poses.append(pose)
        return msg

    def _on_nav2_plan(self, msg: NavPath) -> None:
        """Republish Nav2's live replans on a task-specific RViz topic."""
        self.active_path_pub.publish(msg)

    def publish_route_visualization(self) -> None:
        stamp = self.get_clock().now().to_msg()
        combined = []
        for path in self.planned_paths.values():
            if combined and path:
                combined.extend(path[1:])
            else:
                combined.extend(path)
        if combined:
            self.route_path_pub.publish(self._nav_path(combined, stamp))
        markers = MarkerArray()
        colors = {"A": (1.0, 0.15, 0.10), "B": (0.10, 0.45, 1.0), "C": (0.10, 0.9, 0.25)}
        for index, name in enumerate(("A", "B", "C")):
            if name not in self.task_points:
                continue
            x, y = self.task_points[name]
            sphere = Marker()
            sphere.header.frame_id = "map"; sphere.header.stamp = stamp
            sphere.ns = "task1_checkpoints"; sphere.id = index * 2
            sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
            sphere.pose.position.x = float(x); sphere.pose.position.y = float(y); sphere.pose.position.z = 0.18
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
            sphere.color.r, sphere.color.g, sphere.color.b = colors[name]; sphere.color.a = 0.95
            markers.markers.append(sphere)
            label = Marker()
            label.header = sphere.header; label.ns = "task1_labels"; label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
            label.pose.position.x = float(x); label.pose.position.y = float(y); label.pose.position.z = 0.55
            label.pose.orientation.w = 1.0; label.scale.z = 0.35
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = name; markers.markers.append(label)
        self.marker_pub.publish(markers)

    def preview_route(self, route) -> None:
        """Plan the complete ordered route once for visualization and audit."""
        self.plan_continuous_route(route)
        self.get_logger().info(
            "published complete A->B->C route on /task1/planned_route; "
            "live Nav2 replans are mirrored to /task1/active_global_path")

    def _on_map(self, msg: OccupancyGrid) -> None:
        grid=np.asarray(msg.data,dtype=np.int16).reshape(msg.info.height,msg.info.width)
        rows,cols=np.nonzero(grid >= self.args.occupied_threshold)
        yaw=2.0*math.atan2(msg.info.origin.orientation.z,msg.info.origin.orientation.w)
        c,s=math.cos(yaw),math.sin(yaw); r=float(msg.info.resolution)
        lx=(cols+0.5)*r; ly=(rows+0.5)*r
        self.occupied_world_xy=np.column_stack((msg.info.origin.position.x+c*lx-s*ly,
                                                msg.info.origin.position.y+s*lx+c*ly))

    def _on_status(self, msg: String) -> None:
        try:
            self.navigation_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.navigation_status = {"parse_error": msg.data}

    def ensure_motion_allowed(self) -> None:
        if (self.last_visual_pose_time is None
                or time.monotonic() - self.last_visual_pose_time > self.args.visual_pose_timeout):
            raise RuntimeError(
                "RTAB-Map visual odometry/TF stale for "
                f"{(time.monotonic() - self.last_visual_pose_time) if self.last_visual_pose_time else float('inf'):.2f}s"
            )
        if self.navigation_status is not None:
            if self.navigation_status.get("collision_stop_latched"):
                detail = self.navigation_status.get("first_collision") or {}
                raise RuntimeError(
                    "Mission failed: robot-environment collision detected "
                    f"(robot={detail.get('robot_geom_name')}, "
                    f"environment={detail.get('environment_geom_name')})"
                )
            if not self.navigation_status.get("motion_allowed", True):
                raise RuntimeError(
                    "Navigation safety interlock stopped motion: "
                    + json.dumps(self.navigation_status, ensure_ascii=False)
                )
        self.ensure_motion_progress()

    def _reset_motion_stall_tracking(self) -> None:
        now = time.monotonic()
        self._mission_has_moved = False
        self._refresh_motion_stall_reference(now)

    def _refresh_motion_stall_reference(self, now: float | None = None) -> None:
        """Reset the stall clock without clearing mission movement history."""
        if now is None:
            now = time.monotonic()
        self._stall_reference_time = now
        self._stall_reference_pose = (
            tuple(float(value) for value in self.pose)
            if self.pose is not None else None
        )

    def ensure_motion_progress(self) -> None:
        """Fail the mission once the robot stops moving before reaching C."""
        if (
            self.args.plan_only
            or self._motion_stall_suspended
            or self.mission_started_monotonic is None
            or "C" in self.coordinate_reach
        ):
            return
        now = time.monotonic()
        if getattr(self, "current_leg_name", "") == "C" and self.pose is not None:
            c_target = getattr(self, "task_points", {}).get("C")
            if c_target is not None:
                c_dist = self.task_distance(self.coverage_target("C", c_target))
                if c_dist <= VALIDATION_C_STALL_GRACE_M:
                    self._refresh_motion_stall_reference(now)
                    return
        if self._stall_reference_time is None:
            self._refresh_motion_stall_reference(now)
            return
        speed = abs(float(self.current_odom_speed))
        pose_delta = 0.0
        if self.pose is not None and self._stall_reference_pose is not None:
            pose_delta = math.hypot(
                self.pose[0] - self._stall_reference_pose[0],
                self.pose[1] - self._stall_reference_pose[1],
            )
        moving = (
            speed >= float(self.args.motion_stall_speed_mps)
            or pose_delta >= float(self.args.motion_stall_displacement_m)
        )
        if moving:
            self._mission_has_moved = True
            self._stall_reference_time = now
            if self.pose is not None:
                self._stall_reference_pose = tuple(float(value) for value in self.pose)
            return
        if not self._mission_has_moved:
            startup_limit = (
                float(self.args.motion_stall_startup_grace_s)
                + float(self.args.motion_stall_timeout_s)
            )
            if now - self.mission_started_monotonic <= startup_limit:
                return
            raise RuntimeError(
                "robot never started moving before mission stall timeout "
                f"({startup_limit:.1f}s)"
            )
        stalled_for = now - self._stall_reference_time
        if stalled_for >= float(self.args.motion_stall_timeout_s):
            raise MotionStalled(
                "robot motion stopped for "
                f"{stalled_for:.1f}s (speed={speed:.3f} m/s, "
                f"displacement={pose_delta:.3f} m)"
            )

    def _on_odom(self, msg: Odometry) -> None:
        # Formal navigation consumes the live global TF directly. RTAB-Map
        # owns map->odom and wheel odometry owns odom->base_link; composing
        # the Odometry payload ourselves can race TF correction updates and
        # silently leave the mission in the local odom frame.
        self.current_odom_speed = float(msg.twist.twist.linear.x)
        if self.args.truth_pose_topic:
            return
        try:
            # Consume only the canonical, metadata-aligned map->base_link TF.
        # The wheel odometry message and RTAB-Map correction can arrive on
        # different callbacks; use the newest complete TF chain.
            # In this simulator the latter can legitimately lag the message by
            # a few hundred milliseconds.  Looking up the exact message stamp
            # therefore causes deterministic ``future extrapolation`` errors.
            # Use the newest available map-frame TF and record its age instead;
            # the visual-pose timeout below still fail-stops stale localization.
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.15))
            tf_age = (self.get_clock().now() - rclpy.time.Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
            if tf_age > self.args.max_tf_age:
                raise RuntimeError(f"map->base_link TF is stale ({tf_age:.3f}s)")
            t, q = tf.transform.translation, tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                             1.0 - 2.0 * (q.y*q.y + q.z*q.z))
            raw = (float(t.x), float(t.y), float(yaw))
            if self.pose is not None:
                jump = math.hypot(raw[0] - self.pose[0], raw[1] - self.pose[1])
                yaw_jump = abs(math.atan2(math.sin(raw[2] - self.pose[2]),
                                          math.cos(raw[2] - self.pose[2])))
                if self.motion_started and (jump > self.args.max_pose_step or
                                             yaw_jump > self.args.max_yaw_step):
                    self.pose_jump_rejections += 1
                    self.get_logger().warning(
                        "rejecting visual pose jump %.3fm/%.1fdeg during motion" %
                        (jump, math.degrees(yaw_jump)))
                    return
                alpha = self.pose_filter_alpha
                dyaw = math.atan2(math.sin(raw[2] - self.pose[2]),
                                  math.cos(raw[2] - self.pose[2]))
                raw = (self.pose[0] + alpha * (raw[0] - self.pose[0]),
                       self.pose[1] + alpha * (raw[1] - self.pose[1]),
                       self.pose[2] + alpha * dyaw)
            self.last_raw_pose = (float(t.x), float(t.y), float(yaw))
            self.pose = raw
            self.pose_update_count += 1
            self.last_visual_pose_time = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"visual odometry TF unavailable: {exc}")

    def _on_truth_pose(self, msg: PoseStamped) -> None:
        """Use authorized root truth transformed by the measured world→map SE(2)."""
        q = msg.pose.orientation
        world_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        tx, ty, theta = self.args.truth_world_to_map
        c, s = math.cos(theta), math.sin(theta)
        world_x = float(msg.pose.position.x)
        world_y = float(msg.pose.position.y)
        self.pose = (
            c * world_x - s * world_y + tx,
            s * world_x + c * world_y + ty,
            math.atan2(
                math.sin(world_yaw + theta), math.cos(world_yaw + theta)
            ),
        )
        self.last_raw_pose = self.pose
        self.pose_update_count += 1
        self.last_visual_pose_time = time.monotonic()

    def _on_truth_body_link_pose(self, msg: PoseStamped) -> None:
        """Official scorer samples body_link1; track its map XY when published."""
        q = msg.pose.orientation
        world_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        tx, ty, theta = self.args.truth_world_to_map
        c, s = math.cos(theta), math.sin(theta)
        world_x = float(msg.pose.position.x)
        world_y = float(msg.pose.position.y)
        self.truth_body_link_map_xy = (
            c * world_x - s * world_y + tx,
            s * world_x + c * world_y + ty,
        )
        self.last_truth_body_link_time = time.monotonic()
        # body_link yaw unused; scoring is XY distance only.
        _ = world_yaw

    def _publish_twist(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)
        self.last_linear_command = linear
        self.last_angular_command = angular
        self.last_command_time = time.monotonic()
        self.audit_path_yaw = None

    def _predict_command_path(
        self,
        linear: float,
        angular: float,
        *,
        straight_audit: bool,
    ) -> list[tuple[float, float, float]]:
        if straight_audit:
            return predict_straight_path(
                self.pose,
                float(self.audit_path_yaw),
                linear,
                self.args.command_safety_horizon,
            )
        return predict_unicycle_path(
            self.pose, linear, angular, self.args.command_safety_horizon,
        )

    def _try_c_leg_safety_fallback(
        self,
        linear: float,
        angular: float,
        *,
        audit_linear: float,
        straight_audit: bool,
        min_clear: float,
    ) -> tuple[float, float] | None:
        """No C-only speed substitute: keep the path-tracking command."""
        del linear, angular, audit_linear, straight_audit, min_clear
        return None

    def _mission_elapsed_s(self) -> float | None:
        if self.mission_started_monotonic is None:
            return None
        return time.monotonic() - self.mission_started_monotonic

    def _cruise_ramp_elapsed_s(self) -> float | None:
        if self.cruise_ramp_started_monotonic is None:
            return None
        return time.monotonic() - self.cruise_ramp_started_monotonic

    def _ensure_cruise_ramp_started(self, linear: float) -> None:
        if self.cruise_ramp_started_monotonic is not None:
            return
        if abs(float(linear)) < OFFICIAL_SCORER_START_MOVE_MIN_MPS:
            return
        self.cruise_ramp_started_monotonic = time.monotonic()

    def _startup_ramp_elapsed_s(self) -> float | None:
        return self._cruise_ramp_elapsed_s()

    def _startup_ramp_scale(self) -> float:
        return startup_ramp_scale(self._startup_ramp_elapsed_s())

    def _apply_startup_ramp(self, linear: float) -> float:
        """Ramp from departure floor to cruise from the first forward command."""
        self._ensure_cruise_ramp_started(linear)
        return apply_startup_ramp(
            linear,
            elapsed_s=self._startup_ramp_elapsed_s(),
            plateau_s=VALIDATION_STARTUP_PLATEAU_S,
        )

    def _score_aware_speed_cap(
        self,
        peak_val: float,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
    ) -> float:
        if next_scored >= len(scored):
            return peak_val
        name, task_target = scored[next_scored]
        site = self.coverage_target(name, task_target)
        distance = self.scoring_distance_to_site(site)
        cap = float(peak_val)
        if name in ("A", "B") and distance <= OFFICIAL_REGION_RADIUS_M * 1.2:
            if distance > OFFICIAL_CHECKPOINT_DEEP_RADIUS_M:
                cap = min(cap, scoring_speed_policy(
                    phase="checkpoint_zone",
                    contract_speed_mps=float(self.args.speed),
                    speed_cap_mps=cap,
                    checkpoint_distance_m=distance,
                ))
        if name == "C" and distance <= OFFICIAL_REGION_RADIUS_M * 1.8:
            cap = min(cap, VALIDATION_C_TERMINAL_CAPTURE_CAP_MPS)
        return cap

    def _checkpoint_flythrough_override(
        self,
        *,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        path: np.ndarray,
        path_index: int,
        limit_index: int,
        speed_cap: float,
        x: float,
        y: float,
        yaw: float,
        signed_cross_track: float,
    ) -> tuple[float, float, str] | None:
        if next_scored >= len(scored):
            return None
        name, task_target = scored[next_scored]
        if name not in ("A", "B"):
            return None
        site = self.coverage_target(name, task_target)
        distance = self.scoring_distance_to_site(site)
        return self.flythrough_checkpoint_command(
            checkpoint_target=site,
            checkpoint_distance=distance,
            checkpoint_tol=self.coordinate_tolerance(name),
            is_terminal_leg=False,
            path=path,
            path_index=path_index,
            limit_index=limit_index,
            speed_cap=speed_cap,
            x=x,
            y=y,
            yaw=yaw,
            signed_cross_track=signed_cross_track,
        )

    def _command_slew_elapsed_s(self) -> float | None:
        return self._startup_ramp_elapsed_s()

    def command(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        *,
        audit_path_yaw: float | None = None,
    ) -> bool:
        if audit_path_yaw is not None:
            self.audit_path_yaw = float(audit_path_yaw)
        if abs(linear) <= 1e-6 and abs(angular) <= 1e-6:
            self.last_linear_command = 0.0
            self.last_angular_command = 0.0
            self.last_command_time = time.monotonic()
        else:
            now = time.monotonic()
            dt = max(0.01, min(0.20, now - self.last_command_time))
            max_da = self.args.steering_ramp_rate * dt
            angular = float(np.clip(
                angular,
                self.last_angular_command - max_da,
                self.last_angular_command + max_da,
            ))
            elapsed = self._command_slew_elapsed_s()
            accel, jerk = validation_command_slew_limits(
                elapsed,
                base_accel_mps2=float(self.args.profile_linear_accel),
                base_jerk_mps3=float(self.args.profile_linear_jerk),
            )
            max_dv = min(accel * dt, jerk * dt * dt + accel * dt)
            linear = float(np.clip(
                linear,
                self.last_linear_command - max_dv,
                self.last_linear_command + max_dv,
            ))
            self.last_linear_command = linear
            self.last_angular_command = angular
            self.last_command_time = now
        msg = Twist()
        if ((abs(linear) > 1e-6 or abs(angular) > 1e-6)
                and self.pose is not None and self.occupied_world_xy is not None):
            audit_linear = float(linear)
            current_speed = abs(float(self.current_odom_speed))
            straight_audit = self.audit_path_yaw is not None
            if audit_linear > current_speed + 0.25:
                # Audit the near-term footprint at the current motion rate, not
                # the full profile cap.  A 20 m/s ceiling otherwise rejects every
                # command while the chassis is still crawling and the robot stalls.
                audit_linear = min(
                    audit_linear,
                    max(current_speed * 2.5 + 0.60, 0.60),
                )
            if straight_audit:
                predicted = predict_straight_path(
                    self.pose,
                    float(self.audit_path_yaw),
                    audit_linear,
                    self.args.command_safety_horizon,
                )
            else:
                predicted = predict_unicycle_path(
                    self.pose, audit_linear, angular, self.args.command_safety_horizon
                )
            safe, report = swept_clear(
                predicted, self.occupied_world_xy, self.safety_footprint,
                self.args.command_safety_clearance,
            )
            self.command_safety_checks += 1
            self.last_command_safety_report = report
            if not safe and not getattr(self, "safety_warning_emitted", False):
                self.safety_warning_emitted = True
                self.get_logger().warning(
                    "predicted footprint clearance below audit threshold; publishing command "
                    "because safety stop is disabled: " + json.dumps(report)
                )
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)
        self.audit_path_yaw = None
        return True

    def settle_checkpoint(self, name: str, target: tuple[float, float]) -> None:
        """Hold still at a scored checkpoint (A/B dwell or optional audit settle)."""
        dwell_s = checkpoint_settle_duration_s(
            name,
            ab_checkpoint_dwell_s=float(
                getattr(self.args, "ab_checkpoint_dwell_s", VALIDATION_AB_CHECKPOINT_DWELL_S)
            ),
            checkpoint_settle_s=float(self.args.checkpoint_settle),
        )
        if dwell_s <= 0.0:
            return
        started = time.time()
        deadline = time.monotonic() + dwell_s
        self._motion_stall_suspended = True
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                self.command()
                rclpy.spin_once(self, timeout_sec=0.05)
                self.ensure_motion_allowed()
        finally:
            self._motion_stall_suspended = False
            self._refresh_motion_stall_reference()
        finished = time.time()
        self.checkpoint_intervals.append({
            "name": name,
            "target_map_xy": [float(target[0]), float(target[1])],
            "settle_started_unix": started,
            "settle_finished_unix": finished,
            "settle_duration_s": finished - started,
            "final_visual_map_pose": (
                [float(value) for value in self.pose] if self.pose is not None else None
            ),
        })
        self.get_logger().info(
            f"checkpoint {name} audit window complete ({finished-started:.2f}s)"
        )

    def _hold_zero_for_local_replan(self, duration_s: float) -> None:
        """Zero-command dwell so the next local plan starts from a settled pose."""
        if duration_s <= 0.0:
            return
        deadline = time.monotonic() + float(duration_s)
        self._motion_stall_suspended = True
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                self.command()
                rclpy.spin_once(self, timeout_sec=0.02)
        finally:
            self._motion_stall_suspended = False
            self._refresh_motion_stall_reference()

    def dwell_after_robot_start(self) -> None:
        """Hold zero command after robot_started.marker before route motion."""
        dwell_s = float(
            getattr(self.args, "post_robot_start_dwell_s", VALIDATION_POST_ROBOT_START_DWELL_S)
        )
        if dwell_s <= 0.0:
            return
        self.get_logger().info(
            f"robot started; holding zero for {dwell_s:.3f}s before route motion"
        )
        self._hold_zero_for_local_replan(dwell_s)

    def begin_route_motion_clock(self) -> None:
        """Start mission timing after post-start dwell (start_move / stall grace)."""
        self.mission_started_monotonic = time.monotonic()
        self._reset_motion_stall_tracking()

    @staticmethod
    def waypoint_goal_yaw(name: str, current_yaw: float | None = None) -> float:
        """Compatibility fallback without waypoint-name-specific headings."""
        del name
        return float(current_yaw) if current_yaw is not None else 0.0

    @staticmethod
    def geometric_goal_yaw(
        current_pose,
        target,
        next_target=None,
    ) -> float:
        """Infer terminal heading from adjacent route geometry."""
        if next_target is not None:
            dx = float(next_target[0]) - float(target[0])
            dy = float(next_target[1]) - float(target[1])
        else:
            dx = float(target[0]) - float(current_pose[0])
            dy = float(target[1]) - float(current_pose[1])
        if math.hypot(dx, dy) <= 1e-9:
            return float(current_pose[2])
        return math.atan2(dy, dx)

    def is_pass_through(self, start, mid, end) -> bool:
        """True when ``mid`` can be scored while driving ``start→end``."""
        radius = max(0.20, self.args.waypoint_coordinate_tolerance * 0.70)
        if point_to_segment_distance(mid, start, end) > radius:
            return False
        start_gap = math.hypot(float(mid[0]) - float(start[0]), float(mid[1]) - float(start[1]))
        end_gap = math.hypot(float(mid[0]) - float(end[0]), float(mid[1]) - float(end[1]))
        span = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        return start_gap > radius and end_gap > radius and span > 2.0 * radius

    def navigation_goal(
        self,
        site: tuple[float, float],
        inbound_pose,
        *,
        terminal: bool,
    ) -> tuple[float, float]:
        """Plan every scored site at its map coordinate; trim C after planning."""
        del inbound_pose, terminal
        return (float(site[0]), float(site[1]))

    @staticmethod
    def split_path_by_sites(
        points: list[tuple[float, float, float]],
        sites: list[tuple[str, tuple[float, float]]],
    ) -> list[tuple[str, list[tuple[float, float, float]]]]:
        """Cut one planned polyline into per-site audit slices."""
        if not points or not sites:
            return []
        if len(sites) == 1:
            return [(sites[0][0], list(points))]
        slices: list[tuple[str, list[tuple[float, float, float]]]] = []
        start = 0
        for name, xy in sites[:-1]:
            remaining = points[start:]
            nearest = start + min(
                range(len(remaining)),
                key=lambda index: math.hypot(
                    remaining[index][0] - float(xy[0]),
                    remaining[index][1] - float(xy[1]),
                ),
            )
            nearest = min(max(nearest, start + 1), len(points) - 1)
            slices.append((name, list(points[start:nearest + 1])))
            start = nearest
        tail = list(points[start:])
        if len(tail) < 2:
            tail = list(points[max(0, len(points) - 2):])
        slices.append((sites[-1][0], tail))
        return slices

    def coordinate_tolerance(self, name: str) -> float:
        return (
            self.args.final_coordinate_tolerance
            if name == "C" else self.args.waypoint_coordinate_tolerance
        )

    def _uses_segmented_speed_profile(self) -> bool:
        return bool(
            getattr(self.args, "use_hybrid_theta_mppi", False)
            or getattr(self.args, "use_nav2_controller", False)
        )

    def _build_nav2_leg_speed_profile(self, ordered: list[str]) -> None:
        """Build a continuous cruise profile without terminal arrival braking."""
        combined_profile: list[dict] = []
        arc_offset = 0.0
        join_speed: float | None = None
        for name in ordered:
            segment = self.planned_paths[name]
            segment_profile = build_path_speed_profile(
                segment,
                self.occupied_world_xy,
                self.safety_footprint,
                **self.profile_speed_kwargs(
                    stop_at_end=False,
                    terminal_arrival=False,
                    depart_from_rest=not combined_profile,
                ),
                join_speed_mps=join_speed,
                handoff_geometry=name != ordered[-1],
            )
            for index, item in enumerate(segment_profile):
                if index == 0 and combined_profile:
                    continue
                entry = dict(item)
                entry["arc_length_m"] = arc_offset + float(item["arc_length_m"])
                entry["index"] = len(combined_profile)
                combined_profile.append(entry)
            arc_offset = float(combined_profile[-1]["arc_length_m"])
            join_speed = float(combined_profile[-1]["speed_cap_mps"])
        self.path_speed_profile = combined_profile

    def flythrough_checkpoint_command(
        self,
        *,
        checkpoint_target: tuple[float, float] | None,
        checkpoint_distance: float,
        checkpoint_tol: float,
        is_terminal_leg: bool,
        path,
        path_index: int,
        limit_index: int,
        speed_cap: float,
        x: float,
        y: float,
        yaw: float,
        signed_cross_track: float,
    ) -> tuple[float, float, str] | None:
        """Steer A/B fly-throughs so body_link1 stays inside the scorer circle."""
        if checkpoint_target is None or is_terminal_leg:
            return None
        if checkpoint_distance > CHECKPOINT_ZONE_RADIUS_M:
            return None
        aligned_target = scorer_aligned_root_target(checkpoint_target, yaw)
        site_alpha = self.wrap_angle(
            math.atan2(aligned_target[1] - y, aligned_target[0] - x) - yaw
        )
        path_yaw = float(path[path_index, 2])
        heading_error = self.wrap_angle(path_yaw - yaw)
        path_curvature = float(self.path_speed_profile[limit_index]["curvature_1pm"])
        linear = scoring_speed_policy(
            phase="checkpoint_zone",
            contract_speed_mps=float(self.args.speed),
            speed_cap_mps=float(speed_cap),
            checkpoint_distance_m=float(checkpoint_distance),
            checkpoint_tol_m=float(checkpoint_tol),
            checkpoint_capture_min_mps=float(self.args.checkpoint_capture_min_speed),
        )
        steer_gain = 2.5 if checkpoint_distance <= checkpoint_tol else 1.5
        if checkpoint_distance <= OFFICIAL_REGION_RADIUS_M * 1.05:
            steer_gain = 3.2 if checkpoint_distance <= checkpoint_tol else 2.2
        if checkpoint_distance <= OFFICIAL_REGION_RADIUS_M:
            steer_gain = max(steer_gain, 3.8)
        site_alpha_gain = 1.00
        if checkpoint_distance <= OFFICIAL_REGION_RADIUS_M:
            site_alpha_gain = 2.20
        angular = float(np.clip(
            linear * path_curvature
            + 1.20 * heading_error
            + 1.00 * signed_cross_track
            + steer_gain * site_alpha_gain * site_alpha,
            -self.args.max_angular,
            self.args.max_angular,
        ))
        return linear, angular, "checkpoint_flythrough"

    def _terminal_capture_command(
        self,
        site: tuple[float, float],
        x: float,
        y: float,
        yaw: float,
        speed_cap: float,
    ) -> tuple[float, float, str]:
        """Steer the root toward the site so body_link1 enters the 0.60 m circle."""
        distance = self.scoring_distance_to_site(site)
        approach_cap = float(speed_cap)
        if distance <= OFFICIAL_REGION_RADIUS_M * 1.5:
            approach_cap = min(
                approach_cap,
                VALIDATION_C_TERMINAL_CAPTURE_CAP_MPS,
            )
        offset_x, offset_y = self.args.judgement_offset_xy
        if abs(offset_x) < 1e-6 and abs(offset_y) < 1e-6:
            target = site
        else:
            target = scorer_aligned_root_target(site, yaw)
        alpha = self.wrap_angle(
            math.atan2(target[1] - y, target[0] - x) - yaw
        )
        kappa_max = float(getattr(
            self.args, "max_path_curvature", VALIDATION_MAX_PATH_CURVATURE_1PM,
        ))
        # A steering base can only realise an arc, so command the pure-pursuit
        # curvature and derive the yaw rate from the speed actually issued.
        # Asking for a yaw rate the crawl speed cannot produce saturates the
        # steering and leaves the robot standing still short of the circle.
        lookahead_m = max(distance, 0.35)
        if abs(alpha) > math.pi / 2.0:
            # Behind the robot, sin(alpha) collapses back toward zero and pure
            # pursuit would drive straight past; commit to the tightest arc.
            curvature = math.copysign(kappa_max, alpha)
        else:
            curvature = float(np.clip(
                2.0 * math.sin(alpha) / lookahead_m, -kappa_max, kappa_max,
            ))
        if distance > 0.35 and abs(alpha) > 0.40:
            linear = min(approach_cap, float(self.args.speed))
            return linear, linear * curvature, "terminal_align"
        linear = min(approach_cap, float(self.args.speed))
        return linear, linear * curvature, "terminal_capture"

    def _crawl_scored_checkpoint_until(
        self,
        name: str,
        target: tuple[float, float],
        deadline: float,
        *,
        max_duration_s: float = 12.0,
    ) -> bool:
        """Final crawl into the official scorer circle when path tracking undershoots."""
        site = self.coverage_target(name, target)
        crawl_deadline = min(deadline, time.monotonic() + max_duration_s)
        speed_cap = float(self.args.speed)
        while rclpy.ok() and time.monotonic() < crawl_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.ensure_motion_allowed()
            if self.pose is None:
                continue
            if self.checkpoint_satisfied(name, site):
                self.mark_reached(name, site)
                self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
                self.checkpoint_pass_speeds_detail[name] = {
                    "odom_mps": abs(self.current_odom_speed),
                    "commanded_mps": 0.0,
                }
                if name not in self.leg_timestamps:
                    self.leg_timestamps[name] = time.monotonic()
                self.get_logger().info(
                    f"terminal crawl captured {name} at "
                    f"error={self.task_distance(site):.3f}m"
                )
                return True
            x, y, yaw = self.pose
            distance = self.task_distance(site)
            linear, angular, control_mode = self._terminal_capture_command(
                site, x, y, yaw, speed_cap,
            )
            judgement_pose = self.judgement_pose()
            self.samples.append((
                time.time(), name, *judgement_pose, distance, linear, angular
            ))
            self._publish_motion_command(
                linear,
                angular,
                control_mode=control_mode,
                speed_cap=speed_cap,
                cap_reason="terminal_capture",
                cross_track_m=0.0,
                audit_path_yaw=None,
            )
        return False

    def _official_circle_capture_and_dwell(
        self,
        name: str,
        task_target: tuple[float, float],
        deadline: float,
    ) -> None:
        """Drive body_link1 into the scorer circle and dwell for official credit."""
        if name != "C":
            return
        site = self.coverage_target(name, task_target)
        if self.scoring_distance_to_site(site) <= OFFICIAL_CHECKPOINT_DEEP_RADIUS_M:
            return
        capture_target = OFFICIAL_REGION_RADIUS_M * OFFICIAL_CIRCLE_CAPTURE_RADIUS_SCALE
        capture_deadline = min(deadline, time.monotonic() + 15.0)
        speed_cap = min(
            float(self.args.speed),
            VALIDATION_C_TERMINAL_CAPTURE_CAP_MPS,
        )
        while rclpy.ok() and time.monotonic() < capture_deadline:
            if self.scoring_distance_to_site(site) <= capture_target:
                break
            rclpy.spin_once(self, timeout_sec=0.02)
            self.ensure_motion_allowed()
            if self.pose is None:
                continue
            cx, cy, cyaw = self.pose
            crawl_linear, crawl_angular, crawl_mode = (
                self._terminal_capture_command(site, cx, cy, cyaw, speed_cap)
            )
            self._publish_motion_command(
                crawl_linear,
                crawl_angular,
                control_mode=crawl_mode,
                speed_cap=speed_cap,
                cap_reason="official_circle_capture",
                cross_track_m=0.0,
                audit_path_yaw=None,
            )
        dwell_deadline = time.monotonic() + 0.8
        hold_break = OFFICIAL_REGION_RADIUS_M * 0.78
        while rclpy.ok() and time.monotonic() < dwell_deadline:
            if self.scoring_distance_to_site(site) > hold_break:
                break
            rclpy.spin_once(self, timeout_sec=0.02)
            self.ensure_motion_allowed()
            if self.pose is None:
                continue
            cx, cy, cyaw = self.pose
            crawl_linear, crawl_angular, crawl_mode = (
                self._terminal_capture_command(site, cx, cy, cyaw, speed_cap)
            )
            self._publish_motion_command(
                crawl_linear,
                crawl_angular,
                control_mode=crawl_mode,
                speed_cap=speed_cap,
                cap_reason="official_circle_dwell",
                cross_track_m=0.0,
                audit_path_yaw=None,
            )

    def judgement_pose(self) -> tuple[float, float, float] | None:
        """Map pose of the g1_omnipicker robot root used for task judgement."""
        if self.pose is None:
            return None
        x, y, yaw = self.pose
        offset_x, offset_y = self.args.judgement_offset_xy
        c, s = math.cos(yaw), math.sin(yaw)
        return (
            x + c * offset_x - s * offset_y,
            y + s * offset_x + c * offset_y,
            yaw,
        )

    def task_distance(self, target: tuple[float, float]) -> float:
        pose = self.judgement_pose()
        return math.inf if pose is None else self.xy_distance(pose, target)

    def scoring_distance_to_site(self, site: tuple[float, float]) -> float:
        """Distance used for official 0.60 m circle checks (body_link1 when known)."""
        if (
            self.truth_body_link_map_xy is not None
            and self.last_truth_body_link_time is not None
            and time.monotonic() - self.last_truth_body_link_time
            <= float(self.args.truth_pose_timeout)
        ):
            return self.xy_distance(self.truth_body_link_map_xy, site)
        return self.task_distance(site)

    def _truth_body_link_fresh(self) -> bool:
        return (
            self.truth_body_link_map_xy is not None
            and self.last_truth_body_link_time is not None
            and time.monotonic() - self.last_truth_body_link_time
            <= float(self.args.truth_pose_timeout)
        )

    def official_site_distance_m(self, site_name: str) -> float:
        """XY distance from scorer-aligned body proxy to the official site."""
        sites = self.task_points or {}
        if site_name not in sites or self.pose is None:
            return math.inf
        return self.task_distance(tuple(sites[site_name]))

    def _start_to_a_elapsed_ok(self) -> bool:
        return start_to_a_motion_elapsed_ok(self._mission_elapsed_s())

    def checkpoint_satisfied(self, name: str, target: tuple[float, float]) -> bool:
        """Pass when body_link1 (official scorer proxy) enters the 0.60 m circle."""
        site = self.coverage_target(name, target)
        if name == "C":
            in_circle = (
                self.scoring_distance_to_site(site) <= VALIDATION_C_CHECKPOINT_PASS_M
            )
        else:
            in_circle = self.scoring_distance_to_site(site) <= (
                OFFICIAL_REGION_RADIUS_M * OFFICIAL_CHECKPOINT_PASS_RADIUS_SCALE
            )
        if not in_circle:
            return False
        if name == "A" and not self._start_to_a_elapsed_ok():
            return False
        return True

    @staticmethod
    def xy_distance(first, second) -> float:
        """XY-plane Euclidean distance; deliberately ignore any Z component."""
        return math.hypot(float(first[0]) - float(second[0]),
                          float(first[1]) - float(second[1]))

    def target_reached(self, name: str, target: tuple[float, float]) -> bool:
        """Judge A/B/C from robot-center XY distance, never footprint overlap."""
        return self.task_distance(target) <= self.coordinate_tolerance(name)

    def coverage_target(self, name: str, fallback: tuple[float, float]) -> tuple[float, float]:
        return tuple(self.task_points.get(name, fallback))

    def mark_reached(self, name: str, target: tuple[float, float]) -> None:
        task = self.coverage_target(name, target)
        judgement_pose = self.judgement_pose()
        error = self.task_distance(task)
        official_error = self.scoring_distance_to_site(task)
        self.coordinate_reach[name] = {
            "task_xy": [float(task[0]), float(task[1])],
            "visual_base_pose": [float(v) for v in self.pose],
            "judgement_entity": "g1_omnipicker",
            "judgement_map_pose": [float(v) for v in judgement_pose],
            "center_error_m": float(error),
            "official_scoring_error_m": float(official_error),
            "tolerance_m": self.coordinate_tolerance(name),
            "reached": bool(error <= self.coordinate_tolerance(name)),
        }

    def wait_for_pose(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and (self.pose is None or self.occupied_world_xy is None) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None or self.occupied_world_xy is None:
            raise RuntimeError("No valid RTAB-Map visual pose/map received")
        # A TF chain can be technically queryable before RTAB-Map has applied
        # the saved-map correction.  Reject that local-odom pose instead of
        # allowing Nav2 to plan from the wrong global origin.
        ix, iy, _ = self.args.initial_map_pose
        if math.hypot(self.pose[0] - ix, self.pose[1] - iy) > self.args.initial_pose_tolerance:
            self.get_logger().warning(
                "initial visual pose is %.3fm from expected map origin; waiting for RTAB-Map correction"
                % math.hypot(self.pose[0] - ix, self.pose[1] - iy))
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.pose is not None and math.hypot(self.pose[0] - ix, self.pose[1] - iy) <= self.args.initial_pose_tolerance:
                    return
            raise RuntimeError(
                "RTAB-Map map->base_link did not converge to the saved-map initial pose: "
                f"pose={self.pose}, expected=({ix:.3f},{iy:.3f})")

    def wait_for_fresh_visual_pose(self, timeout: float = 10.0) -> None:
        """Wait for a continuously fresh global TF before enabling motion.

        During startup RTAB-Map's dynamic ``map->odom`` correction and wheel
        odometry can become available in a different order. ``wait_for_pose``
        may therefore observe one valid historical sample while the next
        callbacks still report a stale TF.  Do not start the mission until a
        callback has refreshed the pose within the same freshness budget used
        by the runtime interlock.
        """
        deadline = time.monotonic() + timeout
        start_updates = self.pose_update_count
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.pose is not None and self.last_visual_pose_time is not None
                    and time.monotonic() - self.last_visual_pose_time
                    <= self.args.visual_pose_timeout
                    and self.pose_update_count - start_updates >= self.args.startup_pose_samples):
                self.get_logger().info("visual localization is fresh; enabling navigation")
                return
        age = (time.monotonic() - self.last_visual_pose_time
               if self.last_visual_pose_time is not None else float("inf"))
        raise RuntimeError(
            "RTAB-Map visual localization did not become fresh during startup "
            f"(age={age:.2f}s, timeout={timeout:.1f}s)")

    def ensure_path_safe(self, name: str, path) -> None:
        safe, report = swept_clear(
            path, self.occupied_world_xy, self.safety_footprint,
            self.args.path_safety_clearance,
        )
        self.path_safety_reports[name] = report
        if not safe:
            raise RuntimeError(
                f"{name} path rejected by swept-footprint safety filter: {report}"
            )
        min_clearance = report.get("minimum_clearance_m")
        if (min_clearance is not None
                and float(min_clearance) < VALIDATION_MIN_PATH_CLEARANCE_M):
            raise RuntimeError(
                f"{name} path minimum clearance {min_clearance:.3f}m "
                f"< {VALIDATION_MIN_PATH_CLEARANCE_M:.3f}m"
            )
        if name in ("A", "B") and "A" in self.task_points and "B" in self.task_points:
            lateral = max_lateral_from_chord(
                path,
                tuple(self.task_points["A"]),
                tuple(self.task_points["B"]),
            )
            report["max_lateral_from_chord_m"] = float(lateral)
            if lateral > VALIDATION_MAX_CORRIDOR_LATERAL_M:
                raise RuntimeError(
                    f"{name} path lateral offset {lateral:.3f}m "
                    f"> {VALIDATION_MAX_CORRIDOR_LATERAL_M:.3f}m from A→B chord"
                )

    def record_execution_trace(
        self,
        *,
        linear: float,
        angular: float,
        speed_cap: float,
        limiting_factor: str,
        cross_track_m: float = 0.0,
    ) -> None:
        if self.pose is None:
            return
        x, y, yaw = self.pose
        sites = self.task_points or {}
        judgement = self.judgement_pose()
        dist_a = (
            self.scoring_distance_to_site(tuple(sites["A"]))
            if "A" in sites else math.inf
        )
        dist_b = (
            self.scoring_distance_to_site(tuple(sites["B"]))
            if "B" in sites else math.inf
        )
        dist_c = (
            self.scoring_distance_to_site(tuple(sites["C"]))
            if "C" in sites else math.inf
        )
        radius = OFFICIAL_REGION_RADIUS_M
        self.execution_trace.append({
            "t": time.time(),
            "leg": self.current_leg_name,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "odom_v": float(abs(self.current_odom_speed)),
            "cmd_v": float(linear),
            "cmd_w": float(angular),
            "control_mode": self.current_control_mode,
            "cross_track_m": float(cross_track_m),
            "dist_A": float(dist_a),
            "dist_B": float(dist_b),
            "dist_C": float(dist_c),
            "speed_cap": float(speed_cap),
            "limiting_factor": str(limiting_factor),
            "in_official_circle_A": dist_a <= radius,
            "in_official_circle_B": dist_b <= radius,
            "in_official_circle_C": dist_c <= radius,
        })

    def build_mission_analysis(self, *, elapsed_s: float, status: str) -> dict:
        control_modes: dict[str, int] = {}
        for row in self.execution_trace:
            mode = str(row.get("control_mode") or "unknown")
            control_modes[mode] = control_modes.get(mode, 0) + 1
        site_frames = {"A": 0, "B": 0, "C": 0}
        for row in self.execution_trace:
            for site in site_frames:
                if row.get(f"in_official_circle_{site}"):
                    site_frames[site] += 1
        min_dist = {"A": math.inf, "B": math.inf, "C": math.inf}
        for row in self.execution_trace:
            for site in min_dist:
                key = f"dist_{site}"
                if key in row:
                    min_dist[site] = min(min_dist[site], float(row[key]))
        leg_durations: dict[str, float | None] = {}
        stamps = sorted(self.leg_timestamps.items(), key=lambda item: item[1])
        for index, (name, stamp) in enumerate(stamps):
            if index + 1 < len(stamps):
                leg_durations[name] = stamps[index + 1][1] - stamp
            else:
                leg_durations[name] = elapsed_s - (stamp - (self.mission_started_monotonic or stamp))
        return {
            "format": "task1_mission_analysis_v1",
            "status": status,
            "elapsed_s": float(elapsed_s),
            "leg_durations_s": leg_durations,
            "control_mode_counts": control_modes,
            "estimated_frames_in_official_circle": site_frames,
            "min_dist_to_official_site_m": {
                key: (None if value is math.inf else float(value))
                for key, value in min_dist.items()
            },
            "checkpoint_pass_speeds_mps": self.checkpoint_pass_speeds,
            "checkpoint_pass_speeds_detail": self.checkpoint_pass_speeds_detail,
            "route_geometry_audits": self.route_geometry_audits,
            "path_safety_reports": self.path_safety_reports,
            "body_link_forward_offset_m": BODY_LINK_FORWARD_OFFSET_M,
        }

    def capture_arc_curvature(self, alpha: float, distance_m: float) -> float:
        """Pure-pursuit arc curvature toward a checkpoint, clipped to the base."""
        kappa_max = float(getattr(
            self.args, "max_path_curvature", VALIDATION_MAX_PATH_CURVATURE_1PM,
        ))
        lookahead_m = max(float(distance_m), 0.35)
        if abs(float(alpha)) > math.pi / 2.0:
            return math.copysign(kappa_max, float(alpha))
        return float(np.clip(
            2.0 * math.sin(float(alpha)) / lookahead_m, -kappa_max, kappa_max,
        ))

    def ackermann_command_envelope(
        self,
        linear: float,
        angular: float,
        *,
        speed_cap: float,
    ) -> tuple[float, float]:
        """Bind the commanded yaw rate to the speed the steering base is given."""
        kappa_max = float(getattr(
            self.args, "max_path_curvature", VALIDATION_MAX_PATH_CURVATURE_1PM,
        ))
        cap = min(float(speed_cap), float(self.args.speed))
        return ackermann_feasible_command(
            linear,
            angular,
            max_curvature_1pm=kappa_max,
            max_angular=float(self.args.max_angular),
            speed_cap_mps=cap,
        )

    def _profile_audit_yaw(self, path: np.ndarray, path_index: int) -> float:
        """Audit commanded motion along the planned path tangent, not a unicycle arc."""
        return float(path[min(path_index, len(path) - 1), 2])

    def _publish_motion_command(
        self,
        linear: float,
        angular: float,
        *,
        control_mode: str,
        speed_cap: float,
        cap_reason: str,
        cross_track_m: float,
        audit_path_yaw: float | None,
    ) -> None:
        self.current_control_mode = control_mode
        self.last_speed_cap = float(speed_cap)
        self.last_limiting_factor = str(cap_reason)
        self.last_cross_track_m = float(cross_track_m)
        linear, angular = self.ackermann_command_envelope(
            linear, angular, speed_cap=speed_cap,
        )
        linear = self._apply_startup_ramp(linear)
        published = self.command(linear, angular, audit_path_yaw=audit_path_yaw)
        if not published:
            self.command(0.0, 0.0, audit_path_yaw=audit_path_yaw)
            linear, angular = 0.0, 0.0
        self.record_execution_trace(
            linear=linear,
            angular=angular,
            speed_cap=speed_cap,
            limiting_factor=cap_reason,
            cross_track_m=cross_track_m,
        )

    def write_analysis_artifacts(self, output_dir: Path, analysis: dict) -> None:
        trace_path = output_dir / "execution_trace.csv"
        if self.execution_trace:
            fieldnames = list(self.execution_trace[0].keys())
            with trace_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.execution_trace)
        (output_dir / "mission_analysis.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8"
        )
        if self.route_geometry_audits:
            (output_dir / "route_geometry_audit.json").write_text(
                json.dumps({
                    "format": "task1_route_geometry_audit_v1",
                    "segments": self.route_geometry_audits,
                }, indent=2),
                encoding="utf-8",
            )

    def enable_motion(self) -> None:
        """Enable jump rejection only after startup localization has settled."""
        self.motion_started = True

    def make_continuous_path(self, path: np.ndarray) -> np.ndarray:
        """Ensure a newly planned leg starts at the current filtered pose."""
        if self.pose is None or len(path) == 0:
            return path
        current = np.asarray(self.pose, dtype=np.float64)
        if float(np.linalg.norm(path[0, :2] - current[:2])) > 0.05:
            path = np.vstack((current, path))
        else:
            path = path.copy()
            path[0, :] = current
        return path

    def write_plan_snapshot(self) -> None:
        temporary = self.args.output_dir / "planned_paths.json.tmp"
        temporary.write_text(json.dumps(self.planned_paths, indent=2), encoding="utf-8")
        temporary.replace(self.args.output_dir / "planned_paths.json")

    def drive_to(self, name: str, target: tuple[float, float], deadline: float) -> None:
        stable = 0
        tolerance = self.args.approach_tolerance if name.startswith("_") else self.args.tolerance
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.ensure_motion_allowed()
            x, y, yaw = self.pose
            dx, dy = target[0] - x, target[1] - y
            distance = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)
            error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
            # Reverse toward targets behind the chassis instead of making a
            # large forward circle. Steering error is measured in the chosen
            # travel direction so both forward and reverse converge.
            travel_sign = 1.0
            if abs(error) > math.pi / 2.0:
                travel_sign = -1.0
                error = math.atan2(math.sin(error - math.pi), math.cos(error - math.pi))
            linear = travel_sign * self.args.speed * max(0.25, math.cos(error))
            angular = travel_sign * max(
                -self.args.max_angular,
                min(self.args.max_angular, self.args.heading_gain * error),
            )
            if name == "_A_exit":
                angular = float(np.clip(angular, -0.22, 0.22))
            if name == "A":
                # From the southern staging point, saturated steering creates
                # a fixed-radius orbit around A. Use a shallow reverse/forward
                # arc so the footprint crosses the target without sweeping
                # back toward DB_04_A.
                linear = float(np.clip(linear, -0.05, 0.05))
                angular = float(np.clip(angular, -0.18, 0.18))
            if self.target_reached(name, self.coverage_target(name, target)):
                stable += 1
                self.command()
                if stable >= 5 and (
                    name != "A" or self._start_to_a_elapsed_ok()
                ):
                    self.mark_reached(name, self.coverage_target(name, target))
                    self.get_logger().info(f"checkpoint {name} reached at ({x:.3f}, {y:.3f})")
                    return
            else:
                stable = 0
                self.command(linear, angular)
            self.samples.append((time.time(), name, x, y, yaw, distance, linear, angular))
        raise TimeoutError(f"Timed out before reaching {name}: pose={self.pose}, target={target}")

    def retreat_along_a_trace(self, target: tuple[float, float], deadline: float) -> None:
        """Reverse only the forward portion of the proven A ingress trace."""
        controls = [(row[6], row[7]) for row in self.samples
                    if row[1] == "A" and math.isfinite(row[6]) and row[6] > 1e-4]
        if not controls:
            raise RuntimeError("forward A ingress trace is unavailable for safe retreat")
        started = time.monotonic()
        for linear, angular in reversed(controls):
            if not rclpy.ok() or time.monotonic() >= deadline:
                break
            rclpy.spin_once(self, timeout_sec=0.0)
            self.ensure_motion_allowed()
            x, y, yaw = self.pose
            distance = math.hypot(x-target[0], y-target[1])
            if distance >= 0.90:
                self.command()
                self.get_logger().info(
                    f"safe A trace retreat complete at ({x:.3f},{y:.3f}), "
                    f"distance_from_A={distance:.3f}m"
                )
                return
            reverse_linear = -min(abs(linear), self.args.speed * 0.45, 0.08)
            # The bridge interprets angular.z as yaw rate. Negating both v and
            # omega preserves the physical steering angle while reversing.
            reverse_angular = -angular
            self.command(reverse_linear, reverse_angular)
            self.samples.append((time.time(), "_A_exit", x, y, yaw, distance,
                                 reverse_linear, reverse_angular))
            time.sleep(0.02)
        self.command()
        raise RuntimeError(
            f"A trace retreat did not reach safe clearance after {time.monotonic()-started:.2f}s"
        )

    def planned_points(self, target: tuple[float, float], goal_yaw: float = 0.0,
                       start_yaw: float | None = None,
                       start_xy: tuple[float, float] | None = None) -> list[tuple[float, float, float]]:
        if not self.planner.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Nav2 /compute_path_to_pose action is unavailable")
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = "map"
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x, goal.goal.pose.position.y = target
        goal.goal.pose.orientation.z = math.sin(goal_yaw * 0.5)
        goal.goal.pose.orientation.w = math.cos(goal_yaw * 0.5)
        if start_yaw is not None:
            goal.use_start = True
            goal.start.header.frame_id = "map"
            goal.start.header.stamp = goal.goal.header.stamp
            sx, sy = start_xy if start_xy is not None else self.pose[:2]
            goal.start.pose.position.x = float(sx)
            goal.start.pose.position.y = float(sy)
            goal.start.pose.orientation.z = math.sin(start_yaw * 0.5)
            goal.start.pose.orientation.w = math.cos(start_yaw * 0.5)
        goal.planner_id = str(self.args.global_planner_id)
        wrapped = None
        # Retries only exist to cover costmap warm-up before the planner has
        # ever answered. Once a path came back, an infeasible goal is final:
        # re-searching the same start/goal pair burns seconds per heading
        # candidate during the route heading search.
        max_attempts = (
            1 if self._planner_produced_path
            else 2 if self._motion_stall_suspended else 3
        )
        for attempt in range(1, max_attempts + 1):
            future = self.planner.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            handle = future.result()
            if handle is not None and handle.accepted:
                result_future = handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
                wrapped = result_future.result()
                if (wrapped is not None and wrapped.result.error_code == 0
                        and wrapped.result.path.poses):
                    self._planner_produced_path = True
                    break
            self.get_logger().warning(f"Nav2 plan attempt {attempt}/{max_attempts} failed for {target}")
            if attempt >= max_attempts:
                break
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.2 if self._motion_stall_suspended else 1.0)
        if wrapped is None or wrapped.result.error_code != 0 or not wrapped.result.path.poses:
            code = wrapped.result.error_code if wrapped is not None else "rejected/timeout"
            raise RuntimeError(f"Nav2 failed to plan to {target}: error_code={code}")
        result = []
        for stamped in wrapped.result.path.poses:
            pose, q = stamped.pose, stamped.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            result.append((pose.position.x, pose.position.y, yaw))
        if self.args.global_planner_plugin in {"theta_star", "smac_2d"}:
            # Grid/any-angle planners return sparse poses whose orientation is
            # not the travel direction; rebuild both before profiling.
            result = densify_path(result, spacing=0.12)
            result = orient_path_for_ackermann(result)
        return result

    def planned_points_with_heading_search(
        self,
        target: tuple[float, float],
        preferred_yaw: float,
        start_pose,
        *,
        terminal_leg: bool = False,
        quick: bool = False,
    ) -> list[tuple[float, float, float]]:
        """Select the fastest safe terminal heading without route labels."""
        offsets = (
            0.0,
            math.pi / 4.0, -math.pi / 4.0,
            math.pi / 2.0, -math.pi / 2.0,
        ) if quick else (
            0.0,
            math.pi / 4.0, -math.pi / 4.0,
            math.pi / 2.0, -math.pi / 2.0,
            3.0 * math.pi / 4.0, -3.0 * math.pi / 4.0,
            math.pi,
        )
        yaw_candidates = [
            self.wrap_angle(preferred_yaw + offset) for offset in offsets
        ]
        if terminal_leg:
            # Arrive facing north into the C alcove instead of the 1.92 rad
            # SW-hugging Hybrid default that pins the last apex to the cabinet.
            for extra in (math.pi / 2.0, 1.70):
                if all(
                    abs(self.wrap_angle(extra - yaw)) > 0.12
                    for yaw in yaw_candidates
                ):
                    yaw_candidates.append(float(extra))
        feasible: list[tuple[
            float,
            list[tuple[float, float, float]],
            dict[str, float | bool] | None,
        ]] = []
        last_error: RuntimeError | None = None
        for yaw in yaw_candidates:
            offset = self.wrap_angle(yaw - preferred_yaw)
            try:
                points = self.planned_points(
                    target, yaw, start_pose[2], start_pose[:2]
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            metrics = terminal_path_metrics(points)
            if not bool(metrics["forward_only"]):
                continue
            if not terminal_leg and abs(offset) <= 1e-9:
                # A pass-through leg keeps its geometric goal yaw whenever that
                # yaw plans a forward-only path: every alternate carries a
                # >=1.5s offset penalty a corridor leg cannot win back.
                return points
            if terminal_leg:
                points, _widen = self._widen_terminal_c_path(points, target)
                metrics = terminal_path_metrics(points)
            profile = build_path_speed_profile(
                points,
                self.occupied_world_xy,
                self.safety_footprint,
                **self.profile_speed_kwargs(
                    stop_at_end=False,
                    terminal_arrival=False,
                ),
            )
            traversal_time = estimated_profile_time(
                profile,
                minimum_speed=self.args.minimum_profile_speed,
            )
            if terminal_leg:
                if not terminal_path_is_trackable(metrics):
                    continue
                minimum_clearance = min(
                    (
                        float(item["clearance_m"])
                        for item in profile
                        if math.isfinite(float(item["clearance_m"]))
                    ),
                    default=math.inf,
                )
                if minimum_clearance < self.args.profile_clearance_hard:
                    continue
                end_arc = float(profile[-1]["arc_length_m"]) if profile else 0.0
                apex_clearance = min(
                    (
                        float(item["clearance_m"])
                        for item in profile
                        if math.isfinite(float(item["clearance_m"]))
                        and float(item["arc_length_m"]) >= end_arc - 4.0
                    ),
                    default=minimum_clearance,
                )
                end_yaw = float(points[-1][2]) if points else preferred_yaw
                north_err = abs(self.wrap_angle(end_yaw - math.pi / 2.0))
                score = (
                    traversal_time
                    + 4.0 * float(metrics["end_vs_inbound_rad"])
                    + 2.0 * float(metrics["yaw_change_rad"])
                    + 2.0 * max(
                        0.0,
                        float(metrics["max_abs_curvature_1pm"]) - 1.5,
                    )
                    + 8.0 * max(
                        0.0,
                        float(self.args.profile_clearance_soft) - minimum_clearance,
                    )
                    + 24.0 * max(
                        0.0,
                        VALIDATION_C_TURN_TARGET_CLEARANCE_M - apex_clearance,
                    )
                    + 8.0 * north_err
                )
            else:
                score = traversal_time + 2.0 * abs(self.wrap_angle(offset))
            feasible.append((score, points, metrics if terminal_leg else None))
        if not feasible:
            raise last_error or RuntimeError(
                f"Nav2 found no feasible terminal heading for {target}"
            )
        selected = min(feasible, key=lambda item: item[0])
        if terminal_leg:
            self.get_logger().info(
                "selected forward terminal approach: "
                + json.dumps(selected[2], sort_keys=True)
            )
        return selected[1]

    @staticmethod
    def _lookahead_index(path: np.ndarray, position: np.ndarray, start: int, distance: float) -> int:
        """Find a forward path point without allowing nearest-point regressions."""
        nearest = start + int(np.argmin(np.linalg.norm(path[start:, :2] - position, axis=1)))
        index = nearest
        travelled = 0.0
        while index + 1 < len(path) and travelled < distance:
            travelled += float(np.linalg.norm(path[index + 1, :2] - path[index, :2]))
            index += 1
        return index

    def _path_curvature_preview(
        self,
        profile_index: int,
        preview_m: float = 1.5,
    ) -> float:
        """Peak |kappa| on the planned path within the next preview distance."""
        profile = getattr(self, "path_speed_profile", None) or []
        if not profile:
            return 0.0
        index = max(0, min(int(profile_index), len(profile) - 1))
        start_arc = float(profile[index]["arc_length_m"])
        peak = abs(float(profile[index]["curvature_1pm"]))
        for item in profile[index:]:
            if float(item["arc_length_m"]) - start_arc > float(preview_m):
                break
            peak = max(peak, abs(float(item["curvature_1pm"])))
        return peak

    def _effective_profile_lookahead(
        self,
        *,
        cross_track_m: float = 0.0,
        speed_mps: float | None = None,
        curvature_1pm: float = 0.0,
    ) -> float:
        speed = max(0.35, float(speed_mps if speed_mps is not None else self.args.speed))
        del cross_track_m
        lookahead = validation_profile_lookahead_m(speed)
        abs_k = abs(float(curvature_1pm))
        if abs_k >= self._straight_curvature_threshold():
            radius = 1.0 / max(abs_k, 1e-6)
            lookahead = min(lookahead, max(0.32, 0.30 * radius))
        return lookahead

    def _pure_pursuit_command(
        self,
        path: np.ndarray,
        position: np.ndarray,
        yaw: float,
        path_index: int,
        speed_cap: float,
        *,
        cross_track_m: float = 0.0,
        curvature_1pm: float = 0.0,
    ) -> tuple[float, float]:
        """Track the planned polyline with pure pursuit at the profile cap."""
        linear = min(float(self.args.speed), float(speed_cap))
        realized = abs(float(getattr(self, "current_odom_speed", 0.0)))
        look_speed = max(0.35, min(linear, max(realized, 0.35)))
        lookahead = self._effective_profile_lookahead(
            cross_track_m=cross_track_m,
            speed_mps=look_speed,
            curvature_1pm=curvature_1pm,
        )
        target_index = self._lookahead_index(path, position, path_index, lookahead)
        target_point = path[target_index, :2]
        alpha = self.wrap_angle(
            math.atan2(target_point[1] - position[1], target_point[0] - position[0])
            - yaw
        )
        target_distance = max(0.20, float(np.linalg.norm(target_point - position)))
        kappa_max = float(getattr(
            self.args, "max_path_curvature", VALIDATION_MAX_PATH_CURVATURE_1PM,
        ))
        curvature = float(np.clip(
            2.0 * math.sin(alpha) / target_distance, -kappa_max, kappa_max,
        ))
        del cross_track_m
        return linear, linear * curvature

    @staticmethod
    def _direction_segments(path: np.ndarray, min_run: int = 3) -> list[tuple[float, int, int]]:
        """Return stable Reeds-Shepp direction runs as (sign, first_edge, last_edge)."""
        edges = path[1:, :2] - path[:-1, :2]
        headings = np.column_stack((np.cos(path[:-1, 2]), np.sin(path[:-1, 2])))
        signs = np.where(np.sum(edges * headings, axis=1) >= 0.0, 1.0, -1.0)
        runs: list[list[float | int]] = []
        for index, sign in enumerate(signs):
            if not runs or runs[-1][0] != sign:
                runs.append([float(sign), index, index])
            else:
                runs[-1][2] = index
        # Hybrid smoothing may create alternating one-edge directions at the
        # goal.  They are not executable cusps; merge them into a neighbour.
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for index, run in enumerate(runs):
                if int(run[2]) - int(run[1]) + 1 >= min_run:
                    continue
                if index == 0:
                    runs[1][1] = run[1]
                    runs.pop(0)
                else:
                    runs[index - 1][2] = run[2]
                    runs.pop(index)
                changed = True
                break
            # Coalesce adjacent runs after a short run was absorbed.
            merged = []
            for run in runs:
                if merged and merged[-1][0] == run[0]:
                    merged[-1][2] = run[2]
                else:
                    merged.append(run)
            runs = merged
        return [(float(sign), int(first), int(last)) for sign, first, last in runs]

    def drive_planned(self, name: str, target: tuple[float, float], deadline: float) -> None:
        # A finishes south along the spawn corridor. B and C finish west
        # along the open hall so the swept footprint stays off the cabinet.
        planning_pose = self.plan_pose if self._zero_motion_plan() else self.pose
        goal_yaw = self.waypoint_goal_yaw(name, planning_pose[2] if planning_pose else None)
        # Supply an explicit start pose. This is important for a sequential
        # preview and also prevents Nav2 from sampling a stale TF between
        # checkpoint legs during live execution.
        start_yaw = planning_pose[2] if planning_pose is not None else None
        start_xy = planning_pose[:2] if planning_pose is not None else None
        path = np.asarray(self.planned_points(target, goal_yaw, start_yaw, start_xy), dtype=np.float64)
        if not self.args.plan_only:
            path = self.make_continuous_path(path)
        self.ensure_path_safe(name,path.tolist())
        self.planned_paths[name] = path.tolist()
        self.write_plan_snapshot()
        self.publish_route_visualization()
        if len(path) < 2:
            raise RuntimeError(f"Nav2 returned a degenerate path to {name}")
        if self.args.plan_only:
            self.plan_pose = (float(path[-1, 0]), float(path[-1, 1]), float(path[-1, 2]))
            self.get_logger().info(
                f"plan-only: {name} path accepted ({len(path)} poses); no cmd_vel published"
            )
            return
        direct_distance = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        path_length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
        if direct_distance > self.args.tolerance * 2 and path_length < direct_distance * 0.7:
            raise RuntimeError(
                f"Nav2 returned an unsafe truncated path to {name}: "
                f"direct={direct_distance:.3f}m, path={path_length:.3f}m"
            )
        self.get_logger().info(
            f"{name} Nav2 global path has {len(path)} poses; plan-only or fallback local track"
        )
        x, y, yaw = self.pose
        direction_index = self._lookahead_index(
            path, np.asarray((x, y)), 0, self.args.direction_lookahead
        )
        initial_heading = math.atan2(path[direction_index, 1] - y,
                                     path[direction_index, 0] - x)
        initial_error = math.atan2(math.sin(initial_heading - yaw), math.cos(initial_heading - yaw))
        forced_direction = self.args.forced_directions.get(name)
        if forced_direction is not None:
            travel_sign = -1.0 if forced_direction == "reverse" else 1.0
        else:
            travel_sign = (
                -1.0
                if (self.args.allow_reverse and name != "C"
                    and abs(initial_error) > self.args.reverse_threshold)
                else 1.0
            )
        # A fixed direction is only a preference.  It must not force this
        # Ackermann-like chassis into a large sweeping turn when a newly
        # introduced obstacle makes the first path segment lie behind it.
        selected_error = math.atan2(
            math.sin(initial_heading - (yaw if travel_sign > 0.0 else yaw + math.pi)),
            math.cos(initial_heading - (yaw if travel_sign > 0.0 else yaw + math.pi)),
        )
        if self.args.allow_reverse and abs(selected_error) > self.args.direction_switch_angle:
            travel_sign *= -1.0
            self.get_logger().warning(
                f"{name} overriding preferred direction: initial tracking error "
                f"{math.degrees(selected_error):.1f} deg exceeds "
                f"{math.degrees(self.args.direction_switch_angle):.1f} deg"
            )
        # NavFn/A* returns a positional path whose orientation fields are not
        # Reeds-Shepp motion directions. Parse cusps only when reverse motion
        # was explicitly enabled for a Hybrid-A* path.
        segments = (self._direction_segments(path, self.args.min_direction_run)
                    if self.args.allow_reverse else [(1.0, 0, len(path) - 2)])
        travel_sign = segments[0][0]
        self.leg_directions[name] = (
            "mixed" if len(segments) > 1 else ("reverse" if travel_sign < 0.0 else "forward")
        )
        self.get_logger().info(
            f"{name} travel direction: {self.leg_directions[name]} "
            f"(path heading error {math.degrees(initial_error):.1f} deg; segments={segments})"
        )
        path_index = 0
        segment_index = 0
        stable = 0
        path_deviation_warned = False
        tolerance = self.args.approach_tolerance if name.startswith("_") else self.args.tolerance
        if name == "C":
            tolerance = self.args.final_tolerance
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.ensure_motion_allowed()
            x, y, yaw = self.pose
            position = np.asarray((x, y))
            distance = math.hypot(target[0] - x, target[1] - y)
            # Once the chassis is close to the exact checkpoint, stop chasing
            # a lookahead point that may already lie behind the vehicle.  The
            # service robot cannot rotate in place, so select forward/reverse
            # from the instantaneous target bearing and taper speed to zero.
            # This terminal controller is what makes centimetre-level A/B
            # capture possible without changing either requested coordinate.
            if distance <= self.args.goal_capture_distance:
                if self.target_reached(name, self.coverage_target(name, target)):
                    stable += 1
                    self.command()
                    self.samples.append((time.time(), name, x, y, yaw, distance, 0.0, 0.0))
                    if stable >= self.args.goal_stable_cycles:
                        self.mark_reached(name, self.coverage_target(name, target))
                        self.get_logger().info(
                            f"checkpoint {name} reached at ({x:.3f}, {y:.3f}), "
                            f"error={distance:.4f}m"
                        )
                        return
                    continue
                stable = 0
                bearing = math.atan2(target[1] - y, target[0] - x)
                forward_error = math.atan2(math.sin(bearing-yaw), math.cos(bearing-yaw))
                # The steering base cannot rotate in place.  Reversing at
                # the final centimetres causes a limit cycle when the goal
                # lies behind the current yaw (observed at A). Keep a slow
                # forward arc and turn toward the exact point instead; the
                # speed is proportional to distance and is clamped very low.
                terminal_sign = 1.0
                terminal_error = forward_error
                terminal_speed = min(
                    self.args.speed * 0.55,
                    max(self.args.goal_min_speed, self.args.goal_position_gain * distance),
                )
                linear = terminal_sign * terminal_speed * max(0.20, math.cos(terminal_error))
                # Keep the same curvature->yaw-rate contract as the main
                # tracker.  A fixed angular command at centimetre-scale
                # linear speed saturates Ackermann steering and creates the
                # observed limit cycle around the checkpoint.
                terminal_curvature = 2.0 * math.sin(terminal_error) / max(distance, 0.20)
                angular = float(np.clip(
                    linear * terminal_curvature * self.args.goal_steering_gain,
                    -self.args.max_angular,
                    self.args.max_angular,
                ))
                self.command(linear, angular)
                self.samples.append((time.time(), name, x, y, yaw, distance, linear, angular))
                continue
            travel_sign, segment_first_edge, segment_last_edge = segments[segment_index]
            cusp_pose_index = segment_last_edge + 1
            segment_distance = float(np.min(np.linalg.norm(
                path[segment_first_edge:cusp_pose_index + 1, :2] - position,
                axis=1,
            )))
            if segment_distance > self.args.path_tolerance:
                # Path deviation remains an audit metric, but task acceptance
                # is defined by g1_omnipicker XY distance to the requested site.
                # Do not abort a recoverable leg just because the
                # vehicle cuts outside the nominal Nav2 centerline.
                if not path_deviation_warned:
                    path_deviation_warned = True
                    self.get_logger().warning(
                        f"{name} path deviation {segment_distance:.3f}m exceeds "
                        f"audit limit {self.args.path_tolerance:.3f}m; continuing "
                        "because g1_omnipicker XY distance is the acceptance rule"
                    )
            cusp_distance = float(np.linalg.norm(path[cusp_pose_index, :2] - position))
            if segment_index + 1 < len(segments) and cusp_distance <= self.args.cusp_tolerance:
                self.command()
                segment_index += 1
                travel_sign = segments[segment_index][0]
                path_index = max(path_index, segments[segment_index][1])
                self.get_logger().info(
                    f"{name} reached Hybrid-A* cusp {segment_index}; switching to "
                    f"{'reverse' if travel_sign < 0.0 else 'forward'}"
                )
                continue
            lookahead = min(distance, self.args.lookahead_min + self.args.lookahead_gain * self.args.speed)
            lookahead = max(lookahead, 0.20)
            path_index = self._lookahead_index(path, position, path_index, lookahead)
            path_index = min(path_index, cusp_pose_index)
            point = path[path_index, :2]
            travel_yaw = yaw if travel_sign > 0.0 else yaw + math.pi
            alpha = math.atan2(math.sin(math.atan2(point[1] - y, point[0] - x) - travel_yaw),
                               math.cos(math.atan2(point[1] - y, point[0] - x) - travel_yaw))
            max_curvature = self.args.final_max_curvature if name == "C" else self.args.max_curvature
            curvature = float(np.clip(2.0 * math.sin(alpha) / max(lookahead, 1e-3),
                                      -max_curvature, max_curvature))
            curve_scale = max(self.args.min_speed_ratio,
                              1.0 - self.args.curvature_slowdown * abs(curvature))
            # Tight startup bends need a short, slow arc.  A long lookahead at
            # full minimum speed cuts across the inflated obstacle boundary.
            heading_scale = max(
                self.args.min_heading_speed_ratio,
                min(1.0, math.cos(min(abs(alpha), math.pi / 2.0)) ** 2),
            )
            goal_scale = max(self.args.min_goal_speed_ratio,
                             min(1.0, distance / self.args.goal_slowdown_distance))
            speed_scale = curve_scale * goal_scale * heading_scale
            linear = travel_sign * self.args.speed * speed_scale
            # ``angular.z`` is a yaw-rate command at the ROS/Nav2 interface.
            # Convert path curvature (1/m) to the physically correct yaw rate
            # using the *actual signed linear command*.  The chassis bridge
            # then applies the Ackermann relation ``delta=atan(L*w/v)``.
            # Passing curvature directly as angular.z made the bridge divide
            # by a very small terminal speed and command near-full steering;
            # the robot consequently entered an expanding orbit around A.
            if name == "A":
                steering_gain = self.args.startup_steering_gain
            elif name == "C":
                steering_gain = self.args.final_steering_gain
            else:
                steering_gain = self.args.steering_gain
            angular = float(np.clip(
                linear * curvature * steering_gain,
                -self.args.max_angular,
                self.args.max_angular,
            ))
            if self.target_reached(name, self.coverage_target(name, target)):
                stable += 1
                self.command()
                if stable >= 5 and (
                    name != "A" or self._start_to_a_elapsed_ok()
                ):
                    self.mark_reached(name, self.coverage_target(name, target))
                    self.get_logger().info(f"checkpoint {name} reached at ({x:.3f}, {y:.3f})")
                    return
            else:
                stable = 0
                self.command(linear, angular)
            self.samples.append((time.time(), name, x, y, yaw, distance, linear, angular))
        raise TimeoutError(f"Timed out before reaching {name}: pose={self.pose}, target={target}")

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def heading_priority_command(
        linear: float,
        alpha: float,
        curvature: float,
        max_angular: float,
        max_linear: float,
    ) -> tuple[float, float]:
        """Drop linear speed when v·|κ| would saturate yaw, then keep a yaw floor."""
        linear = float(linear)
        kappa = float(curvature)
        max_angular = max(float(max_angular), 1e-6)
        max_linear = float(max_linear)
        if abs(kappa) > 1e-6:
            yaw_limited = max_angular / abs(kappa)
            if abs(linear) > yaw_limited:
                linear = math.copysign(yaw_limited, linear) if linear else yaw_limited
        angular = float(np.clip(linear * kappa, -max_angular, max_angular))
        if abs(alpha) <= 0.18:
            return linear, angular
        min_yaw = min(max_angular, max(0.20, 0.90 * abs(alpha)))
        if abs(angular) < min_yaw:
            angular = math.copysign(min_yaw, alpha)
        if linear < 0.18:
            linear = min(max_linear, 0.18)
        return linear, angular

    def align_heading(self, name: str, desired_yaw: float, deadline: float,
                      tolerance: float = 0.35) -> None:
        """Face the next corridor by reversing, not by path-curvature tracking.

        After a south A stop the west B path starts with a Reeds-Shepp cusp.
        Forward steering from that heading cuts south of A. Reverse + yaw-rate
        rotates toward west while backing north along the spawn corridor.
        """
        while rclpy.ok() and time.monotonic() < deadline:
            self.ensure_motion_allowed()
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.pose is None:
                continue
            x, y, yaw = self.pose
            error = self.wrap_angle(desired_yaw - yaw)
            if abs(error) <= tolerance:
                self.command()
                self.get_logger().info(
                    f"{name} heading aligned to {desired_yaw:.3f} at ({x:.3f}, {y:.3f})"
                )
                return
            linear = -min(0.45, max(0.20, self.args.speed * 0.50))
            angular = float(np.clip(self.args.heading_gain * error, -0.55, 0.55))
            self.command(linear, angular)
            self.samples.append((time.time(), f"{name}_align", x, y, yaw, abs(error), linear, angular))
        self.command()
        raise TimeoutError(
            f"Timed out aligning heading for {name}: pose={self.pose}, desired={desired_yaw}"
        )

    def publish_speed_limit(self, speed: float | None, path_index: int = -1,
                            reason: str = "clear") -> None:
        """Publish an absolute Nav2 limit; zero explicitly clears the limit."""
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.percentage = False
        message.speed_limit = 0.0 if speed is None else max(
            self.args.minimum_profile_speed, float(speed)
        )
        self.speed_limit_pub.publish(message)
        self.speed_limit_history.append({
            "unix_time": time.time(),
            "path_index": int(path_index),
            "speed_limit_mps": (
                None if speed is None else float(message.speed_limit)
            ),
            "limiting_factor": reason,
        })

    def profile_speed_kwargs(
        self,
        *,
        stop_at_end: bool,
        terminal_arrival: bool = False,
        depart_from_rest: bool = True,
    ) -> dict[str, float | bool | None]:
        arrival = (
            float(self.args.terminal_arrival_speed)
            if terminal_arrival and stop_at_end
            else 0.0
        )
        peak = self.args.peak_contract_speed
        if peak is not None and peak + 1e-9 < float(self.args.speed):
            peak = float(self.args.speed)
        peak_val = float(peak if peak is not None else self.args.speed)
        profile_accel = scaled_profile_linear_accel(
            self.args.profile_linear_accel, peak_val,
        )
        return {
            "contract_speed": float(self.args.speed),
            "max_angular": self.args.max_angular,
            "lateral_accel_limit": self.args.profile_lateral_accel,
            "accel_limit": profile_accel,
            "jerk_limit": self.args.profile_linear_jerk,
            "clearance_hard": self.args.profile_clearance_hard,
            "clearance_soft": self.args.profile_clearance_soft,
            "minimum_motion_speed": self.args.minimum_profile_speed,
            "stop_at_end": stop_at_end,
            "terminal_arrival_speed_mps": arrival,
            "terminal_zone_m": float(self.args.terminal_zone_m),
            "peak_contract_speed": peak,
            "straight_curvature_1pm": self.args.straight_curvature_1pm,
            "depart_from_rest": depart_from_rest,
            "path_curvature_cap_1pm": VALIDATION_MAX_PATH_CURVATURE_1PM,
        }

    def build_combined_speed_profile(
        self, points: list[tuple[float, float, float]]
    ) -> None:
        ordered = [
            name for name in VALIDATION_ROUTE_ORDER if name in self.planned_paths
        ]
        self.profile_leg_arc_bounds = {}
        hybrid_profile = bool(
            getattr(self.args, "use_hybrid_theta_mppi", False)
            or getattr(self.args, "use_profiled_controller", False)
        )
        if hybrid_profile and len(ordered) >= 2 and ordered[-1] == "C":
            combined_profile: list[dict] = []
            arc_offset = 0.0
            join_speed: float | None = None
            pending_handoff: list[tuple[float, float, float]] | None = None

            def flush_pending_handoff() -> None:
                nonlocal pending_handoff, arc_offset, join_speed, combined_profile
                if pending_handoff is None:
                    return
                segment_profile = build_path_speed_profile(
                    pending_handoff,
                    self.occupied_world_xy,
                    self.safety_footprint,
                    **self.profile_speed_kwargs(
                        stop_at_end=False,
                        terminal_arrival=False,
                        depart_from_rest=not combined_profile,
                    ),
                    join_speed_mps=join_speed,
                    handoff_geometry=True,
                )
                for index, item in enumerate(segment_profile):
                    if index == 0 and combined_profile:
                        continue
                    entry = dict(item)
                    entry["arc_length_m"] = arc_offset + float(item["arc_length_m"])
                    entry["index"] = len(combined_profile)
                    combined_profile.append(entry)
                arc_offset = float(combined_profile[-1]["arc_length_m"])
                join_speed = float(combined_profile[-1]["speed_cap_mps"])
                pending_handoff = None

            for name in ordered:
                leg_start_arc = arc_offset
                segment = self.planned_paths[name]
                subpaths = split_path_by_curvature(
                    segment,
                    straight_threshold_1pm=float(self.args.straight_curvature_1pm),
                )
                for sub_index, subpath in enumerate(subpaths):
                    handoff = (
                        sub_index < len(subpaths) - 1
                        or (len(subpaths) == 1 and name != ordered[-1])
                    )
                    if handoff:
                        if pending_handoff is None:
                            pending_handoff = list(subpath)
                        else:
                            pending_handoff.extend(subpath[1:])
                        continue
                    flush_pending_handoff()
                    segment_profile = build_path_speed_profile(
                        subpath,
                        self.occupied_world_xy,
                        self.safety_footprint,
                        **self.profile_speed_kwargs(
                            stop_at_end=False,
                            terminal_arrival=False,
                            depart_from_rest=not combined_profile,
                        ),
                        join_speed_mps=join_speed,
                    )
                    for index, item in enumerate(segment_profile):
                        if index == 0 and combined_profile:
                            continue
                        entry = dict(item)
                        entry["arc_length_m"] = arc_offset + float(item["arc_length_m"])
                        entry["index"] = len(combined_profile)
                        combined_profile.append(entry)
                    arc_offset = float(combined_profile[-1]["arc_length_m"])
                    join_speed = float(combined_profile[-1]["speed_cap_mps"])
                self.profile_leg_arc_bounds[name] = (leg_start_arc, arc_offset)
            flush_pending_handoff()
            self.path_speed_profile = combined_profile
            self._tune_straight_hall_profile(combined_profile)
        else:
            self.path_speed_profile = build_path_speed_profile(
                points,
                self.occupied_world_xy,
                self.safety_footprint,
                **self.profile_speed_kwargs(
                    stop_at_end=False,
                    terminal_arrival=False,
                ),
            )
        self.path_speed_profile_summary = profile_summary(self.path_speed_profile)

    def _blend_segment_join_braking(
        self, profile: list[dict], *, lookback_m: float = 4.0,
    ) -> None:
        """Brake A/B just enough to meet C's entry speed; do not re-anchor on C stop."""
        if len(profile) < 3 or "C" not in self.planned_paths:
            return
        c_bounds = self.profile_leg_arc_bounds.get("C")
        if c_bounds is not None:
            c_start_arc = float(c_bounds[0])
            c_start = next(
                (
                    index
                    for index, item in enumerate(profile)
                    if float(item["arc_length_m"]) >= c_start_arc - 1e-6
                ),
                len(profile) - 1,
            )
        else:
            offset = 0
            for name in VALIDATION_ROUTE_ORDER:
                if name == "C":
                    break
                count = len(self.planned_paths.get(name, []))
                if not count:
                    continue
                offset += count if offset == 0 else count - 1
            c_start = offset
        if c_start <= 0 or c_start >= len(profile):
            return
        peak_val = float(
            getattr(self.args, "peak_contract_speed", None) or self.args.speed
        )
        accel = scaled_profile_linear_accel(
            self.args.profile_linear_accel, peak_val,
        )
        jerk = float(self.args.profile_linear_jerk)
        entry = float(profile[c_start]["speed_cap_mps"])
        join_arc = float(profile[c_start]["arc_length_m"])
        for index in range(c_start - 1, -1, -1):
            distance = max(
                0.0, join_arc - float(profile[index]["arc_length_m"])
            )
            if distance > lookback_m:
                break
            allowed = _reachable_speed(
                entry, distance, peak_val, accel, jerk,
            )
            current = float(profile[index]["speed_cap_mps"])
            if current > allowed + 1e-6:
                profile[index]["speed_cap_mps"] = allowed
                profile[index]["limiting_factor"] = "segment_join_braking"

    def profile_index_for_pose(self, start: int = 0) -> int:
        if self.pose is None or not self.path_speed_profile:
            return start
        remaining = self.path_speed_profile[start:]
        distances = [
            math.hypot(float(item["x"]) - self.pose[0],
                       float(item["y"]) - self.pose[1])
            for item in remaining
        ]
        return start + int(np.argmin(distances))

    @staticmethod
    def monotonic_path_projection(
        path: np.ndarray,
        position: np.ndarray,
        start_index: int,
        arc_lengths: list[float],
    ) -> tuple[int, float, float, float]:
        """Project onto the path with non-decreasing arc progress."""
        search = path[start_index:, :2]
        local = int(np.argmin(np.linalg.norm(search - position, axis=1)))
        index = start_index + local
        cross_track = float(np.linalg.norm(path[index, :2] - position))
        signed_cross_track = 0.0
        arc_m = float(arc_lengths[index])
        if index + 1 < len(path):
            segment = path[index + 1, :2] - path[index, :2]
            seg_len = float(np.linalg.norm(segment))
            if seg_len > 1e-9:
                progress = float(np.clip(
                    np.dot(position - path[index, :2], segment) / (seg_len * seg_len),
                    0.0,
                    1.0,
                ))
                projected = path[index, :2] + progress * segment
                cross_track = float(np.linalg.norm(projected - position))
                tangent = segment / seg_len
                normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
                signed_cross_track = float(np.dot(position - projected, normal))
                arc_m = float(arc_lengths[index]) + progress * seg_len
        return index, arc_m, cross_track, signed_cross_track

    def _straight_curvature_threshold(self) -> float:
        return float(getattr(self.args, "straight_curvature_1pm", 0.25))

    def _c_leg_committed_turn_active(self, arc_m: float) -> bool:
        """True on a C-leg bend where cross-track braking would kill the apex."""
        if self._leg_name_at_arc(arc_m) != "C":
            return False
        profile = getattr(self, "path_speed_profile", None) or []
        if profile:
            index = min(
                len(profile) - 1,
                max(
                    0,
                    next(
                        (
                            item_index
                            for item_index, item in enumerate(profile)
                            if float(item["arc_length_m"]) >= float(arc_m)
                        ),
                        len(profile) - 1,
                    ),
                ),
            )
            if abs(float(profile[index]["curvature_1pm"])) >= (
                VALIDATION_C_LEG_PRIMARY_TURN_CURVATURE_1PM
            ):
                return True
        bend_arc = self._c_leg_bend_start_arc()
        if bend_arc is None:
            return False
        dist_from_bend = float(arc_m) - float(bend_arc)
        if dist_from_bend >= 0.0:
            return dist_from_bend <= VALIDATION_C_LEG_PRE_BEND_LOOKAHEAD_M
        return -dist_from_bend <= VALIDATION_C_LEG_COMMITTED_TURN_LOOKAHEAD_M

    def _use_straight_cruise_control(
        self,
        leg_name: str | None,
        arc_m: float,
        local_curvature: float,
    ) -> bool:
        """Always track the polyline; do not switch controllers at A/B/C."""
        del leg_name, arc_m, local_curvature
        return False

    def _straight_cruise_command(
        self,
        path: np.ndarray,
        yaw: float,
        path_index: int,
        speed_cap: float,
    ) -> tuple[float, float]:
        """Track a straight chord at the profile cap with tangent heading hold."""
        path_yaw = float(path[min(path_index, len(path) - 1), 2])
        linear = min(float(speed_cap), float(self.args.speed))
        alpha = self.wrap_angle(path_yaw - yaw)
        kappa_max = float(getattr(
            self.args, "max_path_curvature", VALIDATION_MAX_PATH_CURVATURE_1PM,
        ))
        curvature = float(np.clip(
            2.0 * math.sin(alpha) / 0.50, -kappa_max * 0.50, kappa_max * 0.50,
        ))
        return self.heading_priority_command(
            linear,
            alpha,
            curvature,
            self.args.max_angular,
            linear,
        )

    def coordinated_speed_cap(
        self,
        arc_m: float,
        cross_track_m: float,
        *,
        remaining_arc_m: float | None = None,
    ) -> tuple[int, float, str]:
        """Hold the contract cruise on the whole path; no point-wise overlays."""
        del remaining_arc_m, cross_track_m
        peak_val = float(
            getattr(self.args, "peak_contract_speed", None) or self.args.speed
        )
        if not self.path_speed_profile:
            return 0, peak_val, "missing_profile"
        index = min(
            len(self.path_speed_profile) - 1,
            max(
                0,
                next(
                    (
                        item_index
                        for item_index, item in enumerate(self.path_speed_profile)
                        if float(item["arc_length_m"]) >= float(arc_m)
                    ),
                    len(self.path_speed_profile) - 1,
                ),
            ),
        )
        return index, peak_val, "peak_contract"

    def _leg_name_at_arc(self, arc_m: float) -> str | None:
        bounds = getattr(self, "profile_leg_arc_bounds", None) or {}
        if not bounds:
            return None
        for name, (start_arc, end_arc) in bounds.items():
            if float(start_arc) <= float(arc_m) <= float(end_arc) + 1e-3:
                return name
        last_name = next(reversed(bounds))
        return last_name

    def _c_leg_bend_start_arc(
        self, profile: list[dict] | None = None,
    ) -> float | None:
        bounds = getattr(self, "profile_leg_arc_bounds", None) or {}
        bounds = bounds.get("C")
        c_path = self.planned_paths.get("C")
        if bounds is None or not c_path or len(c_path) < 2:
            return None
        start_arc, _end_arc = bounds
        primary = primary_c_turn_start_arc(
            c_path,
            leg_start_arc=float(start_arc),
            heading_change_rad=VALIDATION_C_LEG_TURN_HEADING_CHANGE_RAD,
            window_m=VALIDATION_C_LEG_TURN_ZONE_WINDOW_M,
            min_turn_arc_m=VALIDATION_C_LEG_TURN_MIN_ARC_M,
            min_straight_arc_m=VALIDATION_C_LEG_TURN_MIN_STRAIGHT_M,
            primary_turn_curvature_1pm=VALIDATION_C_LEG_PRIMARY_TURN_CURVATURE_1PM,
        )
        if primary is not None:
            return primary
        active_profile = profile if profile is not None else getattr(
            self, "path_speed_profile", None,
        )
        if not active_profile:
            return None
        for item in active_profile:
            arc = float(item["arc_length_m"])
            if arc < start_arc - 1e-3 or arc > bounds[1] + 1e-3:
                continue
            if abs(float(item["curvature_1pm"])) >= VALIDATION_C_LEG_PRE_BEND_CURVATURE_1PM:
                return arc
        return None

    def _tune_straight_hall_profile(self, profile: list[dict]) -> None:
        """Keep every sample at the contract peak (startup/stop zeros stay)."""
        if not profile:
            return
        peak_val = float(
            getattr(self.args, "peak_contract_speed", None) or self.args.speed
        )
        for item in profile:
            reason = str(item["limiting_factor"])
            if reason in {
                "initial_acceleration",
                "terminal_stop",
                "terminal_arrival",
            }:
                continue
            if float(item["speed_cap_mps"]) + 1e-6 < peak_val:
                item["speed_cap_mps"] = peak_val
                item["limiting_factor"] = "peak_contract"
        self._apply_c_leg_trackable_caps(profile)

    def _c_leg_clearance_is_tight(self, item: dict) -> bool:
        clearance = float(item.get("clearance_m", math.inf))
        return math.isfinite(clearance) and clearance < float(
            getattr(self.args, "profile_clearance_soft", 0.80)
        )

    def _apply_c_leg_trackable_caps(self, profile: list[dict]) -> None:
        """No extra C speed knobs: the contract peak is the only cruise cap."""
        del profile
        return

    def profile_limit_ahead(self, path_index: int) -> tuple[int, float, str]:
        profile = self.path_speed_profile
        target_arc = (
            float(profile[path_index]["arc_length_m"])
            + self.args.profile_lookahead
        )
        lookahead_index = path_index
        while (lookahead_index + 1 < len(profile)
               and float(profile[lookahead_index]["arc_length_m"]) < target_arc):
            lookahead_index += 1
        item = profile[lookahead_index]
        return (
            lookahead_index,
            float(item["speed_cap_mps"]),
            str(item["limiting_factor"]),
        )

    def _route_leg_groups(
        self,
        route,
        planning_pose: tuple[float, float, float],
    ) -> list[tuple[list[tuple[str, tuple[float, float]]], bool, tuple[float, float] | None]]:
        """Split the scored route into Nav2 planner legs (pass-through groups)."""
        groups: list[
            tuple[list[tuple[str, tuple[float, float]]], bool, tuple[float, float] | None]
        ] = []
        index = 0
        pose = planning_pose
        while index < len(route):
            end = index
            while (
                end + 1 < len(route)
                and self.is_pass_through(pose, route[end][1], route[end + 1][1])
            ):
                end += 1
            terminal = end + 1 >= len(route)
            next_target = route[end + 1][1] if not terminal else None
            groups.append((route[index:end + 1], terminal, next_target))
            pose = (
                float(route[end][1][0]),
                float(route[end][1][1]),
                float(pose[2]),
            )
            index = end + 1
        return groups

    def _widen_terminal_c_path(
        self,
        points: list[tuple[float, float, float]],
        site_xy: tuple[float, float],
    ) -> tuple[list[tuple[float, float, float]], dict]:
        """Nudge only the last C cabinet turn off the inner occupied corner."""
        if self.occupied_world_xy is None or self.safety_footprint is None:
            return points, {"applied": False, "skipped": "no_occupancy"}
        widened, audit = widen_last_c_turn(
            points,
            self.occupied_world_xy,
            self.safety_footprint,
            target_clearance_m=VALIDATION_C_TURN_TARGET_CLEARANCE_M,
            max_push_m=VALIDATION_C_TURN_MAX_PUSH_M,
            site_xy=site_xy,
            score_radius_m=float(self.args.score_region_radius),
            path_safety_clearance_m=float(self.args.path_safety_clearance),
        )
        if audit.get("applied"):
            self.get_logger().info(
                "widened last C turn: "
                + json.dumps(audit, sort_keys=True, default=str)
            )
        elif audit.get("rejected"):
            self.get_logger().warning(
                "last C turn widen rejected: "
                + json.dumps(audit, sort_keys=True, default=str)
            )
        return widened, audit

    def _finalize_planned_leg(
        self,
        group: list[tuple[str, tuple[float, float]]],
        points: list[tuple[float, float, float]],
        *,
        terminal: bool,
        nav_goal: tuple[float, float],
    ) -> list[tuple[float, float, float]]:
        goal_name, goal_site = group[-1]
        if terminal:
            points = trim_path_to_score_stop(
                points,
                goal_site,
                stop_offset_m=self.args.score_circle_stop_offset,
                score_radius_m=float(self.args.score_region_radius),
            )
            points, widen_audit = self._widen_terminal_c_path(points, goal_site)
            if widen_audit.get("applied") or widen_audit.get("rejected"):
                self.route_geometry_audits.append({
                    "goal_leg": goal_name,
                    "widen_last_c_turn": widen_audit,
                })
        for name, slice_points in self.split_path_by_sites(points, group):
            self.ensure_path_safe(name, slice_points)
            self.planned_paths[name] = slice_points
        if terminal:
            stop = points[-1]
            self.get_logger().info(
                f"terminal navigation goal {goal_name} at "
                f"({nav_goal[0]:.3f}, {nav_goal[1]:.3f}); "
                f"path stop ({stop[0]:.3f}, {stop[1]:.3f})"
            )
        return points

    def _plan_segmented_c_leg(
        self,
        group: list[tuple[str, tuple[float, float]]],
        planning_pose: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """Two-phase C planning: hall cruise then a dedicated bend approach."""
        goal_name, goal_site = group[-1]
        nav_goal = self.navigation_goal(
            goal_site, planning_pose, terminal=True,
        )
        mono_yaw = self.geometric_goal_yaw(planning_pose, nav_goal, None)
        mono = self.planned_points_with_heading_search(
            nav_goal, mono_yaw, planning_pose,
            terminal_leg=False, quick=True,
        )
        via = c_leg_turn_via_point(
            mono,
            heading_change_rad=VALIDATION_C_LEG_TURN_HEADING_CHANGE_RAD,
            window_m=VALIDATION_C_LEG_TURN_ZONE_WINDOW_M,
            min_turn_arc_m=VALIDATION_C_LEG_TURN_MIN_ARC_M,
            min_straight_arc_m=VALIDATION_C_LEG_TURN_MIN_STRAIGHT_M,
            primary_turn_curvature_1pm=VALIDATION_C_LEG_PRIMARY_TURN_CURVATURE_1PM,
        )
        if via is None:
            raise RuntimeError("monolithic C plan has no detectable turn zone")
        via_yaw = self.geometric_goal_yaw(planning_pose, via, goal_site)
        hall = self.planned_points_with_heading_search(
            via, via_yaw, planning_pose, terminal_leg=False, quick=True,
        )
        turn_pose = tuple(float(value) for value in hall[-1])
        turn_yaw = self.geometric_goal_yaw(turn_pose, nav_goal, None)
        turn = self.planned_points_with_heading_search(
            nav_goal, turn_yaw, turn_pose, terminal_leg=True,
        )
        combined = list(hall)
        combined.extend(turn[1:])
        self.ensure_path_safe("C", combined)
        self.get_logger().info(
            f"segmented C-leg via ({via[0]:.3f},{via[1]:.3f}): "
            f"hall={len(hall)} turn={len(turn)} total={len(combined)}"
        )
        return self._finalize_planned_leg(
            group, combined, terminal=True, nav_goal=nav_goal,
        )

    def _plan_route_leg_group(
        self,
        group: list[tuple[str, tuple[float, float]]],
        planning_pose: tuple[float, float, float],
        *,
        terminal: bool,
        next_target: tuple[float, float] | None,
    ) -> list[tuple[float, float, float]]:
        """Plan one route leg with Nav2 global planner (no hardcoded geometry)."""
        goal_name, goal_site = group[-1]
        nav_goal = self.navigation_goal(
            goal_site, planning_pose, terminal=terminal,
        )
        goal_yaw = self.geometric_goal_yaw(
            planning_pose, nav_goal, next_target,
        )
        points = self.planned_points_with_heading_search(
            nav_goal, goal_yaw, planning_pose,
            terminal_leg=terminal,
        )
        if (
            self.args.global_planner_plugin == "theta_star"
            and goal_name == "B"
        ):
            corridor_sites = None
            if "A" in self.task_points and "B" in self.task_points:
                corridor_sites = (
                    tuple(self.task_points["A"]),
                    tuple(self.task_points["B"]),
                )
            if corridor_sites is not None:
                points, audit = refine_planned_path(
                    points,
                    self.occupied_world_xy,
                    self.safety_footprint,
                    clearance=self.args.command_safety_clearance,
                    corridor_sites=corridor_sites,
                )
                audit["goal_leg"] = goal_name
                self.route_geometry_audits.append(audit)
        return self._finalize_planned_leg(
            group, points, terminal=terminal, nav_goal=nav_goal,
        )

    def _join_preflight_leg_from_pose(
        self,
        leg_name: str,
        planning_pose: tuple[float, float, float],
        target: tuple[float, float],
    ) -> list[tuple[float, float, float]]:
        """Reuse a preflight Nav2 leg, joined from the live pose (not hand-drawn)."""
        if self.args.preplanned_paths is None:
            raise RuntimeError(f"no preflight Nav2 cache available for {leg_name}")
        raw = json.loads(self.args.preplanned_paths.read_text(encoding="utf-8"))
        values = raw.get(leg_name)
        if not isinstance(values, list) or len(values) < 2:
            raise RuntimeError(f"preflight cache has no usable {leg_name} leg")
        points: list[tuple[float, float, float]] = [
            tuple(float(item) for item in row) for row in values
        ]
        nearest = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - planning_pose[0],
                points[index][1] - planning_pose[1],
            ),
        )
        suffix = points[nearest:]
        start_gap = math.hypot(
            suffix[0][0] - planning_pose[0],
            suffix[0][1] - planning_pose[1],
        )
        if nearest == 0 and start_gap > 0.35:
            raise RuntimeError(
                f"preflight {leg_name} join at path start with gap {start_gap:.3f}m"
            )
        if start_gap > self.args.initial_pose_tolerance:
            raise RuntimeError(
                f"preflight {leg_name} join gap {start_gap:.3f}m exceeds tolerance"
            )
        if leg_name == "C":
            suffix = trim_path_to_score_stop(
                suffix,
                target,
                stop_offset_m=self.args.score_circle_stop_offset,
                score_radius_m=float(self.args.score_region_radius),
            )
            suffix, _widen = self._widen_terminal_c_path(suffix, target)
        self.ensure_path_safe(leg_name, suffix)
        self.planned_paths[leg_name] = suffix
        self.get_logger().warning(
            f"joined preflight Nav2 {leg_name} leg at index {nearest} "
            f"(gap={start_gap:.3f}m, poses={len(suffix)})"
        )
        return suffix

    def _plan_mission_leg_points(
        self,
        group: list[tuple[str, tuple[float, float]]],
        planning_pose: tuple[float, float, float],
        *,
        terminal: bool,
        next_target: tuple[float, float] | None,
    ) -> list[tuple[float, float, float]]:
        """Plan one mission leg from the live pose (Nav2 global planner only)."""
        goal_name = group[-1][0]
        self._motion_stall_suspended = True
        try:
            try:
                return self._plan_route_leg_group(
                    group, planning_pose, terminal=terminal, next_target=next_target,
                )
            except RuntimeError as exc:
                if (
                    terminal
                    and goal_name == "C"
                    and self.args.preplanned_paths is not None
                ):
                    self.get_logger().warning(
                        f"live Nav2 C-leg planning failed ({exc}); "
                        "retrying preflight planner cache"
                    )
                    return self._join_preflight_leg_from_pose(
                        "C", planning_pose, group[-1][1],
                    )
                raise
        finally:
            self._motion_stall_suspended = False
            self._refresh_motion_stall_reference()

    def plan_continuous_route(self, route) -> list[tuple[float, float, float]]:
        """Plan fly-through scored circles; stop only inside the last circle."""
        planning_pose = tuple(float(value) for value in self.pose)
        combined: list[tuple[float, float, float]] = []
        for group, terminal, next_target in self._route_leg_groups(
            route, planning_pose,
        ):
            points = self._plan_route_leg_group(
                group, planning_pose, terminal=terminal, next_target=next_target,
            )
            combined.extend(points if not combined else points[1:])
            planning_pose = tuple(float(value) for value in points[-1])
        self.build_combined_speed_profile(combined)
        self.write_plan_snapshot()
        self.publish_route_visualization()
        return combined

    def load_preplanned_route(
        self,
        route,
        path_file: Path,
    ) -> list[tuple[float, float, float]]:
        """Load and revalidate the zero-motion preflight route."""
        raw = json.loads(path_file.read_text(encoding="utf-8"))
        expected_names = [name for name, _target in route]
        if not isinstance(raw, dict) or list(raw) != expected_names:
            raise RuntimeError(
                "preplanned route waypoint order does not match current task"
            )
        combined: list[tuple[float, float, float]] = []
        previous_end = tuple(float(value) for value in self.pose)
        for name, target in route:
            values = raw.get(name)
            if not isinstance(values, list) or len(values) < 2:
                raise RuntimeError(f"preplanned route has no usable path for {name}")
            points: list[tuple[float, float, float]] = []
            for value in values:
                if (not isinstance(value, (list, tuple)) or len(value) != 3
                        or not all(math.isfinite(float(item)) for item in value)):
                    raise RuntimeError(f"preplanned route contains invalid pose for {name}")
                points.append(tuple(float(item) for item in value))
            start_gap = math.hypot(
                points[0][0] - previous_end[0],
                points[0][1] - previous_end[1],
            )
            if start_gap > self.args.initial_pose_tolerance:
                raise RuntimeError(
                    f"preplanned {name} starts {start_gap:.3f}m from current route pose"
                )
            endpoint_error = math.hypot(
                points[-1][0] - float(target[0]),
                points[-1][1] - float(target[1]),
            )
            endpoint_limit = 0.15
            if name == route[-1][0]:
                endpoint_limit = max(
                    0.15,
                    self.coordinate_tolerance(name),
                    float(getattr(self.args, "score_circle_stop_offset", 0.25)) + 0.05,
                )
            if endpoint_error > endpoint_limit:
                raise RuntimeError(
                    f"preplanned {name} endpoint differs from task point by "
                    f"{endpoint_error:.3f}m"
                )
            self.ensure_path_safe(name, points)
            self.planned_paths[name] = points
            combined.extend(points if not combined else points[1:])
            previous_end = points[-1]
        self.build_combined_speed_profile(combined)
        self.write_plan_snapshot()
        self.publish_route_visualization()
        self.get_logger().info(
            f"reused and revalidated preflight route from {path_file}"
        )
        return combined

    def _advance_scored_checkpoints(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        *,
        linear: float = 0.0,
        angular: float = 0.0,
    ) -> tuple[int, list[str], bool]:
        """Advance A/B/C bookkeeping when the robot enters a scored circle."""
        reached: list[str] = []
        while next_scored < len(scored):
            name, task_target = scored[next_scored]
            site = self.coverage_target(name, task_target)
            distance = self.scoring_distance_to_site(site)
            judgement_pose = self.judgement_pose()
            self.samples.append((
                time.time(), name, *judgement_pose, distance, linear, angular
            ))
            if not self.checkpoint_satisfied(name, task_target):
                break
            self.mark_reached(name, site)
            self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
            self.checkpoint_pass_speeds_detail[name] = {
                "odom_mps": abs(self.current_odom_speed),
                "commanded_mps": abs(linear),
            }
            if name not in self.leg_timestamps:
                self.leg_timestamps[name] = time.monotonic()
            reached.append(name)
            next_scored += 1
            self.get_logger().info(
                f"{'arrived at' if name == 'C' else 'passed'} {name} "
                f"at speed={abs(self.current_odom_speed):.3f}m/s, "
                f"error={distance:.3f}m"
            )
            if name in ("A", "B"):
                self.settle_checkpoint(name, site)
            if name == self.route_terminal_name:
                for _ in range(3):
                    self.command()
                    rclpy.spin_once(self, timeout_sec=0.02)
                self.publish_speed_limit(None)
                return next_scored, reached, True
        return next_scored, reached, False

    def drive_nav2_subpath(
        self,
        points: list[tuple[float, float, float]],
        deadline: float,
        *,
        segment_label: str,
        join_speed_mps: float | None = None,
        terminal_stop: bool = False,
    ) -> None:
        """Track one corner arc with Nav2 MPPI Ackermann FollowPath."""
        if len(points) < 2:
            return
        corner_profile = build_path_speed_profile(
            points,
            self.occupied_world_xy,
            self.safety_footprint,
            **self.profile_speed_kwargs(
                stop_at_end=terminal_stop,
                terminal_arrival=terminal_stop,
                depart_from_rest=join_speed_mps is None,
            ),
            join_speed_mps=join_speed_mps,
            handoff_geometry=not terminal_stop,
        )
        saved_profile = self.path_speed_profile
        self.path_speed_profile = corner_profile
        if not self.controller.wait_for_server(timeout_sec=10.0):
            self.path_speed_profile = saved_profile
            raise RuntimeError("Nav2 /follow_path action is unavailable")
        path_msg = self._nav_path(points)
        path_msg.header.stamp.sec = 0
        path_msg.header.stamp.nanosec = 0
        goal = FollowPath.Goal()
        goal.path = path_msg
        goal.controller_id = "FollowPath"
        sent = self.controller.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(
                f"Nav2 FollowPath rejected corner segment {segment_label}"
            )
        result_future = handle.get_result_async()
        # MPPI scales wz_max by speed_limit/vx_max; do not throttle mid-track.
        self.publish_speed_limit(None)
        try:
            while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                self.ensure_motion_allowed()
            if not result_future.done():
                handle.cancel_goal_async()
                if VALIDATION_MISSION_DEADLINE_ENABLED:
                    raise TimeoutError(
                        f"MPPI corner segment {segment_label} timed out"
                    )
            wrapped = result_future.result()
            if wrapped is None or wrapped.result.error_code != 0:
                code = wrapped.result.error_code if wrapped is not None else "missing"
                raise RuntimeError(
                    f"MPPI corner segment {segment_label} failed: error_code={code}"
                )
        finally:
            self.publish_speed_limit(None)
            self.path_speed_profile = saved_profile

    def _drive_profiled_subpath(
        self,
        path: np.ndarray,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        *,
        finish_at_end: bool,
    ) -> tuple[int, list[str], bool]:
        """Profiled straight tracking on one curvature segment."""
        if len(path) < 2:
            return next_scored, [], False
        path_arc_lengths = cumulative_arc_lengths(
            (tuple(row) for row in path)
        )
        nearest_index = 0
        last_published_speed_cap = None
        all_reached: list[str] = []
        end_arc = float(path_arc_lengths[-1])
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.ensure_motion_allowed()
            if self.pose is None:
                continue
            x, y, yaw = self.pose
            position = np.asarray((x, y))
            path_index, arc_m, cross_track_error, signed_cross_track = (
                self.monotonic_path_projection(
                    path, position, nearest_index, path_arc_lengths,
                )
            )
            nearest_index = max(nearest_index, path_index)
            if not finish_at_end and arc_m >= end_arc - 0.08:
                break
            remaining_arc = max(0.0, end_arc - arc_m)
            leg_name = self._leg_name_at_arc(arc_m)
            self.current_leg_name = leg_name
            limit_index, speed_cap, cap_reason = self.coordinated_speed_cap(
                arc_m, cross_track_error, remaining_arc_m=remaining_arc,
            )
            speed_cap = self._score_aware_speed_cap(speed_cap, scored, next_scored)
            if (last_published_speed_cap is None
                    or abs(speed_cap - last_published_speed_cap) >= 0.05):
                self.publish_speed_limit(speed_cap, limit_index, cap_reason)
                last_published_speed_cap = speed_cap
            local_curvature = abs(
                float(self.path_speed_profile[limit_index]["curvature_1pm"])
            )
            preview_curvature = self._path_curvature_preview(limit_index, preview_m=1.5)
            flythrough = self._checkpoint_flythrough_override(
                scored=scored,
                next_scored=next_scored,
                path=path,
                path_index=path_index,
                limit_index=limit_index,
                speed_cap=speed_cap,
                x=x,
                y=y,
                yaw=yaw,
                signed_cross_track=signed_cross_track,
            )
            if flythrough is not None:
                linear, angular, control_mode = flythrough
            else:
                linear, angular = self._pure_pursuit_command(
                    path, position, yaw, path_index, speed_cap,
                    cross_track_m=cross_track_error,
                    curvature_1pm=max(local_curvature, preview_curvature),
                )
                control_mode = "pure_pursuit"
            next_scored, reached, finished_c = self._advance_scored_checkpoints(
                scored, next_scored, linear=linear, angular=angular,
            )
            all_reached.extend(reached)
            if finished_c:
                return next_scored, all_reached, True
            audit_yaw = self._profile_audit_yaw(path, path_index)
            self._publish_motion_command(
                linear,
                angular,
                control_mode=control_mode,
                speed_cap=speed_cap,
                cap_reason=cap_reason,
                cross_track_m=cross_track_error,
                audit_path_yaw=audit_yaw,
            )
        return next_scored, all_reached, False

    def _segment_arc_length(
        self, subpath: list[tuple[float, float, float]]
    ) -> float:
        arcs = cumulative_arc_lengths(subpath)
        return float(arcs[-1]) if arcs else 0.0

    def _segment_is_corner(
        self,
        subpath: list[tuple[float, float, float]],
        *,
        straight_threshold: float,
    ) -> bool:
        peak_curvature = segment_peak_curvature(subpath)
        arc_length = self._segment_arc_length(subpath)
        # Theta* polyline vertices can create bogus κ>5 1/m spikes; only hand
        # real bends to MPPI Ackermann.
        return (
            straight_threshold <= peak_curvature < 3.0
            and arc_length >= 1.0
        )

    def _drive_profiled_segment(
        self,
        subpath: list[tuple[float, float, float]],
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        *,
        join_speed: float | None,
        finish_at_end: bool,
        depart_from_rest: bool,
        force_leg: str | None = None,
    ) -> tuple[int, list[str], bool, float | None]:
        profile_kwargs = self.profile_speed_kwargs(
            stop_at_end=False,
            terminal_arrival=False,
            depart_from_rest=depart_from_rest,
        )
        # Keep the real path curvature. Do not rebuild a stopping envelope:
        # cruise stays at the contract peak through C capture.
        segment_profile = build_path_speed_profile(
            subpath,
            self.occupied_world_xy,
            self.safety_footprint,
            **profile_kwargs,
            join_speed_mps=join_speed,
            handoff_geometry=False,
        )
        saved_profile = self.path_speed_profile
        saved_bounds = dict(self.profile_leg_arc_bounds)
        self.path_speed_profile = segment_profile
        if force_leg and segment_profile:
            self.profile_leg_arc_bounds = {
                force_leg: (0.0, float(segment_profile[-1]["arc_length_m"])),
            }
        elif finish_at_end and segment_profile and "C" not in self.profile_leg_arc_bounds:
            self.profile_leg_arc_bounds = {
                "C": (0.0, float(segment_profile[-1]["arc_length_m"])),
            }
        if segment_profile and self._leg_name_at_arc(0.0) == "C":
            self._tune_straight_hall_profile(segment_profile)
        try:
            next_scored, advanced, finished_c = self._drive_profiled_subpath(
                np.asarray(subpath, dtype=np.float64),
                scored,
                next_scored,
                deadline,
                finish_at_end=finish_at_end,
            )
        finally:
            self.path_speed_profile = saved_profile
            self.profile_leg_arc_bounds = saved_bounds
        return (
            next_scored,
            advanced,
            finished_c,
            float(segment_profile[-1]["speed_cap_mps"]),
        )

    def drive_hybrid_theta_mppi_route(self, route, deadline: float) -> list[str]:
        """Theta* straights via profiled cruise; corner arcs via MPPI Ackermann."""
        scored = [(name, target) for name, target in route if not name.startswith("_")]
        if not scored or scored[-1][0] != self.route_terminal_name:
            raise RuntimeError(
                f"continuous route must end at scored point {self.route_terminal_name}"
            )
        points = self._continuous_route_points(route)
        if not self.path_speed_profile:
            self.build_combined_speed_profile(points)
        execution_segments = hybrid_execution_segments(
            self.planned_paths,
            contract_speed_mps=float(self.args.speed),
        )
        reached: list[str] = []
        next_scored = 0
        join_speed: float | None = None
        for segment_index, (label, subpath, is_turn) in enumerate(execution_segments):
            peak_curvature = segment_peak_curvature(subpath)
            arc_length = self._segment_arc_length(subpath)
            mode = "mppi_corner" if is_turn else "profiled_straight"
            self.get_logger().info(
                f"hybrid route {label} ({mode}): poses={len(subpath)} "
                f"peak_curvature={peak_curvature:.3f} 1/m arc={arc_length:.2f}m"
            )
            finish_at_end = segment_index == len(execution_segments) - 1
            if is_turn:
                try:
                    self.drive_nav2_subpath(
                        subpath, deadline, segment_label=label, join_speed_mps=join_speed,
                    )
                    join_speed = abs(float(self.current_odom_speed))
                except RuntimeError as exc:
                    self.get_logger().warning(
                        f"MPPI {label} failed ({exc}); falling back to profiled arc"
                    )
                    next_scored, advanced, finished_c, join_speed = (
                        self._drive_profiled_segment(
                            subpath, scored, next_scored, deadline,
                            join_speed=join_speed,
                            finish_at_end=finish_at_end,
                            depart_from_rest=join_speed is None,
                        )
                    )
                    reached.extend(advanced)
                    if finished_c:
                        return reached
                    continue
                next_scored, advanced, finished_c = self._advance_scored_checkpoints(
                    scored, next_scored,
                )
                reached.extend(advanced)
                if finished_c:
                    return reached
                continue
            next_scored, advanced, finished_c, join_speed = self._drive_profiled_segment(
                subpath, scored, next_scored, deadline,
                join_speed=join_speed,
                finish_at_end=finish_at_end,
                depart_from_rest=join_speed is None,
            )
            reached.extend(advanced)
            if finished_c:
                return reached
        missing = [name for name, _target in scored[next_scored:]]
        if missing and VALIDATION_MISSION_DEADLINE_ENABLED:
            raise TimeoutError(
                f"hybrid Theta*/MPPI route timed out before {missing}"
            )
        return reached

    FOLLOW_PATH_INVALID_PATH = 103
    FOLLOW_PATH_BC_HANDOFF = -103

    def _nav2_c_leg_start_index(
        self, points: list[tuple[float, float, float]]
    ) -> int:
        if "C" not in self.planned_paths:
            return max(0, len(points) - 1)
        offset = 0
        for name in VALIDATION_ROUTE_ORDER:
            if name == "C":
                break
            count = len(self.planned_paths.get(name, []))
            if not count:
                continue
            offset += count if offset == 0 else count - 1
        return max(0, min(offset, len(points) - 1))

    def _slice_path_speed_profile(self, start_index: int) -> None:
        """Re-base the stored profile to a path suffix without rebuilding caps."""
        if not self.path_speed_profile or start_index <= 0:
            return
        start_index = max(0, min(start_index, len(self.path_speed_profile) - 1))
        start_arc = float(self.path_speed_profile[start_index]["arc_length_m"])
        sliced: list[dict] = []
        for offset, item in enumerate(self.path_speed_profile[start_index:]):
            entry = dict(item)
            entry["arc_length_m"] = float(item["arc_length_m"]) - start_arc
            entry["index"] = offset
            sliced.append(entry)
        self.path_speed_profile = sliced
        self.path_speed_profile_summary = profile_summary(self.path_speed_profile)

    def _set_nav2_follow_path_profile(
        self,
        points: list[tuple[float, float, float]],
        *,
        terminal_approach: bool = False,
    ) -> None:
        """Monolithic caps for nav2_follow_path (matches session 153431)."""
        del terminal_approach
        self.path_speed_profile = build_path_speed_profile(
            points,
            self.occupied_world_xy,
            self.safety_footprint,
            **self.profile_speed_kwargs(
                stop_at_end=False,
                terminal_arrival=False,
            ),
        )
        self.path_speed_profile_summary = profile_summary(self.path_speed_profile)

    def _drive_nav2_follow_path_segment(
        self,
        points: list[tuple[float, float, float]],
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        reached: list[str],
        *,
        segment_end_name: str | None = None,
    ) -> tuple[int, list[str], int | None]:
        """Track one FollowPath segment and advance A/B/C bookkeeping."""
        if len(points) < 2:
            raise RuntimeError("continuous FollowPath segment is degenerate")
        path = np.asarray(points, dtype=np.float64)
        path_arc_lengths = cumulative_arc_lengths(
            (tuple(row) for row in path)
        )
        self._nav2_path_arc_lengths = path_arc_lengths
        nearest_index = max(0, int(self._nav2_monotonic_index))
        if not self.controller.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Nav2 /follow_path action is unavailable")
        path_msg = self._nav_path(points)
        path_msg.header.stamp.sec = 0
        path_msg.header.stamp.nanosec = 0
        goal = FollowPath.Goal()
        goal.path = path_msg
        goal.controller_id = "FollowPath"
        sent = self.controller.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
        handle = sent.result()
        if handle is None:
            raise RuntimeError(
                "Nav2 FollowPath timed out waiting for goal acceptance"
            )
        if not handle.accepted:
            raise RuntimeError("Nav2 FollowPath rejected continuous route segment")
        result_future = handle.get_result_async()
        # MPPI scales wz_max by speed_limit/vx_max, so a running /speed_limit
        # both throttles cruise and strips the yaw authority needed for the
        # B->C bend. Clear it once and let the controller pick its own speed:
        # it only slows where curvature or the goal make it necessary.
        self.publish_speed_limit(None)
        cancelled_at_c = False
        cancelled_at_segment_end = False
        error_code: int | None = None
        try:
            while (rclpy.ok() and time.monotonic() < deadline
                   and not result_future.done()):
                rclpy.spin_once(self, timeout_sec=0.05)
                self.ensure_motion_allowed()
                if self.pose is None:
                    continue
                position = np.asarray(self.pose[:2])
                path_index, _arc_m, _cross_track_error, _signed = (
                    self.monotonic_path_projection(
                        path, position, nearest_index, path_arc_lengths,
                    )
                )
                nearest_index = max(nearest_index, path_index)
                self._nav2_monotonic_index = nearest_index
                if next_scored >= len(scored):
                    self.current_leg_name = "C"
                    continue
                name, target = scored[next_scored]
                self.current_leg_name = name
                task_target = self.coverage_target(name, target)
                distance = self.task_distance(task_target)
                judgement_pose = self.judgement_pose()
                self.samples.append((
                    time.time(), name, *judgement_pose, distance, math.nan, math.nan
                ))
                if distance > self.coordinate_tolerance(name):
                    continue
                if name == "A" and not self._start_to_a_elapsed_ok():
                    continue
                self.mark_reached(name, task_target)
                self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
                reached.append(name)
                next_scored += 1
                self.get_logger().info(
                    f"{'arrived at' if name == 'C' else 'passed'} {name} "
                    f"at speed={abs(self.current_odom_speed):.3f}m/s, "
                    f"error={distance:.3f}m"
                )
                if name == "C":
                    handle.cancel_goal_async()
                    cancelled_at_c = True
                    break
                if (
                    segment_end_name is not None
                    and name == segment_end_name
                    and next_scored < len(scored)
                ):
                    handle.cancel_goal_async()
                    cancelled_at_segment_end = True
                    break
            if cancelled_at_c or cancelled_at_segment_end:
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=2.0)
            elif not result_future.done():
                handle.cancel_goal_async()
                if VALIDATION_MISSION_DEADLINE_ENABLED:
                    raise TimeoutError("continuous FollowPath timed out before C stop")
            elif next_scored < len(scored):
                name, target = scored[next_scored]
                if name == "C" and self.target_reached(
                    name, self.coverage_target(name, target)
                ):
                    self.mark_reached(name, self.coverage_target(name, target))
                    self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
                    reached.append(name)
                    next_scored += 1
            if next_scored < len(scored) and result_future.done():
                wrapped = result_future.result()
                if wrapped is not None:
                    error_code = int(wrapped.result.error_code)
            return next_scored, reached, error_code
        finally:
            self.publish_speed_limit(None)

    def _nav2_resume_index(self, points: list[tuple[float, float, float]]) -> int:
        if self.pose is None or len(points) < 2:
            return 0
        path = np.asarray(points, dtype=np.float64)
        arcs = cumulative_arc_lengths((tuple(row) for row in path))
        start = max(0, int(self._nav2_monotonic_index))
        index, _arc, _ct, _signed = self.monotonic_path_projection(
            path,
            np.asarray(self.pose[:2]),
            start,
            arcs,
        )
        return max(0, min(index, len(points) - 2))

    def _drive_profiled_c_leg(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        reached: list[str],
    ) -> tuple[int, list[str]]:
        """Track the B->C bend with the profiled pure-pursuit controller."""
        if next_scored >= len(scored):
            return next_scored, reached
        c_points = self.planned_paths.get("C")
        if not c_points or len(c_points) < 2:
            raise RuntimeError("profiled C leg requires a preplanned C path")
        dense_c = densify_path(
            [tuple(map(float, row)) for row in c_points],
            spacing=0.12,
        )
        join_speed = max(
            float(self.args.checkpoint_capture_min_speed),
            abs(self.current_odom_speed),
        )
        self.get_logger().info(
            f"handing B->C bend to profiled tracker ({len(dense_c)} poses)"
        )
        next_scored, advanced, finished_c, _ = self._drive_profiled_segment(
            dense_c,
            scored,
            next_scored,
            deadline,
            join_speed=join_speed,
            finish_at_end=True,
            depart_from_rest=False,
            force_leg="C",
        )
        for name in advanced:
            if name not in reached:
                reached.append(name)
        if finished_c:
            return next_scored, reached
        return next_scored, reached

    def _drive_nav2_c_leg_subpath(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        reached: list[str],
    ) -> tuple[int, list[str], int | None]:
        c_points = self.planned_paths.get("C")
        if not c_points or len(c_points) < 2:
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        dense_c = densify_path(
            [tuple(map(float, row)) for row in c_points],
            spacing=0.12,
        )
        join_speed = max(
            float(self.args.checkpoint_capture_min_speed),
            abs(self.current_odom_speed),
        )
        saved_profile = self.path_speed_profile
        self._motion_stall_suspended = True
        try:
            self.drive_nav2_subpath(
                dense_c,
                deadline,
                segment_label="C",
                join_speed_mps=join_speed,
                terminal_stop=True,
            )
        except RuntimeError as exc:
            self.get_logger().warning(f"C leg MPPI subpath failed: {exc}")
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        finally:
            self.path_speed_profile = saved_profile
            self._refresh_motion_stall_reference()
        if next_scored >= len(scored):
            return next_scored, reached, 0
        name, target = scored[next_scored]
        task_target = self.coverage_target(name, target)
        if not self.target_reached(name, task_target):
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        if name == "A" and not self._start_to_a_elapsed_ok():
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        self.mark_reached(name, task_target)
        self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
        reached.append(name)
        self.get_logger().info(
            f"arrived at {name} at speed={abs(self.current_odom_speed):.3f}m/s, "
            f"error={self.task_distance(task_target):.3f}m"
        )
        return next_scored + 1, reached, 0

    def _drive_nav2_c_leg_fallback(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        reached: list[str],
    ) -> tuple[int, list[str], int | None]:
        c_points = self.planned_paths.get("C")
        if not c_points or len(c_points) < 2:
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        self.get_logger().warning(
            "FollowPath continuous route failed; retrying C leg only"
        )
        return self._drive_nav2_c_leg_subpath(
            scored, next_scored, deadline, reached,
        )

    def _zero_motion_plan(self) -> bool:
        return bool(
            self.args.plan_only or getattr(self.args, "inline_preflight", False)
        )

    def _continuous_route_points(self, route) -> list[tuple[float, float, float]]:
        """Reuse an in-memory or on-disk preflight plan; replan live if unusable."""
        scored_names = [
            name for name, _target in route if not name.startswith("_")
        ]
        if scored_names and all(name in self.planned_paths for name in scored_names):
            combined: list[tuple[float, float, float]] = []
            for name in scored_names:
                path = self.planned_paths[name]
                combined.extend(path if not combined else path[1:])
            if not self.path_speed_profile:
                self.build_combined_speed_profile(combined)
            self.write_plan_snapshot()
            self.publish_route_visualization()
            return combined
        if self.args.preplanned_paths is not None:
            try:
                return self.load_preplanned_route(route, self.args.preplanned_paths)
            except (OSError, ValueError, RuntimeError) as exc:
                self.get_logger().warning(
                    f"preflight Nav2 route unusable ({exc}); planning from live pose"
                )
        return self.plan_continuous_route(route)

    def _recover_nav2_route_after_invalid_path(
        self,
        route,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
        reached: list[str],
    ) -> tuple[int, list[str], int | None]:
        """Replan unreached legs from the live pose after FollowPath INVALID_PATH."""
        if self.pose is None or next_scored >= len(scored):
            return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
        self.get_logger().warning(
            "FollowPath continuous route aborted (INVALID_PATH); "
            "replanning remainder from live pose"
        )
        self._motion_stall_suspended = True
        try:
            planning_pose = tuple(float(value) for value in self.pose)
            remaining_route = [
                item for item in route
                if not item[0].startswith("_")
                and item[0] in {name for name, _ in scored[next_scored:]}
            ]
            if not remaining_route:
                remaining_route = [scored[i] for i in range(next_scored, len(scored))]
            combined: list[tuple[float, float, float]] = []
            for group, terminal, next_target in self._route_leg_groups(
                remaining_route, planning_pose,
            ):
                leg_points = self._plan_route_leg_group(
                    group,
                    planning_pose,
                    terminal=terminal,
                    next_target=next_target,
                )
                combined.extend(leg_points if not combined else leg_points[1:])
                planning_pose = tuple(float(value) for value in leg_points[-1])
            if len(combined) < 2:
                return next_scored, reached, self.FOLLOW_PATH_INVALID_PATH
            self._set_nav2_follow_path_profile(combined)
            self._nav2_monotonic_index = 0
            return self._drive_nav2_follow_path_segment(
                combined, scored, next_scored, deadline, reached,
            )
        finally:
            self._motion_stall_suspended = False
            self._refresh_motion_stall_reference()

    def _attempt_terminal_c_finish(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        reached: list[str],
        deadline: float,
    ) -> list[str]:
        """Crawl into the route terminal when FollowPath undershoots the checkpoint."""
        terminal = self.route_terminal_name
        if terminal in reached or self.pose is None:
            return reached
        t_name, t_target = scored[-1]
        if t_name != terminal:
            return reached
        site = self.coverage_target(t_name, t_target)
        distance = self.task_distance(site)
        if distance > 2.0:
            return reached
        self.get_logger().info(
            f"terminal finish: {distance:.3f}m from {terminal}; crawling into tolerance"
        )
        if self._crawl_scored_checkpoint_until(t_name, t_target, deadline):
            if terminal not in reached:
                reached.append(terminal)
            if terminal == "C":
                self._official_circle_capture_and_dwell(t_name, t_target, deadline)
        return reached

    def drive_nav2_through_route(self, route, deadline: float) -> list[str]:
        """Track the full preflight route with Nav2 FollowPath."""
        scored = [(name, target) for name, target in route if not name.startswith("_")]
        if not scored or scored[-1][0] != self.route_terminal_name:
            raise RuntimeError(
                f"continuous route must end at scored point {self.route_terminal_name}"
            )
        if self.pose is None:
            raise RuntimeError("cannot execute Nav2 route without localization pose")
        reached: list[str] = []
        next_scored = 0
        error_code: int | None = None
        self._motion_stall_suspended = True
        try:
            points = self._continuous_route_points(route)
            self.get_logger().info(
                f"Nav2 continuous route: {len(points)} poses from "
                f"{self.args.global_planner_plugin} global planner"
            )
            self._set_nav2_follow_path_profile(points)
            self._nav2_monotonic_index = 0
            next_scored, reached, error_code = (
                self._drive_nav2_follow_path_segment(
                    points, scored, next_scored, deadline, reached,
                )
            )
            if (
                next_scored < len(scored)
                and error_code == self.FOLLOW_PATH_INVALID_PATH
                and self.pose is not None
            ):
                next_scored, reached, error_code = (
                    self._recover_nav2_route_after_invalid_path(
                        route, scored, next_scored, deadline, reached,
                    )
                )
            if next_scored != len(scored):
                missing = [name for name, _ in scored[next_scored:]]
                raise RuntimeError(
                    f"continuous Nav2 route missed {missing}: "
                    f"error_code={error_code if error_code is not None else 'missing'}"
                )
            return reached
        finally:
            self._motion_stall_suspended = False
            self._refresh_motion_stall_reference()

    def remaining_path_arc_m(self, path_index: int) -> float:
        if not self.path_speed_profile:
            return math.inf
        end_arc = float(self.path_speed_profile[-1]["arc_length_m"])
        start_arc = float(self.path_speed_profile[path_index]["arc_length_m"])
        return max(0.0, end_arc - start_arc)

    def _remaining_c_path_from_pose(
        self,
        planning_pose: tuple[float, float, float],
    ) -> list[tuple[float, float, float]] | None:
        """Suffix of the global C path nearest the live pose, if the join is tight."""
        cached = self.planned_paths.get("C") or []
        if len(cached) < 2:
            return None
        nearest = min(
            range(len(cached)),
            key=lambda index: math.hypot(
                cached[index][0] - planning_pose[0],
                cached[index][1] - planning_pose[1],
            ),
        )
        gap = math.hypot(
            cached[nearest][0] - planning_pose[0],
            cached[nearest][1] - planning_pose[1],
        )
        if gap > float(self.args.initial_pose_tolerance):
            return None
        suffix = [tuple(map(float, row)) for row in cached[nearest:]]
        if len(suffix) < 2:
            return None
        self.get_logger().info(
            f"C-leg joined global path at index {nearest} "
            f"(gap={gap:.3f}m, poses={len(suffix)})"
        )
        return suffix

    def _drive_c_leg_hybrid(
        self,
        scored: list[tuple[str, tuple[float, float]]],
        next_scored: int,
        deadline: float,
    ) -> tuple[int, list[str], bool]:
        """Settle, take the remaining C path, track it with the profiled controller.

        MPPI FollowPath stalled the 2WS chassis on C in sessions 110758 and
        112714 (crawl then 5 s at ~0.02 m/s). A/B already succeed with this
        same profiled tracker, so C stays on it after a 50 ms settle.
        """
        c_item = next((item for item in scored if item[0] == "C"), None)
        if c_item is None:
            raise RuntimeError("C-leg local execution requires scored point C")
        _, c_target = c_item
        if self.pose is None:
            raise RuntimeError("cannot start C-leg without localization pose")
        self.current_leg_name = "C"
        self._hold_zero_for_local_replan(VALIDATION_LOCAL_REPLAN_SETTLE_S)
        planning_pose = tuple(float(value) for value in self.pose)
        odom_speed = abs(float(self.current_odom_speed))
        join_speed: float | None = max(
            float(self.args.checkpoint_capture_min_speed),
            odom_speed,
        )
        points = self._remaining_c_path_from_pose(planning_pose)
        if points is None:
            self.get_logger().info(
                "C-leg join gap too large; local-replanning C from live pose"
            )
            points = self._plan_mission_leg_points(
                [("C", c_target)],
                planning_pose,
                terminal=True,
                next_target=None,
            )
        if len(points) < 2:
            raise RuntimeError("C-leg local plan is degenerate")
        points = densify_path(points, spacing=0.12)
        self.get_logger().info(
            f"C-leg local profiled track: {len(points)} poses from live pose "
            f"({planning_pose[0]:.3f},{planning_pose[1]:.3f})"
        )
        next_scored, advanced, finished_c, _join = self._drive_profiled_segment(
            points,
            scored,
            next_scored,
            deadline,
            join_speed=join_speed,
            finish_at_end=True,
            depart_from_rest=odom_speed < 0.05,
            force_leg="C",
        )
        if finished_c:
            self._official_circle_capture_and_dwell("C", c_target, deadline)
            return next_scored, advanced, True
        if self._crawl_scored_checkpoint_until("C", c_target, deadline):
            next_scored, segment_advanced, finished_c = (
                self._advance_scored_checkpoints(scored, next_scored)
            )
            advanced.extend(segment_advanced)
            if finished_c:
                self._official_circle_capture_and_dwell("C", c_target, deadline)
                return next_scored, advanced, True
        return next_scored, advanced, False

    def drive_profiled_through_route(self, route, deadline: float) -> list[str]:
        """Track the preflight path directly with its jerk-aware speed caps."""
        scored = [(name, target) for name, target in route if not name.startswith("_")]
        if not scored or scored[-1][0] != self.route_terminal_name:
            raise RuntimeError(
                f"continuous route must end at scored point {self.route_terminal_name}"
            )
        points = self._continuous_route_points(route)
        path = np.asarray(points, dtype=np.float64)
        if len(path) < 2:
            raise RuntimeError("continuous preflight path is degenerate")
        path_arc_lengths = cumulative_arc_lengths(
            (tuple(row) for row in path)
        )
        reached: list[str] = []
        next_scored = 0
        nearest_index = 0
        last_published_speed_cap = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.ensure_motion_allowed()
            if self.pose is None:
                continue
            x, y, yaw = self.pose
            position = np.asarray((x, y))
            path_index, arc_m, cross_track_error, signed_cross_track = (
                self.monotonic_path_projection(
                    path, position, nearest_index, path_arc_lengths,
                )
            )
            nearest_index = max(nearest_index, path_index)
            leg_name = self._leg_name_at_arc(arc_m)
            self.current_leg_name = leg_name
            remaining_arc = max(
                0.0,
                float(path_arc_lengths[-1]) - arc_m,
            )
            limit_index, speed_cap, cap_reason = self.coordinated_speed_cap(
                arc_m, cross_track_error, remaining_arc_m=remaining_arc,
            )
            speed_cap = self._score_aware_speed_cap(
                speed_cap,
                scored,
                next_scored,
            )
            if (last_published_speed_cap is None
                    or abs(speed_cap - last_published_speed_cap) >= 0.05):
                self.publish_speed_limit(speed_cap, limit_index, cap_reason)
                last_published_speed_cap = speed_cap
            local_curvature = abs(
                float(self.path_speed_profile[limit_index]["curvature_1pm"])
            )
            preview_curvature = self._path_curvature_preview(limit_index, preview_m=1.5)
            flythrough = self._checkpoint_flythrough_override(
                scored=scored,
                next_scored=next_scored,
                path=path,
                path_index=path_index,
                limit_index=limit_index,
                speed_cap=speed_cap,
                x=x,
                y=y,
                yaw=yaw,
                signed_cross_track=signed_cross_track,
            )
            if flythrough is not None:
                linear, angular, control_mode = flythrough
            else:
                linear, angular = self._pure_pursuit_command(
                    path, position, yaw, path_index, speed_cap,
                    cross_track_m=cross_track_error,
                    curvature_1pm=max(local_curvature, preview_curvature),
                )
                control_mode = "pure_pursuit"
            if next_scored < len(scored):
                name, task_target = scored[next_scored]
                distance = self.task_distance(
                    self.coverage_target(name, task_target)
                )
                judgement_pose = self.judgement_pose()
                self.samples.append((
                    time.time(), name, *judgement_pose, distance, linear, angular
                ))
                if self.checkpoint_satisfied(name, self.coverage_target(name, task_target)):
                    self.mark_reached(name, self.coverage_target(name, task_target))
                    self.checkpoint_pass_speeds[name] = abs(self.current_odom_speed)
                    self.checkpoint_pass_speeds_detail[name] = {
                        "odom_mps": abs(self.current_odom_speed),
                        "commanded_mps": abs(linear),
                    }
                    if name not in self.leg_timestamps:
                        self.leg_timestamps[name] = time.monotonic()
                    reached.append(name)
                    next_scored += 1
                    self.get_logger().info(
                        f"{'arrived at' if name == 'C' else 'passed'} {name} "
                        f"at speed={abs(self.current_odom_speed):.3f}m/s, "
                        f"error={distance:.3f}m"
                    )
                    if name in ("A", "B"):
                        self.settle_checkpoint(
                            name, self.coverage_target(name, task_target),
                        )
                    if name == self.route_terminal_name:
                        for _ in range(3):
                            self.command()
                            rclpy.spin_once(self, timeout_sec=0.02)
                        self.publish_speed_limit(None)
                        return reached
            audit_yaw = self._profile_audit_yaw(path, path_index)
            self._publish_motion_command(
                linear,
                angular,
                control_mode=control_mode,
                speed_cap=speed_cap,
                cap_reason=cap_reason,
                cross_track_m=cross_track_error,
                audit_path_yaw=audit_yaw,
            )
        self.command()
        self.publish_speed_limit(None)
        missing = [name for name, _target in scored[next_scored:]]
        if missing and VALIDATION_MISSION_DEADLINE_ENABLED:
            raise TimeoutError(f"profiled route timed out before {missing}")
        return reached

    def drive_nav2(self, name: str, target: tuple[float, float], deadline: float) -> None:
        """Execute the Nav2 global path with the Nav2 MPPI FollowPath controller."""
        goal_yaw = self.waypoint_goal_yaw(name, self.pose[2] if self.pose else None)
        if name in ("B", "_B_exit", "_A_exit") and self.pose is not None:
            heading_error = abs(self.wrap_angle(goal_yaw - self.pose[2]))
            if heading_error > 0.40:
                self.align_heading(name, goal_yaw, deadline)
        planning_pose = self.pose
        goal_yaw = self.waypoint_goal_yaw(name, planning_pose[2] if planning_pose else None)
        start_yaw = planning_pose[2] if planning_pose is not None else None
        start_xy = planning_pose[:2] if planning_pose is not None else None
        points = self.planned_points(target, goal_yaw, start_yaw, start_xy)
        self.ensure_path_safe(name, points)
        self.planned_paths[name] = points
        self.write_plan_snapshot()
        self.publish_route_visualization()
        if not self.controller.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Nav2 /follow_path action is unavailable")
        path = NavPath()
        path.header.frame_id = "map"
        # Zero requests the latest available transform and avoids a startup
        # race against the latched map->odom transform.
        path.header.stamp.sec = 0
        path.header.stamp.nanosec = 0
        from geometry_msgs.msg import PoseStamped
        for x, y, yaw in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            path.poses.append(pose)
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = "FollowPath"
        send = self.controller.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Nav2 FollowPath rejected path to {name}")
        result_future = handle.get_result_async()
        cancelled_for_capture = False
        while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.ensure_motion_allowed()
            if self.pose is not None:
                x, y, yaw = self.pose
                task_target = self.coverage_target(name, target)
                distance = self.task_distance(task_target)
                self.samples.append((time.time(), name, x, y, yaw, distance, math.nan, math.nan))
                if distance <= self.coordinate_tolerance(name):
                    handle.cancel_goal_async()
                    cancelled_for_capture = True
                    break
        if cancelled_for_capture:
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=2.0)
            self.finish_nav2_capture(name, target, deadline, "FollowPath")
            return
        if not result_future.done():
            handle.cancel_goal_async()
            if not self.target_reached(name, self.coverage_target(name, target)):
                raise TimeoutError(f"Nav2 FollowPath timed out before {name}")
            self.get_logger().warning(
                f"Nav2 FollowPath timed out within coordinate tolerance at {name}"
            )
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code != 0:
            code = wrapped.result.error_code if wrapped is not None else "missing"
            message = wrapped.result.error_msg if wrapped is not None else "missing result"
            if not self.target_reached(name, self.coverage_target(name, target)):
                raise RuntimeError(f"Nav2 FollowPath failed to {name}: {code} {message}")
            self.get_logger().warning(
                f"Nav2 FollowPath returned {code} at {name}, but pose is within approach envelope; "
                "accepting robot-center coordinate tolerance"
            )
        self.finish_nav2_capture(name, target, deadline, "FollowPath")

    def navigate_nav2(self, name: str, target: tuple[float, float], deadline: float) -> None:
        """Run Nav2's replanning behavior tree for dynamic RGB-D obstacles."""
        if not self.navigator.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("Nav2 /navigate_to_pose action is unavailable")
        goal_yaw = self.waypoint_goal_yaw(name, self.pose[2] if self.pose else None)
        if name in ("B", "_B_exit", "_A_exit") and self.pose is not None:
            heading_error = abs(self.wrap_angle(goal_yaw - self.pose[2]))
            if heading_error > 0.40:
                self.align_heading(name, goal_yaw, deadline)
                goal_yaw = self.waypoint_goal_yaw(name, self.pose[2] if self.pose else None)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp.sec = 0
        goal.pose.header.stamp.nanosec = 0
        goal.pose.pose.position.x, goal.pose.pose.position.y = target
        goal.pose.pose.orientation.z = math.sin(goal_yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(goal_yaw * 0.5)
        sent = self.navigator.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=15.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Nav2 rejected NavigateToPose goal {name}")
        result_future = handle.get_result_async()
        cancelled_for_capture = False
        while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.ensure_motion_allowed()
            if self.pose is not None:
                x, y, yaw = self.pose
                task_target = self.coverage_target(name, target)
                distance = self.task_distance(task_target)
                self.samples.append((time.time(), name, x, y, yaw, distance, math.nan, math.nan))
                if distance <= self.coordinate_tolerance(name):
                    handle.cancel_goal_async()
                    cancelled_for_capture = True
                    break
        if cancelled_for_capture:
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=2.0)
            self.finish_nav2_capture(name, target, deadline, "NavigateToPose")
            return
        if not result_future.done():
            handle.cancel_goal_async()
            if not self.target_reached(name, self.coverage_target(name, target)):
                raise TimeoutError(f"NavigateToPose timed out before {name}")
            self.get_logger().warning(
                f"NavigateToPose timed out within coordinate tolerance at {name}"
            )
        wrapped = result_future.result()
        if wrapped is None or wrapped.status != 4:
            status = wrapped.status if wrapped is not None else "missing"
            code = wrapped.result.error_code if wrapped is not None else "missing"
            if not self.target_reached(name, self.coverage_target(name, target)):
                raise RuntimeError(f"NavigateToPose failed to {name}: status={status}, code={code}")
            self.get_logger().warning(
                f"NavigateToPose returned status={status} at {name}; "
                "accepting robot-center coordinate tolerance"
            )
        self.finish_nav2_capture(name, target, deadline, "NavigateToPose")

    def finish_nav2_capture(self, name: str, target: tuple[float, float],
                            deadline: float, source: str) -> None:
        x, y, _ = self.pose
        task_target = self.coverage_target(name, target)
        remaining = self.task_distance(task_target)
        if self.target_reached(name, task_target):
            self.get_logger().info(
                f"Nav2 {source} reached {name} by robot center at "
                f"({x:.3f}, {y:.3f}), error={remaining:.3f}m"
            )
            self.mark_reached(name, task_target)
            return
        raise RuntimeError(
            f"Nav2 {source} finished {remaining:.3f}m from {name}, outside "
            f"robot-center tolerance {self.coordinate_tolerance(name):.3f}m: "
            f"pose={self.pose}"
        )


def build_route_preview(
    node: "ABCNavigator",
    route: list[tuple[str, tuple[float, float]]],
) -> tuple[dict[str, Any], bool]:
    route_preview: dict[str, Any] = {}
    previous_end: tuple[float, float] | None = None
    route_continuous = True
    for name, path in node.planned_paths.items():
        if not path:
            route_continuous = False
            continue
        xy = [(float(q[0]), float(q[1])) for q in path]
        length = sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(xy, xy[1:])
        )
        start_gap = (
            None if previous_end is None else
            math.hypot(xy[0][0] - previous_end[0], xy[0][1] - previous_end[1])
        )
        if start_gap is not None and start_gap > 0.10:
            route_continuous = False
        route_preview[name] = {
            "start_xy": list(xy[0]),
            "end_xy": list(xy[-1]),
            "length_m": float(length),
            "bbox_xy": [
                min(x for x, _ in xy), max(x for x, _ in xy),
                min(y for _, y in xy), max(y for _, y in xy),
            ],
            "start_gap_from_previous_m": start_gap,
            "pose_count": len(path),
            "minimum_clearance_m": node.path_safety_reports.get(name, {}).get(
                "minimum_clearance_m"
            ),
        }
        previous_end = xy[-1]
    return route_preview, route_continuous


def write_robot_started_marker(
    output_dir: Path,
    start_pose: tuple[float, float, float] | None,
    logger: Any,
) -> None:
    logger.info("robot_started")
    (output_dir / ROBOT_STARTED_MARKER_NAME).write_text(
        json.dumps(
            {
                "started_at_unix": time.time(),
                "start_pose": [float(v) for v in start_pose] if start_pose else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def export_zero_motion_preflight(
    node: "ABCNavigator",
    args: argparse.Namespace,
    route: list[tuple[str, tuple[float, float]]],
    points: dict[str, tuple[float, float]],
    output_dir: Path,
    start_pose: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """Persist zero-motion route planning artifacts for the validation parent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "planned_paths.json").write_text(
        json.dumps(node.planned_paths, indent=2), encoding="utf-8"
    )
    (output_dir / "path_speed_profile.json").write_text(
        json.dumps(node.path_speed_profile, indent=2), encoding="utf-8"
    )
    route_preview, route_continuous = build_route_preview(node, route)
    result = {
        "status": "succeeded",
        "error": None,
        "points": points,
        "start_pose": start_pose,
        "planned_path_pose_counts": {
            name: len(path) for name, path in node.planned_paths.items()
        },
        "path_safety_reports": node.path_safety_reports,
        "route_preview": {
            "ordered_names": [name for name, _ in route],
            "continuous": route_continuous,
            "segments": route_preview,
            "note": "zero-motion route geometry; no chassis commands are published",
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def drive_continuous_route(
    node: "ABCNavigator",
    route: list[tuple[str, tuple[float, float]]],
    deadline: float,
    args: argparse.Namespace,
) -> list[str]:
    if args.use_hybrid_theta_mppi:
        return node.drive_hybrid_theta_mppi_route(route, deadline)
    if args.use_profiled_controller:
        return node.drive_profiled_through_route(route, deadline)
    return node.drive_nav2_through_route(route, deadline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument(
        "--truth-pose-topic", default="",
        help="authorized g1_omnipicker world-pose aid; empty disables it",
    )
    parser.add_argument(
        "--truth-body-link-topic", default="",
        help="body_link1 world pose for official scorer alignment; empty disables",
    )
    parser.add_argument(
        "--truth-world-to-map", default="0,0,0",
        help="comma-separated world→map TX,TY,YAW so negative values parse reliably",
    )
    parser.add_argument("--localization-pose-topic", default="/rtabmap/localization_pose")
    parser.add_argument("--global-pose-topic", default="/rtabmap/global_pose")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--speed-limit-topic", default="/speed_limit")
    parser.add_argument("--status-topic", default="/task1/navigation_status")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--occupied-threshold",type=int,default=65)
    parser.add_argument("--path-safety-clearance",type=float,default=0.03)
    parser.add_argument("--command-safety-clearance", type=float, default=0.05)
    parser.add_argument("--command-safety-horizon", type=float, default=1.50)
    parser.add_argument(
        "--safety-stop", action=argparse.BooleanOptionalAction, default=True,
        help="stop on predicted map-footprint clearance violation (disable for footprint-only validation)",
    )
    parser.add_argument("--truth-pose-timeout", type=float, default=2.0,
                        help="deprecated compatibility alias; visual pose timeout is used")
    parser.add_argument("--visual-pose-timeout", type=float, default=0.5)
    parser.add_argument("--startup-pose-samples", type=int, default=5)
    parser.add_argument("--max-pose-step", type=float, default=0.10)
    parser.add_argument("--max-yaw-step", type=float, default=0.08)
    parser.add_argument("--max-tf-age", type=float, default=1.0,
                        help="maximum age of latest map->base_link TF")
    parser.add_argument("--visual-max-position-variance", type=float, default=0.25)
    parser.add_argument("--visual-max-yaw-variance", type=float, default=0.25)
    parser.add_argument(
        "--waypoint-coordinate-tolerance", type=float, default=0.60,
        help="robot-center XY radius used to count passage through A and B",
    )
    parser.add_argument(
        "--final-coordinate-tolerance", type=float, default=0.60,
        help="robot-center XY radius used to count arrival at C",
    )
    parser.add_argument(
        "--score-region-radius", type=float, default=0.60,
        help="official XY region radius; navigation stop stays inside this circle",
    )
    parser.add_argument(
        "--score-circle-stop-offset", type=float, default=0.25,
        help="how far short of the final site the planned stop is placed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.24)
    parser.add_argument("--max-angular", type=float, default=1.0)
    parser.add_argument(
        "--max-path-curvature", type=float,
        default=VALIDATION_MAX_PATH_CURVATURE_1PM,
        help=("steering-base curvature ceiling in 1/m; every commanded yaw rate "
              "is bound to linear_speed * this value so v and w stay consistent"),
    )
    parser.add_argument("--heading-gain", type=float, default=1.4)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--approach-tolerance", type=float, default=0.10)
    parser.add_argument("--final-tolerance", type=float, default=0.12)
    parser.add_argument("--path-spacing", type=float, default=0.40)
    parser.add_argument("--path-tolerance", type=float, default=0.28)
    parser.add_argument("--lookahead-min", type=float, default=0.55)
    parser.add_argument("--lookahead-gain", type=float, default=0.5)
    parser.add_argument("--steering-ramp-rate", type=float, default=1.0)
    parser.add_argument("--max-curvature", type=float, default=2.5)
    parser.add_argument("--steering-gain", type=float, default=0.30)
    parser.add_argument("--startup-steering-gain", type=float, default=0.85,
                        help="higher gain for the initial A ingress bend")
    parser.add_argument("--final-steering-gain", type=float, default=0.65,
                        help="lower gain for the long straight C corridor")
    parser.add_argument("--final-max-curvature", type=float, default=1.2,
                        help="curvature cap for the long straight C corridor")
    parser.add_argument("--curvature-slowdown", type=float, default=0.35)
    parser.add_argument("--min-speed-ratio", type=float, default=0.60)
    parser.add_argument("--goal-slowdown-distance", type=float, default=0.80)
    parser.add_argument("--min-goal-speed-ratio", type=float, default=0.10)
    parser.add_argument("--goal-capture-distance", type=float, default=0.30)
    parser.add_argument("--goal-position-gain", type=float, default=0.35)
    parser.add_argument("--goal-min-speed", type=float, default=0.012)
    parser.add_argument("--goal-steering-gain", type=float, default=0.45)
    parser.add_argument("--goal-stable-cycles", type=int, default=30)
    parser.add_argument("--reverse-threshold", type=float, default=math.pi / 2.0)
    parser.add_argument("--direction-switch-angle", type=float, default=math.radians(80.0))
    parser.add_argument("--min-heading-speed-ratio", type=float, default=0.25)
    parser.add_argument("--min-direction-run", type=int, default=3)
    parser.add_argument("--cusp-tolerance", type=float, default=0.22)
    parser.add_argument("--direction-lookahead", type=float, default=0.50)
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--initial-map-pose", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--judgement-offset-xy", nargs=2, type=float, default=(0.0, 0.0),
        help="fixed g1_omnipicker root XY offset in navigation base frame",
    )
    parser.add_argument("--initial-pose-tolerance", type=float, default=2.0,
                        help="maximum startup distance from saved map pose before motion")
    parser.add_argument("--forced-directions-json", default="{}")
    parser.add_argument("--use-nav2-path", action="store_true")
    parser.add_argument("--use-nav2-controller", action="store_true")
    parser.add_argument(
        "--use-profiled-controller",
        action="store_true",
        help="track the validated global path directly using dynamic speed caps",
    )
    parser.add_argument(
        "--use-hybrid-theta-mppi",
        action="store_true",
        help="Theta* straights via profiled cruise; corner arcs via MPPI Ackermann",
    )
    parser.add_argument(
        "--global-planner-id",
        default="GridBased",
        help="Nav2 planner_server plugin namespace (default GridBased)",
    )
    parser.add_argument(
        "--global-planner-plugin",
        choices=("smac_hybrid", "theta_star", "smac_2d"),
        default="smac_hybrid",
        help="global planner backend used by /compute_path_to_pose",
    )
    parser.add_argument("--use-nav2-navigate", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="compute and safety-check every route path without publishing motion commands")
    parser.add_argument(
        "--inline-preflight",
        action="store_true",
        help="plan the route in-process, export preflight artifacts, then execute",
    )
    parser.add_argument(
        "--preflight-output-dir",
        type=Path,
        help="directory for zero-motion preflight artifacts when using --inline-preflight",
    )
    parser.add_argument(
        "--preplanned-paths",
        type=Path,
        help="reuse and revalidate planned_paths.json from zero-motion preflight",
    )
    parser.add_argument(
        "--route-stop-at", default="C", choices=("A", "B", "C"),
        help="terminal scored checkpoint for this mission (default: C)",
    )
    parser.add_argument(
        "--continuous-route", action=argparse.BooleanOptionalAction, default=True,
        help="use one FollowPath through intermediate points; stop only at route terminal",
    )
    parser.add_argument("--profile-lateral-accel", type=float, default=1.2)
    parser.add_argument("--profile-linear-accel", type=float, default=1.2)
    parser.add_argument("--profile-linear-jerk", type=float, default=2.0)
    parser.add_argument("--profile-clearance-hard", type=float, default=0.08)
    parser.add_argument("--profile-clearance-soft", type=float, default=0.45)
    parser.add_argument("--minimum-profile-speed", type=float, default=0.18)
    parser.add_argument("--profile-lookahead", type=float, default=0.65)
    parser.add_argument("--final-stop-speed", type=float, default=0.08)
    parser.add_argument(
        "--terminal-arrival-speed", type=float, default=0.45,
        help="non-zero speed cap at the trimmed path end (m/s)",
    )
    parser.add_argument(
        "--terminal-zone-m", type=float, default=VALIDATION_TERMINAL_ZONE_M,
        help="arc length over which terminal arrival speed ramps in (m)",
    )
    parser.add_argument(
        "--checkpoint-capture-min-speed", type=float, default=0.35,
        help="minimum linear speed while steering into a scored checkpoint",
    )
    parser.add_argument(
        "--peak-contract-speed", type=float, default=None,
        help="optional higher straight-line contract; curves still use --speed",
    )
    parser.add_argument(
        "--straight-curvature-1pm", type=float, default=0.25,
        help="curvature threshold for peak-contract straights (1/m)",
    )
    parser.add_argument("--timeout", type=float, default=VALIDATION_MISSION_TIMEOUT_S)
    parser.add_argument(
        "--motion-stall-speed-mps", type=float,
        default=VALIDATION_MOTION_STALL_SPEED_MPS,
        help="odom speed below this counts as stopped once the mission is underway",
    )
    parser.add_argument(
        "--motion-stall-displacement-m", type=float,
        default=VALIDATION_MOTION_STALL_DISPLACEMENT_M,
        help="map-frame displacement that refreshes the motion-progress clock",
    )
    parser.add_argument(
        "--motion-stall-timeout-s", type=float,
        default=VALIDATION_MOTION_STALL_TIMEOUT_S,
        help="fail the mission after this many seconds without motion progress",
    )
    parser.add_argument(
        "--motion-stall-startup-grace-s", type=float,
        default=VALIDATION_MOTION_STALL_STARTUP_GRACE_S,
        help="allow Nav2/controller startup before requiring first movement",
    )
    parser.add_argument(
        "--checkpoint-settle", type=float, default=0.0,
        help="zero-command dwell recorded after non-A/B waypoints (C audit window)",
    )
    parser.add_argument(
        "--ab-checkpoint-dwell", type=float, default=VALIDATION_AB_CHECKPOINT_DWELL_S,
        help="zero-command hold at scored A/B fly-through points (seconds)",
    )
    parser.add_argument(
        "--post-robot-start-dwell", type=float, default=VALIDATION_POST_ROBOT_START_DWELL_S,
        help="zero-command hold after robot_started.marker before route motion (seconds)",
    )
    parser.add_argument(
        "--waypoints-json",
        help='ordered object of waypoint names to [x,y], e.g. {"A":[1,0],"B":[2,0]}',
    )
    parser.add_argument("--task-points-json", default="{}",
                        help="exact task coordinates used for robot-center distance checks")
    args = parser.parse_args()
    try:
        truth_se2 = [float(part) for part in str(args.truth_world_to_map).split(",")]
        if len(truth_se2) != 3:
            raise ValueError
        args.truth_world_to_map = tuple(truth_se2)
    except ValueError:
        parser.error("--truth-world-to-map must be TX,TY,YAW")
    args.forced_directions = json.loads(args.forced_directions_json)
    args.task_points = {name: tuple(map(float, xy))
                        for name, xy in json.loads(args.task_points_json).items()}
    if any(value not in {"forward", "reverse"} for value in args.forced_directions.values()):
        parser.error("--forced-directions-json values must be forward or reverse")
    if args.checkpoint_settle < 0.0:
        parser.error("--checkpoint-settle must be non-negative")
    if args.ab_checkpoint_dwell < 0.0:
        parser.error("--ab-checkpoint-dwell must be non-negative")
    if args.post_robot_start_dwell < 0.0:
        parser.error("--post-robot-start-dwell must be non-negative")
    if args.preplanned_paths is not None and not args.preplanned_paths.is_file():
        parser.error(f"preplanned path file does not exist: {args.preplanned_paths}")
    if args.inline_preflight:
        if args.plan_only:
            parser.error("--inline-preflight cannot be combined with --plan-only")
        if args.preplanned_paths is not None:
            parser.error("--inline-preflight cannot be combined with --preplanned-paths")
        if args.preflight_output_dir is None:
            parser.error("--inline-preflight requires --preflight-output-dir")
    if args.command_safety_clearance < 0.0 or args.command_safety_horizon <= 0.0:
        parser.error("command safety clearance/horizon configuration is invalid")
    if (args.waypoint_coordinate_tolerance <= 0.0 or
            args.final_coordinate_tolerance <= 0.0 or
            args.score_region_radius <= 0.0 or
            args.score_circle_stop_offset < 0.0 or
            args.score_circle_stop_offset >= args.score_region_radius):
        parser.error("coordinate tolerances and score-circle stop must be valid")
    if (args.profile_lateral_accel <= 0.0 or args.profile_linear_accel <= 0.0
            or args.profile_linear_jerk <= 0.0 or args.minimum_profile_speed <= 0.0
            or args.profile_lookahead < 0.0 or args.final_stop_speed < 0.0
            or args.terminal_arrival_speed < 0.0
            or args.terminal_zone_m < 0.0
            or args.checkpoint_capture_min_speed < 0.0
            or args.straight_curvature_1pm < 0.0
            or not 0.0 <= args.profile_clearance_hard < args.profile_clearance_soft):
        parser.error("path speed profile configuration is invalid")
    if (args.peak_contract_speed is not None
            and args.peak_contract_speed + 1e-9 < args.speed):
        parser.error("--peak-contract-speed must be >= --speed")
    controller_flags = sum(bool(flag) for flag in (
        args.use_nav2_controller,
        args.use_profiled_controller,
        args.use_hybrid_theta_mppi,
    ))
    if controller_flags > 1:
        parser.error(
            "choose only one of --use-nav2-controller, --use-profiled-controller, "
            "or --use-hybrid-theta-mppi"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = {"A": (0.0, 0.0), "B": (0.0, 1.2), "D": (-0.8, 1.2), "C": (0.0, 2.4)}
    route = [("D", points["D"]), ("C", points["C"])]
    if args.waypoints_json:
        supplied = json.loads(args.waypoints_json)
        if not isinstance(supplied, dict) or not supplied:
            parser.error("--waypoints-json must be a non-empty JSON object")
        points = {name: tuple(map(float, xy)) for name, xy in supplied.items()}
        if any(len(xy) != 2 for xy in points.values()):
            parser.error("each waypoint must contain exactly [x, y]")
        route = []
        for name, xy in points.items():
            route.append((name, xy))
            # Do not inject a synthetic departure waypoint.  A/B are exact
            # competition coordinates and the saved-map Hybrid-A* planner
            # must choose the valid departure direction for the current map.

    rclpy.init()
    node = ABCNavigator(args)
    node.task_points = args.task_points
    node.points = points
    status, error = "failed", None
    start_pose = None
    final_pose = None
    reached = []
    started = time.monotonic()
    try:
        node.wait_for_pose()
        # A pose sample can arrive before the static map->odom bridge and the
        # corrected odom->base_link TF have settled.  Require a fresh sample
        # immediately before the first actuator command; runtime checks remain
        # strict and still fail-stop if localization is later lost.
        node.wait_for_fresh_visual_pose(timeout=10.0)
        node.enable_motion()
        start_pose = node.pose
        zero_motion_plan = args.plan_only or args.inline_preflight
        if zero_motion_plan:
            node.plan_pose = tuple(float(v) for v in start_pose)
        elif args.use_nav2_navigate:
            # Publish the complete nominal route before motion. Nav2 may
            # replan the active leg around live RGB-D obstacles; both paths
            # remain visible in RViz on separate topics.
            node.preview_route(route)
        deadline = (
            started + args.timeout
            if VALIDATION_MISSION_DEADLINE_ENABLED
            else math.inf
        )
        continuous = args.continuous_route and (
            args.use_nav2_controller
            or args.use_profiled_controller
            or args.use_hybrid_theta_mppi
            or args.inline_preflight
            or (args.plan_only and args.use_nav2_path)
        )
        if continuous:
            if zero_motion_plan:
                node.plan_continuous_route(route)
                for name, path in node.planned_paths.items():
                    node.get_logger().info(
                        f"route plan: {name} accepted ({len(path)} poses)"
                    )
                if args.inline_preflight:
                    export_zero_motion_preflight(
                        node, args, route, points, args.preflight_output_dir, start_pose,
                    )
            if args.plan_only:
                reached = [
                    name for name, _target in route if not name.startswith("_")
                ]
            else:
                write_robot_started_marker(
                    args.output_dir, start_pose, node.get_logger(),
                )
                node.dwell_after_robot_start()
                node.begin_route_motion_clock()
                reached = drive_continuous_route(node, route, deadline, args)
        else:
            write_robot_started_marker(
                args.output_dir, start_pose, node.get_logger(),
            )
            node.dwell_after_robot_start()
            node.begin_route_motion_clock()
            for name, target in route:
                if args.use_nav2_navigate:
                    node.navigate_nav2(name, target, deadline)
                elif args.use_nav2_controller:
                    node.drive_nav2(name, target, deadline)
                elif args.use_nav2_path:
                    node.drive_planned(name, target, deadline)
                else:
                    node.drive_to(name, target, deadline)
                if not name.startswith("_"):
                    reached.append(name)
                    node.settle_checkpoint(name, target)
            if args.plan_only and args.continuous_route:
                combined = []
                for path in node.planned_paths.values():
                    combined.extend(path if not combined else path[1:])
                node.build_combined_speed_profile(combined)
        status = "succeeded"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        if rclpy.ok():
            for _ in range(10):
                node.command()
                rclpy.spin_once(node, timeout_sec=0.02)
        final_pose = node.pose
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("unix_time", "leg", "x", "y", "yaw", "distance", "linear", "angular"))
        writer.writerows(node.samples)
    (args.output_dir / "planned_paths.json").write_text(
        json.dumps(node.planned_paths, indent=2), encoding="utf-8"
    )
    (args.output_dir / "path_speed_profile.json").write_text(
        json.dumps(node.path_speed_profile, indent=2), encoding="utf-8"
    )
    b = points.get("B")
    min_b = min((math.hypot(row[2] - b[0], row[3] - b[1]) for row in node.samples), default=None) if b else None
    reached_names = list(node.coordinate_reach)
    final_name = reached_names[-1] if reached_names else route[0][0]
    final_target = node.coverage_target(final_name, points[final_name])
    route_preview, route_continuous = build_route_preview(node, route)
    result = {
        "status": status, "error": error, "points": points,
        "pass_rule": {
            "type": "g1_omnipicker_xy_euclidean_tolerance",
            "description": (
                "A/B passage and C arrival use g1_omnipicker root XY Euclidean distance "
                "sqrt((x1-x2)^2+(y1-y2)^2); Z is ignored and footprint overlap is not used."
            ),
            "judgement_entity": "g1_omnipicker",
            "waypoint_tolerance_m": args.waypoint_coordinate_tolerance,
            "final_tolerance_m": args.final_coordinate_tolerance,
            "collision_is_independent_failure": True,
        },
        "command_safety": {
            "prediction_horizon_s": args.command_safety_horizon,
            "clearance_m": args.command_safety_clearance,
            "check_count": node.command_safety_checks,
            "last_report": node.last_command_safety_report,
            "safety_footprint_xy_m": [list(point) for point in node.safety_footprint],
        },
        "events": [
            "robot_started_moving",
            *[("target_cabinet_reached" if name == "C" else f"checkpoint_{name}_reached")
              for name in reached_names],
        ],
        "start_pose": start_pose, "final_pose": final_pose,
        "final_judgement_pose": node.judgement_pose(),
        "final_waypoint": final_name,
        "final_error_m": (
            node.task_distance(final_target)
            if final_pose is not None else None
        ),
        "minimum_distance_to_B_m": min_b,
        "leg_directions": node.leg_directions,
        "navigation_status": node.navigation_status,
        "localization_filter": {
            "pose_filter_alpha": node.pose_filter_alpha,
            "pose_jump_rejections": node.pose_jump_rejections,
            "max_pose_step_m": args.max_pose_step,
            "max_yaw_step_rad": args.max_yaw_step,
            "startup_pose_samples": args.startup_pose_samples,
        },
        "planned_path_pose_counts": {
            name: len(path) for name, path in node.planned_paths.items()
        },
        "path_safety_reports": node.path_safety_reports,
        "route_preview": {
            "ordered_names": [name for name, _ in route],
            "continuous": route_continuous,
            "segments": route_preview,
            "note": "plan-only route geometry; no chassis commands are published",
        },
        "coordinate_reach": node.coordinate_reach,
        "checkpoint_pass_speeds_mps": node.checkpoint_pass_speeds,
        "checkpoint_pass_speeds_detail": node.checkpoint_pass_speeds_detail,
        "path_speed_profile_summary": node.path_speed_profile_summary,
        "speed_limit_history": node.speed_limit_history,
        "checkpoint_intervals": node.checkpoint_intervals,
        "sample_count": len(node.samples), "elapsed_s": time.monotonic() - started,
    }
    elapsed_s = time.monotonic() - started
    analysis = node.build_mission_analysis(elapsed_s=elapsed_s, status=status)
    node.write_analysis_artifacts(args.output_dir, analysis)
    result["mission_analysis"] = str((args.output_dir / "mission_analysis.json").resolve())
    result["execution_trace"] = str((args.output_dir / "execution_trace.csv").resolve())
    if node.route_geometry_audits:
        result["route_geometry_audit"] = str(
            (args.output_dir / "route_geometry_audit.json").resolve()
        )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if status != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
