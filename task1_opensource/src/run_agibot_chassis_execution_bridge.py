#!/usr/bin/env python3
"""RGB-D task-1 chassis execution bridge.

The task-1 production path is RGB-D/RTAB-Map visual navigation. This module
contains chassis actuation, live-pose arm holding, freshness interlocks and
post-run MuJoCo auditing; it does not provide a production LiDAR localization
source.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import grpc
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from orca_gym.protos import mjc_message_pb2, mjc_message_pb2_grpc
from chassis_motion_profile import (
    LINEAR_ACCEL_LIMIT,
    LINEAR_JERK_LIMIT,
    step_linear_scurve,
    summarize_speed_profile,
)
from task1_scene_contract import (
    CHASSIS_DRIVE_WHEELS,
    CHASSIS_MAX_STEER_RAD,
    CHASSIS_STEER_WHEELS,
    CHASSIS_WHEELBASE_M,
    VALIDATION_MIN_TURNING_RADIUS_M,
)
from wheel_odometry import MujocoWheelOdomPublisher

TIME_STEP = 0.001
FRAME_SKIP = 20
ARM_ZERO_QPOS = np.zeros(7, dtype=np.float64)
ARM_MAPPING_QPOS = {
    "L": np.array([0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0]),
    "R": np.array([0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0, 0.0]),
}


@dataclass(frozen=True)
class ChassisActuator:
    name: str
    index: int
    joint_name: str
    ctrl_range: tuple[float, float]
    wheel: str


def classify_robot_environment_contact(
    geom1: int,
    geom2: int,
    robot_geom_ids: set[int],
    collision_robot_geom_ids: set[int],
    ground_geom_ids: set[int],
) -> tuple[int, int] | None:
    """Return (robot, environment) for a chassis/environment collision."""
    belongs1 = geom1 in robot_geom_ids
    belongs2 = geom2 in robot_geom_ids
    if belongs1 == belongs2:
        return None
    robot_geom = geom1 if belongs1 else geom2
    environment_geom = geom2 if belongs1 else geom1
    if robot_geom not in collision_robot_geom_ids:
        return None
    if environment_geom in ground_geom_ids:
        return None
    return robot_geom, environment_geom


def find_chassis_actuators(model, robot_name: str) -> tuple[list[ChassisActuator], dict[str, ChassisActuator]]:
    """Bind front-wheel drive and front-wheel steer; hold unused rear steer at zero."""
    drives: list[ChassisActuator] = []
    steering: dict[str, ChassisActuator] = {}
    for actuator_name, actuator in model.get_actuator_dict().items():
        joint_name = str(actuator.get("JointName", ""))
        if not (actuator_name.startswith(robot_name + "_") or joint_name.startswith(robot_name + "_")):
            continue
        wheel_match = re.search(r"wheel_(fl|fr|bl|br)_", actuator_name.lower())
        if wheel_match is None:
            continue
        ctrl_range = tuple(float(value) for value in actuator.get("CtrlRange", ()))
        if len(ctrl_range) != 2 or ctrl_range[0] == ctrl_range[1]:
            continue
        item = ChassisActuator(
            actuator_name, model.actuator_name2id(actuator_name), joint_name, ctrl_range,
            wheel_match.group(1),
        )
        if actuator_name.endswith("_joint_mctrl") and item.wheel in CHASSIS_DRIVE_WHEELS:
            drives.append(item)
        elif actuator_name.endswith("_steer_joint_pctrl"):
            steering[item.wheel] = item
    if ({item.wheel for item in drives} != set(CHASSIS_DRIVE_WHEELS)
            or not set(CHASSIS_STEER_WHEELS) <= set(steering)):
        raise RuntimeError(
            f"{robot_name} is not fully compiled: expected FL/FR drive plus FL/FR steering; "
            f"model nu={model.nu}. Open task_1/move.json and start Play before launching this program."
        )
    return drives, steering


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ensure_simulation(_root: Path, timeout: float) -> None:
    """Wait for the user-started scene; never start, stop or replace it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(50051):
            return
        time.sleep(1.0)
    raise RuntimeError("OrcaGym service 50051 is offline; open task_1/move.json and click Play")


def _snapshot_live_state(address: str) -> tuple[np.ndarray, np.ndarray]:
    channel = grpc.insecure_channel(address)
    try:
        response = mjc_message_pb2_grpc.GrpcServiceStub(channel).QueryAllQposQvelQacc(
            mjc_message_pb2.QueryAllQposQvelQaccRequest(), timeout=3.0)
        return np.asarray(response.qpos, dtype=float), np.asarray(response.qvel, dtype=float)
    finally:
        channel.close()


def _restore_live_state(env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    if qpos.shape != (env.model.nq,) or qvel.shape != (env.model.nv,):
        raise RuntimeError("Scene model changed while connecting")
    env.gym._mjData.qpos[:] = qpos
    env.gym._mjData.qvel[:] = qvel
    mujoco.mj_forward(env.gym._mjModel, env.gym._mjData)
    env.loop.run_until_complete(env.gym.set_qpos(qpos))
    env.loop.run_until_complete(env.gym.set_qvel(qvel))
    env.gym.mj_forward()
    env.gym.update_data()
    env.render()


def _yaw_from_matrix(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class AgiBotChassisNavigation(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("agibot_g1_chassis_execution_bridge")
        from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

        self.args = args
        saved_qpos, saved_qvel = _snapshot_live_state(args.orcagym_addr)
        self.env = OrcaGymLocalEnv(
            frame_skip=1,
            orcagym_addr=args.orcagym_addr,
            agent_names=[args.robot_name],
            time_step=TIME_STEP,
        )
        # Constructing the local environment is enough for model metadata.
        # In formal RTAB-Map validation the remote scene has just been reset
        # and odometry is already anchored; writing qpos again here produces
        # an identity/high-variance odom reset. Preserve the live state when
        # requested and use the snapshot only for diagnostics.
        if not args.preserve_live_state:
            _restore_live_state(self.env, saved_qpos, saved_qvel)
        self.model = self.env.gym._mjModel
        self.data = self.env.gym._mjData
        self.drive_wheels, self.steering_wheels = find_chassis_actuators(
            self.env.model, args.robot_name
        )
        if getattr(args, "publish_wheel_odom", False) and getattr(args, "publish_truth_odom", True):
            raise RuntimeError(
                "wheel odometry cannot run with MuJoCo truth odometry; choose one odom->base_link publisher"
            )
        self.wheel_odom = None
        if getattr(args, "publish_wheel_odom", False):
            calibration = Path(args.wheel_odom_calibration).resolve()
            self.wheel_odom = MujocoWheelOdomPublisher(
                self, self.model, self.data, calibration,
                odom_topic=args.wheel_odom_topic,
                odom_frame=args.odom_frame,
                base_frame=args.base_frame,
                publish_tf=True,
            )
        self.arm_joints = self._bind_arm_joints()
        self.arm_joint1_ranges = self._arm_joint1_ranges()
        self.arm_qpos_at_connection = self._read_arm_qpos()
        self.arm_targets = self._arm_targets(args.arm_posture)
        self.arm_position_initialization_applied = args.arm_posture != "current"
        if self.arm_position_initialization_applied:
            self._hold_arm_pose()
        self.holder_name = f"{args.robot_name}_robot_holder1"
        self.holder_id = self.env.model.body_name2id(self.holder_name)
        self.body_link_name = f"{args.robot_name}_body_link1"
        try:
            self.body_link_id = self.env.model.body_name2id(self.body_link_name)
        except ValueError:
            self.body_link_id = None
        self.robot_body_ids = {
            body_id for body_id in range(self.model.nbody)
            if (self.model.body(body_id).name or "").startswith(args.robot_name + "_")
        }
        self.robot_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in self.robot_body_ids
        }
        # Any robot geom touching the environment is a hard mission failure.
        self.collision_robot_geom_ids = set(self.robot_geom_ids)
        self.ground_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
        }

        # Formal visual mode must not expose a MuJoCo ray/laser source to ROS.
        self.scan_pub = (self.create_publisher(LaserScan, args.scan_topic, 10)
                         if getattr(args, "publish_truth_odom", True) else None)
        # The MuJoCo pose stream is an audit aid.  Formal visual-navigation
        # runs disable these publishers so it cannot accidentally become the
        # localization source for Nav2.
        self.odom_pub = (self.create_publisher(Odometry, args.odom_topic, 20)
                         if getattr(args, "publish_truth_odom", True) else None)
        self.tf_pub = TransformBroadcaster(self) if getattr(args, "publish_truth_odom", True) else None
        self.static_tf_pub = (StaticTransformBroadcaster(self)
                              if getattr(args, "publish_truth_odom", True) else None)
        self.create_subscription(Twist, args.cmd_vel_topic, self._on_cmd_vel, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)

        self.latest_cmd_vel = (0.0, 0.0)
        self.latest_cmd_time = -math.inf
        self._last_output = (0.0, 0.0)
        self._last_linear_accel = 0.0
        self._last_output_time = time.monotonic()
        self._profile_elapsed = 0.0
        self._linear_command_samples: list[tuple[float, float]] = []
        # Command-path telemetry is deliberately kept at the actuator bridge
        # boundary.  A mission CSV only contains the command generated by the
        # follower; it cannot tell whether Collision Monitor delivered that
        # command or whether the bridge converted it to a non-zero actuator
        # target.  These counters make an apparent "turn in place" diagnosable
        # without using MuJoCo pose as a control input.
        self.command_stats = {
            "input_count": 0,
            "input_nonzero_count": 0,
            "input_zero_count": 0,
            "input_min_linear": 0.0,
            "input_max_abs_linear": 0.0,
            "input_max_abs_angular": 0.0,
            "output_count": 0,
            "output_nonzero_count": 0,
            "output_zero_count": 0,
            "output_min_linear": 0.0,
            "output_max_abs_linear": 0.0,
            "output_max_abs_steer": 0.0,
            "last_input_linear": 0.0,
            "last_input_angular": 0.0,
            "last_output_linear": 0.0,
            "last_output_steer": 0.0,
            "last_applied_linear": 0.0,
            "last_applied_steer": 0.0,
            "speed_profile": None,
        }
        if getattr(args, "publish_truth_odom", True):
            self.start_xy, self.start_yaw = self._holder_planar_pose()
            self.start_world_xy = self.start_xy.copy()
        else:
            # Pure visual mode has no runtime MuJoCo pose input. These values
            # are unused for control/TF and remain placeholders for the audit
            # report written at shutdown.
            self.start_xy = np.zeros(2, dtype=np.float64)
            self.start_yaw = 0.0
            self.start_world_xy = np.full(2, np.nan, dtype=np.float64)
        self.last_xy = self.start_xy.copy()
        self.last_yaw = self.start_yaw
        self.last_pose_time = time.monotonic()
        self.last_scan_time = -math.inf
        self.last_camera_render_time = -math.inf
        self.camera_render_count = 0
        self.scan_count = 0
        self.scan_finite_min = None
        self.scan_finite_max = None
        self.last_scan_finite_count = 0
        self.forward_obstacle_distance = math.inf
        self.reverse_obstacle_distance = math.inf
        self.minimum_motion_obstacle_distance = None
        self.safety_slowdown_count = 0
        self.emergency_stop_count = 0
        self._safety_state = "clear"
        self.contact_sample_count = 0
        self.robot_environment_contact_count = 0
        self.robot_environment_contacts = []
        self.first_collision = None
        self.collision_stop_latched = False
        self.truth_audit_samples = []
        self.last_truth_audit_time = -math.inf
        self.truth_pose_pub = (
            self.create_publisher(PoseStamped, args.truth_pose_topic, 20)
            if args.publish_truth_assist else None
        )
        self.truth_body_link_pub = (
            self.create_publisher(
                PoseStamped, args.truth_body_link_topic, 20,
            )
            if args.publish_truth_assist and self.body_link_id is not None else None
        )
        self.started_at = datetime.now().astimezone().isoformat()
        self.session_dir = args.output_dir or (
            Path(__file__).resolve().parent / "data" /
            f"agibot_navigation_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_scene_inventory()
        if self.static_tf_pub is not None:
            self._publish_laser_static_tf()
        # A modest ROS timer cadence is sufficient for the physical chassis;
        # the simulator itself advances FRAME_SKIP integration steps inside
        # each callback. Keep the historical 20 ms cadence for compatibility.
        self.create_timer(TIME_STEP * FRAME_SKIP, self._tick)
        self.get_logger().info(
            f"Chassis execution bridge ready (production localization: RGB-D/RTAB-Map); "
            f"range={args.range_min:.2f}-{args.range_max:.2f} m, arm_posture={args.arm_posture}, "
            f"arm_position_initialization_applied={self.arm_position_initialization_applied}, "
            "autonomous /cmd_vel control only; interactive keyboard control disabled; "
            f"output={self.session_dir}"
        )

    def _bind_arm_joints(self) -> dict[str, list[tuple[str, int, int]]]:
        joints: dict[str, list[tuple[str, int, int]]] = {}
        for key, side, prefix in (("L", "l", 2), ("R", "r", 6)):
            joints[key] = []
            for number in range(1, 8):
                name = f"{self.args.robot_name}_idx{prefix}{number}_arm_{side}_joint{number}"
                joint_id = self.env.model.joint_name2id(name)
                joints[key].append((
                    name,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                ))
        return joints

    def _arm_joint1_ranges(self) -> dict[str, tuple[float, float]]:
        ranges = {}
        for key in ("L", "R"):
            joint_id = self.env.model.joint_name2id(self.arm_joints[key][0][0])
            ranges[key] = tuple(float(value) for value in self.model.jnt_range[joint_id])
        return ranges

    def _read_arm_qpos(self) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(
                [self.data.qpos[qpos_adr]
                 for _name, qpos_adr, _dof_adr in self.arm_joints[key]],
                dtype=np.float64,
            )
            for key in ("L", "R")
        }

    def _arm_targets(self, posture: str) -> dict[str, np.ndarray]:
        self._recorded_body_quats = {}
        self._recorded_body_positions = {}
        if posture == "current":
            # Preserve the pose configured in OrcaLab. Capture it after the
            # live scene snapshot is restored and do not load a project preset.
            return {key: values.copy()
                    for key, values in self.arm_qpos_at_connection.items()}
        if posture == "mapping":
            targets = {key: value.copy() for key, value in ARM_MAPPING_QPOS.items()}
            # The prefab elbow limits only allow folding upward. Mapping needs
            # the full chain downward, so widen only this process-local model;
            # the prefab, scene JSON, and editor properties remain untouched.
            for key in ("L", "R"):
                elbow_name = self.arm_joints[key][3][0]
                elbow_id = self.env.model.joint_name2id(elbow_name)
                self.model.jnt_range[elbow_id] = np.array([-np.pi, np.pi])
            return targets
        if self.args.robot_name == "service_robot_1":
            posture_file = Path(__file__).with_name("service_robot_navigation_posture.json")
            recorded = json.loads(posture_file.read_text(encoding="utf-8"))
            values = recorded["arm_joint_qpos_rad"]
            targets = {
                "L": np.asarray(values["left"], dtype=np.float64),
                "R": np.asarray(values["right"], dtype=np.float64),
            }
            self._recorded_body_quats = recorded["body_parent_relative_quat_wxyz"]
            self._recorded_body_positions = recorded.get("body_parent_relative_xyz", {})
            return targets
        return {"L": ARM_ZERO_QPOS.copy(), "R": ARM_ZERO_QPOS.copy()}

    def _hold_arm_pose(self) -> None:
        for key in ("L", "R"):
            for target, (_name, qpos_adr, dof_adr) in zip(self.arm_targets[key], self.arm_joints[key]):
                self.data.qpos[qpos_adr] = target
                self.data.qvel[dof_adr] = 0.0
        self.env.mj_forward()
        for body_name, expected in getattr(self, "_recorded_body_quats", {}).items():
            body_id = self.env.model.body_name2id(body_name)
            actual = np.asarray(self.model.body_quat[body_id], dtype=np.float64)
            expected = np.asarray(expected, dtype=np.float64)
            # q and -q encode the same rotation.
            error = min(np.linalg.norm(actual-expected), np.linalg.norm(actual+expected))
            if error > 1e-5:
                raise RuntimeError(
                    f"Recorded navigation posture no longer matches asset rotation for "
                    f"{body_name}: quaternion error={error:.3g}"
                )
        for body_name, expected in getattr(self, "_recorded_body_positions", {}).items():
            body_id = self.env.model.body_name2id(body_name)
            actual = np.asarray(self.model.body_pos[body_id], dtype=np.float64)
            error = np.linalg.norm(actual-np.asarray(expected, dtype=np.float64))
            if error > 1e-6:
                raise RuntimeError(
                    f"Recorded navigation posture no longer matches asset XYZ for "
                    f"{body_name}: position error={error:.3g}m"
                )

    def _holder_planar_pose(self) -> tuple[np.ndarray, float]:
        self.env.mj_forward()
        position = np.asarray(self.data.xpos[self.holder_id], dtype=np.float64)
        matrix = np.asarray(self.data.xmat[self.holder_id], dtype=np.float64).reshape(3, 3)
        return position[:2].copy(), _yaw_from_matrix(matrix)

    def _body_link_planar_pose(self) -> tuple[np.ndarray, float]:
        self.env.mj_forward()
        position = np.asarray(self.data.xpos[self.body_link_id], dtype=np.float64)
        matrix = np.asarray(
            self.data.xmat[self.body_link_id], dtype=np.float64,
        ).reshape(3, 3)
        return position[:2].copy(), _yaw_from_matrix(matrix)

    def _publish_laser_static_tf(self) -> None:
        if self.static_tf_pub is None:
            return
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.base_frame
        msg.child_frame_id = self.args.laser_frame
        msg.transform.translation.x = self.args.laser_x
        msg.transform.translation.y = self.args.laser_y
        msg.transform.translation.z = self.args.laser_z
        msg.transform.rotation.w = 1.0
        self.static_tf_pub.sendTransform(msg)

    def _on_cmd_vel(self, msg: Twist) -> None:
        linear, angular = float(msg.linear.x), float(msg.angular.z)
        self.latest_cmd_vel = (linear, angular)
        self.latest_cmd_time = time.monotonic()
        stats = self.command_stats
        stats["input_count"] += 1
        if abs(linear) > 1e-5 or abs(angular) > 1e-5:
            stats["input_nonzero_count"] += 1
        else:
            stats["input_zero_count"] += 1
        stats["input_min_linear"] = min(stats["input_min_linear"], linear)
        stats["input_max_abs_linear"] = max(stats["input_max_abs_linear"], abs(linear))
        stats["input_max_abs_angular"] = max(stats["input_max_abs_angular"], abs(angular))
        stats["last_input_linear"], stats["last_input_angular"] = linear, angular

    @staticmethod
    def _control_value(ctrl_range: tuple[float, float], normalized: float) -> float:
        low, high = ctrl_range
        return (low + high) * 0.5 + float(np.clip(normalized, -1.0, 1.0)) * (high - low) * 0.5

    def _commands(self) -> tuple[float, float]:
        if time.monotonic() - self.latest_cmd_time <= self.args.cmd_vel_timeout:
            linear, angular = self.latest_cmd_vel
            # The OmniPicker steers its wheels but cannot rotate in place. Nav2
            # recovery/controller commands often request angular.z with zero
            # linear.x, so convert that request to a slow forward arc.
            if abs(angular) >= self.args.arc_turn_threshold and abs(linear) < self.args.arc_turn_speed:
                # The forward speed injected here sets the arc the wheels will
                # actually cut, so size it to the requested yaw rate. A fixed
                # crawl makes tight requests saturate the steering and stall.
                arc_speed = max(
                    self.args.arc_turn_speed,
                    abs(angular) * self.args.min_turning_radius,
                )
                linear = math.copysign(
                    min(arc_speed, self.args.max_linear_speed),
                    linear if abs(linear) > 1e-9 else 1.0,
                )
            # Nav2 publishes yaw rate, while this chassis actuator expects a
            # normalized steering angle. Convert through Ackermann curvature
            # instead of using a speed-independent angular gain.
            # Ackermann yaw rate is v*tan(delta)/L. Preserve the sign of v:
            # a reverse segment needs the opposite steering angle for the
            # same requested yaw rate. Using abs(linear) made every
            # Reeds-Shepp reverse turn bend toward the wrong side.
            signed_speed = math.copysign(
                max(abs(linear), self.args.ackermann_min_speed),
                linear if abs(linear) > 1e-9 else 1.0,
            )
            steering_angle = math.atan(
                self.args.wheelbase * angular / signed_speed
            )
            target = (
                float(np.clip(linear / self.args.max_linear_speed, -1.0, 1.0)) * self.args.speed,
                float(np.clip(steering_angle / self.args.max_steering_angle, -1.0, 1.0))
                * self.args.turn_speed,
            )
        else:
            target = (0.0, 0.0)
        now = time.monotonic(); dt = max(1e-3, now - self._last_output_time)
        # Jerk-limit linear command; first-order slew is enough for steering.
        linear, self._last_linear_accel = step_linear_scurve(
            self._last_output[0],
            self._last_linear_accel,
            target[0],
            dt,
            accel_limit=self.args.linear_slew_rate,
            jerk_limit=self.args.linear_jerk_limit,
        )
        steer = float(np.clip(
            target[1],
            self._last_output[1] - self.args.steering_slew_rate * dt,
            self._last_output[1] + self.args.steering_slew_rate * dt,
        ))
        out = (linear, steer)
        self._last_output, self._last_output_time = out, now
        self._profile_elapsed += dt
        self._linear_command_samples.append((self._profile_elapsed, linear))
        stats = self.command_stats
        stats["output_count"] += 1
        if abs(out[0]) > 1e-5 or abs(out[1]) > 1e-5:
            stats["output_nonzero_count"] += 1
        else:
            stats["output_zero_count"] += 1
        stats["output_min_linear"] = min(stats["output_min_linear"], out[0])
        stats["output_max_abs_linear"] = max(stats["output_max_abs_linear"], abs(out[0]))
        stats["output_max_abs_steer"] = max(stats["output_max_abs_steer"], abs(out[1]))
        stats["last_output_linear"], stats["last_output_steer"] = out
        return out

    def _apply_obstacle_protection(self, forward: float, turn: float) -> tuple[float, float]:
        """Apply direction-aware braking using the most recent official-model ray scan."""
        if self.collision_stop_latched:
            return 0.0, 0.0
        if abs(forward) < 1e-6:
            self._safety_state = "clear"
            return forward, turn
        distance = self.forward_obstacle_distance if forward > 0.0 else self.reverse_obstacle_distance
        if math.isfinite(distance):
            self.minimum_motion_obstacle_distance = (
                distance if self.minimum_motion_obstacle_distance is None
                else min(self.minimum_motion_obstacle_distance, distance)
            )
        if distance <= self.args.obstacle_stop_distance:
            if self._safety_state != "stopped":
                self.emergency_stop_count += 1
                self.get_logger().warning(
                    f"Obstacle safety stop: {distance:.3f} m in travel direction"
                )
            self._safety_state = "stopped"
            return 0.0, 0.0
        if distance < self.args.obstacle_slow_distance:
            ratio = (distance - self.args.obstacle_stop_distance) / (
                self.args.obstacle_slow_distance - self.args.obstacle_stop_distance
            )
            scale = max(self.args.obstacle_min_speed_ratio, min(1.0, ratio))
            if self._safety_state != "slowing":
                self.safety_slowdown_count += 1
            self._safety_state = "slowing"
            return forward * scale, turn
        self._safety_state = "clear"
        return forward, turn

    def _monitor_contacts(self) -> None:
        """Record robot/environment contacts while excluding normal wheel-to-ground contact."""
        self.contact_sample_count += 1
        for contact in self.env.query_contact_simple():
            contact_id = int(contact["ID"])
            # MuJoCo reports speculative contacts inside the geom margin. They
            # are not physical collisions. Count only actual touch/penetration
            # (non-positive signed distance) in the acceptance audit.
            contact_dist = float(self.data.contact[contact_id].dist)
            # MuJoCo margin contacts are positive; latch penetration and
            # near-touch (<=1 mm) so table-edge grazes are not missed.
            if contact_dist > 0.001:
                continue
            geom1, geom2 = int(contact["Geom1"]), int(contact["Geom2"])
            classified = classify_robot_environment_contact(
                geom1,
                geom2,
                self.robot_geom_ids,
                self.collision_robot_geom_ids,
                self.ground_geom_ids,
            )
            if classified is None:
                continue
            robot_geom, environment_geom = classified
            self.robot_environment_contact_count += 1
            # A real robot/environment contact is a hard safety failure in
            # every mode.  The contact pose remains audit-only (it is never
            # published as odometry), but the latch must immediately force
            # zero actuator commands so visual localization cannot continue
            # after a collision.
            self.collision_stop_latched = True
            if self.first_collision is None:
                xy, yaw = self._holder_planar_pose()
                self.first_collision = {
                    "contact_id": contact_id,
                    "contact_dist_m": contact_dist,
                    "robot_geom_id": robot_geom,
                    "robot_geom_name": self.model.geom(robot_geom).name or "",
                    "environment_geom_id": environment_geom,
                    "environment_geom_name": self.model.geom(environment_geom).name or "",
                    "world_xy": xy.tolist(),
                    "world_yaw_rad": float(yaw),
                    "unix_time": time.time(),
                }
                self.get_logger().error(
                    "FIRST COLLISION: task failed immediately; "
                    f"robot={self.first_collision['robot_geom_name']}, "
                    f"environment={self.first_collision['environment_geom_name']}, "
                    f"pose=({xy[0]:.3f},{xy[1]:.3f},{yaw:.3f})"
                )
                self._publish_navigation_status()
            if len(self.robot_environment_contacts) < self.args.max_recorded_contacts:
                self.robot_environment_contacts.append({
                    "contact_id": contact_id,
                    "contact_dist_m": contact_dist,
                    "robot_geom_id": robot_geom,
                    "robot_geom_name": self.model.geom(robot_geom).name or "",
                    "environment_geom_id": environment_geom,
                    "environment_geom_name": self.model.geom(environment_geom).name or "",
                })

    @staticmethod
    def _json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): AgiBotChassisNavigation._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AgiBotChassisNavigation._json_value(item) for item in value]
        return value

    def _write_scene_inventory(self) -> None:
        """Snapshot bodies and collision geoms through OrcaGym's documented query API."""
        bodies = self.env.gym.query_all_bodies()
        geoms = self.env.gym.query_all_geoms()
        for name, body in bodies.items():
            body_id = int(body["ID"])
            body["WorldPos"] = np.asarray(self.data.xpos[body_id], dtype=np.float64)
            body["WorldMat"] = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            body["WorldQuat"] = np.asarray(self.data.xquat[body_id], dtype=np.float64)
        for geom_id in range(self.model.ngeom):
            name = self.model.geom(geom_id).name or f"__unnamed_geom_{geom_id}"
            geom = geoms.get(name)
            if geom is None:
                continue
            geom["ID"] = geom_id
            geom["WorldPos"] = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            geom["WorldMat"] = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        payload = {
            "api": {
                "bodies": "OrcaGym query_all_bodies()",
                "geoms": "OrcaGym query_all_geoms()",
                "contacts": "OrcaGymLocalEnv.query_contact_simple()",
                "rays": "MuJoCo mj_ray() on OrcaGym's compiled model",
            },
            "body_count": len(bodies),
            "geom_count": len(geoms),
            "robot_body_count": len(self.robot_body_ids),
            "robot_geom_count": len(self.robot_geom_ids),
            "collision_robot_geom_count": len(self.collision_robot_geom_ids),
            "bodies": bodies,
            "geoms": geoms,
        }
        (self.session_dir / "scene_inventory.json").write_text(
            json.dumps(self._json_value(payload), indent=2), encoding="utf-8"
        )

    def _publish_navigation_status(self) -> None:
        """Publish the safety interlock consumed by ABCNavigator.ensure_motion_allowed."""
        monitor_active = not self.args.disable_collision_monitor
        payload = {
            "motion_allowed": monitor_active and not self.collision_stop_latched,
            "collision_stop_latched": self.collision_stop_latched,
            "first_collision": self.first_collision,
            "api_collision_monitor_active": monitor_active,
            "robot_environment_contact_count": self.robot_environment_contact_count,
            "failure_reason": (
                "collision" if self.collision_stop_latched else None
            ),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _ctrl(self, forward: float, turn: float) -> np.ndarray:
        ctrl = np.zeros(self.env.model.nu, dtype=np.float64)
        for wheel in self.drive_wheels + list(self.steering_wheels.values()):
            ctrl[wheel.index] = sum(wheel.ctrl_range) * 0.5
        speed_scale = (
            float(forward) / float(self.args.speed)
            if abs(float(self.args.speed)) > 1e-6
            else 0.0
        )
        for wheel in self.drive_wheels:
            sign = self.args.left_sign if wheel.wheel == "fl" else self.args.right_sign
            ctrl[wheel.index] = self._control_value(
                wheel.ctrl_range, speed_scale * sign,
            )
        for name, wheel in self.steering_wheels.items():
            # Rear steer hinges may exist in the compiled model; this chassis
            # is front 2WS, so only FL/FR receive the Ackermann angle.
            command = turn if name in CHASSIS_STEER_WHEELS else 0.0
            ctrl[wheel.index] = self._control_value(wheel.ctrl_range, command)
        return ctrl

    def _tick(self) -> None:
        forward, turn = self._commands()
        forward, turn = self._apply_obstacle_protection(forward, turn)
        self.command_stats["last_applied_linear"] = float(forward)
        self.command_stats["last_applied_steer"] = float(turn)
        self._hold_arm_pose()
        self.env.do_simulation(self._ctrl(forward, turn), FRAME_SKIP)
        # Pin all shoulder-through-wrist joints again after physics so the
        # state submitted to OrcaLab and sampled by LiDAR cannot gravity-sag.
        self._hold_arm_pose()
        # Sample contacts continuously. They are never used as odometry, but
        # a real contact latches the common hard stop for the next tick.
        if not self.args.disable_collision_monitor:
            self._monitor_contacts()
        self._publish_navigation_status()
        if self.args.record_truth_audit:
            now_wall = time.time()
            if now_wall - self.last_truth_audit_time >= 1.0 / self.args.truth_audit_hz:
                # This sample is written only to the shutdown report. It is
                # never published, transformed, or read by command generation.
                xy, yaw = self._holder_planar_pose()
                self.truth_audit_samples.append({
                    "wall_time": now_wall,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "yaw": float(yaw),
                })
                if self.truth_pose_pub is not None:
                    pose = PoseStamped()
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.header.frame_id = "world"
                    pose.pose.position.x = float(xy[0])
                    pose.pose.position.y = float(xy[1])
                    pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
                    pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
                    self.truth_pose_pub.publish(pose)
                if self.truth_body_link_pub is not None:
                    body_xy, body_yaw = self._body_link_planar_pose()
                    body_pose = PoseStamped()
                    body_pose.header.stamp = self.get_clock().now().to_msg()
                    body_pose.header.frame_id = "world"
                    body_pose.pose.position.x = float(body_xy[0])
                    body_pose.pose.position.y = float(body_xy[1])
                    body_pose.pose.orientation.z = math.sin(float(body_yaw) * 0.5)
                    body_pose.pose.orientation.w = math.cos(float(body_yaw) * 0.5)
                    self.truth_body_link_pub.publish(body_pose)
                self.last_truth_audit_time = now_wall
        # Camera capture reads the most recently submitted OrcaGym render; it
        # does not submit a new render when the physical chassis moves.  The
        # former visual-mode branch skipped render() completely, so RGB-D
        # files kept arriving while showing a stale camera pose and visual
        # odometry stayed near its initial pose.  Render every tick in legacy
        # truth mode and at the camera rate in formal visual mode, avoiding
        # the 50 Hz RPC contention that originally motivated throttling.
        now = time.monotonic()
        truth_mode = getattr(self.args, "publish_truth_odom", True)
        if (truth_mode or
                now - self.last_camera_render_time >= 1.0 / self.args.camera_render_hz):
            self.env.render()
            self.last_camera_render_time = now
            self.camera_render_count += 1
        stamp = self.get_clock().now().to_msg()
        self._publish_odom(stamp, now)
        if self.wheel_odom is not None:
            self.wheel_odom.publish(stamp)
        if self.scan_pub is not None and now - self.last_scan_time >= 1.0 / self.args.scan_hz:
            self.last_scan_time = now
            self._publish_scan(stamp)

    def _publish_odom(self, stamp, now: float) -> None:
        if self.odom_pub is None or self.tf_pub is None:
            return
        xy, yaw = self._holder_planar_pose()
        dt = max(now - self.last_pose_time, 1e-6)
        yaw_delta = math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
        world_velocity = (xy - self.last_xy) / dt
        c, s = math.cos(yaw), math.sin(yaw)
        linear_x = c * world_velocity[0] + s * world_velocity[1]
        angular_z = yaw_delta / dt
        rel_xy = xy - self.start_xy
        rel_yaw = math.atan2(math.sin(yaw - self.start_yaw), math.cos(yaw - self.start_yaw))
        c0, s0 = math.cos(self.start_yaw), math.sin(self.start_yaw)
        odom_xy = np.array([c0 * rel_xy[0] + s0 * rel_xy[1], -s0 * rel_xy[0] + c0 * rel_xy[1]])
        qx, qy, qz, qw = _yaw_quaternion(rel_yaw)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.odom_frame
        msg.child_frame_id = self.args.base_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y = map(float, odom_xy)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = qx, qy
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = qz, qw
        msg.twist.twist.linear.x = float(linear_x)
        msg.twist.twist.angular.z = float(angular_z)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 1e-4
        msg.pose.covariance[35] = 1e-4
        msg.twist.covariance[0] = msg.twist.covariance[7] = 1e-3
        msg.twist.covariance[35] = 1e-3
        self.odom_pub.publish(msg)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.args.odom_frame
        tf.child_frame_id = self.args.base_frame
        tf.transform.translation.x, tf.transform.translation.y = map(float, odom_xy)
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)
        self.last_xy, self.last_yaw, self.last_pose_time = xy, yaw, now

    def _ray_distance(self, origin: np.ndarray, direction: np.ndarray) -> float:
        travelled = 0.0
        ray_origin = origin.copy()
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(self.args.max_self_hits + 1):
            distance = float(mujoco.mj_ray(
                self.model, self.data, ray_origin, direction, None, 1, -1, geom_id, None
            ))
            if distance < 0.0 or travelled + distance > self.args.range_max:
                return math.inf
            travelled += distance
            if int(geom_id[0]) not in self.robot_geom_ids:
                return travelled if travelled >= self.args.range_min else math.inf
            advance = max(self.args.self_hit_epsilon, distance + self.args.self_hit_epsilon)
            travelled += self.args.self_hit_epsilon
            ray_origin = ray_origin + direction * advance
        return math.inf

    def _publish_scan(self, stamp) -> None:
        if self.scan_pub is None:
            return
        matrix = np.asarray(self.data.xmat[self.holder_id], dtype=np.float64).reshape(3, 3)
        holder_pos = np.asarray(self.data.xpos[self.holder_id], dtype=np.float64)
        mount_local = np.array([self.args.laser_x, self.args.laser_y, self.args.laser_z])
        origin = holder_pos + matrix @ mount_local
        angle_increment = (self.args.angle_max - self.args.angle_min) / self.args.scan_beams
        angles = self.args.angle_min + np.arange(self.args.scan_beams) * angle_increment
        ranges = []
        for angle in angles:
            local_direction = np.array([math.cos(angle), math.sin(angle), 0.0])
            direction = matrix @ local_direction
            direction /= np.linalg.norm(direction)
            ranges.append(self._ray_distance(origin, direction))

        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.laser_frame
        msg.angle_min = float(self.args.angle_min)
        msg.angle_max = float(angles[-1])
        msg.angle_increment = float(angle_increment)
        msg.scan_time = float(1.0 / self.args.scan_hz)
        msg.time_increment = msg.scan_time / self.args.scan_beams
        msg.range_min = float(self.args.range_min)
        msg.range_max = float(self.args.range_max)
        msg.ranges = ranges
        self.scan_pub.publish(msg)
        self.scan_count += 1
        finite = [value for value in ranges if math.isfinite(value)]
        self.last_scan_finite_count = len(finite)
        if finite:
            current_min, current_max = min(finite), max(finite)
            self.scan_finite_min = current_min if self.scan_finite_min is None else min(self.scan_finite_min, current_min)
            self.scan_finite_max = current_max if self.scan_finite_max is None else max(self.scan_finite_max, current_max)
        sector = math.radians(self.args.obstacle_sector_degrees)
        forward_ranges = [value for angle, value in zip(angles, ranges) if abs(angle) <= sector]
        reverse_ranges = [
            value for angle, value in zip(angles, ranges)
            if abs(abs(angle) - math.pi) <= sector
        ]
        self.forward_obstacle_distance = min(forward_ranges, default=math.inf)
        self.reverse_obstacle_distance = min(reverse_ranges, default=math.inf)

    def _finalize_command_telemetry(self) -> dict:
        stats = dict(self.command_stats)
        stats["speed_profile"] = summarize_speed_profile(
            self._linear_command_samples,
            accel_limit=self.args.linear_slew_rate,
            jerk_limit=self.args.linear_jerk_limit,
        )
        self.command_stats = stats
        return stats

    def close(self) -> None:
        end_xy, end_yaw = self._holder_planar_pose()
        arm_vectors = self._arm_unit_vectors()
        arm_qpos_at_end = self._read_arm_qpos()
        arm_max_change = max(
            float(np.max(np.abs(
                arm_qpos_at_end[key] - self.arm_qpos_at_connection[key]
            )))
            for key in ("L", "R")
        )
        payload = {
            "started_at": self.started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "scan_count": self.scan_count,
            "camera_render_hz_target": self.args.camera_render_hz,
            "camera_render_count": self.camera_render_count,
            "last_scan_finite_count": self.last_scan_finite_count,
            "finite_range_min_observed": self.scan_finite_min,
            "finite_range_max_observed": self.scan_finite_max,
            "api_scene_inventory": "scene_inventory.json",
            "api_collision_monitor_active": not self.args.disable_collision_monitor,
            "contact_audit_only": not bool(getattr(self.args, "publish_truth_odom", True)),
            "truth_pose_audit_only": bool(self.args.record_truth_audit),
            "truth_pose_published": bool(self.args.publish_truth_assist),
            "truth_audit_hz_target": self.args.truth_audit_hz,
            "truth_audit_samples": self.truth_audit_samples,
            "contact_sample_count": self.contact_sample_count,
            "robot_environment_contact_count": self.robot_environment_contact_count,
            "robot_environment_contacts": self.robot_environment_contacts,
            "first_collision": self.first_collision,
            "collision_stop_latched": self.collision_stop_latched,
            "command_telemetry": self._finalize_command_telemetry(),
            "drive_actuators": [
                {"wheel": w.wheel, "name": w.name, "index": w.index,
                 "ctrl_range": list(w.ctrl_range)} for w in self.drive_wheels
            ],
            "steering_actuators": {
                name: {"name": w.name, "index": w.index,
                       "ctrl_range": list(w.ctrl_range)}
                for name, w in self.steering_wheels.items()
            },
            "emergency_stop_count": self.emergency_stop_count,
            "safety_slowdown_count": self.safety_slowdown_count,
            "minimum_motion_obstacle_distance_m": self.minimum_motion_obstacle_distance,
            "obstacle_stop_distance_m": self.args.obstacle_stop_distance,
            "obstacle_slow_distance_m": self.args.obstacle_slow_distance,
            "rgbd_rate_hz": getattr(self, "rgbd_rate_hz", None),
            "rgbd_rate_min_hz": getattr(self.args, "min_rgbd_rate", None),
            "rgbd_rate_stop_latched": getattr(self, "rgbd_rate_stop_latched", False),
            "robot_world_xy_start": self.start_world_xy.tolist(),
            "robot_world_xy_end": end_xy.tolist(),
            "robot_displacement_m": float(np.linalg.norm(end_xy - self.start_world_xy)),
            "robot_yaw_change_rad": float(math.atan2(
                math.sin(end_yaw - self.start_yaw), math.cos(end_yaw - self.start_yaw)
            )),
            "arm_posture": self.args.arm_posture,
            "arm_pose_source": (
                "live OrcaLab state captured at bridge connection"
                if self.args.arm_posture == "current" else "project posture preset"
            ),
            "arm_position_initialization_applied": self.arm_position_initialization_applied,
            "arm_qpos_at_connection": {
                "left": self.arm_qpos_at_connection["L"].tolist(),
                "right": self.arm_qpos_at_connection["R"].tolist(),
            },
            "arm_qpos_hold_target": {
                "left": self.arm_targets["L"].tolist(),
                "right": self.arm_targets["R"].tolist(),
            },
            "left_arm_qpos": arm_qpos_at_end["L"].tolist(),
            "right_arm_qpos": arm_qpos_at_end["R"].tolist(),
            "arm_max_abs_qpos_change_from_connection_rad": arm_max_change,
            "arm_unit_vectors": arm_vectors,
            "scan_topic": self.args.scan_topic,
            "odom_topic": self.args.odom_topic,
            "wheel_odom": self.wheel_odom.diagnostics() if self.wheel_odom is not None else None,
            "robot_asset_unchanged": "assets/e071469a36d3c8aa/default_project/prefabs/g1_omnipicker_usda",
        }
        (self.session_dir / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            self.env.do_simulation(self._ctrl(0.0, 0.0), 1)
        finally:
            self.env.close()

    def _arm_unit_vectors(self) -> dict[str, list[list[float]]]:
        result = {}
        for key, side in (("left", "l"), ("right", "r")):
            names = [
                f"{self.args.robot_name}_arm_{side}_link1",
                f"{self.args.robot_name}_arm_{side}_link3",
                f"{self.args.robot_name}_arm_{side}_link5",
                f"{self.args.robot_name}_arm_{side}_end_link",
                f"{self.args.robot_name}_gripper_{side}_outer_link1",
                f"{self.args.robot_name}_gripper_{side}_inner_link1",
            ]
            body_ids = [self.env.model.body_name2id(name) for name in names]
            points = np.asarray(self.data.xpos[body_ids], dtype=np.float64)
            chain = np.vstack((points[:4], (points[4] + points[5]) * 0.5))
            vectors = np.diff(chain, axis=0)
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            result[key] = vectors.tolist()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgiBot autonomous LiDAR, odometry, TF and cmd_vel bridge")
    parser.add_argument("--orcagym-addr", default="127.0.0.1:50051")
    parser.add_argument("--robot-name", default="industrial_collaborative_robot_1")
    parser.add_argument(
        "--arm-posture", choices=("current", "mapping", "navigation"), default="current",
        help=("current preserves the live OrcaLab joint pose; legacy mapping/navigation "
              "values explicitly load project presets"),
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--speed", type=float, default=0.60)
    parser.add_argument("--turn-speed", type=float, default=0.60)
    parser.add_argument("--arm-target-step", type=float, default=0.01)
    parser.add_argument(
        "--preserve-live-state", action="store_true",
        help="do not write a qpos/qvel snapshot back during bridge startup",
    )
    parser.add_argument("--left-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--right-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    truth_group = parser.add_mutually_exclusive_group()
    truth_group.add_argument(
        "--publish-truth-odom", dest="publish_truth_odom", action="store_true",
        help="publish MuJoCo truth odometry (legacy audit mode)",
    )
    truth_group.add_argument(
        "--no-publish-truth-odom", dest="publish_truth_odom", action="store_false",
        help="disable MuJoCo truth odometry/laser publishers",
    )
    parser.set_defaults(publish_truth_odom=True)
    parser.add_argument(
        "--publish-wheel-odom", action="store_true",
        help="publish calibrated wheel /odom and odom->base_link from wheel joint qpos",
    )
    parser.add_argument(
        "--wheel-odom-topic", default="/odom",
        help="canonical wheel odometry topic (must match Nav2/RTAB-Map)",
    )
    parser.add_argument(
        "--wheel-odom-calibration", type=Path,
        default=Path(__file__).with_name("config") / "service_robot_1_wheel_odom_calibration.json",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--status-topic", default="/task1/navigation_status",
        help="JSON safety interlock published for the mission navigator",
    )
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--laser-frame", default="laser")
    parser.add_argument("--laser-x", type=float, default=0.0)
    parser.add_argument("--laser-y", type=float, default=0.0)
    parser.add_argument("--laser-z", type=float, default=0.45)
    parser.add_argument("--scan-hz", type=float, default=10.0)
    parser.add_argument(
        "--camera-render-hz", type=float, default=15.0,
        help="submit live OrcaGym camera renders at this rate in formal visual mode",
    )
    parser.add_argument("--min-rgbd-rate", type=float, default=11.0,
                        help="hard minimum head RGB-D callback rate in Hz")
    parser.add_argument("--rgbd-rate-window", type=float, default=3.0,
                        help="rolling window used by the RGB-D rate gate")
    parser.add_argument("--rgbd-rate-warmup", type=float, default=8.0,
                        help="startup grace period before enforcing RGB-D rate")
    parser.add_argument("--rgbd-rate-failure-hold", type=float, default=2.0,
                        help="seconds below minimum before latching the stop")
    parser.add_argument("--scan-beams", type=int, default=360)
    parser.add_argument("--angle-min", type=float, default=-math.pi)
    parser.add_argument("--angle-max", type=float, default=math.pi)
    parser.add_argument("--range-min", type=float, default=0.10)
    parser.add_argument("--range-max", type=float, default=15.0)
    parser.add_argument("--max-self-hits", type=int, default=8)
    parser.add_argument("--self-hit-epsilon", type=float, default=0.002)
    parser.add_argument("--obstacle-sector-degrees", type=float, default=35.0)
    parser.add_argument("--obstacle-stop-distance", type=float, default=0.70)
    parser.add_argument("--obstacle-slow-distance", type=float, default=1.10)
    parser.add_argument("--obstacle-min-speed-ratio", type=float, default=0.20)
    parser.add_argument("--max-recorded-contacts", type=int, default=100)
    parser.add_argument(
        "--disable-collision-monitor", action="store_true",
        help="disable MuJoCo contact auditing and collision-stop latching",
    )
    parser.add_argument("--max-linear-speed", type=float, default=0.60)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument(
        "--wheelbase", type=float, default=CHASSIS_WHEELBASE_M,
        help="2WS Ackermann axle length (front-to-rear); g1_omnipicker is 0.42 m",
    )
    parser.add_argument(
        "--max-steering-angle", type=float, default=CHASSIS_MAX_STEER_RAD,
    )
    parser.add_argument("--ackermann-min-speed", type=float, default=0.08)
    parser.add_argument("--cmd-vel-timeout", type=float, default=0.5)
    parser.add_argument("--linear-slew-rate", type=float, default=LINEAR_ACCEL_LIMIT,
                        help="linear acceleration limit in command units per second")
    parser.add_argument("--linear-jerk-limit", type=float, default=LINEAR_JERK_LIMIT,
                        help="linear jerk limit in command units per second cubed")
    parser.add_argument("--steering-slew-rate", type=float, default=3.0)
    parser.add_argument("--arc-turn-threshold", type=float, default=0.05)
    parser.add_argument(
        "--arc-turn-speed", type=float, default=0.12,
        help="forward m/s injected for angular-only Nav2 commands on the non-holonomic steering base",
    )
    parser.add_argument(
        "--min-turning-radius", type=float,
        default=VALIDATION_MIN_TURNING_RADIUS_M,
        help="curvature envelope used to size the injected arc speed for angular-only commands",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--record-truth-audit", action="store_true",
        help="record non-published world poses for offline validation only",
    )
    parser.add_argument(
        "--publish-truth-assist", action="store_true",
        help="publish the MuJoCo root world pose as an authorized navigation aid",
    )
    parser.add_argument("--truth-pose-topic", default="/task1/truth_pose")
    parser.add_argument(
        "--truth-body-link-topic", default="/task1/truth_body_link_pose",
    )
    parser.add_argument("--truth-audit-hz", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.scan_beams < 2 or args.scan_hz <= 0 or args.camera_render_hz <= 0 or
            args.min_rgbd_rate <= 0 or args.rgbd_rate_window <= 0 or
            args.rgbd_rate_warmup < 0 or args.rgbd_rate_failure_hold <= 0 or
            args.truth_audit_hz <= 0 or
            args.range_min < 0 or args.range_max <= args.range_min):
        parser.error("invalid LiDAR scan configuration")
    if not (0.0 < args.obstacle_sector_degrees <= 90.0):
        parser.error("--obstacle-sector-degrees must be in (0, 90]")
    if not (args.range_min <= args.obstacle_stop_distance < args.obstacle_slow_distance <= args.range_max):
        parser.error("obstacle distances must satisfy range-min <= stop < slow <= range-max")
    if not (0.0 < args.obstacle_min_speed_ratio <= 1.0) or args.max_recorded_contacts < 0:
        parser.error("invalid obstacle speed ratio or contact record limit")
    return args


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    _ensure_simulation(root, args.startup_timeout)
    rclpy.init()
    node = AgiBotChassisNavigation(args)
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
