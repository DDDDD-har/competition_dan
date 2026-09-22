#!/usr/bin/env python3
"""Localize in a saved RTAB-Map database from a one-shot world pose.

This entry point intentionally does not use wheel odometry, continuous MuJoCo
truth odometry, AMCL, or a static map->odom bridge.  It starts the RGB-D
pipeline and RTAB-Map components directly.  MuJoCo truth is sampled only at
well-defined audit points:

* once before startup to seed ``rgbd_odometry.initial_pose``;
* once while stationary after confirmed image-to-database localization to
  estimate a world->map SE(2) calibration; and
* once after an optional motion probe to measure independent terminal error.

The saved mapping database is copied into the session before it is opened, so
the source mapping asset cannot be modified by a localization run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
PROJECT = ROOT.parent
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
ROS2 = "/opt/ros/jazzy/bin/ros2"
SYSTEM_PYTHON = "/usr/bin/python3"
ORCALAB_PYTHON = "/home/dan/miniconda3/envs/orcalab/bin/python"
ORCALAB_SITE = "/home/dan/miniconda3/envs/orcalab/lib/python3.12/site-packages"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chassis_motion_profile import (
    LINEAR_ACCEL_LIMIT,
    LINEAR_JERK_LIMIT,
    profile_within_jerk_limit,
)
from official_scorer import OfficialTaskScorer
from task1_scene_contract import (
    DEFAULT_MAPPING_DIR,
    OFFICIAL_REGION_RADIUS_M,
    ROBOT_NAME,
    VALIDATION_AUGMENT_SCENE_GEOMETRY,
    VALIDATION_COORDINATE_TOLERANCE_M,
    VALIDATION_MAX_ANGULAR_RADPS,
    VALIDATION_MAX_POINT_SHIFT_M,
    VALIDATION_MAX_SPEED_M,
    CHASSIS_MAX_STEER_RAD,
    CHASSIS_WHEELBASE_M,
    VALIDATION_MIN_TURNING_RADIUS_M,
    VALIDATION_POINT_CLEARANCE_M,
    VALIDATION_PROFILE_LATERAL_ACCEL_MPS2,
    VALIDATION_SCORE_CIRCLE_STOP_OFFSET_M,
    VALIDATION_SPEED_M,
    VALIDATION_STRAIGHT_CURVATURE_1PM,
    VALIDATION_TERMINAL_ARRIVAL_SPEED_MPS,
    VALIDATION_TERMINAL_ZONE_M,
    VALIDATION_PROFILE_LOOKAHEAD_M,
    VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS,
    VALIDATION_AB_CHECKPOINT_DWELL_S,
    VALIDATION_POST_ROBOT_START_DWELL_S,
    VALIDATION_HIGH_SPEED_THRESHOLD_MPS,
    VALIDATION_HIGH_SPEED_PROFILE_ACCEL_MPS2,
    VALIDATION_MISSION_TIMEOUT_S,
    VALIDATION_MISSION_DEADLINE_ENABLED,
    VALIDATION_NAV2_STARTUP_TIMEOUT_S,
    VALIDATION_START_POSE_MAX_SHIFT_M,
    VALIDATION_STARTUP_POSE_SAMPLES,
    VALIDATION_STATIC_GATE_MIN_SAMPLES,
    VALIDATION_STATIC_GATE_WINDOW_S,
    VALIDATION_PLANNER_COST_PENALTY,
    VALIDATION_PLANNER_NON_STRAIGHT_PENALTY,
    VALIDATION_PROFILE_CLEARANCE_HARD_M,
    VALIDATION_PROFILE_CLEARANCE_SOFT_M,
    VALIDATION_PROFILE_LINEAR_ACCEL_MPS2,
    scaled_profile_linear_accel,
    contract_map_points_from_safety_report,
    order_route_points,
    play_scene_hint,
    ROBOT_STARTED_MARKER_NAME,
    scored_route_points,
    validation_route_world_points,
    validation_route_order,
    world_points_xy_json,
)


# Nominal spacing of the SMAC Hybrid poses handed to FollowPath. MPPI critics
# address the path by index, so the index offsets below are derived from it.
PLANNED_PATH_SPACING_M = 0.12
# The chassis tops out well below the 20 m/s contract number; MPPI needs the
# real ceiling because every constraint ratio is computed against vx_max.
MPPI_CRUISE_CEILING_MPS = 5.0
# Slowest speed the controller is expected to hold while still tracking (the
# tight B->C bend). Critic offsets must stay usable down here.
MPPI_MIN_TRACKING_MPS = 0.35
# How far past the sampled trajectory end PathFollowCritic aims. Large values
# put the target across a bend and the robot cuts the corner.
MPPI_PATH_FOLLOW_LOOKAHEAD_M = 1.2
# Distance over which GoalCritic takes over and brakes into C.
MPPI_GOAL_BRAKING_M = 2.0


def mppi_cruise_speed(contract_speed: float) -> float:
    """Base vx_max for MPPI: the speed the chassis can actually hold."""
    return float(min(max(0.5, contract_speed), MPPI_CRUISE_CEILING_MPS))


def mppi_path_offset(distance_m: float) -> int:
    """Convert a lookahead distance into a MPPI critic path index offset."""
    return int(max(2, round(distance_m / PLANNED_PATH_SPACING_M)))


def mppi_path_align_offset(min_tracking_speed: float, horizon_s: float) -> int:
    """Largest align offset that still engages at the slowest tracking speed.

    PathAlignCritic bails out while ``furthest_reached_path_point`` (how far the
    sampled trajectories reach, i.e. speed * horizon) is below this offset, so a
    value tuned for cruise silently disables path following whenever the robot
    slows for a bend - exactly when alignment matters most.
    """
    reachable = mppi_path_offset(min_tracking_speed * horizon_s)
    return int(max(2, reachable // 2))


def mppi_local_costmap_extent(cruise: float, horizon_s: float) -> int:
    """Rolling window wide enough that MPPI previews a full braking horizon.

    The window is centred on the robot and MPPI truncates its path preview at
    the border, so the usable lookahead is only half the extent.
    """
    lookahead = cruise * horizon_s * 2.0
    return int(min(48, max(12, 2 * math.ceil(lookahead))))


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def mean_angle(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty angle list")
    return math.atan2(sum(math.sin(v) for v in values),
                      sum(math.cos(v) for v in values))


def pose_statistics(samples: list[dict[str, float]]) -> dict[str, float] | None:
    if not samples:
        return None
    xs = [float(item["x"]) for item in samples]
    ys = [float(item["y"]) for item in samples]
    yaws = [float(item["yaw"]) for item in samples]
    yaw_mean = mean_angle(yaws)
    xy_radii = [math.hypot(x - statistics.fmean(xs), y - statistics.fmean(ys))
                for x, y in zip(xs, ys)]
    yaw_errors = [abs(wrap_angle(yaw - yaw_mean)) for yaw in yaws]
    return {
        "x": statistics.fmean(xs),
        "y": statistics.fmean(ys),
        "yaw": yaw_mean,
        "xy_rms_m": math.sqrt(statistics.fmean([r * r for r in xy_radii])),
        "xy_max_m": max(xy_radii),
        "yaw_rms_rad": math.sqrt(statistics.fmean([e * e for e in yaw_errors])),
        "yaw_max_rad": max(yaw_errors),
        "sample_count": len(samples),
    }


def pose_from_world_record(record: dict[str, Any]) -> dict[str, float]:
    position = record["position_xyz"]
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "yaw": float(record["yaw_rad"]),
    }


def derive_world_to_map(world_pose: dict[str, float],
                        map_pose: dict[str, float]) -> dict[str, Any]:
    """Return T_map_world from one complete planar pose correspondence."""
    theta = wrap_angle(map_pose["yaw"] - world_pose["yaw"])
    c = math.cos(theta)
    s = math.sin(theta)
    tx = map_pose["x"] - (c * world_pose["x"] - s * world_pose["y"])
    ty = map_pose["y"] - (s * world_pose["x"] + c * world_pose["y"])
    return {
        "format": "task1_world_to_map_se2_v1",
        "source_frame": "world",
        "target_frame": "map",
        "translation_xy": [tx, ty],
        "yaw_rad": theta,
        "method": "single stationary full-pose correspondence after RTAB-Map localization",
    }


def world_to_map_pose(world_pose: dict[str, float],
                      calibration: dict[str, Any]) -> dict[str, float]:
    theta = float(calibration["yaw_rad"])
    tx, ty = [float(v) for v in calibration["translation_xy"]]
    c = math.cos(theta)
    s = math.sin(theta)
    return {
        "x": c * world_pose["x"] - s * world_pose["y"] + tx,
        "y": s * world_pose["x"] + c * world_pose["y"] + ty,
        "yaw": wrap_angle(world_pose["yaw"] + theta),
    }


def map_to_world_pose(map_pose: dict[str, float],
                      calibration: dict[str, Any]) -> dict[str, float]:
    theta = float(calibration["yaw_rad"])
    tx, ty = [float(v) for v in calibration["translation_xy"]]
    dx = map_pose["x"] - tx
    dy = map_pose["y"] - ty
    c = math.cos(theta)
    s = math.sin(theta)
    return {
        "x": c * dx + s * dy,
        "y": -s * dx + c * dy,
        "yaw": wrap_angle(map_pose["yaw"] - theta),
    }


def calibration_delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    a = [float(v) for v in first["translation_xy"]]
    b = [float(v) for v in second["translation_xy"]]
    return {
        "translation_m": math.hypot(a[0] - b[0], a[1] - b[1]),
        "yaw_rad": abs(wrap_angle(float(first["yaw_rad"]) -
                                  float(second["yaw_rad"]))),
    }


def fit_world_to_map_se2(
    correspondences: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit the least-squares planar rigid transform from world to map."""
    if len(correspondences) < 2:
        raise ValueError("at least two world/map correspondences are required")
    world = [item["world_pose"] for item in correspondences]
    mapped = [item["map_pose"] for item in correspondences]
    world_cx = statistics.fmean(float(item["x"]) for item in world)
    world_cy = statistics.fmean(float(item["y"]) for item in world)
    map_cx = statistics.fmean(float(item["x"]) for item in mapped)
    map_cy = statistics.fmean(float(item["y"]) for item in mapped)
    dot = 0.0
    cross = 0.0
    spread = 0.0
    for source, target in zip(world, mapped):
        wx = float(source["x"]) - world_cx
        wy = float(source["y"]) - world_cy
        mx = float(target["x"]) - map_cx
        my = float(target["y"]) - map_cy
        dot += wx * mx + wy * my
        cross += wx * my - wy * mx
        spread += wx * wx + wy * wy
    if spread <= 1e-9:
        raise ValueError("world correspondences have insufficient spatial spread")
    theta = math.atan2(cross, dot)
    c = math.cos(theta)
    s = math.sin(theta)
    tx = map_cx - (c * world_cx - s * world_cy)
    ty = map_cy - (s * world_cx + c * world_cy)
    return {
        "format": "task1_world_to_map_se2_v2",
        "source_frame": "world",
        "target_frame": "map",
        "translation_xy": [tx, ty],
        "yaw_rad": theta,
        "method": "multi-point least-squares position fit with independent yaw residual audit",
        "correspondence_count": len(correspondences),
        "world_position_spread_sum_sq_m2": spread,
    }


def transform_residuals(
    correspondences: list[dict[str, Any]], calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for item in correspondences:
        predicted = world_to_map_pose(item["world_pose"], calibration)
        measured = item["map_pose"]
        dx = predicted["x"] - float(measured["x"])
        dy = predicted["y"] - float(measured["y"])
        yaw_error = wrap_angle(predicted["yaw"] - float(measured["yaw"]))
        result.append({
            "name": item["name"],
            "predicted_map_pose": predicted,
            "measured_map_pose": measured,
            "position_error_xy_m": [dx, dy],
            "position_error_m": math.hypot(dx, dy),
            "yaw_error_rad": yaw_error,
        })
    return result


def convex_hull_area(points: list[tuple[float, float]]) -> float:
    unique = sorted(set(points))
    if len(unique) < 3:
        return 0.0

    def cross(origin: tuple[float, float], first: tuple[float, float],
              second: tuple[float, float]) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1]) -
                (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(hull, hull[1:] + hull[:1])
    )) * 0.5


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ros_environment() -> dict[str, str]:
    env = os.environ.copy()
    output = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         f"source {ROS_SETUP} && env -0"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    for item in output.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
    env["PYTHONPATH"] = ORCALAB_SITE + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def require_port(port: int) -> None:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            raise RuntimeError(
                f"OrcaLab service {port} is unavailable; {play_scene_hint()} first"
            )


def restart_ros_daemon(env: dict[str, str]) -> None:
    for action in ("stop", "start"):
        subprocess.run(
            [ROS2, "daemon", action], cwd=PROJECT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=8.0, check=False,
        )


def set_ros_param(
    node: str,
    name: str,
    value: str,
    env: dict[str, str],
    *,
    timeout_s: float = 30.0,
    interval_s: float = 0.5,
) -> subprocess.CompletedProcess[str]:
    """Set a ROS parameter, retrying until the node is discoverable."""
    deadline = time.monotonic() + timeout_s
    last: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        last = subprocess.run(
            [ROS2, "param", "set", node, name, value],
            cwd=PROJECT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False,
        )
        if (last.returncode == 0 and
                "successful" in last.stdout.lower()):
            return last
        time.sleep(interval_s)
    if last is None:
        raise RuntimeError(f"ros2 param set {node} {name} did not run")
    return last


def process_log_tail(session: Path, name: str, *, limit: int = 40) -> str:
    path = session / f"{name}.log"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def publisher_count(topic: str, env: dict[str, str]) -> int:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = subprocess.run(
                [ROS2, "topic", "info", topic], cwd=PROJECT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                timeout=5.0, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            restart_ros_daemon(env)
            continue
        for line in result.stdout.splitlines():
            if line.startswith("Publisher count:"):
                try:
                    return int(line.split(":", 1)[1])
                except ValueError:
                    return 0
        return 0
    raise RuntimeError(
        f"ros2 topic info {topic} timed out after restarting the ROS daemon"
    ) from last_error


def wait_for_publisher(topic: str, env: dict[str, str], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if publisher_count(topic, env) > 0:
            return True
        time.sleep(0.25)
    return False


def assert_topics_unclaimed(env: dict[str, str]) -> None:
    occupied = []
    for topic in ("/camera/color/image_raw", "/odom", "/map"):
        count = publisher_count(topic, env)
        if count:
            occupied.append(f"{topic} ({count} publisher(s))")
    if occupied:
        raise RuntimeError(
            "Another RGB-D/odometry/map pipeline is already running: " +
            ", ".join(occupied)
        )


def start_process(name: str, command: list[str], env: dict[str, str],
                  session: Path, processes: dict[str, subprocess.Popen]) -> None:
    log = (session / f"{name}.log").open("w", encoding="utf-8")
    processes[name] = subprocess.Popen(
        command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _proc_parent_and_group(pid: int) -> tuple[int, int] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    fields = text[close + 2:].split() if close >= 0 else []
    try:
        return int(fields[1]), int(fields[2])
    except (IndexError, ValueError):
        return None


def process_tree_pids(root_pid: int) -> set[int]:
    """Snapshot a process and all descendants, including new ROS process groups."""
    children: dict[int, list[int]] = {}
    groups: dict[int, int] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        identity = _proc_parent_and_group(pid)
        if identity is None:
            continue
        parent, group = identity
        children.setdefault(parent, []).append(pid)
        groups[pid] = group
    found: set[int] = {root_pid} if root_pid in groups else set()
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


def _pid_running(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    close = text.rfind(")")
    fields = text[close + 2:].split() if close >= 0 else []
    return bool(fields) and fields[0] != "Z"


def _signal_managed_tree(root_pid: int, tracked: set[int], sig: int) -> None:
    tracked.update(process_tree_pids(root_pid))
    # Every process started here is a session leader, so its PID remains the
    # original process-group ID even if the ros2 wrapper exits first.
    try:
        os.killpg(root_pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    for pid in sorted(tracked, reverse=True):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate(proc: subprocess.Popen | None, timeout: float = 20.0) -> list[int]:
    """Terminate the complete launched tree and return any surviving PIDs."""
    if proc is None:
        return []
    tracked = process_tree_pids(proc.pid)
    for sig, wait_s in (
        (signal.SIGINT, timeout),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 5.0),
    ):
        _signal_managed_tree(proc.pid, tracked, sig)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            running = [pid for pid in tracked if _pid_running(pid)]
            if not running:
                return []
            time.sleep(0.05)
    return sorted(pid for pid in tracked if _pid_running(pid))


def topic_ready(topic: str, message_type: str, env: dict[str, str],
                timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if publisher_count(topic, env) > 0:
            break
        time.sleep(0.25)
    else:
        return False
    result = subprocess.run(
        [ROS2, "topic", "echo", "--once", "--timeout", str(timeout),
         "--qos-profile", "best_available", topic, message_type],
        cwd=PROJECT, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def read_log_count(path: Path, phrase: str) -> int:
    try:
        return path.read_text(encoding="utf-8", errors="replace").count(phrase)
    except FileNotFoundError:
        return 0


def recent_pose(status: dict[str, Any]) -> dict[str, float] | None:
    value = status.get("recent_map_base_pose")
    if not isinstance(value, dict) or int(value.get("sample_count", 0)) < 5:
        return None
    return {"x": float(value["x"]), "y": float(value["y"]),
            "yaw": float(value["yaw"])}


def wait_for_initial_gate(status_path: Path, rtabmap_log: Path,
                          timeout: float, min_matches: int,
                          min_odom_hz: float, max_lost_ratio: float,
                          max_static_xy_rms: float,
                          max_static_yaw_rms: float,
                          expected_map_pose: dict[str, float] | None = None,
                          max_seed_position_error: float = math.inf,
                          max_seed_yaw_error: float = math.inf,
                          require_global_match: bool = True,
                          min_static_samples: int = 20) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.5)
            continue
        pose = last_status.get("recent_map_base_pose") or {}
        match_count = max(
            int(last_status.get("global_match_events", 0)),
            read_log_count(rtabmap_log, "Localization was good"),
        )
        checks = {
            # Global validation is explicitly world-seeded and disables
            # appearance detection before motion. In that mode a stable,
            # seed-consistent map->base_link is the initial localization
            # evidence; other modes still require an RTAB-Map match event.
            "global_match": (not require_global_match) or match_count >= min_matches,
            "odom_rate": float(last_status.get("recent_odom_rate_hz", 0.0)) >= min_odom_hz,
            "odom_not_lost": float(last_status.get("recent_odom_lost_ratio", 1.0)) <= max_lost_ratio,
            "map_base_tf_fresh": float(last_status.get("map_base_tf_age_s", 999.0)) <= 0.5,
            "static_samples": int(pose.get("sample_count", 0)) >= min_static_samples,
            "static_xy": float(pose.get("xy_rms_m", 999.0)) <= max_static_xy_rms,
            "static_yaw": float(pose.get("yaw_rms_rad", 999.0)) <= max_static_yaw_rms,
        }
        if expected_map_pose is not None:
            pose_x = float(pose.get("x", math.nan))
            pose_y = float(pose.get("y", math.nan))
            pose_yaw = float(pose.get("yaw", math.nan))
            checks["initial_seed_position_consistency"] = (
                math.isfinite(pose_x) and math.isfinite(pose_y) and
                math.hypot(pose_x - expected_map_pose["x"],
                           pose_y - expected_map_pose["y"]) <= max_seed_position_error
            )
            checks["initial_seed_yaw_consistency"] = (
                math.isfinite(pose_yaw) and
                abs(wrap_angle(pose_yaw - expected_map_pose["yaw"])) <= max_seed_yaw_error
            )
        if all(checks.values()):
            return {"passed": True, "checks": checks, "status": last_status,
                    "global_match_count": match_count}
        time.sleep(0.5)
    return {"passed": False, "checks": checks if last_status else None,
            "status": last_status,
            "global_match_count": max(
                int(last_status.get("global_match_events", 0)),
                read_log_count(rtabmap_log, "Localization was good"),
            )}


def monitor_worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="RTAB-Map localization monitor")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window", type=float, default=5.0)
    args = parser.parse_args(argv)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from rtabmap_msgs.msg import Info, OdomInfo
    from tf2_ros import Buffer, TransformException, TransformListener

    class LocalizationMonitor(Node):
        def __init__(self) -> None:
            super().__init__("world_anchored_localization_monitor")
            self.started_wall = time.time()
            self.odom: list[dict[str, Any]] = []
            self.odom_info: list[dict[str, Any]] = []
            self.info_count = 0
            self.global_events: list[dict[str, Any]] = []
            self.last_stats: dict[str, float] = {}
            self.map_base: list[dict[str, float]] = []
            self.map_odom: list[dict[str, float]] = []
            self.last_tf_wall = 0.0
            self.buffer = Buffer(cache_time=Duration(seconds=30.0))
            self.listener = TransformListener(self.buffer, self)
            self.create_subscription(Odometry, "/odom", self.on_odom, 100)
            self.create_subscription(OdomInfo, "/odom_info", self.on_odom_info, 100)
            # Direct ``ros2 run rtabmap_slam rtabmap`` publishes this public
            # topic at /info.  /rtabmap/info would only exist with a matching
            # node namespace/remap and silently misses all localization events.
            self.create_subscription(Info, "/info", self.on_info, 100)
            self.create_timer(0.1, self.sample_tf)
            self.create_timer(0.5, self.write_status)

        def on_odom(self, message: Odometry) -> None:
            stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            pose = message.pose.pose
            self.odom.append({
                "wall_time": time.time(), "stamp": stamp,
                "x": float(pose.position.x), "y": float(pose.position.y),
                "yaw": quaternion_yaw(pose.orientation.x, pose.orientation.y,
                                      pose.orientation.z, pose.orientation.w),
            })

        def on_odom_info(self, message: OdomInfo) -> None:
            self.odom_info.append({
                "wall_time": time.time(), "lost": bool(message.lost),
                "matches": int(message.matches), "inliers": int(message.inliers),
                "features": int(message.features),
            })

        def on_info(self, message: Info) -> None:
            self.info_count += 1
            stats = {key: float(value) for key, value in
                     zip(message.stats_keys, message.stats_values)}
            self.last_stats = stats
            if message.loop_closure_id > 0 or message.proximity_detection_id > 0:
                self.global_events.append({
                    "wall_time": time.time(), "ref_id": int(message.ref_id),
                    "loop_closure_id": int(message.loop_closure_id),
                    "proximity_detection_id": int(message.proximity_detection_id),
                    "localization_stats": {
                        key: value for key, value in stats.items()
                        if "Loop" in key or "Proximity" in key or "Localization" in key
                    },
                })

        def lookup(self, parent: str, child: str) -> dict[str, float]:
            transform = self.buffer.lookup_transform(parent, child, Time())
            t = transform.transform.translation
            q = transform.transform.rotation
            return {
                "wall_time": time.time(), "x": float(t.x), "y": float(t.y),
                "yaw": quaternion_yaw(q.x, q.y, q.z, q.w),
                "stamp": float(transform.header.stamp.sec) +
                         float(transform.header.stamp.nanosec) * 1e-9,
            }

        def sample_tf(self) -> None:
            try:
                self.map_base.append(self.lookup("map", "base_link"))
                self.map_odom.append(self.lookup("map", "odom"))
                self.last_tf_wall = time.time()
            except TransformException:
                return

        def status(self) -> dict[str, Any]:
            now = time.time()
            cutoff = now - args.window
            recent_odom = [item for item in self.odom if item["wall_time"] >= cutoff]
            recent_info = [item for item in self.odom_info if item["wall_time"] >= cutoff]
            recent_poses = [item for item in self.map_base if item["wall_time"] >= cutoff]
            rate = 0.0
            if len(recent_odom) >= 2:
                span = recent_odom[-1]["wall_time"] - recent_odom[0]["wall_time"]
                rate = (len(recent_odom) - 1) / span if span > 0.0 else 0.0
            lost_ratio = 1.0
            inliers_median = 0.0
            if recent_info:
                lost_ratio = sum(int(item["lost"]) for item in recent_info) / len(recent_info)
                inliers_median = statistics.median(item["inliers"] for item in recent_info)
            return {
                "format": "world_anchored_localization_monitor_status_v1",
                "wall_time": now,
                "elapsed_s": now - self.started_wall,
                "odom_messages": len(self.odom),
                "odom_info_messages": len(self.odom_info),
                "rtabmap_info_messages": self.info_count,
                "global_match_events": len(self.global_events),
                "recent_odom_rate_hz": rate,
                "recent_odom_lost_ratio": lost_ratio,
                "recent_odom_inliers_median": inliers_median,
                "map_base_tf_age_s": now - self.last_tf_wall if self.last_tf_wall else 999.0,
                "recent_map_base_pose": pose_statistics(recent_poses),
                "last_localization_stats": {
                    key: value for key, value in self.last_stats.items()
                    if "Loop" in key or "Proximity" in key or "Localization" in key
                },
            }

        def write_status(self) -> None:
            atomic_json(args.status, self.status())

        def report(self) -> dict[str, Any]:
            return {
                "format": "world_anchored_localization_monitor_report_v1",
                "started_wall": self.started_wall,
                "finished_wall": time.time(),
                "odom": self.odom,
                "odom_info": self.odom_info,
                "global_events": self.global_events,
                "map_base_samples": self.map_base,
                "map_odom_samples": self.map_odom,
                "final_status": self.status(),
            }

    rclpy.init(args=None)
    node = LocalizationMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_status()
        atomic_json(args.report, node.report())
        node.destroy_node()
        rclpy.shutdown()
    return 0


def truth_tf_worker(argv: list[str]) -> int:
    """Fuse authorized root truth with visual odometry into map->odom."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation", nargs=2, type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--truth-topic", default="/task1/truth_pose")
    parser.add_argument("--odom-topic", default="/odom")
    args = parser.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster

    class TruthMapOdom(Node):
        def __init__(self) -> None:
            super().__init__("task1_truth_map_odom")
            self.odom_pose = None
            self.broadcaster = TransformBroadcaster(self)
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, 20)
            self.create_subscription(PoseStamped, args.truth_topic, self.on_truth, 20)

        @staticmethod
        def yaw_from_quaternion(q) -> float:
            return math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )

        def on_odom(self, msg: Odometry) -> None:
            p = msg.pose.pose.position
            self.odom_pose = (
                float(p.x), float(p.y),
                self.yaw_from_quaternion(msg.pose.pose.orientation),
            )

        def on_truth(self, msg: PoseStamped) -> None:
            if self.odom_pose is None:
                return
            p = msg.pose.position
            world_yaw = self.yaw_from_quaternion(msg.pose.orientation)
            c, s = math.cos(args.yaw), math.sin(args.yaw)
            map_x = c * float(p.x) - s * float(p.y) + args.translation[0]
            map_y = s * float(p.x) + c * float(p.y) + args.translation[1]
            map_yaw = wrap_angle(world_yaw + args.yaw)
            odom_x, odom_y, odom_yaw = self.odom_pose
            map_odom_yaw = wrap_angle(map_yaw - odom_yaw)
            co, so = math.cos(map_odom_yaw), math.sin(map_odom_yaw)
            map_odom_x = map_x - (co * odom_x - so * odom_y)
            map_odom_y = map_y - (so * odom_x + co * odom_y)

            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = "map"
            tf.child_frame_id = "odom"
            tf.transform.translation.x = map_odom_x
            tf.transform.translation.y = map_odom_y
            tf.transform.rotation.z = math.sin(map_odom_yaw * 0.5)
            tf.transform.rotation.w = math.cos(map_odom_yaw * 0.5)
            self.broadcaster.sendTransform(tf)

    rclpy.init()
    node = TruthMapOdom()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def post_run_metrics(report: dict[str, Any], calibration_time: float,
                     motion_time: float | None) -> dict[str, Any]:
    poses = [item for item in report.get("map_base_samples", [])
             if float(item["wall_time"]) >= calibration_time]
    odom_info = [item for item in report.get("odom_info", [])
                 if float(item["wall_time"]) >= calibration_time]
    events = [item for item in report.get("global_events", [])
              if motion_time is not None and float(item["wall_time"]) >= motion_time]
    translation_steps: list[float] = []
    yaw_steps: list[float] = []
    for first, second in zip(poses, poses[1:]):
        dt = float(second["wall_time"]) - float(first["wall_time"])
        if dt <= 0.0 or dt > 0.5:
            continue
        translation_steps.append(math.hypot(float(second["x"]) - float(first["x"]),
                                            float(second["y"]) - float(first["y"])))
        yaw_steps.append(abs(wrap_angle(float(second["yaw"]) - float(first["yaw"]))))
    return {
        "post_calibration_map_pose_samples": len(poses),
        "post_calibration_odom_info_samples": len(odom_info),
        "post_calibration_lost_ratio": (
            sum(int(item["lost"]) for item in odom_info) / len(odom_info)
            if odom_info else 1.0
        ),
        "post_motion_global_match_events": len(events),
        "max_consecutive_translation_step_m": max(translation_steps, default=0.0),
        "max_consecutive_yaw_step_rad": max(yaw_steps, default=0.0),
    }


NAV2_VALIDATION_STACK_NODES = (
    "/controller_server",
    "/planner_server",
)


def query_lifecycle_state(node: str, env: dict[str, str]) -> str:
    """Return the lifecycle state line for ``node``, or an error placeholder."""
    try:
        result = subprocess.run(
            [ROS2, "lifecycle", "get", node], cwd=PROJECT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=5.0, check=False,
        )
    except subprocess.TimeoutExpired:
        return "query_timeout"
    if result.returncode != 0:
        return (result.stdout or result.stderr or "unavailable").strip()
    return (result.stdout or "").strip()


def lifecycle_is_active(state: str) -> bool:
    """``ros2 lifecycle get`` prints ``active [3]``; ``inactive [2]`` must not match."""
    token = state.strip().split(" ", 1)[0].lower()
    return token == "active"


def wait_for_lifecycle_active(
    node: str,
    env: dict[str, str],
    timeout: float,
    *,
    retries: int = 3,
) -> None:
    last_error = ""
    for attempt in range(1, max(1, int(retries)) + 1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            last_error = query_lifecycle_state(node, env)
            if lifecycle_is_active(last_error):
                if attempt > 1:
                    print(
                        f"Nav2 lifecycle {node} became active on attempt {attempt}",
                        flush=True,
                    )
                return
            time.sleep(0.5)
        print(
            f"Nav2 lifecycle {node} not active after attempt {attempt}/{retries}; "
            f"last_state={last_error!r}",
            flush=True,
        )
        time.sleep(1.0)
    raise RuntimeError(
        f"Nav2 lifecycle node did not become active: {node} "
        f"(last_state={last_error!r})"
    )


def wait_for_nav2_stack_ready(
    env: dict[str, str],
    session: Path,
    timeout: float,
    processes: dict[str, subprocess.Popen],
    *,
    log_name: str = "validation_nav2",
    nodes: tuple[str, ...] = NAV2_VALIDATION_STACK_NODES,
    poll_s: float = 1.0,
) -> None:
    """Wait until the full Nav2 bringup chain reaches active (not one node at a time)."""
    deadline = time.monotonic() + float(timeout)
    last_states: dict[str, str] = {}
    last_report = 0.0
    while time.monotonic() < deadline:
        nav2_proc = processes.get(log_name)
        if nav2_proc is not None and nav2_proc.poll() is not None:
            tail = process_log_tail(session, log_name, limit=60)
            raise RuntimeError(
                f"Nav2 launch exited before the stack became active "
                f"(code={nav2_proc.returncode}); see {log_name}.log:\n{tail}"
            )
        all_active = True
        for node in nodes:
            state = query_lifecycle_state(node, env)
            last_states[node] = state
            if not lifecycle_is_active(state):
                all_active = False
        if all_active:
            print(f"Nav2 stack ready: {last_states}", flush=True)
            return
        now = time.monotonic()
        if now - last_report >= 15.0:
            print(
                f"Nav2 bringup in progress ({int(now - (deadline - timeout))}s): "
                f"{last_states}",
                flush=True,
            )
            last_report = now
        time.sleep(poll_s)
    tail = process_log_tail(session, log_name, limit=60)
    raise RuntimeError(
        f"Nav2 stack did not become active within {timeout:.0f}s: {last_states}. "
        f"See {log_name}.log tail:\n{tail}"
    )


def assert_validation_start_pose(
    pose: dict[str, float],
    expected: dict[str, float],
    *,
    max_shift_m: float,
    scene_reset: bool,
) -> None:
    """Fail fast before route preflight when the robot is not at the spawn corridor."""
    shift = math.hypot(
        float(pose["x"]) - float(expected["x"]),
        float(pose["y"]) - float(expected["y"]),
    )
    if shift <= float(max_shift_m):
        return
    if scene_reset:
        hint = (
            "Scene reset was requested but localization converged away from spawn; "
            "check RTAB-Map initial pose and mapping database."
        )
    else:
        hint = (
            "Re-run with --reset-scene so the robot returns to the south spawn "
            "before route preflight."
        )
    raise RuntimeError(
        "validation start pose is "
        f"{shift:.2f}m from expected map seed "
        f"({float(expected['x']):.2f}, {float(expected['y']):.2f}); "
        f"limit is {float(max_shift_m):.2f}m. {hint}"
    )


def wait_process_ready(
    name: str,
    processes: dict[str, subprocess.Popen],
    session: Path,
    failure: str,
    *,
    ready: Callable[[], bool] | None = None,
    timeout: float = 30.0,
) -> None:
    """Poll a freshly started helper until it signals readiness.

    Blind start-up sleeps both waste seconds on a fast host and hide a crash
    until the next stage fails with an unrelated error.
    """
    deadline = time.monotonic() + timeout
    while True:
        if processes[name].poll() is not None:
            tail = process_log_tail(session, name)
            raise RuntimeError(failure + (":\n" + tail if tail else ""))
        if ready is None or ready():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{failure} (not ready after {timeout:.1f}s)")
        time.sleep(0.05)


def start_validation_bridge(
    args: argparse.Namespace, env: dict[str, str], session: Path,
    processes: dict[str, subprocess.Popen],
) -> None:
    """Start the chassis bridge before RGB-D/RTAB-Map nodes.

    OrcaGymLocalEnv construction performs a simulator initialization/reset.
    Starting this bridge after RTAB-Map would invalidate the active odom
    chain. The bridge restores the already-reset live state once, before
    visual odometry starts.
    """
    bridge_dir = session / "validation_chassis_bridge"
    bridge_command = [
        ORCALAB_PYTHON, str(SRC / "run_agibot_chassis_execution_bridge.py"),
        "--robot-name", args.robot_name,
        "--arm-posture", "current",
        "--no-publish-truth-odom",
        "--cmd-vel-topic", "/cmd_vel",
        "--camera-render-hz", str(args.camera_render_hz),
        "--max-linear-speed", str(max(args.validation_speed, 0.08)),
        "--wheelbase", str(CHASSIS_WHEELBASE_M),
        "--max-steering-angle", str(CHASSIS_MAX_STEER_RAD),
        "--min-turning-radius", str(VALIDATION_MIN_TURNING_RADIUS_M),
        "--speed", str(args.validation_speed),
        "--linear-slew-rate", str(args.validation_linear_accel_limit),
        "--linear-jerk-limit", str(args.validation_linear_jerk_limit),
        "--turn-speed", str(args.validation_turn_speed),
        "--record-truth-audit",
        "--publish-truth-assist",
        "--truth-audit-hz", str(args.validation_truth_audit_hz),
        "--truth-body-link-topic", "/task1/truth_body_link_pose",
        "--output-dir", str(bridge_dir),
    ]
    if not args.validation_contact_monitor:
        bridge_command.append("--disable-collision-monitor")
    start_process("validation_bridge", bridge_command, env, session, processes)
    # The bridge snapshots the scene through OrcaGym right after the simulator
    # connection is live, so the inventory file marks the end of its reset.
    inventory = bridge_dir / "scene_inventory.json"
    wait_process_ready(
        "validation_bridge", processes, session,
        "validation chassis bridge exited during startup",
        ready=inventory.is_file, timeout=60.0,
    )


def wait_for_robot_started(
    marker: Path,
    process: subprocess.Popen | None,
    *,
    timeout_s: float = 120.0,
    poll_s: float = 0.05,
) -> bool:
    """Block until the mission writes robot_started.marker or the process exits."""
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if marker.is_file():
            return True
        if process is not None and process.poll() is not None:
            return marker.is_file()
        time.sleep(poll_s)
    return marker.is_file()


def load_zero_motion_preflight(preflight_dir: Path) -> dict[str, Any]:
    result_path = preflight_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError("inline route preflight did not write result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "succeeded":
        raise RuntimeError(
            "inline route preflight failed: "
            + str(result.get("error") or "unknown error")
        )
    return result


def run_global_validation_route(
    args: argparse.Namespace,
    env: dict[str, str],
    session: Path,
    processes: dict[str, subprocess.Popen],
    initial_world: dict[str, float],
    initial_map: dict[str, float],
    calibration: dict[str, Any],
    official_scorer: OfficialTaskScorer | None = None,
    *,
    localization_pose: dict[str, float] | None = None,
) -> dict[str, Any]:
    world_points_raw = json.loads(args.validation_world_points_json)
    route_stop = str(getattr(args, "validation_route_stop", "C") or "C").upper()
    expected_route = validation_route_order(route_stop)
    if not isinstance(world_points_raw, dict) or len(world_points_raw) < len(expected_route):
        raise RuntimeError(
            f"global validation requires at least {len(expected_route)} named world points "
            f"for route stop {route_stop}"
        )
    if any(not isinstance(values, list) or len(values) != 2
           for values in world_points_raw.values()):
        raise RuntimeError("each validation world point must contain exactly [x, y]")
    world_points = order_route_points(world_points_raw)
    safety_report_path = session / "validation_safety_map.json"
    safety_map_command = [
        ORCALAB_PYTHON, str(SRC / "run_world_anchored_validation_safety_map.py"),
        "--robot-name", args.robot_name,
        "--mapping-yaml", str(args.validation_map_yaml.resolve()),
        "--calibration", str((session / "world_to_map_calibration.json").resolve()),
        "--world-points-json", json.dumps(world_points, separators=(",", ":")),
        "--start-world-xy", str(initial_world["x"]), str(initial_world["y"]),
        "--point-clearance", str(args.validation_point_clearance),
        "--max-point-shift", str(args.validation_max_point_shift),
        "--topic", "/validation/safety_map",
        "--report", str(safety_report_path),
    ]
    if args.validation_augment_scene_geometry:
        safety_map_command.append("--augment-scene-geometry")
    start_process("validation_safety_map", safety_map_command, env, session, processes)
    if not topic_ready(
        "/validation/safety_map", "nav_msgs/msg/OccupancyGrid", env, 30.0
    ):
        raise RuntimeError("validation safety map did not become ready")
    safety_report = json.loads(safety_report_path.read_text(encoding="utf-8"))
    if bool(safety_report.get("scene_geometry_applied_to_grid")) != bool(
        args.validation_augment_scene_geometry
    ):
        raise RuntimeError("validation safety-map scene geometry mode mismatch")
    map_points = safety_report["resolved_map_points"]
    resolved_world_points = safety_report["resolved_world_points"]
    route_map_points = order_route_points(
        contract_map_points_from_safety_report(safety_report)
    )
    contract_map_points = scored_route_points(route_map_points)
    evidence_bag = session / "rviz_evidence_bag"
    evidence_topics = [
        "/validation/safety_map",
        "/global_costmap/costmap", "/global_costmap/costmap_updates",
        "/global_costmap/published_footprint",
        "/local_costmap/costmap", "/local_costmap/costmap_updates",
        "/local_costmap/published_footprint",
        "/task1/planned_route", "/task1/active_global_path",
        "/task1/route_markers", "/task1/navigation_status",
        "/plan", "/odom", "/odom_info", "/info", "/cmd_vel",
        "/tf", "/tf_static",
    ]
    start_process("validation_bag", [
        ROS2, "bag", "record", "-o", str(evidence_bag), *evidence_topics,
    ], env, session, processes)
    wait_process_ready(
        "validation_bag", processes, session,
        "RViz evidence rosbag exited during startup",
        ready=evidence_bag.is_dir, timeout=20.0,
    )
    route_manifest = {
        "format": "world_anchored_global_validation_route_v1",
        "requested_world_points": world_points,
        "world_points": resolved_world_points,
        "map_targets_from_safety_map": map_points,
        "task_coverage_map_points": contract_map_points,
        "route_map_points": route_map_points,
        "safety_map": str(safety_report_path.resolve()),
        "safety_map_point_audit": safety_report["point_audit"],
        "static_map_contract": safety_report["static_grid_mode"],
        "scene_geometry_audit_only": not args.validation_augment_scene_geometry,
        "scene_geometry_applied_to_grid": args.validation_augment_scene_geometry,
        "scene_cells_added": safety_report["scene_cells_added"],
        "scene_cells_not_in_source_map": safety_report["scene_cells_not_in_source_map"],
        "execution_backend": (
            "nav2_navigate_to_pose"
            if args.validation_adaptive_route_profile
            else "hybrid_theta_mppi"
            if getattr(args, "validation_theta_star_planner", False)
            else "nav2_follow_path"
        ),
        "adaptive_route_profile": bool(args.validation_adaptive_route_profile),
        "validation_speed_m": args.validation_speed,
        "linear_accel_limit_mps2": args.validation_linear_accel_limit,
        "linear_jerk_limit_mps3": args.validation_linear_jerk_limit,
        "path_speed_profile": {
            "enabled": True,
            "lateral_accel_limit_mps2": args.validation_profile_lateral_accel,
            "clearance_hard_m": args.validation_profile_clearance_hard,
            "clearance_soft_m": args.validation_profile_clearance_soft,
            "lookahead_m": args.validation_profile_lookahead,
        },
        "contact_monitor_enabled": bool(args.validation_contact_monitor),
        "waypoint_coordinate_tolerance_m":
            args.validation_waypoint_coordinate_tolerance,
        "final_coordinate_tolerance_m":
            args.validation_final_coordinate_tolerance,
        "rviz_config": str((ROOT / "task1_world_anchored_validation.rviz").resolve()),
        "rviz_evidence_bag": str(evidence_bag.resolve()),
        "rviz_evidence_topics": evidence_topics,
        "truth_role": "non-published offline audit samples only",
        "control_pose_source": "map->odom->base_link from RGB-D RTAB-Map localization",
    }
    atomic_json(session / "global_validation_route.json", route_manifest)

    import yaml
    nav2_config = yaml.safe_load(
        (ROOT / "config" / "nav2_truth.yaml").read_text(encoding="utf-8")
    )
    grid = nav2_config["planner_server"]["ros__parameters"]["GridBased"]
    use_theta_star = bool(getattr(args, "validation_theta_star_planner", False))
    use_astar = not use_theta_star and bool(
        getattr(args, "validation_astar_planner", False)
    )
    if use_astar:
        # Nav2 SmacPlanner2D (grid A*). Optional; ABC direct B→C defaults to Hybrid.
        for key in list(grid):
            grid.pop(key, None)
        grid.update({
            "plugin": "nav2_smac_planner::SmacPlanner2D",
            "tolerance": 0.12,
            "downsample_costmap": False,
            "downsampling_factor": 1,
            "allow_unknown": True,
            "max_iterations": 1000000,
            "max_on_approach_iterations": 1000,
            "max_planning_time": 5.0,
            "terminal_checking_interval": 5000,
            "cost_travel_multiplier": 2.0,
            "use_final_approach_orientation": False,
            "smoother": {
                "max_iterations": 1000,
                "w_smooth": 0.3,
                "w_data": 0.2,
                "tolerance": 1.0e-10,
                "do_refinement": True,
            },
        })
        route_manifest["global_planner"] = "nav2_smac_planner::SmacPlanner2D"
    elif use_theta_star:
        grid["plugin"] = "nav2_theta_star_planner::ThetaStarPlanner"
        for key in list(grid):
            if key in {
                "plugin", "tolerance", "how_many_corners",
                "w_euc_cost", "w_traversal_cost",
            }:
                continue
            grid.pop(key, None)
        grid["tolerance"] = 0.12
        grid["how_many_corners"] = 8
        grid["w_euc_cost"] = 1.2
        grid["w_traversal_cost"] = 1.4
        route_manifest["global_planner"] = "nav2_theta_star_planner::ThetaStarPlanner"
    else:
        # Task 1 forbids reverse motion throughout the inspection route.
        grid["motion_model_for_search"] = "DUBIN"
        grid["tolerance"] = 0.12
        grid["cost_penalty"] = float(VALIDATION_PLANNER_COST_PENALTY)
        grid["non_straight_penalty"] = float(VALIDATION_PLANNER_NON_STRAIGHT_PENALTY)
        grid["analytic_expansion_max_length"] = 4.0
        # Keep Hybrid smooth_path off: it cuts the last C apex into the cabinet.
        grid["smooth_path"] = False
        # Session 123224: raising this to 1.35 globally overshot the first C
        # bend (robot 0.50 m north of the west hall). Last-cabinet clearance
        # is recovered by widen_last_c_turn + north terminal headings, not by
        # a coarser Dubins radius on A/B/C.
        grid["minimum_turning_radius"] = float(args.validation_min_turning_radius)
        route_manifest["global_planner"] = "nav2_smac_planner::SmacPlannerHybrid"
        route_manifest["planner_cost_penalty"] = float(VALIDATION_PLANNER_COST_PENALTY)
        route_manifest["planner_minimum_turning_radius_m"] = float(
            grid["minimum_turning_radius"]
        )
        route_manifest["chassis_minimum_turning_radius_m"] = float(
            args.validation_min_turning_radius
        )
        route_manifest["chassis_wheelbase_m"] = float(CHASSIS_WHEELBASE_M)
        route_manifest["chassis_max_steer_rad"] = float(CHASSIS_MAX_STEER_RAD)
    goal_checker = nav2_config["controller_server"]["ros__parameters"][
        "general_goal_checker"
    ]
    goal_checker["xy_goal_tolerance"] = min(
        0.25, float(args.validation_final_coordinate_tolerance)
    )
    goal_checker["yaw_goal_tolerance"] = 0.40
    follow = nav2_config["controller_server"]["ros__parameters"]["FollowPath"]
    # MPPI derives every constraint ratio from base vx_max, so it must be the
    # speed the chassis is actually expected to hold. A fictitious ceiling (the
    # 20 m/s contract number) turns any absolute /speed_limit into a
    # limit/vx_max ratio that also divides wz_max, leaving the robot unable to
    # steer through the B->C bend.
    cruise = mppi_cruise_speed(float(args.validation_speed))
    horizon_s = float(follow["time_steps"]) * float(follow["model_dt"])
    follow["vx_max"] = cruise
    follow["vx_min"] = 0.0
    follow["vx_std"] = max(0.30, cruise * 0.30)
    follow["iteration_count"] = 2 if cruise > 1.4 else 1
    # MPPI's Ackermann model clamps wz to vx / min_turning_r. When that radius
    # is finer than the one the planner and the chassis honour, MPPI scores
    # trajectories that pivot at almost zero speed, commands them, and the
    # steering saturates without producing the turn. Sharing the planner radius
    # keeps the sampled yaw rate and linear speed mutually executable.
    turning_radius = float(args.validation_min_turning_radius)
    follow["AckermannConstraints"]["min_turning_r"] = turning_radius
    follow["wz_max"] = min(
        float(args.validation_max_angular), cruise / turning_radius,
    )
    follow["wz_std"] = max(0.2, follow["wz_max"] * 0.35)
    follow["prune_distance"] = max(12.0, cruise * horizon_s * 2.0)
    # PathAlignCritic scores the mean trajectory-to-path distance, so a stopped
    # robot always scores best. Keep it well below PathFollowCritic and add a
    # forward-progress bias so MPPI does not crawl to a halt on the A/B straights.
    follow["PathFollowCritic"]["cost_weight"] = 18.0
    follow["PathFollowCritic"]["offset_from_furthest"] = mppi_path_offset(
        MPPI_PATH_FOLLOW_LOOKAHEAD_M,
    )
    follow["PathAlignCritic"]["cost_weight"] = 3.0
    follow["PathAlignCritic"]["offset_from_furthest"] = mppi_path_align_offset(
        MPPI_MIN_TRACKING_MPS, horizon_s,
    )
    follow["PathAlignCritic"]["max_path_occupancy_ratio"] = 0.2
    follow["PreferForwardCritic"]["enabled"] = True
    follow["PreferForwardCritic"]["cost_weight"] = 2.0
    follow["PreferForwardCritic"]["threshold_to_consider"] = 0.15
    follow["PathAngleCritic"]["cost_weight"] = 4.0
    follow["GoalCritic"]["cost_weight"] = 5.0
    follow["GoalCritic"]["threshold_to_consider"] = MPPI_GOAL_BRAKING_M
    follow["GoalAngleCritic"]["threshold_to_consider"] = 0.35
    follow["CostCritic"]["near_goal_distance"] = 0.6
    follow["ax_max"] = float(args.validation_linear_accel_limit)
    follow["ax_min"] = -float(args.validation_linear_accel_limit)
    smoother = nav2_config["velocity_smoother"]["ros__parameters"]
    smoother["max_velocity"] = [cruise, 0.0, float(follow["wz_max"])]
    smoother["min_velocity"] = [0.0, 0.0, -float(follow["wz_max"])]
    smoother["max_accel"] = [
        float(args.validation_linear_accel_limit), 0.0,
        float(smoother["max_accel"][2]),
    ]
    smoother["max_decel"] = [
        -float(args.validation_linear_accel_limit), 0.0,
        float(smoother["max_decel"][2]),
    ]
    collision = nav2_config["collision_monitor"]["ros__parameters"]
    # Publish the Nav2 chain onto /cmd_vel. The current RGB-D bridge does not
    # emit /camera/obstacles, so disable that source instead of stopping the
    # chassis on a sensor timeout.
    collision["cmd_vel_out_topic"] = "cmd_vel"
    collision["source_timeout"] = 3600.0
    collision["head"]["enabled"] = False
    collision["left"]["enabled"] = False
    collision["right"]["enabled"] = False
    for costmap in ("local_costmap", "global_costmap"):
        nav2_config[costmap][costmap]["ros__parameters"]["static_layer"][
            "map_topic"
        ] = "/validation/safety_map"
    # MPPI truncates its path preview at the local costmap border, so a 6 m
    # rolling window caps the lookahead at 3 m (and raises INVALID_PATH once the
    # robot drifts outside it). Size the window to the braking/preview horizon.
    local_costmap = nav2_config["local_costmap"]["local_costmap"]["ros__parameters"]
    local_extent = mppi_local_costmap_extent(cruise, horizon_s)
    local_costmap["width"] = local_extent
    local_costmap["height"] = local_extent
    local_costmap["resolution"] = 0.1
    local_costmap["update_frequency"] = 10.0
    nav2_config_path = session / "nav2_global_validation.yaml"
    nav2_config_path.write_text(yaml.safe_dump(nav2_config, sort_keys=False),
                                encoding="utf-8")

    mission_dir = session / "validation_mission"
    if "validation_bridge" not in processes:
        raise RuntimeError("validation chassis bridge must be prestarted before RGB-D localization")

    disable_rtabmap_tf = set_ros_param(
        "/rtabmap", "publish_tf", "false", env,
    )
    if (disable_rtabmap_tf.returncode != 0 or
            "successful" not in disable_rtabmap_tf.stdout.lower()):
        raise RuntimeError(
            "failed to transfer map->odom authority from RTAB-Map: "
            + (disable_rtabmap_tf.stderr or disable_rtabmap_tf.stdout).strip()
        )
    start_process("truth_tf", [
        ORCALAB_PYTHON, str(Path(__file__).resolve()), "--truth-tf-worker",
        "--translation",
        f"{float(calibration['translation_xy'][0]):.12f}",
        f"{float(calibration['translation_xy'][1]):.12f}",
        f"--yaw={float(calibration['yaw_rad']):.12f}",
    ], env, session, processes)
    # Nav2 bringup takes seconds to reach the active lifecycle state, so let it
    # boot alongside the TF publisher instead of waiting on the publisher first.
    start_process("validation_nav2", [
        ROS2, "launch", "nav2_bringup", "navigation_launch.py",
        f"params_file:={nav2_config_path}",
        "use_sim_time:=false", "autostart:=true",
    ], env, session, processes)
    wait_process_ready(
        "truth_tf", processes, session,
        "truth-assisted map->odom publisher exited during startup",
    )
    nav2_nodes = list(NAV2_VALIDATION_STACK_NODES)
    if args.validation_adaptive_route_profile:
        nav2_nodes.append("/bt_navigator")
    wait_for_nav2_stack_ready(
        env,
        session,
        args.validation_nav2_timeout,
        processes,
        nodes=tuple(nav2_nodes),
    )
    if not topic_ready(
        "/global_costmap/costmap", "nav_msgs/msg/OccupancyGrid", env, 30.0
    ):
        raise RuntimeError(
            "Nav2 global costmap did not publish the validation safety map"
        )
    if "rviz" not in processes:
        rviz_config = ROOT / "task1_world_anchored_validation.rviz"
        start_process(
            "rviz", ["rviz2", "-d", str(rviz_config)], env, session, processes,
        )
        route_manifest["rviz_config"] = str(rviz_config.resolve())

    start_pose = localization_pose or initial_map
    assert_validation_start_pose(
        start_pose,
        initial_map,
        max_shift_m=VALIDATION_START_POSE_MAX_SHIFT_M,
        scene_reset=bool(args.reset_scene),
    )

    navigator_args = [
        str(SRC / "run_agibot_abc_demo.py"),
        "--use-nav2-path",
        "--odom-topic", "/odom",
        "--truth-pose-topic", "/task1/truth_pose",
        "--truth-body-link-topic", "/task1/truth_body_link_pose",
        "--truth-world-to-map={:.12f},{:.12f},{:.12f}".format(
            float(calibration["translation_xy"][0]),
            float(calibration["translation_xy"][1]),
            float(calibration["yaw_rad"]),
        ),
        "--initial-map-pose", str(initial_map["x"]), str(initial_map["y"]),
        str(initial_map["yaw"]),
        "--initial-pose-tolerance", str(VALIDATION_START_POSE_MAX_SHIFT_M),
        "--cmd-vel-topic", "/cmd_vel",
        "--map-topic", "/validation/safety_map",
        "--waypoints-json", json.dumps(route_map_points, separators=(",", ":")),
        "--task-points-json", json.dumps(contract_map_points, separators=(",", ":")),
        "--judgement-offset-xy",
        str(args.validation_judgement_offset_xy[0]),
        str(args.validation_judgement_offset_xy[1]),
        "--timeout", str(args.validation_route_timeout),
        "--speed", str(args.validation_speed),
        "--max-angular", str(args.validation_max_angular),
        "--waypoint-coordinate-tolerance",
        str(args.validation_waypoint_coordinate_tolerance),
        "--final-coordinate-tolerance",
        str(args.validation_final_coordinate_tolerance),
        "--score-region-radius", str(args.validation_score_region_radius),
        "--score-circle-stop-offset",
        str(args.validation_score_circle_stop_offset),
        # Curvature tracking is retained only for non-Nav2 compatibility.
        "--max-pose-step", str(max(0.18, args.validation_speed * 0.40)),
        "--max-yaw-step", str(math.radians(8.0)),
        "--startup-pose-samples", str(VALIDATION_STARTUP_POSE_SAMPLES),
        "--max-tf-age", "1.50",
        "--path-tolerance", str(args.validation_path_tolerance),
        "--command-safety-clearance", "0.05",
        "--command-safety-horizon", "1.50",
        "--visual-pose-timeout", str(args.validation_visual_pose_timeout),
        "--checkpoint-settle", str(args.validation_checkpoint_settle),
        "--ab-checkpoint-dwell", str(args.validation_ab_checkpoint_dwell),
        "--post-robot-start-dwell", str(args.validation_post_robot_start_dwell),
        "--continuous-route",
        "--profile-lateral-accel", str(args.validation_profile_lateral_accel),
        "--profile-linear-accel", str(args.validation_linear_accel_limit),
        "--profile-linear-jerk", str(args.validation_linear_jerk_limit),
        "--profile-clearance-hard", str(args.validation_profile_clearance_hard),
        "--profile-clearance-soft", str(args.validation_profile_clearance_soft),
        "--profile-lookahead", str(args.validation_profile_lookahead),
        "--minimum-profile-speed", str(args.validation_minimum_profile_speed),
        "--final-stop-speed", str(args.validation_final_stop_speed),
        "--terminal-arrival-speed", str(args.validation_terminal_arrival_speed),
        "--terminal-zone-m", str(args.validation_terminal_zone_m),
        "--checkpoint-capture-min-speed",
        str(args.validation_checkpoint_capture_min_speed),
        "--route-stop-at", route_stop,
        "--straight-curvature-1pm", str(args.validation_straight_curvature_1pm),
        "--global-planner-plugin",
        "theta_star" if use_theta_star else "smac_2d" if use_astar else "smac_hybrid",
    ]
    if args.validation_peak_contract_speed is not None:
        navigator_args.extend([
            "--peak-contract-speed",
            str(args.validation_peak_contract_speed),
        ])
    preflight_dir = session / "validation_route_preflight"
    execution_flag = (
        "--use-nav2-navigate"
        if args.validation_adaptive_route_profile
        else "--use-hybrid-theta-mppi"
        if getattr(args, "validation_theta_star_planner", False)
        else "--use-profiled-controller"
    )
    mission_command = [
        SYSTEM_PYTHON, *navigator_args,
        "--inline-preflight",
        "--preflight-output-dir", str(preflight_dir),
        execution_flag,
        "--output-dir", str(mission_dir),
    ]
    marker = mission_dir / ROBOT_STARTED_MARKER_NAME
    start_process("validation_mission", mission_command, env, session, processes)
    scorer_ready = wait_for_robot_started(
        marker,
        processes.get("validation_mission"),
        timeout_s=min(args.validation_route_timeout, VALIDATION_MISSION_TIMEOUT_S),
    )
    if not scorer_ready:
        raise RuntimeError("validation mission did not publish robot_started.marker")
    preflight_result = load_zero_motion_preflight(preflight_dir)
    route_manifest["zero_motion_preflight"] = {
        "passed": True,
        "result": str((preflight_dir / "result.json").resolve()),
        "planned_paths": str((preflight_dir / "planned_paths.json").resolve()),
        "route_preview": preflight_result.get("route_preview"),
        "path_safety_reports": preflight_result.get("path_safety_reports"),
    }
    atomic_json(session / "global_validation_route.json", route_manifest)
    if official_scorer is not None:
        official_scorer.start()
    try:
        if VALIDATION_MISSION_DEADLINE_ENABLED:
            processes["validation_mission"].wait(
                timeout=args.validation_route_timeout
                + args.validation_nav2_timeout
            )
        else:
            processes["validation_mission"].wait()
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "global validation mission exceeded its process timeout"
        ) from exc
    if processes["validation_mission"].returncode != 0:
        raise RuntimeError("global validation mission failed; see validation_mission.log")
    return route_manifest


def build_global_geometry_report(
    initial_world: dict[str, float],
    initial_map: dict[str, float],
    initial_gate: dict[str, Any] | None,
    initial_calibration: dict[str, Any],
    motion_time: float | None,
    monitor_report: dict[str, Any],
    mission_result: dict[str, Any],
    bridge_report: dict[str, Any],
    expected_names: list[str],
    sample_window: float,
    max_position_rms: float,
    max_position: float,
    max_yaw_rms: float,
    max_yaw: float,
    min_coverage: float,
) -> dict[str, Any]:
    map_samples = monitor_report.get("map_base_samples", [])
    odom_samples = monitor_report.get("odom_info", [])
    global_events = monitor_report.get("global_events", [])
    truth_samples = bridge_report.get("truth_audit_samples", [])
    correspondences = [{
        "name": "START",
        "world_pose": dict(initial_world),
        "map_pose": dict(initial_map),
        "map_pose_statistics": (
            (initial_gate or {}).get("status", {}).get("recent_map_base_pose")
        ),
        "world_pose_statistics": None,
        "global_match_events_since_previous": int(
            (initial_gate or {}).get("global_match_count", 0)
        ),
    }]
    checkpoints = []
    previous_boundary = float(motion_time or 0.0)
    intervals = mission_result.get("checkpoint_intervals", [])
    for interval in intervals:
        finished = float(interval["settle_finished_unix"])
        started = max(float(interval["settle_started_unix"]), finished - sample_window)
        local_map = [item for item in map_samples
                     if started <= float(item["wall_time"]) <= finished]
        local_truth = [item for item in truth_samples
                       if started <= float(item["wall_time"]) <= finished]
        local_odom = [item for item in odom_samples
                      if started <= float(item["wall_time"]) <= finished]
        leg_events = [item for item in global_events
                      if previous_boundary <= float(item["wall_time"]) <= finished]
        map_stats = pose_statistics(local_map)
        truth_stats = pose_statistics(local_truth)
        lost_ratio = (sum(int(item["lost"]) for item in local_odom) / len(local_odom)
                      if local_odom else 1.0)
        checkpoint = {
            "name": interval["name"],
            "audit_window_unix": [started, finished],
            "map_pose_statistics": map_stats,
            "world_pose_statistics": truth_stats,
            "map_sample_count": len(local_map),
            "truth_sample_count": len(local_truth),
            "odom_info_sample_count": len(local_odom),
            "odom_lost_ratio": lost_ratio,
            "global_match_events_since_previous": len(leg_events),
            "global_match_ref_ids": [int(item["ref_id"]) for item in leg_events],
        }
        checkpoints.append(checkpoint)
        if map_stats is not None and truth_stats is not None:
            correspondences.append({
                "name": interval["name"],
                "world_pose": {key: float(truth_stats[key]) for key in ("x", "y", "yaw")},
                "map_pose": {key: float(map_stats[key]) for key in ("x", "y", "yaw")},
                "map_pose_statistics": map_stats,
                "world_pose_statistics": truth_stats,
                "global_match_events_since_previous": len(leg_events),
            })
        previous_boundary = finished

    report: dict[str, Any] = {
        "format": "world_anchored_global_geometry_acceptance_v1",
        "expected_checkpoints": expected_names,
        "checkpoint_audits": checkpoints,
        "correspondences": correspondences,
        "truth_usage": "offline report only; never published to ROS or consumed by control",
        "thresholds": {
            "position_rms_m": max_position_rms,
            "position_max_m": max_position,
            "yaw_rms_rad": max_yaw_rms,
            "yaw_max_rad": max_yaw,
            "minimum_spatial_coverage_m": min_coverage,
        },
    }
    if len(correspondences) >= 2:
        fitted = fit_world_to_map_se2(correspondences)
        fitted_residuals = transform_residuals(correspondences, fitted)
        initial_residuals = transform_residuals(correspondences, initial_calibration)
        position_errors = [float(item["position_error_m"]) for item in fitted_residuals]
        yaw_errors = [abs(float(item["yaw_error_rad"])) for item in fitted_residuals]
        points = [(float(item["world_pose"]["x"]), float(item["world_pose"]["y"]))
                  for item in correspondences]
        pair_distances = [math.hypot(a[0] - b[0], a[1] - b[1])
                          for index, a in enumerate(points) for b in points[index + 1:]]
        report.update({
            "fitted_world_to_map": fitted,
            "fitted_vs_initial_calibration": calibration_delta(initial_calibration, fitted),
            "fitted_residuals": fitted_residuals,
            "initial_calibration_residuals": initial_residuals,
            "position_rms_m": math.sqrt(statistics.fmean(
                error * error for error in position_errors
            )),
            "position_max_m": max(position_errors),
            "yaw_rms_rad": math.sqrt(statistics.fmean(
                error * error for error in yaw_errors
            )),
            "yaw_max_rad": max(yaw_errors),
            "spatial_coverage_max_pair_distance_m": max(pair_distances, default=0.0),
            "spatial_coverage_convex_hull_area_m2": convex_hull_area(points),
        })
    checks = {
        "all_checkpoints_sampled": (
            [item["name"] for item in checkpoints] == expected_names and
            len(correspondences) == len(expected_names) + 1
        ),
        "checkpoint_map_pose_stable": bool(checkpoints) and all(
            item["map_pose_statistics"] is not None and
            int(item["map_sample_count"]) >= 20 and
            float(item["map_pose_statistics"]["xy_rms_m"]) <= 0.03 and
            float(item["map_pose_statistics"]["yaw_rms_rad"]) <= math.radians(2.0)
            for item in checkpoints
        ),
        "checkpoint_truth_pose_stable": bool(checkpoints) and all(
            item["world_pose_statistics"] is not None and
            int(item["truth_sample_count"]) >= 20 and
            float(item["world_pose_statistics"]["xy_rms_m"]) <= 0.01 and
            float(item["world_pose_statistics"]["yaw_rms_rad"]) <= math.radians(0.5)
            for item in checkpoints
        ),
        "database_match_each_leg": bool(checkpoints) and all(
            int(item["global_match_events_since_previous"]) >= 1 for item in checkpoints
        ),
        "checkpoint_odom_not_lost": bool(checkpoints) and all(
            float(item["odom_lost_ratio"]) == 0.0 for item in checkpoints
        ),
        "spatial_coverage": float(
            report.get("spatial_coverage_max_pair_distance_m", 0.0)
        ) >= min_coverage,
        "position_rms": float(report.get("position_rms_m", math.inf)) <= max_position_rms,
        "position_max": float(report.get("position_max_m", math.inf)) <= max_position,
        "yaw_rms": float(report.get("yaw_rms_rad", math.inf)) <= max_yaw_rms,
        "yaw_max": float(report.get("yaw_max_rad", math.inf)) <= max_yaw,
    }
    report["checks"] = checks
    report["non_rigid_geometry_detected"] = not (
        checks["position_rms"] and checks["position_max"] and
        checks["yaw_rms"] and checks["yaw_max"]
    )
    report["passed"] = all(checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default=ROBOT_NAME)
    parser.add_argument(
        "--reset-scene", action=argparse.BooleanOptionalAction, default=None,
        help="pause, load the OrcaLab initial frame, then resume before startup; "
             "enabled by default for global_validation",
    )
    parser.add_argument("--database", type=Path,
                        default=DEFAULT_MAPPING_DIR / "rtabmap.db")
    parser.add_argument("--mapping-manifest", type=Path,
                        default=DEFAULT_MAPPING_DIR / "mapping_manifest.json")
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--calibration", type=Path,
                        help="existing world->map SE(2) calibration used for the initial seed")
    parser.add_argument("--calibration-output", type=Path,
                        default=DEFAULT_MAPPING_DIR / "world_to_map_calibration.json")
    parser.add_argument("--initial-world-pose", nargs=3, type=float,
                        metavar=("X", "Y", "YAW"))
    parser.add_argument("--rgbd-fps", type=float, default=30.0)
    # Rendering above the RGB-D capture rate only adds simulator RPC contention
    # with the camera bridge, which is what throttles localization warm-up.
    parser.add_argument("--camera-render-hz", type=float, default=30.0)
    parser.add_argument("--detection-rate", type=float, default=2.0)
    parser.add_argument("--localization-timeout", type=float, default=90.0)
    parser.add_argument("--static-window", type=float, default=5.0)
    parser.add_argument("--min-global-matches", type=int, default=1)
    parser.add_argument("--min-odom-hz", type=float, default=8.0)
    parser.add_argument("--max-lost-ratio", type=float, default=0.10)
    parser.add_argument("--max-static-xy-rms", type=float, default=0.03)
    parser.add_argument("--max-static-yaw-rms-deg", type=float, default=2.0)
    parser.add_argument("--max-terminal-position-error", type=float, default=0.10)
    parser.add_argument("--max-terminal-yaw-error-deg", type=float, default=5.0)
    parser.add_argument("--max-pose-jump", type=float, default=0.20)
    parser.add_argument("--max-yaw-jump-deg", type=float, default=10.0)
    parser.add_argument("--max-calibration-change", type=float, default=0.25)
    parser.add_argument("--max-calibration-yaw-change-deg", type=float, default=5.0)
    parser.add_argument("--input", choices=("none", "desktop", "terminal", "orcalab",
                                              "hybrid", "probe", "straight_probe",
                                              "u_probe", "rectangle_probe",
                                              "global_validation"), default="none")
    parser.add_argument("--run-seconds", type=float, default=10.0,
                        help="stationary observation after calibration when --input=none")
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--turn-speed", type=float, default=0.10)
    parser.add_argument("--linear-ramp-rate", type=float, default=0.8)
    parser.add_argument("--steering-ramp-rate", type=float, default=1.0)
    parser.add_argument("--probe-distance", type=float, default=0.5)
    parser.add_argument("--probe-timeout", type=float, default=45.0)
    parser.add_argument("--probe-startup-delay", type=float, default=2.0)
    parser.add_argument("--loop-segment-length", type=float, default=0.5)
    parser.add_argument("--post-motion-settle", type=float, default=6.0)
    parser.add_argument(
        "--validation-world-points-json",
        # Task coordinates are world-frame contract values. They must remain
        # unchanged when the robot start pose is varied; the validation safety
        # map applies the one-shot world->map calibration internally.
        default=world_points_xy_json(),
    )
    parser.add_argument(
        "--validation-map-yaml", type=Path,
        default=DEFAULT_MAPPING_DIR / "map" / "task1_rtabmap.yaml",
    )
    parser.add_argument(
        "--validation-point-clearance",
        type=float,
        default=VALIDATION_POINT_CLEARANCE_M,
        help="scene-adapted centre clearance used to resolve safe waypoint stops (m)",
    )
    parser.add_argument(
        "--validation-max-point-shift",
        type=float,
        default=VALIDATION_MAX_POINT_SHIFT_M,
        help="maximum audited waypoint micro-adjustment from the world contract (m)",
    )
    parser.add_argument("--validation-speed", type=float, default=VALIDATION_SPEED_M)
    parser.add_argument(
        "--validation-max-angular", type=float, default=VALIDATION_MAX_ANGULAR_RADPS
    )
    parser.add_argument(
        "--validation-linear-accel-limit", type=float,
        default=LINEAR_ACCEL_LIMIT,
        help="shared MPPI, velocity-smoother and chassis linear acceleration limit",
    )
    parser.add_argument(
        "--validation-linear-jerk-limit", type=float,
        default=LINEAR_JERK_LIMIT,
        help="chassis output linear jerk limit and acceptance threshold",
    )
    parser.add_argument(
        "--validation-contact-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="audit MuJoCo contacts and latch a hard stop on real environment contact",
    )
    parser.add_argument(
        "--validation-waypoint-coordinate-tolerance", type=float,
        default=VALIDATION_COORDINATE_TOLERANCE_M,
        help="robot-center passage radius for A and B",
    )
    parser.add_argument(
        "--validation-final-coordinate-tolerance", type=float,
        default=VALIDATION_COORDINATE_TOLERANCE_M,
        help="robot-center arrival radius for C",
    )
    parser.add_argument(
        "--validation-adaptive-route-profile",
        action="store_true",
        help="adaptive scheme: Nav2 NavigateToPose with BT replanning; default is FollowPath",
    )
    parser.add_argument(
        "--validation-theta-star-planner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use Nav2 Theta* global planner with hybrid profiled/MPPI execution",
    )
    parser.add_argument(
        "--validation-astar-planner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use Nav2 SmacPlanner2D (grid A*); default is SmacPlannerHybrid for B→C",
    )
    parser.add_argument(
        "--validation-augment-scene-geometry",
        action=argparse.BooleanOptionalAction,
        default=VALIDATION_AUGMENT_SCENE_GEOMETRY,
        help="union audited scene collision AABBs into the Nav2 static grid",
    )
    parser.add_argument(
        "--validation-turn-speed", type=float, default=0.55,
        help="steering response scale used by the chassis bridge during speed tests",
    )
    parser.add_argument("--validation-path-tolerance", type=float, default=0.25,
                        help="maximum lateral deviation from the planned validation path (m)")
    parser.add_argument(
        "--validation-profile-lateral-accel", type=float,
        default=VALIDATION_PROFILE_LATERAL_ACCEL_MPS2,
    )
    parser.add_argument(
        "--validation-min-turning-radius", type=float,
        default=VALIDATION_MIN_TURNING_RADIUS_M,
        help="Hybrid-A* / MPPI minimum turning radius from 2WS chassis kinematics",
    )
    parser.add_argument(
        "--validation-score-region-radius", type=float,
        default=OFFICIAL_REGION_RADIUS_M,
        help="official XY region radius used to place the C navigation stop",
    )
    parser.add_argument(
        "--validation-score-circle-stop-offset", type=float,
        default=VALIDATION_SCORE_CIRCLE_STOP_OFFSET_M,
        help="planned C stop distance short of the scored site",
    )
    parser.add_argument(
        "--validation-profile-clearance-hard", type=float,
        default=VALIDATION_PROFILE_CLEARANCE_HARD_M,
    )
    parser.add_argument(
        "--validation-profile-clearance-soft", type=float,
        default=VALIDATION_PROFILE_CLEARANCE_SOFT_M,
    )
    parser.add_argument("--validation-profile-lookahead", type=float,
                        default=VALIDATION_PROFILE_LOOKAHEAD_M)
    parser.add_argument("--validation-minimum-profile-speed", type=float, default=0.18)
    parser.add_argument("--validation-final-stop-speed", type=float, default=0.08)
    parser.add_argument(
        "--validation-terminal-arrival-speed", type=float,
        default=VALIDATION_TERMINAL_ARRIVAL_SPEED_MPS,
    )
    parser.add_argument(
        "--validation-terminal-zone-m", type=float,
        default=VALIDATION_TERMINAL_ZONE_M,
        help="arc length over which terminal arrival speed ramps in (m)",
    )
    parser.add_argument(
        "--validation-checkpoint-capture-min-speed", type=float,
        default=VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS,
        help="minimum linear speed while capturing a scored checkpoint",
    )
    parser.add_argument(
        "--validation-peak-contract-speed", type=float, default=None,
        help="optional straight-line peak above --validation-speed",
    )
    parser.add_argument(
        "--validation-straight-curvature-1pm", type=float,
        default=VALIDATION_STRAIGHT_CURVATURE_1PM,
    )
    parser.add_argument("--validation-visual-pose-timeout", type=float, default=2.0,
                        help="maximum visual TF age allowed by the validation follower (s)")
    parser.add_argument("--validation-checkpoint-settle", type=float, default=0.0)
    parser.add_argument(
        "--validation-ab-checkpoint-dwell",
        type=float,
        default=VALIDATION_AB_CHECKPOINT_DWELL_S,
        help="zero-command hold at scored A/B fly-through points (seconds)",
    )
    parser.add_argument(
        "--validation-post-robot-start-dwell",
        type=float,
        default=VALIDATION_POST_ROBOT_START_DWELL_S,
        help="zero-command hold after robot_started.marker before route motion (seconds)",
    )
    parser.add_argument("--validation-sample-window", type=float, default=0.8)
    parser.add_argument("--validation-truth-audit-hz", type=float, default=10.0)
    parser.add_argument(
        "--validation-nav2-timeout", type=float,
        default=VALIDATION_NAV2_STARTUP_TIMEOUT_S,
        help="seconds to wait for Nav2 lifecycle stack (Hybrid first configure can be slow)",
    )
    parser.add_argument("--validation-route-timeout", type=float,
                        default=VALIDATION_MISSION_TIMEOUT_S)
    parser.add_argument(
        "--validation-route-stop",
        default="C",
        choices=("A", "B", "C"),
        help="stop validation after this scored checkpoint (default: full A→B→C)",
    )
    parser.add_argument("--validation-max-position-rms", type=float, default=0.05)
    parser.add_argument("--validation-max-position", type=float, default=0.10)
    parser.add_argument("--validation-max-yaw-rms-deg", type=float, default=2.0)
    parser.add_argument("--validation-max-yaw-deg", type=float, default=5.0)
    parser.add_argument("--validation-min-coverage", type=float, default=8.0)
    parser.add_argument(
        "--official-scorer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start official ScorerClient task1_inspection after localization "
             "(internal radius matches the official 0.60 m circle)",
    )
    parser.add_argument(
        "--official-score-video",
        action="store_true",
        help="enable official scorer screen capture (off by default to spare GPU)",
    )
    parser.add_argument(
        "--official-score-central",
        action="store_true",
        help="upload official scorer results to the central server "
             "(default: local-only on 127.0.0.1:9000)",
    )
    parser.add_argument(
        "--official-score-team-id",
        default=os.environ.get("ORCA_SCORING_TEAM_ID"),
        help="ScorerClient team id (default: ORCA_SCORING_TEAM_ID env or orca_scoring.yaml)",
    )
    parser.add_argument(
        "--official-score-team-token",
        default=os.environ.get("ORCA_SCORING_TEAM_TOKEN"),
        help="ScorerClient team token (default: ORCA_SCORING_TEAM_TOKEN env or orca_scoring.yaml)",
    )
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--rtabmapviz", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.reset_scene is None:
        args.reset_scene = args.input == "global_validation"

    if not args.database.is_file():
        parser.error(f"mapping database does not exist: {args.database}")
    if args.rgbd_fps <= 0.0 or args.camera_render_hz <= 0.0:
        parser.error("RGB-D and render rates must be positive")
    if not 0.0 < args.speed <= VALIDATION_MAX_SPEED_M:
        parser.error(f"--speed must be in (0, {VALIDATION_MAX_SPEED_M}]")
    if not 0.0 < args.validation_speed <= VALIDATION_MAX_SPEED_M:
        parser.error(f"--validation-speed must be in (0, {VALIDATION_MAX_SPEED_M}]")
    if (args.validation_linear_accel_limit <= 0.0 or
            args.validation_linear_jerk_limit <= 0.0 or
            args.validation_waypoint_coordinate_tolerance <= 0.0 or
            args.validation_final_coordinate_tolerance <= 0.0):
        parser.error("validation accel, jerk and coordinate tolerances must be positive")
    if (args.validation_profile_lateral_accel <= 0.0
            or args.validation_minimum_profile_speed <= 0.0
            or args.validation_profile_lookahead < 0.0
            or args.validation_final_stop_speed < 0.0
            or args.validation_terminal_arrival_speed < 0.0
            or args.validation_terminal_zone_m < 0.0
            or args.validation_checkpoint_capture_min_speed < 0.0
            or args.validation_straight_curvature_1pm < 0.0
            or args.validation_min_turning_radius <= 0.0
            or args.validation_score_region_radius <= 0.0
            or args.validation_score_circle_stop_offset < 0.0
            or args.validation_score_circle_stop_offset
            >= args.validation_score_region_radius
            or (
                args.validation_peak_contract_speed is not None
                and args.validation_peak_contract_speed + 1e-9
                < args.validation_speed
            )
            or not 0.0 <= args.validation_profile_clearance_hard
            < args.validation_profile_clearance_soft):
        parser.error("validation path speed profile configuration is invalid")
    if not args.validation_map_yaml.is_file():
        sibling = args.database.expanduser().resolve().parent / "map" / "task1_rtabmap.yaml"
        if sibling.is_file():
            args.validation_map_yaml = sibling
    if (not args.validation_map_yaml.is_file() or
            args.validation_point_clearance <= 0.0 or args.validation_max_point_shift < 0.0):
        parser.error(
            "validation map/point-clearance configuration is invalid: "
            f"map yaml {args.validation_map_yaml} "
            f"(exists={args.validation_map_yaml.is_file()})"
        )
    if (args.validation_checkpoint_settle < 0.0
            or args.validation_ab_checkpoint_dwell < 0.0
            or args.validation_post_robot_start_dwell < 0.0
            or args.validation_sample_window <= 0.0
            or args.validation_truth_audit_hz <= 0.0):
        parser.error("validation settle, sample window, or audit rate is invalid")

    mapping_manifest_data: dict[str, Any] = {}
    if args.mapping_manifest.is_file():
        mapping_manifest_data = json.loads(args.mapping_manifest.read_text(encoding="utf-8"))
        manifest_database = mapping_manifest_data.get("database")
        if manifest_database:
            manifest_database_path = Path(str(manifest_database))
            if not manifest_database_path.is_absolute():
                manifest_database_path = args.mapping_manifest.parent / manifest_database_path
        if manifest_database and manifest_database_path.resolve() != args.database.resolve():
            parser.error(
                "--database and --mapping-manifest refer to different mapping assets: "
                f"{args.database} vs {manifest_database}"
            )
    grid_parameters = mapping_manifest_data.get("occupancy_grid_parameters", {})
    grid_range_max = float(grid_parameters.get("range_max_m", 3.0))
    grid_max_ground_height = float(grid_parameters.get("max_ground_height_m", 0.05))
    grid_max_obstacle_height = float(grid_parameters.get("max_obstacle_height_m", 0.40))
    if grid_range_max <= 0.0 or grid_max_obstacle_height <= 0.0:
        parser.error("mapping manifest contains invalid occupancy-grid parameters")

    if args.validation_peak_contract_speed is None:
        args.validation_peak_contract_speed = float(args.validation_speed)
    if args.input == "global_validation":
        args.speed = float(args.validation_speed)
    if args.validation_speed > 2.5:
        args.validation_linear_accel_limit = min(
            VALIDATION_PROFILE_LINEAR_ACCEL_MPS2,
            max(
                float(args.validation_linear_accel_limit),
                scaled_profile_linear_accel(
                    float(args.validation_linear_accel_limit),
                    float(args.validation_speed),
                ),
            ),
        )
    require_port(50051)
    require_port(50151)
    reset_pose_record = None
    if args.reset_scene:
        from orcalab_scene_reset import restart_current_scene
        restart_current_scene("127.0.0.1:50051", timeout=10.0, robot_name=args.robot_name)
        from run_world_anchored_rtabmap_mapping import configure_head_rgbd_camera, read_world_pose
        configure_head_rgbd_camera(args.robot_name)
        reset_pose_record = read_world_pose(args.robot_name)
    from run_world_anchored_rtabmap_mapping import read_task_sites_and_robot_root
    task_site_contract = read_task_sites_and_robot_root(args.robot_name)
    site_points = task_site_contract["world_points_xyz"]
    args.validation_world_points_json = json.dumps(
        validation_route_world_points(
            {
                name: [float(values[0]), float(values[1])]
                for name, values in site_points.items()
            },
            stop_at=args.validation_route_stop,
        ),
        separators=(",", ":"),
    )
    args.validation_judgement_offset_xy = [
        float(value) for value in
        task_site_contract["judgement_offset_in_navigation_base_xyz"][:2]
    ]
    # Scene contract documents holder==body_link1 planar offset; do not invent
    # a forward offset that misaligns internal scoring with the official scorer.
    env = ros_environment()
    from run_world_anchored_rtabmap_mapping import clear_leftover_mapping_pipelines
    clear_leftover_mapping_pipelines(env)
    assert_topics_unclaimed(env)

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    session = (args.session_dir or ROOT / "data" /
               f"world_anchored_localization_{stamp}").resolve()
    session.mkdir(parents=True, exist_ok=False)
    atomic_json(session / "task_site_contract.json", task_site_contract)
    if reset_pose_record is not None:
        atomic_json(session / "scene_reset.json", {
            "format": "world_anchored_scene_reset_v1",
            "operation": ["PAUSED", "LoadInitialFrame", "RUNNING"],
            "verified_world_pose_after_reset": reset_pose_record,
            "verified_before_rgbd_rtabmap_start": True,
        })
    session_database = session / "localization.db"
    subprocess.run(["cp", "--reflink=auto", "--preserve=timestamps",
                    str(args.database.resolve()), str(session_database)], check=True)

    from run_world_anchored_rtabmap_mapping import read_camera_extrinsic, read_world_pose

    initial_world_record = read_world_pose(args.robot_name)
    if args.initial_world_pose:
        initial_world_record["position_xyz"][:2] = [args.initial_world_pose[0],
                                                     args.initial_world_pose[1]]
        initial_world_record["yaw_rad"] = args.initial_world_pose[2]
        initial_world_record["source"] = "command-line one-shot override"
    initial_world = pose_from_world_record(initial_world_record)
    prior_calibration_path = args.calibration
    if prior_calibration_path is None and args.calibration_output.is_file():
        prior_calibration_path = args.calibration_output
    prior_calibration = None
    if prior_calibration_path is not None:
        prior_calibration = json.loads(prior_calibration_path.read_text(encoding="utf-8"))
    mapping_seed_calibration = None
    mapping_initial_world = mapping_manifest_data.get("initial_world_pose")
    mapping_initial_map_text = mapping_manifest_data.get("rgbd_odometry_initial_pose")
    if mapping_initial_world and mapping_initial_map_text:
        values = [float(value) for value in str(mapping_initial_map_text).split()]
        if len(values) != 6:
            raise RuntimeError(
                "mapping rgbd_odometry_initial_pose must contain x y z roll pitch yaw"
            )
        mapping_seed_calibration = derive_world_to_map(
            pose_from_world_record(mapping_initial_world),
            {"x": values[0], "y": values[1], "yaw": values[5]},
        )
        mapping_seed_calibration["method"] = (
            "mapping initial world pose to recorded RGB-D map seed"
        )
    if args.input == "global_validation" and mapping_seed_calibration is not None:
        # Transform the changed scene start through the correspondence captured
        # when this exact scan database was built. A live database match below
        # then estimates the residual before the costmap is published.
        initial_map = world_to_map_pose(initial_world, mapping_seed_calibration)
    elif prior_calibration is not None:
        initial_map = world_to_map_pose(initial_world, prior_calibration)
    else:
        initial_map = dict(initial_world)

    extrinsic = read_camera_extrinsic(args.robot_name)
    initial_pose_string = (
        f"{initial_map['x']:.12g} {initial_map['y']:.12g} 0 0 0 "
        f"{initial_map['yaw']:.12g}"
    )
    manifest: dict[str, Any] = {
        "format": "world_anchored_rtabmap_localization_v1",
        "status": "starting",
        "source_database": str(args.database.resolve()),
        "session_database": str(session_database),
        "source_database_size": args.database.stat().st_size,
        "source_database_mtime_ns": args.database.stat().st_mtime_ns,
        "source_database_isolation": "session copy",
        "initial_world_pose": initial_world_record,
        "task_site_contract": task_site_contract,
        "initial_map_seed": initial_map,
        "rgbd_odometry_initial_pose": initial_pose_string,
        "prior_calibration": str(prior_calibration_path.resolve())
        if prior_calibration_path else None,
        "prior_calibration_used_for_initial_seed": bool(
            prior_calibration is not None
            and not (
                args.input == "global_validation"
                and mapping_seed_calibration is not None
            )
        ),
        "mapping_seed_calibration": mapping_seed_calibration,
        "odom_source": "RTAB-Map rgbd_odometry",
        "wheel_odom_used": False,
        "continuous_truth_odom_used": False,
        "amcl_used": False,
        "static_map_to_odom_used": False,
        "truth_usage": "one-shot seed and offline acceptance samples only",
        "optimizer_strategy": 1,
        "occupancy_grid_parameters": {
            "range_max_m": grid_range_max,
            "max_ground_height_m": grid_max_ground_height,
            "max_obstacle_height_m": grid_max_obstacle_height,
        },
    }
    atomic_json(session / "localization_manifest.json", manifest)

    processes: dict[str, subprocess.Popen] = {}
    process_cleanup: dict[str, Any] = {
        "attempted": False,
        "survivors": {},
        "passed": False,
    }
    calibration: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    calibration_time = 0.0
    motion_time: float | None = None
    final_world_record: dict[str, Any] | None = None
    final_map: dict[str, float] | None = None
    global_route_manifest: dict[str, Any] | None = None
    official_scorer: OfficialTaskScorer | None = None
    official_score: dict[str, Any] | None = None
    runtime_error: str | None = None
    status_path = session / "monitor_status.json"
    report_path = session / "monitor_report.json"
    try:
        if args.input == "global_validation":
            # Must precede RGB-D/RTAB-Map startup because OrcaGym environment
            # construction performs a simulator initialization/reset.
            start_validation_bridge(args, env, session, processes)
        start_process("rgbd", [SYSTEM_PYTHON, str(SRC / "three_camera_rgbd_bridge.py"),
            "--head-only", "--vertical-fov", str(extrinsic["vertical_fov_deg"]),
            "--fps", str(args.rgbd_fps),
            "--start-index", str(6_000_000 + int(time.time()) % 1_000_000),
            "--output-dir", f"/tmp/world_anchored_localization_rgbd_{stamp}",
            "--report", str(session / "rgbd_report.json")], env, session, processes)
        t = extrinsic["translation"]
        q = extrinsic["quaternion_xyzw"]
        start_process("camera_tf", [ROS2, "run", "tf2_ros", "static_transform_publisher",
            "--x", str(t[0]), "--y", str(t[1]), "--z", str(t[2]),
            "--qx", str(q[0]), "--qy", str(q[1]), "--qz", str(q[2]), "--qw", str(q[3]),
            "--frame-id", "base_link", "--child-frame-id", extrinsic["camera_frame"]],
            env, session, processes)
        if not topic_ready("/camera/color/image_raw", "sensor_msgs/msg/Image", env, 30.0):
            raise RuntimeError("RGB-D color topic did not become ready")
        if not topic_ready("/camera/depth/image_raw", "sensor_msgs/msg/Image", env, 30.0):
            raise RuntimeError("RGB-D depth topic did not become ready")
        start_process("rgbd_sync", [ROS2, "run", "rtabmap_sync", "rgbd_sync", "--ros-args",
            "-p", "approx_sync:=false", "-p", "sync_queue_size:=30", "-p", "qos:=1",
            "-r", "rgb/image:=/camera/color/image_raw",
            "-r", "rgb/camera_info:=/camera/color/camera_info",
            "-r", "depth/image:=/camera/depth/image_raw",
            "-r", "depth/camera_info:=/camera/depth/camera_info",
            "-r", "rgbd_image:=/rtabmap/rgbd_image"], env, session, processes)
        start_process("rgbd_odometry", [ROS2, "run", "rtabmap_odom", "rgbd_odometry",
            "--Odom/ResetCountdown", "3", "--Vis/MinInliers", "20",
            "--Vis/PnPReprojError", "2.0", "--ros-args",
            "-p", "frame_id:=base_link", "-p", "odom_frame_id:=odom",
            "-p", "publish_tf:=true", "-p", "initial_pose:=" + initial_pose_string,
            "-p", "subscribe_rgb:=true", "-p", "subscribe_depth:=true",
            "-p", "subscribe_rgbd:=false", "-p", "approx_sync:=false",
            "-p", "sync_queue_size:=30", "-p", "topic_queue_size:=30",
            "-p", "qos:=1", "-p", "wait_for_transform:=0.2",
            "-r", "rgb/image:=/camera/color/image_raw",
            "-r", "rgb/camera_info:=/camera/color/camera_info",
            "-r", "depth/image:=/camera/depth/image_raw",
            "-r", "odom:=/odom", "-r", "odom_info:=/odom_info"],
            env, session, processes)
        # Start the monitor before RTAB-Map so the first localization event is
        # not lost while the database is being opened.
        monitor_window = (
            VALIDATION_STATIC_GATE_WINDOW_S
            if args.input == "global_validation"
            else args.static_window
        )
        start_process("monitor", [SYSTEM_PYTHON, str(Path(__file__).resolve()),
            "--monitor-worker", "--status", str(status_path), "--report", str(report_path),
            "--window", str(monitor_window)], env, session, processes)
        start_process("rtabmap", [ROS2, "run", "rtabmap_slam", "rtabmap",
            "--Mem/IncrementalMemory", "false", "--Mem/InitWMWithAllNodes", "true",
            "--Mem/LocalizationDataSaved", "false", "--Reg/Force3DoF", "true",
            "--Grid/FromDepth", "true", "--Grid/3D", "false",
            "--Grid/RangeMax", str(grid_range_max),
            "--Grid/MaxGroundHeight", str(grid_max_ground_height),
            "--Grid/MaxObstacleHeight", str(grid_max_obstacle_height),
            "--Optimizer/Strategy", "1", "--Optimizer/PriorsIgnored", "false",
            # Use tightly local scan-database registration after the initial
            # world seed so visual-odometry drift cannot move the robot away
            # from the otherwise-correct scanned occupancy map.
            "--RGBD/ProximityBySpace", "true",
            # The small search radius and odometry guess exclude distant,
            # visually repetitive nodes while retaining local corrections.
            "--RGBD/LocalRadius", "0.50",
            "--RGBD/ProximityMaxGraphDepth", "1",
            "--RGBD/ProximityOdomGuess", "true",
            "--RGBD/MaxLoopClosureDistance", "0.35",
            "--RGBD/StartAtOrigin", "true",
            # Do not fall back to unconstrained global image matching when
            # proximity links are unavailable. The world-seeded pose is
            # trusted at startup; this fallback caused a static 0.45 m jump
            # to a visually similar cabinet node.
            "--RGBD/LocalizationSecondTryWithoutProximityLinks", "false",
            # A stationary startup cannot provide the second displaced frame
            # requested by an odom cache.  Disable that cache for the initial
            # global match, then enforce independent TF/pose stability gates.
            # A stationary validation preflight must not wait for a second
            # displaced odom sample. With a nonzero cache RTAB-Map can reset
            # odom during the zero-motion preflight and snap map->base_link
            # back to the origin.
            "--RGBD/MaxOdomCacheSize", "0", "--RGBD/OptimizeMaxError", "0.20",
            "--RGBD/AggressiveLoopThr", "1.0",
            "--Rtabmap/LoopThr", "1.0",
            # Global validation is world-seeded and must not process a queued
            # appearance hypothesis during startup. Repetitive scene texture
            # can otherwise create a wrong loop before the post-gate freeze.
            "--Rtabmap/DetectionRate",
            str(0.0 if args.input == "global_validation" else args.detection_rate),
            "--ros-args",
            "-p", "frame_id:=base_link", "-p", "map_frame_id:=map",
            "-p", "initial_pose:=" + initial_pose_string,
            "-p", "publish_tf:=true", "-p", "subscribe_rgbd:=true",
            "-p", "subscribe_odom:=true", "-p", "subscribe_odom_info:=true",
            "-p", "subscribe_scan:=false", "-p", "database_path:=" + str(session_database),
            "-r", "rgbd_image:=/rtabmap/rgbd_image", "-r", "odom:=/odom",
            "-r", "odom_info:=/odom_info", "-r", "map:=/map"],
            env, session, processes)
        if not topic_ready("/odom", "nav_msgs/msg/Odometry", env, 30.0):
            raise RuntimeError("rgbd_odometry did not publish /odom")
        if not wait_for_publisher("/info", env, 30.0):
            raise RuntimeError("RTAB-Map did not advertise /info")
        if args.input == "global_validation":
            manifest["rviz_config"] = str(
                (ROOT / "task1_world_anchored_validation.rviz").resolve()
            )
        elif args.rviz or args.visualize:
            start_process("rviz", ["rviz2", "-d", str(ROOT / "task1_rtabmap_mapping.rviz")],
                          env, session, processes)
        if args.rtabmapviz or args.visualize:
            start_process("rtabmapviz", [ROS2, "run", "rtabmap_viz", "rtabmap_viz",
                "--ros-args", "-p", "frame_id:=base_link", "-p", "odom_frame_id:=odom",
                "-p", "map_frame_id:=map", "-p", "subscribe_rgbd:=true",
                "-p", "subscribe_odom_info:=true", "-p", "subscribe_scan:=false",
                "-r", "rgbd_image:=/rtabmap/rgbd_image", "-r", "odom:=/odom",
                "-r", "odom_info:=/odom_info"], env, session, processes)

        manifest["status"] = "waiting_for_global_localization"
        atomic_json(session / "localization_manifest.json", manifest)
        gate = wait_for_initial_gate(
            status_path, session / "rtabmap.log", args.localization_timeout,
            args.min_global_matches, args.min_odom_hz, args.max_lost_ratio,
            args.max_static_xy_rms, math.radians(args.max_static_yaw_rms_deg),
            initial_map, args.max_calibration_change,
            math.radians(args.max_calibration_yaw_change_deg),
            require_global_match=(args.input != "global_validation"),
            min_static_samples=(
                VALIDATION_STATIC_GATE_MIN_SAMPLES
                if args.input == "global_validation"
                else 20
            ),
        )
        atomic_json(session / "initial_localization_gate.json", gate)
        if not gate["passed"]:
            raise RuntimeError("initial global localization gate did not pass")

        # Startup processing is frozen above to avoid a queued appearance
        # hypothesis before calibration. Resume database matching after the
        # gate: global appearance fallback remains disabled, so only the local
        # proximity window can update map->odom from the scanned data.
        resume_detection = set_ros_param(
            "/rtabmap", "Rtabmap/DetectionRate", f"'{args.detection_rate}'", env,
        )
        manifest["post_initial_detection_policy"] = {
            "detection_rate": args.detection_rate,
            "reason": "local scan-database registration with global fallback disabled",
            "param_command_returncode": resume_detection.returncode,
            "param_command_output": resume_detection.stdout[-500:],
        }
        atomic_json(session / "localization_manifest.json", manifest)

        calibration_world_record = read_world_pose(args.robot_name)
        calibration_world = pose_from_world_record(calibration_world_record)
        map_pose = recent_pose(gate["status"])
        if map_pose is None:
            raise RuntimeError("monitor did not provide a stable map pose")
        localization_pose = dict(map_pose)
        calibration = derive_world_to_map(calibration_world, map_pose)
        calibration.update({
            "created_at": datetime.now().astimezone().isoformat(),
            "database": str(args.database.resolve()),
            "mapping_manifest": str(args.mapping_manifest.resolve()),
            "world_pose_sample": calibration_world_record,
            "map_pose_sample": map_pose,
            "localization_gate": {
                "global_match_count": gate["global_match_count"],
                "recent_pose_statistics": gate["status"]["recent_map_base_pose"],
            },
            "truth_feedback_used": args.input == "global_validation",
        })
        if prior_calibration is not None:
            delta = calibration_delta(prior_calibration, calibration)
            calibration["prior_calibration_delta"] = delta
            scan_matched = int(gate["global_match_count"]) > 0
            calibration["prior_calibration_policy"] = (
                "scan_match_authoritative"
                if scan_matched else "prior_consistency_required"
            )
            if (not scan_matched and
                    (delta["translation_m"] > args.max_calibration_change or
                     delta["yaw_rad"] >
                     math.radians(args.max_calibration_yaw_change_deg))):
                raise RuntimeError(
                    "new calibration disagrees with prior calibration: " + json.dumps(delta)
                )
        atomic_json(session / "world_to_map_calibration.json", calibration)
        # A global validation run must not replace the accepted calibration
        # with its self-fitted startup sample. Only the independently fitted
        # multi-point result is promoted after every geometry gate passes.
        if args.input != "global_validation":
            atomic_json(args.calibration_output.resolve(), calibration)
        calibration_time = time.time()
        manifest["status"] = "localized_and_calibrated"
        manifest["calibration"] = str(args.calibration_output.resolve())
        manifest["calibration_time"] = calibration_time
        atomic_json(session / "localization_manifest.json", manifest)

        if args.input == "none":
            time.sleep(args.run_seconds)
        elif args.input == "global_validation":
            if args.official_scorer:
                official_scorer = OfficialTaskScorer(
                    session,
                    record_video=args.official_score_video,
                    local_only=not args.official_score_central,
                    team_id=args.official_score_team_id,
                    team_token=args.official_score_team_token,
                )
                if not official_scorer.connect():
                    print(
                        "[official-scorer] warning: SDK connect failed; "
                        "validation will continue without official scoring",
                        flush=True,
                    )
                    official_scorer = None
            motion_time = time.time()
            global_route_manifest = run_global_validation_route(
                args, env, session, processes, initial_world, initial_map,
                calibration, official_scorer,
                localization_pose=localization_pose,
            )
            terminate(processes.get("validation_mission"))
            terminate(processes.get("validation_nav2"))
            time.sleep(args.post_motion_settle)
        else:
            motion_time = time.time()
            start_process("driver", [ORCALAB_PYTHON,
                str(SRC / "run_world_anchored_rtabmap_mapping.py"), "--motion-worker",
                "--robot-name", args.robot_name, "--input", args.input,
                "--speed", str(args.speed), "--turn-speed", str(args.turn_speed),
                "--camera-render-hz", str(args.camera_render_hz),
                "--linear-ramp-rate", str(args.linear_ramp_rate),
                "--steering-ramp-rate", str(args.steering_ramp_rate),
                "--probe-distance", str(args.probe_distance),
                "--probe-timeout", str(args.probe_timeout),
                "--probe-startup-delay", str(args.probe_startup_delay),
                "--loop-segment-length", str(args.loop_segment_length),
                "--metadata", str(session / "driver_metadata.json")],
                env, session, processes)
            processes["driver"].wait()
            if processes["driver"].returncode != 0:
                raise RuntimeError("motion worker failed; see driver.log")
            time.sleep(args.post_motion_settle)

        final_world_record = read_world_pose(args.robot_name)
        final_status = json.loads(status_path.read_text(encoding="utf-8"))
        final_map = recent_pose(final_status)
        if final_map is None:
            raise RuntimeError("no stable final map pose available")
        manifest["status"] = "runtime_complete"
    except KeyboardInterrupt:
        runtime_error = "KeyboardInterrupt"
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        if official_scorer is not None:
            official_score = official_scorer.finish()
        process_cleanup["attempted"] = True
        for name in ("driver", "validation_mission",
                     "validation_nav2",
                     "validation_bag",
                     "validation_safety_map",
                     "validation_bridge",
                     "truth_tf"):
            survivors = terminate(processes.get(name))
            if survivors:
                process_cleanup["survivors"][name] = survivors
        for name in ("rtabmapviz", "rviz", "monitor", "rtabmap",
                     "rgbd_odometry", "rgbd_sync", "camera_tf", "rgbd"):
            survivors = terminate(processes.get(name))
            if survivors:
                process_cleanup["survivors"][name] = survivors
        process_cleanup["passed"] = not process_cleanup["survivors"]

    source_unchanged = (
        args.database.stat().st_size == manifest["source_database_size"] and
        args.database.stat().st_mtime_ns == manifest["source_database_mtime_ns"]
    )
    monitor_report: dict[str, Any] = {}
    if report_path.is_file():
        monitor_report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = post_run_metrics(monitor_report, calibration_time, motion_time) \
        if calibration_time and monitor_report else {}
    rgbd_report = None
    if (session / "rgbd_report.json").is_file():
        rgbd_report = json.loads((session / "rgbd_report.json").read_text(encoding="utf-8"))
    mission_result: dict[str, Any] = {}
    bridge_report: dict[str, Any] = {}
    global_geometry: dict[str, Any] | None = None
    if args.input == "global_validation":
        mission_path = session / "validation_mission" / "result.json"
        bridge_path = session / "validation_chassis_bridge" / "session.json"
        if mission_path.is_file():
            mission_result = json.loads(mission_path.read_text(encoding="utf-8"))
        analysis_artifacts: dict[str, str] = {}
        mission_dir = session / "validation_mission"
        for artifact_name in (
            "mission_analysis.json",
            "execution_trace.csv",
            "route_geometry_audit.json",
        ):
            artifact_path = mission_dir / artifact_name
            if artifact_path.is_file():
                analysis_artifacts[artifact_name] = str(artifact_path.resolve())
        if bridge_path.is_file():
            bridge_report = json.loads(bridge_path.read_text(encoding="utf-8"))
        speed_profile = (
            (bridge_report.get("command_telemetry") or {}).get("speed_profile")
        )
        if speed_profile and mission_result:
            mission_result["speed_profile"] = speed_profile
            atomic_json(mission_path, mission_result)
        if (calibration is not None and gate is not None and monitor_report and
                mission_result and bridge_report and global_route_manifest is not None):
            global_geometry = build_global_geometry_report(
                pose_from_world_record(calibration["world_pose_sample"]),
                {key: float(calibration["map_pose_sample"][key])
                 for key in ("x", "y", "yaw")},
                gate, calibration, motion_time, monitor_report,
                mission_result, bridge_report,
                list(global_route_manifest["world_points"]),
                args.validation_sample_window,
                args.validation_max_position_rms,
                args.validation_max_position,
                math.radians(args.validation_max_yaw_rms_deg),
                math.radians(args.validation_max_yaw_deg),
                args.validation_min_coverage,
            )
            atomic_json(session / "global_geometry_acceptance.json", global_geometry)

    acceptance: dict[str, Any] = {
        "format": "world_anchored_localization_acceptance_v1",
        "session": str(session),
        "mode": "motion" if args.input != "none" else "static",
        "runtime_error": runtime_error,
        "initial_gate": gate,
        "calibration_created": calibration is not None,
        "source_database_unchanged": source_unchanged,
        "metrics": metrics,
        "rgbd_report": rgbd_report,
        "global_geometry": global_geometry,
        "official_score": official_score,
        "process_cleanup": process_cleanup,
        "validation_mission": mission_result if args.input == "global_validation" else None,
        "analysis_artifacts": (
            analysis_artifacts if args.input == "global_validation" else None
        ),
        "validation_chassis_audit": ({
            key: bridge_report.get(key) for key in (
                "contact_audit_only", "truth_pose_audit_only", "truth_pose_published",
                "api_collision_monitor_active", "contact_sample_count",
                "robot_environment_contact_count", "collision_stop_latched",
                "first_collision", "camera_render_count", "command_telemetry",
            )
        } if args.input == "global_validation" else None),
    }
    checks: dict[str, bool] = {
        "runtime_completed": runtime_error is None,
        "initial_global_localization": bool(gate and gate.get("passed")),
        "calibration_created": calibration is not None,
        "source_database_unchanged": source_unchanged,
        "post_calibration_odom_not_lost":
            float(metrics.get("post_calibration_lost_ratio", 1.0)) <= args.max_lost_ratio,
        "no_translation_jump":
            float(metrics.get("max_consecutive_translation_step_m", 999.0)) <= args.max_pose_jump,
        "no_yaw_jump":
            float(metrics.get("max_consecutive_yaw_step_rad", 999.0)) <=
            math.radians(args.max_yaw_jump_deg),
        "rgbd_no_failed_groups": bool(rgbd_report is not None and
                                      int(rgbd_report.get("failed_groups", -1)) == 0),
        "rgbd_rate": bool(rgbd_report is not None and
                          float(rgbd_report.get("group_publish_fps", 0.0)) >= args.min_odom_hz),
    }
    accuracy_calibration = (
        global_geometry.get("fitted_world_to_map")
        if global_geometry and global_geometry.get("fitted_world_to_map") else calibration
    )
    if accuracy_calibration and final_world_record and final_map:
        predicted_world = map_to_world_pose(final_map, accuracy_calibration)
        measured_world = pose_from_world_record(final_world_record)
        position_error = math.hypot(predicted_world["x"] - measured_world["x"],
                                    predicted_world["y"] - measured_world["y"])
        yaw_error = abs(wrap_angle(predicted_world["yaw"] - measured_world["yaw"]))
        terminal = {
            "calibration_source": (
                "multi-point global fit" if accuracy_calibration is not calibration
                else "startup full-pose correspondence"
            ),
            "map_pose": final_map,
            "predicted_world_pose": predicted_world,
            "measured_world_pose": measured_world,
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
        }
        acceptance["terminal_accuracy"] = terminal
        checks["terminal_position_accuracy"] = position_error <= args.max_terminal_position_error
        checks["terminal_yaw_accuracy"] = yaw_error <= math.radians(args.max_terminal_yaw_error_deg)
    else:
        checks["terminal_position_accuracy"] = False
        checks["terminal_yaw_accuracy"] = False
    if args.input != "none":
        checks["post_motion_global_match"] = int(
            metrics.get("post_motion_global_match_events", 0)) >= 1
    if args.input == "global_validation":
        checks.update({
            "global_geometry_passed": bool(global_geometry and global_geometry.get("passed")),
            "validation_mission_completed": mission_result.get("status") == "succeeded",
            "no_robot_environment_collision": (
                bool(bridge_report.get("api_collision_monitor_active")) and
                int(bridge_report.get("contact_sample_count", 0)) > 0 and
                int(bridge_report.get("robot_environment_contact_count", -1)) == 0 and
                not bool(bridge_report.get("collision_stop_latched", True))
            ),
            "linear_jerk_within_limit": profile_within_jerk_limit(
                (bridge_report.get("command_telemetry") or {}).get("speed_profile")
                or mission_result.get("speed_profile"),
                args.validation_linear_jerk_limit,
            ),
            "truth_assist_policy_satisfied": (
                bool(bridge_report.get("truth_pose_audit_only")) and
                bridge_report.get("truth_pose_published") is True and
                bool(bridge_report.get("contact_audit_only"))
            ),
        })
    acceptance["checks"] = checks
    if args.input == "global_validation":
        # Task acceptance is defined by robot-center coordinate tolerances.
        # Path deviation, terminal fit residual, global geometry residual and
        # optional post-motion loop matches remain diagnostic metrics, not
        # independent failure gates.
        required = (
            "runtime_completed", "initial_global_localization",
            "calibration_created", "source_database_unchanged",
            "post_calibration_odom_not_lost", "rgbd_no_failed_groups",
            "rgbd_rate", "validation_mission_completed",
            "no_robot_environment_collision", "linear_jerk_within_limit",
            "truth_assist_policy_satisfied",
        )
        acceptance["acceptance_rule"] = {
            "type": "g1_omnipicker_xy_tolerance_and_smooth_speed",
            "required_checks": list(required),
            "diagnostic_only_checks": [key for key in checks if key not in required],
        }
        acceptance["passed"] = all(checks[key] for key in required)
    else:
        acceptance["passed"] = all(checks.values())
    if (acceptance["passed"] and global_geometry is not None
            and "fitted_world_to_map" in global_geometry):
        promoted = dict(global_geometry["fitted_world_to_map"])
        promoted.update({
            "created_at": datetime.now().astimezone().isoformat(),
            "database": str(args.database.resolve()),
            "mapping_manifest": str(args.mapping_manifest.resolve()),
            "global_geometry_acceptance": str(
                (session / "global_geometry_acceptance.json").resolve()
            ),
            "truth_feedback_used": True,
            "truth_used_for_offline_calibration": True,
        })
        atomic_json(args.calibration_output.resolve(), promoted)
        acceptance["promoted_calibration"] = str(args.calibration_output.resolve())
    atomic_json(session / "localization_acceptance.json", acceptance)
    manifest["status"] = "succeeded" if acceptance["passed"] else "failed"
    manifest["acceptance"] = str(session / "localization_acceptance.json")
    manifest["runtime_error"] = runtime_error
    manifest["source_database_unchanged"] = source_unchanged
    atomic_json(session / "localization_manifest.json", manifest)
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    if "--monitor-worker" in sys.argv:
        index = sys.argv.index("--monitor-worker")
        raise SystemExit(monitor_worker(sys.argv[index + 1:]))
    if "--truth-tf-worker" in sys.argv:
        index = sys.argv.index("--truth-tf-worker")
        raise SystemExit(truth_tf_worker(sys.argv[index + 1:]))
    raise SystemExit(main())
