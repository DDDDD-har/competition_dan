#!/usr/bin/env python3
"""Build an RGB-D RTAB-Map database from a world-coordinate initial pose.

This is a deliberately self-contained mapping entry point.  It starts the
upstream RTAB-Map building blocks directly (``rgbd_sync``, ``rgbd_odometry``
and ``rtabmap``), uses a dedicated in-file motion worker for chassis control,
and uses the existing RGB-D bridge only to acquire camera messages.  The
worker imports only generic chassis actuator helpers; it does not import or
execute any previous RTAB-Map driver.

The g1_omnipicker root pose is published as external odometry so the SLAM graph
stays aligned with the simulator world while RGB-D supplies appearance,
geometry and loop-closure observations. Visual odometry remains diagnostic.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import glob
import hashlib
import json
import math
import os
import select
import signal
import shutil
import socket
import subprocess
import sys
import termios
import time
import tty
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
PROJECT = ROOT.parent
ORCALAB_PYTHON = "/home/dan/miniconda3/envs/orcalab/bin/python"
ORCALAB_SITE = "/home/dan/miniconda3/envs/orcalab/lib/python3.12/site-packages"
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
ROS2 = "/opt/ros/jazzy/bin/ros2"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if ORCALAB_SITE not in sys.path:
    sys.path.insert(0, ORCALAB_SITE)

from task1_scene_contract import (
    ROBOT_NAME,
    play_scene_hint,
)

MAPPING_INPUTS = (
    "desktop",
    "terminal",
    "orcalab",
    "hybrid",
    "live",
    "probe",
    "straight_probe",
    "u_probe",
    "rectangle_probe",
    "task1_wrapup",
)


def ros_environment() -> dict[str, str]:
    """Return the current environment augmented with ROS 2 Jazzy."""
    env = os.environ.copy()
    output = subprocess.run(
        ["bash", "-lc", f"source {ROS_SETUP} && env -0"],
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


def configure_head_rgbd_camera(robot_name: str) -> None:
    """Restore the live head CameraSensor after a scene reset."""
    command = [
        ORCALAB_PYTHON,
        str(ROOT / "enable_robot_cameras.py"),
        "--robot-actor", f"/{robot_name}",
        "--head-depth",
        "--disable-wrist-cameras",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60.0,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to configure the live head RGB-D camera:\n" + result.stdout
        )
    print(result.stdout, end="", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAPPING_SESSION_MARKERS = (
    "world_anchored_rtabmap_",
    "visual_rtabmap_",
    "/tmp/world_anchored_rgbd_",
    "/tmp/visual_rtabmap_rgbd_",
    "/tmp/world_anchored_localization_rgbd_",
    "world_anchored_localization_",
    "g1_button_speed_",
    "g1_official_",
    "g1_newmap_",
    "rviz_evidence_bag",
)


def _cmdline_of(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="ignore"
        )
    except OSError:
        return ""


def is_leftover_mapping_command(cmdline: str) -> bool:
    """True for this project's mapping or leftover localization nodes."""
    if not cmdline or "cursorsandbox" in cmdline or "AGENT_LOOP" in cmdline:
        return False
    if "three_camera_rgbd_bridge.py" in cmdline and (
        any(marker in cmdline for marker in MAPPING_SESSION_MARKERS)
        or "task_1/src/three_camera_rgbd_bridge.py" in cmdline
    ):
        return True
    if "record_mapping_waypoints.py" in cmdline:
        return True
    if (
        ("/rtabmap_slam/rtabmap" in cmdline or "rtabmap_slam rtabmap" in cmdline)
        and any(marker in cmdline for marker in MAPPING_SESSION_MARKERS)
    ):
        return True
    if "task1_rtabmap_mapping.rviz" in cmdline:
        return True
    if "task1_world_anchored_validation.rviz" in cmdline:
        return True
    if "ros2 bag record" in cmdline and (
        "task_1/data" in cmdline or "rviz_evidence_bag" in cmdline
    ):
        return True
    if any(name in cmdline for name in (
        "run_agibot_chassis_execution_bridge.py",
        "run_agibot_abc_demo.py",
        "run_world_anchored_validation_safety_map.py",
    )):
        return True
    if "run_world_anchored_rtabmap_localization.py" in cmdline and (
        "--monitor-worker" in cmdline or "--truth-tf-worker" in cmdline
    ):
        return True
    if "nav2_bringup" in cmdline and "navigation_launch.py" in cmdline:
        return True
    if (
        "static_transform_publisher" in cmdline
        and "camera_head_optical_frame" in cmdline
    ):
        return True
    return False


def is_mapping_support_command(cmdline: str) -> bool:
    if not cmdline or "cursorsandbox" in cmdline:
        return False
    if "rgbd_sync" in cmdline and "rgbd_image:=/rtabmap/rgbd_image" in cmdline:
        return True
    if "rgbd_odometry" in cmdline and "odom:=/odom" in cmdline:
        return True
    if (
        "static_transform_publisher" in cmdline
        and "camera_head_optical_frame" in cmdline
    ):
        return True
    if "rtabmap_viz" in cmdline and "subscribe_rgbd:=true" in cmdline:
        return True
    return False


def leftover_mapping_pids(skip: set[int] | None = None) -> list[int]:
    skip = set(skip or ())
    skip.add(os.getpid())
    core: list[int] = []
    support: list[int] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in skip:
            continue
        cmdline = _cmdline_of(pid)
        if is_leftover_mapping_command(cmdline):
            core.append(pid)
        elif is_mapping_support_command(cmdline):
            support.append(pid)
    if not core:
        return []
    return sorted(set(core + support))


def stop_leftover_mapping_pipelines() -> list[int]:
    """Stop an orphaned mapping stack that still owns /camera /odom /map."""
    pids = leftover_mapping_pids()
    if not pids:
        return []
    for pid in pids:
        _signal_tree(pid, signal.SIGTERM)
    time.sleep(0.4)
    remain = leftover_mapping_pids()
    for pid in remain:
        _signal_tree(pid, signal.SIGKILL)
    time.sleep(0.2)
    return leftover_mapping_pids()


def clear_fastrtps_shared_memory() -> int:
    """Remove stale FastRTPS shared-memory segments left by crashed ROS nodes."""
    removed = 0
    for path in glob.glob("/dev/shm/fastrtps_*"):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed


def restart_ros_daemon(env: dict[str, str], *, settle_s: float = 2.0) -> None:
    """Stop and restart the ros2 CLI daemon, clearing any orphaned daemon process."""
    subprocess.run(
        [ROS2, "daemon", "stop"], cwd=PROJECT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=15.0, check=False,
    )
    subprocess.run(
        ["pkill", "-f", "_ros2_daemon"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(0.5)
    subprocess.run(
        [ROS2, "daemon", "start"], cwd=PROJECT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=20.0, check=False,
    )
    if settle_s > 0:
        time.sleep(settle_s)


def cleanup_ros_runtime(env: dict[str, str] | None = None) -> dict[str, int]:
    """Stop orphaned pipelines, reset ros2 daemon, and clear stale DDS shared memory."""
    leftover = leftover_mapping_pids()
    if leftover:
        print(
            "Stopping leftover mapping processes that still own /camera /odom /map: "
            + ", ".join(str(pid) for pid in leftover),
            flush=True,
        )
        still = stop_leftover_mapping_pipelines()
        if still:
            raise RuntimeError(
                "could not stop leftover mapping processes: "
                + ", ".join(str(pid) for pid in still)
            )
    shm_removed = clear_fastrtps_shared_memory()
    if env is not None:
        restart_ros_daemon(env)
    return {"leftover_processes": len(leftover), "fastrtps_shm_removed": shm_removed}


def clear_leftover_mapping_pipelines(env: dict[str, str] | None = None) -> None:
    cleanup_ros_runtime(env)


def rtabmap_binary_running() -> bool:
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        cmdline = _cmdline_of(int(entry.name))
        if "cursorsandbox" in cmdline:
            continue
        if "/rtabmap_slam/rtabmap" in cmdline or "rtabmap_slam rtabmap" in cmdline:
            return True
    return False


def assert_mapping_topics_unclaimed(env: dict[str, str]) -> None:
    """Refuse to start when another mapping pipeline owns global topics."""
    checks = {
        "/camera/color/image_raw": "RGB-D bridge",
        "/odom": "odometry",
        "/map": "RTAB-Map",
    }
    occupied: list[str] = []
    for topic, owner in checks.items():
        result = subprocess.run(
            [ROS2, "topic", "info", topic], cwd=PROJECT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=5.0, check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Publisher count:"):
                try:
                    count = int(line.split(":", 1)[1])
                except ValueError:
                    count = 0
                if count:
                    if topic == "/map" and not rtabmap_binary_running():
                        break
                    occupied.append(f"{topic} ({count} existing {owner} publisher(s))")
                break
    if occupied:
        raise RuntimeError(
            "Another mapping pipeline is already running: " + ", ".join(occupied) +
            ". Stop it before starting a new session; otherwise RGB-D stamps, /odom, "
            "/map and TF will be mixed."
        )


def terminate(proc: subprocess.Popen | None, timeout: float = 2.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    _signal_tree(pid, signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_tree(pid, signal.SIGTERM)
    try:
        proc.wait(timeout=min(1.5, timeout))
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_tree(pid, signal.SIGKILL)
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        return


def _signal_tree(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        _signal_pid(pid, sig)
    for child in descendant_pids(pid):
        _signal_pid(child, sig)


def _proc_ppid(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2:].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        child = int(entry.name)
        parent = _proc_ppid(child)
        if parent is not None:
            children[parent].append(child)
    found: list[int] = []
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, ()))
    return found


def command_mentions_session(cmdline: str, session: Path) -> bool:
    needle = str(session.resolve())
    return bool(needle) and needle in cmdline


def session_child_pids(session: Path, skip: set[int] | None = None) -> list[int]:
    skip = set(skip or ())
    skip.add(os.getpid())
    pids: list[int] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in skip:
            continue
        try:
            cmdline = entry.joinpath("cmdline").read_bytes().replace(b"\x00", b" ")
            text = cmdline.decode("utf-8", errors="ignore")
        except OSError:
            continue
        if command_mentions_session(text, session):
            pids.append(pid)
    return pids


def stop_session_children(session: Path, extra_pids: list[int] | None = None) -> None:
    pids = set(extra_pids or ())
    pids.update(session_child_pids(session))
    pids.discard(os.getpid())
    for pid in sorted(pids):
        _signal_tree(pid, signal.SIGTERM)
    time.sleep(0.3)
    remain = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
    remain.extend(session_child_pids(session))
    for pid in sorted(set(remain) - {os.getpid()}):
        _signal_tree(pid, signal.SIGKILL)


_CLEANUP = {"done": False, "processes": None, "session": None, "env": None, "map_saved": False}


def reset_mapping_cleanup() -> None:
    _CLEANUP["done"] = False
    _CLEANUP["processes"] = None
    _CLEANUP["session"] = None
    _CLEANUP["env"] = None
    _CLEANUP["map_saved"] = False


def cleanup_mapping_session(
    processes: dict[str, subprocess.Popen] | None,
    session: Path | None,
    env: dict[str, str] | None = None,
    save_map: bool = False,
) -> bool:
    """Stop the driver, optionally save the grid, then kill every mapping child.

    Safe to call twice. A second Ctrl+C during map-save skips the save and
    still sweeps RTAB-Map / RGB-D / waypoint recorder leftovers.
    """
    if _CLEANUP["done"]:
        return bool(_CLEANUP.get("map_saved"))
    _CLEANUP["done"] = True
    processes = processes or {}
    map_saved = False
    try:
        terminate(processes.get("driver"), timeout=2.0)
        rtabmap = processes.get("rtabmap")
        if save_map and env is not None and session is not None:
            if rtabmap is not None and rtabmap.poll() is None:
                try:
                    map_saved = publish_map_and_save(
                        session, env, session / "map_save.log"
                    )
                except (KeyboardInterrupt, subprocess.TimeoutExpired):
                    map_saved = False
        for name in ("waypoint_recorder", "rtabmapviz", "rviz", "rtabmap",
                     "rgbd_odometry", "rgbd_sync", "camera_tf", "rgbd"):
            terminate(processes.get(name), timeout=2.0)
        extras = [proc.pid for proc in processes.values() if proc is not None]
        if session is not None:
            stop_session_children(session, extras)
    except KeyboardInterrupt:
        extras = [proc.pid for proc in processes.values() if proc is not None]
        if session is not None:
            stop_session_children(session, extras)
    _CLEANUP["map_saved"] = map_saved
    return map_saved


def install_mapping_cleanup(
    processes: dict[str, subprocess.Popen],
    session: Path,
    env: dict[str, str],
) -> None:
    _CLEANUP["processes"] = processes
    _CLEANUP["session"] = session
    _CLEANUP["env"] = env

    def _atexit_cleanup() -> None:
        cleanup_mapping_session(
            _CLEANUP["processes"], _CLEANUP["session"], _CLEANUP["env"],
            save_map=False,
        )

    def _on_signal(signum: int, _frame) -> None:
        cleanup_mapping_session(
            _CLEANUP["processes"], _CLEANUP["session"], _CLEANUP["env"],
            save_map=False,
        )
        raise SystemExit(128 + signum)

    atexit.register(_atexit_cleanup)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)


def read_world_pose(robot: str) -> dict[str, object]:
    """Read one world pose without advancing the simulation."""
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from task_2.drive_robot_wasd import snapshot_live_state, restore_live_state

    asyncio.set_event_loop(asyncio.new_event_loop())
    saved_qpos, saved_qvel = snapshot_live_state("127.0.0.1:50051")
    env = OrcaGymLocalEnv(
        frame_skip=1, orcagym_addr="127.0.0.1:50051",
        agent_names=[robot], time_step=0.001,
    )
    try:
        # OrcaGymLocalEnv initialization resets the remote simulation. Restore
        # the pre-connect state before sampling so this is a true live read.
        restore_live_state(env, saved_qpos, saved_qvel)
        bodies = env.model.get_body_dict()
        holder_id = int(bodies[f"{robot}_robot_holder1"]["ID"])
        env.mj_forward()
        pos = np.asarray(env.gym._mjData.xpos[holder_id], dtype=float).copy()
        rot = np.asarray(env.gym._mjData.xmat[holder_id], dtype=float).reshape(3, 3)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        return {
            "position_xyz": [float(v) for v in pos],
            "yaw_rad": float(yaw),
            "frame": "world",
            "source": "MuJoCo initial read only",
        }
    finally:
        env.close()


def read_task_sites_and_robot_root(robot: str) -> dict[str, object]:
    """Query scene A/B/C sites and the robot-root judgement frame."""
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from task_2.drive_robot_wasd import snapshot_live_state, restore_live_state

    site_names = {
        "A": "Static_Location_A_site",
        "B": "Static_Location_B_site",
        "C": "Static_Location_C_site",
    }
    asyncio.set_event_loop(asyncio.new_event_loop())
    saved_qpos, saved_qvel = snapshot_live_state("127.0.0.1:50051")
    env = OrcaGymLocalEnv(
        frame_skip=1, orcagym_addr="127.0.0.1:50051",
        agent_names=[robot], time_step=0.001,
    )
    try:
        restore_live_state(env, saved_qpos, saved_qvel)
        env.mj_forward()
        queried = env.query_site_pos_and_mat(list(site_names.values()))
        points = {}
        for name, site_name in site_names.items():
            if site_name not in queried:
                raise RuntimeError(f"scene task site is unavailable: {site_name}")
            position = np.asarray(queried[site_name]["xpos"], dtype=float)
            points[name] = [float(position[0]), float(position[1]), float(position[2])]
        bodies = env.model.get_body_dict()
        holder_name = f"{robot}_robot_holder1"
        if holder_name not in bodies:
            raise RuntimeError(f"robot root body is unavailable: {holder_name}")
        return {
            "source": "OrcaGymLocalEnv.query_site_pos_and_mat initialization read",
            "site_names": site_names,
            "world_points_xyz": points,
            "judgement_entity": robot,
            "model_root_body": holder_name,
            "judgement_offset_in_navigation_base_xyz": [0.0, 0.0, 0.0],
        }
    finally:
        env.close()


def read_camera_extrinsic(robot: str) -> dict[str, object]:
    """Compute the fixed base->optical transform used by the RGB-D bridge."""
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from task_2.drive_robot_wasd import snapshot_live_state, restore_live_state

    asyncio.set_event_loop(asyncio.new_event_loop())
    saved_qpos, saved_qvel = snapshot_live_state("127.0.0.1:50051")
    env = OrcaGymLocalEnv(
        frame_skip=1, orcagym_addr="127.0.0.1:50051",
        agent_names=[robot], time_step=0.001,
    )
    try:
        restore_live_state(env, saved_qpos, saved_qvel)
        bodies = env.model.get_body_dict()
        holder_id = int(bodies[f"{robot}_robot_holder1"]["ID"])
        camera_id = int(bodies[f"{robot}_camera_head_body2"]["ID"])
        env.mj_forward()
        data = env.gym._mjData
        holder_pos = np.asarray(data.xpos[holder_id], dtype=float).copy()
        holder_rot = np.asarray(data.xmat[holder_id], dtype=float).reshape(3, 3).copy()
        camera_pos = np.asarray(data.xpos[camera_id], dtype=float).copy()

        # The render camera is an external CameraSensor.  Its calibrated
        # optical direction is base +X pitched down by 30 degrees.
        pitch = math.radians(30.0)
        optical_x = np.asarray((0.0, -1.0, 0.0))
        optical_z = np.asarray((math.cos(pitch), 0.0, -math.sin(pitch)))
        optical_y = np.cross(optical_z, optical_x)
        base_optical = np.column_stack((optical_x, optical_y, optical_z))
        base_translation = holder_rot.T @ (camera_pos - holder_pos)
        q = matrix_to_xyzw(base_optical)
        return {
            "camera_frame": "camera_head_optical_frame",
            "base_frame": "base_link",
            "translation": [float(v) for v in base_translation],
            "quaternion_xyzw": [float(v) for v in q],
            "vertical_fov_deg": 90.0,
        }
    finally:
        env.close()


def matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized ROS xyzw quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x, y, z = 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x, y, z = (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x, y, z = (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    q = np.asarray((x, y, z, w), dtype=float)
    return q / np.linalg.norm(q)


class TerminalKeys:
    """Small watchdog-based WASD reader used only by the motion worker."""

    def __init__(self, hold_seconds: float = 0.30) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("terminal input requires a TTY")
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        current = termios.tcgetattr(self.fd)
        current[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, current)
        self.hold_seconds = hold_seconds
        self.last = {key: 0.0 for key in "WASD"}
        self.pending_marks: list[str] = []

    def get_state(self) -> dict[str, int]:
        now = time.monotonic()
        while select.select([self.fd], [], [], 0.0)[0]:
            char = os.read(self.fd, 1).decode(errors="ignore").upper()
            if char in {"\x03", "\x1b"}:
                raise KeyboardInterrupt
            mark = mark_name_from_key(char)
            if mark is not None:
                self.pending_marks.append(mark)
            if char in self.last:
                self.last[char] = now
        return {key: int(now - stamp < self.hold_seconds)
                for key, stamp in self.last.items()}

    def consume_waypoint_marks(self) -> list[str]:
        marks, self.pending_marks = self.pending_marks, []
        return marks

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)


class MappingDesktopKeys:
    """WASD drive plus edge-triggered 1/2/3 waypoint marks. Esc stops."""

    def __init__(self) -> None:
        from pynput import keyboard

        self.keyboard = keyboard
        self.state = {key: 0 for key in "WASD"}
        self.stop_requested = False
        self.pending_marks: list[str] = []
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.start()

    def _char(self, key) -> str | None:
        try:
            char = key.char
        except (AttributeError, TypeError):
            return None
        if not char:
            return None
        return char.upper()

    def _press(self, key) -> None:
        if key == self.keyboard.Key.esc:
            self.stop_requested = True
            return
        char = self._char(key)
        if char is None:
            return
        mark = mark_name_from_key(char)
        if mark is not None:
            self.pending_marks.append(mark)
        if char in self.state:
            self.state[char] = 1

    def _release(self, key) -> None:
        char = self._char(key)
        if char in self.state:
            self.state[char] = 0

    def get_state(self) -> dict[str, int]:
        if self.stop_requested:
            raise KeyboardInterrupt
        return {key: int(self.state[key]) for key in "WASD"}

    def consume_waypoint_marks(self) -> list[str]:
        marks, self.pending_marks = self.pending_marks, []
        return marks

    def close(self) -> None:
        self.listener.stop()
        self.listener.join(timeout=1.0)


class CombinedKeys:
    """Combine terminal and OrcaLab viewport input without any ROS access."""

    def __init__(self, address: str) -> None:
        from task_2.drive_robot_wasd import SceneKeyboard

        self.terminal = TerminalKeys() if sys.stdin.isatty() else None
        self.scene = SceneKeyboard(address)

    def get_state(self) -> dict[str, int]:
        state = (self.terminal.get_state() if self.terminal else
                 {key: 0 for key in "WASD"})
        scene = self.scene.get_state()
        return {key: int(bool(state[key] or scene[key])) for key in "WASD"}

    def consume_waypoint_marks(self) -> list[str]:
        if self.terminal is None:
            return []
        return self.terminal.consume_waypoint_marks()

    def close(self) -> None:
        if self.terminal:
            self.terminal.close()
        self.scene.close()


class LiveWasd:
    """Read W/A/S/D from a JSON file so an external process can steer live."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pending_marks: list[str] = []
        if not self.path.exists():
            self.path.write_text(
                '{"W":0,"A":0,"S":0,"D":0,"stop":false,"mark":""}\n', encoding="utf-8"
            )

    def get_state(self) -> dict[str, int]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {key: 0 for key in "WASD"}
        if data.get("stop"):
            raise KeyboardInterrupt
        mark = str(data.get("mark", "")).upper()
        if mark in {"A", "B", "C"}:
            self.pending_marks.append(mark)
            data["mark"] = ""
            try:
                self.path.write_text(json.dumps(data) + "\n", encoding="utf-8")
            except OSError:
                pass
        return {key: int(bool(data.get(key, 0))) for key in "WASD"}

    def consume_waypoint_marks(self) -> list[str]:
        marks, self.pending_marks = self.pending_marks, []
        return marks

    def close(self) -> None:
        return None


def motion_worker(argv: list[str]) -> int:
    """Drive the chassis and optionally publish root-truth external odometry."""
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from task_2.drive_robot_wasd import (
        FRAME_SKIP, SceneKeyboard, TIME_STEP, bind_chassis,
        bind_pose_holds, lock_pose, restore_live_state, scaled_value,
        snapshot_live_state,
    )

    parser = argparse.ArgumentParser(description="Pure OrcaLab mapping motion worker")
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--input", choices=MAPPING_INPUTS, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--turn-speed", type=float, required=True)
    parser.add_argument("--linear-ramp-rate", type=float, default=0.8)
    parser.add_argument("--steering-ramp-rate", type=float, default=1.0)
    parser.add_argument("--camera-render-hz", type=float, required=True)
    parser.add_argument("--probe-distance", type=float, required=True)
    parser.add_argument("--probe-timeout", type=float, required=True)
    parser.add_argument("--probe-startup-delay", type=float, required=True)
    parser.add_argument("--loop-segment-length", type=float, default=1.5)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--live-command-file", type=Path)
    parser.add_argument("--waypoint-request-file", type=Path)
    parser.add_argument("--publish-truth-odom", action="store_true")
    args = parser.parse_args(argv)
    waypoint_request = args.waypoint_request_file

    ros_node = None
    odom_pub = None
    tf_pub = None
    if args.publish_truth_odom:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from tf2_ros import TransformBroadcaster

        rclpy.init()
        ros_node = Node("task1_mapping_truth_odometry")
        odom_pub = ros_node.create_publisher(Odometry, "/odom", 20)
        tf_pub = TransformBroadcaster(ros_node)

    address = "127.0.0.1:50051"
    saved_qpos, saved_qvel = snapshot_live_state(address)
    env = OrcaGymLocalEnv(frame_skip=FRAME_SKIP, orcagym_addr=address,
                          agent_names=[args.robot_name], time_step=TIME_STEP)
    keyboard = None
    metadata: dict[str, object] = {
        "format": "pure_orcalab_mapping_motion_v1",
        "robot_name": args.robot_name,
        "ros_node_created": bool(ros_node),
        "odom_published": bool(odom_pub),
        "world_pose_usage": (
            "g1_omnipicker root external odometry"
            if odom_pub else "automatic path endpoint control only"
        ),
        "requested_speed_command": args.speed,
    }
    last_world_pose: dict[str, object] | None = None
    exit_code = 0
    try:
        restore_live_state(env, saved_qpos, saved_qvel)
        drives, steering = bind_chassis(env.model, args.robot_name)
        holds = bind_pose_holds(env, args.robot_name)
        metadata["joint_pose_initialization_applied"] = False
        metadata["joint_pose_hold_source"] = "live OrcaLab state captured after connection"
        metadata["held_non_chassis_joint_count"] = len(holds)
        metadata["held_non_chassis_qpos"] = [float(target) for _q, _v, target in holds]
        holder_id = int(env.model.get_body_dict()[
            f"{args.robot_name}_robot_holder1"]["ID"])
        env.mj_forward()
        initial_pos = np.asarray(env.gym._mjData.xpos[holder_id], dtype=float).copy()
        initial_rot = np.asarray(env.gym._mjData.xmat[holder_id], dtype=float).reshape(3, 3)
        initial_yaw = math.atan2(float(initial_rot[1, 0]), float(initial_rot[0, 0]))
        last_world_pose = {
            "position_xyz": [float(v) for v in initial_pos],
            "yaw_rad": float(initial_yaw),
        }
        metadata["initial_world_pose"] = last_world_pose.copy()
        if args.input == "desktop":
            keyboard = MappingDesktopKeys()
        elif args.input == "terminal":
            keyboard = TerminalKeys()
        elif args.input == "orcalab":
            keyboard = SceneKeyboard(address)
        elif args.input == "hybrid":
            keyboard = CombinedKeys(address)
        elif args.input == "live":
            command_file = args.live_command_file or args.metadata.with_name("live_wasd.json")
            keyboard = LiveWasd(command_file)

        started = time.monotonic()
        phase = "settle"
        phase_started = started
        max_forward = 0.0
        loop_count = 3 if args.input == "u_probe" else 4
        loop_index = 0
        loop_phase = "segment"
        segment_start = initial_pos.copy()
        segment_yaw = initial_yaw
        turn_start_yaw = initial_yaw
        applied_forward = 0.0
        applied_turn = 0.0
        last_command = started
        last_render = 0.0
        last_pose_write = 0.0
        last_odom_pos = initial_pos.copy()
        last_odom_yaw = initial_yaw
        last_odom_time = started
        wrapup_index = 0
        wrapup_arrived_at = None
        wrapup_progress_path = args.metadata.with_name("wrapup_progress.json")
        last_wrapup_log = 0.0
        print(
            "Motion worker ready"
            + (" (publishing root-truth /odom)." if odom_pub else "."),
            flush=True,
        )
        print(KEY_HELP, flush=True)
        if args.input == "task1_wrapup":
            print(
                f"task1_wrapup: {len(WRAPUP_WAYPOINTS)} waypoints "
                f"spawn→A→B→C plus cabinet outline",
                flush=True,
            )
        while True:
            tick = time.monotonic()
            env.mj_forward()
            pos = np.asarray(env.gym._mjData.xpos[holder_id], dtype=float).copy()
            rot = np.asarray(env.gym._mjData.xmat[holder_id], dtype=float).reshape(3, 3)
            yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
            if odom_pub is not None and tf_pub is not None and ros_node is not None:
                from geometry_msgs.msg import TransformStamped
                from nav_msgs.msg import Odometry

                stamp = ros_node.get_clock().now().to_msg()
                odom_dt = max(tick - last_odom_time, 1e-6)
                world_velocity = (pos[:2] - last_odom_pos[:2]) / odom_dt
                yaw_delta = math.atan2(
                    math.sin(yaw - last_odom_yaw),
                    math.cos(yaw - last_odom_yaw),
                )
                cy, sy = math.cos(yaw), math.sin(yaw)
                msg = Odometry()
                msg.header.stamp = stamp
                msg.header.frame_id = "odom"
                msg.child_frame_id = "base_link"
                msg.pose.pose.position.x = float(pos[0])
                msg.pose.pose.position.y = float(pos[1])
                msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
                msg.pose.pose.orientation.w = math.cos(yaw * 0.5)
                msg.twist.twist.linear.x = float(
                    cy * world_velocity[0] + sy * world_velocity[1]
                )
                msg.twist.twist.angular.z = float(yaw_delta / odom_dt)
                msg.pose.covariance[0] = msg.pose.covariance[7] = 1e-6
                msg.pose.covariance[35] = 1e-6
                odom_pub.publish(msg)

                tf = TransformStamped()
                tf.header.stamp = stamp
                tf.header.frame_id = "odom"
                tf.child_frame_id = "base_link"
                tf.transform.translation.x = float(pos[0])
                tf.transform.translation.y = float(pos[1])
                tf.transform.rotation.z = math.sin(yaw * 0.5)
                tf.transform.rotation.w = math.cos(yaw * 0.5)
                tf_pub.sendTransform(tf)
                rclpy.spin_once(ros_node, timeout_sec=0.0)
                last_odom_pos = pos.copy()
                last_odom_yaw = yaw
                last_odom_time = tick
            last_world_pose = {
                "position_xyz": [float(v) for v in pos],
                "yaw_rad": float(yaw),
            }
            if args.input == "live" and tick - last_pose_write >= 0.5:
                last_pose_write = tick
                args.metadata.with_name("live_pose.json").write_text(
                    json.dumps({
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "yaw": float(yaw),
                        "elapsed_s": tick - started,
                    }) + "\n",
                    encoding="utf-8",
                )
            elapsed = tick - started
            delta = pos[:2] - initial_pos[:2]
            forward_projection = float(math.cos(initial_yaw) * delta[0] +
                                       math.sin(initial_yaw) * delta[1])
            lateral = float(-math.sin(initial_yaw) * delta[0] +
                            math.cos(initial_yaw) * delta[1])
            max_forward = max(max_forward, forward_projection)

            if args.input == "straight_probe":
                if elapsed < args.probe_startup_delay:
                    state = {key: 0 for key in "WASD"}
                elif phase == "settle":
                    phase, phase_started = "outbound", tick
                    metadata["motion_started_unix"] = time.time()
                    state = {"W": 1, "A": 0, "S": 0, "D": 0}
                elif phase == "outbound" and forward_projection < args.probe_distance:
                    if tick - phase_started > args.probe_timeout:
                        raise RuntimeError("straight probe outbound motion timed out")
                    state = {"W": 1, "A": 0, "S": 0, "D": 0}
                elif phase == "outbound":
                    phase, phase_started = "pause", tick
                    state = {key: 0 for key in "WASD"}
                elif phase == "pause" and tick - phase_started < 1.0:
                    state = {key: 0 for key in "WASD"}
                elif phase == "pause":
                    phase, phase_started = "return", tick
                    state = {"W": 0, "A": 0, "S": 1, "D": 0}
                elif forward_projection > 0.05:
                    if tick - phase_started > args.probe_timeout:
                        raise RuntimeError("straight probe return motion timed out")
                    state = {"W": 0, "A": 0, "S": 1, "D": 0}
                else:
                    metadata["straight_probe"] = {
                        "requested_outbound_distance_m": args.probe_distance,
                        "max_forward_projection_m": max_forward,
                        "final_forward_projection_m": forward_projection,
                        "final_lateral_error_m": lateral,
                        "final_yaw_error_rad": math.atan2(math.sin(yaw-initial_yaw),
                                                           math.cos(yaw-initial_yaw)),
                        "completed": True,
                    }
                    break
            elif args.input in {"u_probe", "rectangle_probe"}:
                if elapsed < args.probe_startup_delay:
                    state = {key: 0 for key in "WASD"}
                elif loop_phase == "segment":
                    along = (math.cos(segment_yaw) * (pos[0]-segment_start[0]) +
                             math.sin(segment_yaw) * (pos[1]-segment_start[1]))
                    if along < args.loop_segment_length:
                        if tick - phase_started > args.probe_timeout:
                            raise RuntimeError("loop segment motion timed out")
                        state = {"W": 1, "A": 0, "S": 0, "D": 0}
                    else:
                        loop_index += 1
                        if loop_index >= loop_count and args.input == "u_probe":
                            metadata["loop_probe"] = {"shape": args.input,
                                "segments": loop_count, "segments_completed": loop_index,
                                "motion_completed": True}
                            break
                        loop_phase = "turn"
                        turn_start_yaw = yaw
                        phase_started = tick
                        state = {"W": 1, "A": 1, "S": 0, "D": 0}
                else:
                    progress = math.atan2(math.sin(yaw-turn_start_yaw),
                                          math.cos(yaw-turn_start_yaw))
                    if progress < math.pi/2.0 - 0.08:
                        if tick - phase_started > args.probe_timeout:
                            raise RuntimeError("loop turn motion timed out")
                        state = {"W": 1, "A": 1, "S": 0, "D": 0}
                    else:
                        segment_start = pos.copy()
                        segment_yaw = yaw
                        loop_phase = "segment"
                        phase_started = tick
                        state = {"W": 1, "A": 0, "S": 0, "D": 0}
                        if loop_index >= loop_count and args.input == "rectangle_probe":
                            metadata["loop_probe"] = {"shape": args.input,
                                "segments": loop_count, "segments_completed": loop_index,
                                "motion_completed": True}
                            break
            elif args.input == "probe":
                active = elapsed - args.probe_startup_delay
                if active < 0.0:
                    state = {key: 0 for key in "WASD"}
                elif active < 3.0:
                    state = {"W": 1, "A": 0, "S": 0, "D": 0}
                elif active < 6.0:
                    state = {"W": 1, "A": 1, "S": 0, "D": 0}
                elif active < 7.0:
                    state = {key: 0 for key in "WASD"}
                else:
                    break
            elif args.input == "task1_wrapup":
                if wrapup_index >= len(WRAPUP_WAYPOINTS):
                    metadata["wrapup"] = {
                        "completed": True,
                        "waypoints": len(WRAPUP_WAYPOINTS),
                    }
                    break
                name, target_x, target_y = WRAPUP_WAYPOINTS[wrapup_index]
                choice = wrapup_wasd(float(pos[0]), float(pos[1]), float(yaw),
                                     float(target_x), float(target_y))
                if name.endswith("_reverse") and not choice["arrived"]:
                    choice["W"] = 0
                    choice["S"] = 1
                dwell_s = 1.2 if name in {"A", "B", "C", "A_on_again", "B_on_again", "C_on_again"} else 0.25
                if choice["arrived"]:
                    if wrapup_arrived_at is None:
                        wrapup_arrived_at = tick
                        if name in {"A", "B", "C"}:
                            write_waypoint_request(waypoint_request, name)
                    state = {key: 0 for key in "WASD"}
                    if tick - wrapup_arrived_at >= dwell_s:
                        print(
                            f"task1_wrapup reached {name} "
                            f"xy=({pos[0]:.2f},{pos[1]:.2f}) dist={choice['dist']:.2f}",
                            flush=True,
                        )
                        wrapup_index += 1
                        wrapup_arrived_at = None
                else:
                    wrapup_arrived_at = None
                    state = {key: int(choice[key]) for key in "WASD"}
                if tick - last_wrapup_log >= 2.0:
                    last_wrapup_log = tick
                    progress = {
                        "waypoint_index": wrapup_index,
                        "waypoint": name,
                        "target_xy": [target_x, target_y],
                        "pose_xy": [float(pos[0]), float(pos[1])],
                        "yaw_rad": float(yaw),
                        "dist_m": float(choice["dist"]),
                        "wasd": {key: int(state[key]) for key in "WASD"},
                        "elapsed_s": elapsed,
                    }
                    wrapup_progress_path.write_text(
                        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
                    )
                    print(
                        f"task1_wrapup {wrapup_index+1}/{len(WRAPUP_WAYPOINTS)} "
                        f"{name} dist={choice['dist']:.2f} "
                        f"xy=({pos[0]:.2f},{pos[1]:.2f})",
                        flush=True,
                    )
            else:
                state = keyboard.get_state()
            consume_marks = getattr(keyboard, "consume_waypoint_marks", None)
            if consume_marks is not None:
                for name in consume_marks():
                    write_waypoint_request(waypoint_request, name)
                    print(f"queued map waypoint {name} (waiting for TF recorder)", flush=True)

            target_forward = (state["W"] - state["S"]) * args.speed
            if state["A"] or state["D"]:
                target_forward *= 0.65
            target_turn = (state["A"] - state["D"]) * args.turn_speed
            dt = max(1e-3, tick - last_command)
            last_command = tick
            applied_forward += float(np.clip(target_forward-applied_forward,
                                              -args.linear_ramp_rate*dt,
                                              args.linear_ramp_rate*dt))
            applied_turn += float(np.clip(target_turn-applied_turn,
                                           -args.steering_ramp_rate*dt,
                                           args.steering_ramp_rate*dt))
            lock_pose(env, holds)
            control = np.zeros(env.model.nu, dtype=float)
            for actuator in drives:
                control[actuator.index] = scaled_value(actuator, applied_forward)
            for wheel, actuator in steering.items():
                phase_sign = 1.0 if wheel in {"fl", "fr"} else -1.0
                control[actuator.index] = scaled_value(
                    actuator, applied_turn * phase_sign)
            env.do_simulation(control, FRAME_SKIP)
            lock_pose(env, holds)
            if last_render == 0.0 or tick-last_render >= 1.0/args.camera_render_hz:
                env.render()
                last_render = tick
            delay = TIME_STEP*FRAME_SKIP - (time.monotonic()-tick)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        metadata["motion_finished_unix"] = time.time()
        if last_world_pose is not None:
            metadata["final_world_pose"] = last_world_pose
        args.metadata.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        if keyboard:
            keyboard.close()
        try:
            env.do_simulation(np.zeros(env.model.nu, dtype=float), FRAME_SKIP)
        except Exception:
            pass
        env.close()
        if ros_node is not None:
            ros_node.destroy_node()
            rclpy.shutdown()
    return exit_code


def start_process(name: str, command: list[str], env: dict[str, str], log_dir: Path,
                  processes: dict[str, subprocess.Popen]) -> None:
    log = (log_dir / f"{name}.log").open("w", encoding="utf-8")
    processes[name] = subprocess.Popen(
        command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def topic_ready(topic: str, env: dict[str, str], timeout: float = 2.0,
                message_type: str | None = None) -> bool:
    """Wait for one message, with an explicit type to avoid discovery races."""
    command = [ROS2, "topic", "echo", "--once", "--timeout", str(timeout),
               "--qos-profile", "best_available", topic]
    if message_type:
        command.append(message_type)
    return subprocess.run(
        command,
        cwd=PROJECT, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def publish_map_and_save(session: Path, env: dict[str, str], log: Path) -> bool:
    """Hold a /map subscriber, publish the optimized global grid, then save.

    RTAB-Map skips PublishMap when nobody is subscribed. map_saver_cli must
    start only after that service returns, or it snapshots the old local grid.
    """
    prefix = session / "map" / "task1_rtabmap"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    saver_log = log.open("a", encoding="utf-8")
    holder = subprocess.Popen(
        [ROS2, "topic", "echo", "--qos-durability", "transient_local",
         "/map", "nav_msgs/msg/OccupancyGrid"],
        cwd=PROJECT, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        try:
            published = subprocess.run(
                [ROS2, "service", "call", "/rtabmap/publish_map",
                 "rtabmap_msgs/srv/PublishMap",
                 "{global_map: true, optimized: true, graph_only: false}"],
                cwd=PROJECT, env=env, stdout=saver_log, stderr=subprocess.STDOUT,
                timeout=60.0, check=False,
            )
        except subprocess.TimeoutExpired:
            saver_log.write("publish_map timed out\n")
            saver_log.flush()
            return False
        if published.returncode != 0:
            return False
        time.sleep(2.0)
        saver = subprocess.Popen(
            [ROS2, "run", "nav2_map_server", "map_saver_cli", "-t", "/map", "-f", str(prefix)],
            cwd=PROJECT, env=env, stdout=saver_log, stderr=subprocess.STDOUT,
        )
        try:
            saver.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            saver.terminate()
            saver.wait(timeout=5.0)
        return saver.returncode == 0 and prefix.with_suffix(".yaml").is_file()
    finally:
        if holder.poll() is None:
            holder.terminate()
            try:
                holder.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                holder.kill()
        saver_log.close()


def export_occupancy_from_database(session_dir: Path) -> int:
    """Rebuild the 2D grid from rtabmap.db without driving or touching the db."""
    session = session_dir.expanduser().resolve()
    database = session / "rtabmap.db"
    if not database.is_file():
        raise SystemExit(f"no rtabmap.db in {session}")
    env = ros_environment()
    clear_leftover_mapping_pipelines(env)
    processes: dict[str, subprocess.Popen] = {}
    reset_mapping_cleanup()
    try:
        start_process("rtabmap", [
            ROS2, "run", "rtabmap_slam", "rtabmap",
            "--Mem/IncrementalMemory", "false",
            "--Mem/InitWMWithAllNodes", "true",
            "--Mem/LocalizationDataSaved", "false",
            "--RGBD/Enabled", "false",
            "--Optimizer/Strategy", "1",
            "--Reg/Force3DoF", "true",
            "--Grid/FromDepth", "true",
            "--Grid/3D", "false",
            "--Grid/RangeMax", "3.0",
            "--Grid/MaxGroundHeight", "0.05",
            "--Grid/MaxObstacleHeight", "1.5",
            "--ros-args",
            "-p", "frame_id:=base_link",
            "-p", "map_frame_id:=map",
            "-p", "publish_tf:=false",
            "-p", "subscribe_rgb:=false",
            "-p", "subscribe_depth:=false",
            "-p", "subscribe_rgbd:=false",
            "-p", "subscribe_odom:=false",
            "-p", "subscribe_scan:=false",
            "-p", "database_path:=" + str(database),
            "-r", "map:=/map",
        ], env, session, processes)
        if not topic_ready("/map", env, 45.0, "nav_msgs/msg/OccupancyGrid"):
            raise RuntimeError("rtabmap did not publish /map from the database")
        saved = publish_map_and_save(session, env, session / "map_export.log")
        if not saved:
            raise RuntimeError("failed to write map/task1_rtabmap after PublishMap")
        print(json.dumps({
            "session": str(session),
            "database": str(database),
            "map_yaml": str(session / "map" / "task1_rtabmap.yaml"),
            "rewritten": True,
        }, indent=2))
        return 0
    finally:
        terminate(processes.get("rtabmap"), timeout=5.0)
        stop_session_children(session, [proc.pid for proc in processes.values()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default=ROBOT_NAME)
    parser.add_argument(
        "--reset-scene", action=argparse.BooleanOptionalAction, default=True,
        help="pause, LoadInitialFrame, then resume before the one-shot world pose seed",
    )
    parser.add_argument("--input", choices=MAPPING_INPUTS, default="desktop")
    parser.add_argument("--session-dir", type=Path)
    database_group = parser.add_mutually_exclusive_group()
    database_group.add_argument("--database", type=Path)
    database_group.add_argument(
        "--resume-from", type=Path,
        help=("copy an existing rtabmap.db (or mapping session containing it) "
              "into the new session and continue mapping without modifying the source"),
    )
    parser.add_argument("--rgbd-fps", type=float, default=20.0,
                        help="RGB-D capture rate; 20 Hz is recommended at 0.45 m/s")
    parser.add_argument("--camera-render-hz", type=float, default=30.0)
    parser.add_argument(
        "--grid-max-obstacle-height", type=float, default=1.50,
        help=("maximum obstacle height included in the saved 2D occupancy grid; "
              "must cover the configured robot collision envelope"),
    )
    parser.add_argument("--initial-world-pose", nargs=3, type=float, metavar=("X", "Y", "YAW"),
                        help="override the one-shot MuJoCo pose used to initialize rgbd_odometry")
    parser.add_argument("--speed", type=float, default=0.45,
                        help="target chassis speed command (calibrated about 0.45 m/s; max 0.45)")
    parser.add_argument("--turn-speed", type=float, default=0.10)
    parser.add_argument("--linear-ramp-rate", type=float, default=0.8,
                        help="normalized acceleration command per second")
    parser.add_argument("--steering-ramp-rate", type=float, default=1.0,
                        help="normalized steering ramp command per second")
    parser.add_argument("--probe-distance", type=float, default=1.5)
    parser.add_argument("--probe-timeout", type=float, default=45.0)
    parser.add_argument("--probe-startup-delay", type=float, default=10.0)
    parser.add_argument("--loop-segment-length", type=float, default=1.5)
    parser.add_argument("--disable-loop-closure", action="store_true",
                        help="disable RTAB-Map spatial loop-closure candidates")
    parser.add_argument("--rviz", action="store_true",
                        help="open the task RGB-D mapping RViz layout")
    parser.add_argument("--rtabmapviz", action="store_true",
                        help="open RTAB-Map's native 3D map viewer (rtabmap_viz)")
    parser.add_argument("--visualize", action="store_true",
                        help="open both RViz2 and RTAB-Map's native 3D map viewer")
    parser.add_argument("--live-command-file", type=Path,
                        help="JSON W/A/S/D file used when --input live")
    parser.add_argument(
        "--export-grid", type=Path, metavar="SESSION_DIR",
        help="reload an existing rtabmap.db and rewrite map/task1_rtabmap.pgm",
    )
    args = parser.parse_args()
    if args.export_grid:
        return export_occupancy_from_database(args.export_grid)
    if not 0.0 < args.speed <= 0.45:
        parser.error("--speed must be in the safe command range (0, 0.45]")
    if args.rgbd_fps <= 0 or args.camera_render_hz <= 0:
        parser.error("--rgbd-fps and --camera-render-hz must be positive")
    if not 0.40 <= args.grid_max_obstacle_height <= 3.0:
        parser.error("--grid-max-obstacle-height must be in [0.40, 3.0] m")
    require_port(50051); require_port(50151)
    if args.reset_scene:
        from orcalab_scene_reset import restart_current_scene
        restart_current_scene("127.0.0.1:50051", timeout=10.0, robot_name=args.robot_name)
    configure_head_rgbd_camera(args.robot_name)
    env = ros_environment()
    clear_leftover_mapping_pipelines(env)
    assert_mapping_topics_unclaimed(env)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    session = (args.session_dir or ROOT / "data" / f"world_anchored_rtabmap_{stamp}").resolve()
    session.mkdir(parents=True, exist_ok=False)
    (session / "map").mkdir()
    resume_source: Path | None = None
    if args.resume_from:
        resume_source = args.resume_from.expanduser().resolve()
        if resume_source.is_dir():
            resume_source = resume_source / "rtabmap.db"
        if not resume_source.is_file():
            parser.error(f"resume database does not exist: {resume_source}")
        database = session / "rtabmap.db"
        shutil.copy2(resume_source, database)
        source_sha256 = sha256_file(resume_source)
        copied_sha256 = sha256_file(database)
        if source_sha256 != copied_sha256:
            raise RuntimeError("resumed database copy failed SHA-256 verification")
    else:
        database = (args.database or session / "rtabmap.db").resolve()
    initial = read_world_pose(args.robot_name)
    if args.initial_world_pose:
        initial["position_xyz"][:2] = [float(args.initial_world_pose[0]), float(args.initial_world_pose[1])]
        initial["yaw_rad"] = float(args.initial_world_pose[2])
        initial["source"] = "command-line override"
    extrinsic = read_camera_extrinsic(args.robot_name)
    odom_initial_pose = (
        f"{float(initial['position_xyz'][0]):.12g} "
        f"{float(initial['position_xyz'][1]):.12g} 0 0 0 "
        f"{float(initial['yaw_rad']):.12g}"
    )
    metadata = {
        "format": "world_anchored_rtabmap_mapping_v1",
        "scene": SCENE_NAME,
        "robot_name": args.robot_name,
        "database": str(database),
        "mapping_mode": "resume" if resume_source else "new",
        "resume": ({
            "source_database": str(resume_source),
            "source_sha256": source_sha256,
            "copied_database_sha256": copied_sha256,
            "source_database_modified": False,
        } if resume_source else None),
        "initial_world_pose": initial,
        "rgbd_odometry_initial_pose": odom_initial_pose,
        "initial_pose_contract": "x y z roll pitch yaw in the RTAB-Map odom frame",
        "post_mapping_world_to_map_calibration_required": True,
        "camera_extrinsic": extrinsic,
        "odom_source": "g1_omnipicker root truth",
        "visual_odom_topic": "/visual_odom",
        "wheel_odom_used": False,
        "truth_odom_used": True,
        "joint_pose_initialization_applied": False,
        "joint_pose_hold_source": "live OrcaLab state captured at motion-worker connection",
        "waypoint_file": str((session / DEFAULT_WAYPOINT_NAME).resolve()),
        "waypoint_request_file": str((session / "waypoint_request.json").resolve()),
        "waypoint_keys": "1=A 2=B 3=C; services /task1/record_waypoint_{a,b,c}",
        "occupancy_grid_parameters": {
            "from_depth": True,
            "range_max_m": 3.0,
            "max_ground_height_m": 0.05,
            "max_obstacle_height_m": args.grid_max_obstacle_height,
        },
    }
    (session / "mapping_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    processes: dict[str, subprocess.Popen] = {}
    reset_mapping_cleanup()
    install_mapping_cleanup(processes, session, env)
    try:
        # Existing camera bridge is only an RGB-D message acquisition stage.
        start_process("rgbd", ["/usr/bin/python3", str(SRC / "three_camera_rgbd_bridge.py"),
            "--head-only", "--vertical-fov", str(extrinsic["vertical_fov_deg"]),
            "--fps", str(args.rgbd_fps), "--start-index", str(5_000_000 + int(time.time()) % 1_000_000),
            "--output-dir", f"/tmp/world_anchored_rgbd_{stamp}",
            "--report", str(session / "rgbd_report.json")], env, session, processes)
        t = extrinsic["translation"]; q = extrinsic["quaternion_xyzw"]
        start_process("camera_tf", [ROS2, "run", "tf2_ros", "static_transform_publisher",
            "--x", str(t[0]), "--y", str(t[1]), "--z", str(t[2]),
            "--qx", str(q[0]), "--qy", str(q[1]), "--qz", str(q[2]), "--qw", str(q[3]),
            "--frame-id", "base_link", "--child-frame-id", extrinsic["camera_frame"]], env, session, processes)
        if not topic_ready("/camera/color/image_raw", env, 30.0,
                           "sensor_msgs/msg/Image"):
            raise RuntimeError("RGB-D color topic did not become ready")
        if not topic_ready("/camera/depth/image_raw", env, 30.0,
                           "sensor_msgs/msg/Image"):
            raise RuntimeError("RGB-D depth topic did not become ready")
        start_process("rgbd_sync", [ROS2, "run", "rtabmap_sync", "rgbd_sync", "--ros-args",
            "-p", "approx_sync:=false", "-p", "sync_queue_size:=30", "-p", "qos:=1",
            "-r", "rgb/image:=/camera/color/image_raw", "-r", "rgb/camera_info:=/camera/color/camera_info",
            "-r", "depth/image:=/camera/depth/image_raw", "-r", "depth/camera_info:=/camera/depth/camera_info",
            "-r", "rgbd_image:=/rtabmap/rgbd_image"], env, session, processes)
        start_process("rgbd_odometry", [ROS2, "run", "rtabmap_odom", "rgbd_odometry",
            "--Odom/ResetCountdown", "3", "--Vis/MinInliers", "20",
            "--Vis/PnPReprojError", "2.0", "--Reg/Force3DoF", "true",
            "--ros-args",
            "-p", "frame_id:=base_link", "-p", "odom_frame_id:=visual_odom",
            "-p", "publish_tf:=false",
            "-p", "initial_pose:=" + odom_initial_pose,
            "-p", "subscribe_rgb:=false", "-p", "subscribe_depth:=false",
            "-p", "subscribe_rgbd:=true", "-p", "topic_queue_size:=30",
            "-p", "qos:=1", "-p", "wait_for_transform:=0.2",
            "-r", "rgbd_image:=/rtabmap/rgbd_image",
            "-r", "odom:=/visual_odom", "-r", "odom_info:=/visual_odom_info"],
            env, session, processes)
        loop_closure_args = [] if args.disable_loop_closure else [
            "--RGBD/ProximityBySpace", "true",
            "--RGBD/ProximityMaxGraphDepth", "0",
        ]
        database_reset_args = [] if resume_source else ["-d"]
        waypoint_file = session / DEFAULT_WAYPOINT_NAME
        waypoint_request = session / "waypoint_request.json"
        live_command = args.live_command_file or (session / "live_wasd.json")
        if args.input == "live":
            live_command.write_text(
                '{"W":0,"A":0,"S":0,"D":0,"stop":false}\n', encoding="utf-8"
            )
        motion_command = [
            ORCALAB_PYTHON, str(Path(__file__).resolve()), "--motion-worker",
            "--robot-name", args.robot_name, "--input", args.input,
            "--speed", str(args.speed), "--turn-speed", str(args.turn_speed),
            "--camera-render-hz", str(args.camera_render_hz),
            "--linear-ramp-rate", str(args.linear_ramp_rate),
            "--steering-ramp-rate", str(args.steering_ramp_rate),
            "--probe-distance", str(args.probe_distance),
            "--probe-timeout", str(args.probe_timeout),
            "--probe-startup-delay", str(args.probe_startup_delay),
            "--loop-segment-length", str(getattr(args, "loop_segment_length", 1.5)),
            "--metadata", str(session / "driver_metadata.json"),
            "--live-command-file", str(live_command),
            "--waypoint-request-file", str(waypoint_request),
            "--publish-truth-odom",
        ]
        start_process("rtabmap", [ROS2, "run", "rtabmap_slam", "rtabmap",
            *database_reset_args,
            "--Optimizer/Strategy", "1",
            "--Reg/Force3DoF", "true", "--Grid/FromDepth", "true", "--Grid/3D", "false",
            "--Grid/RangeMax", "3.0", "--Grid/MaxGroundHeight", "0.05",
            "--Grid/MaxObstacleHeight", str(args.grid_max_obstacle_height),
            "--Mem/IncrementalMemory", "true",
            "--Mem/InitWMWithAllNodes", "true" if resume_source else "false",
            *loop_closure_args, "--ros-args",
            "-p", "frame_id:=base_link", "-p", "map_frame_id:=map", "-p", "publish_tf:=true",
            "-p", "subscribe_rgbd:=true", "-p", "subscribe_odom:=true",
            "-p", "subscribe_odom_info:=false",
            "-p", "approx_sync:=true", "-p", "sync_queue_size:=30",
            "-p", "odom_sensor_sync:=true",
            "-p", "subscribe_scan:=false", "-p", "database_path:=" + str(database),
            "-r", "rgbd_image:=/rtabmap/rgbd_image", "-r", "odom:=/odom",
            "-r", "map:=/map"], env, session, processes)
        start_process("driver", motion_command, env, session, processes)
        if not topic_ready("/odom", env, 30.0,
                           "nav_msgs/msg/Odometry"):
            raise RuntimeError("mapping truth odometry did not publish /odom")
        start_process("waypoint_recorder", [
            "/usr/bin/python3", str(SRC / "record_mapping_waypoints.py"),
            "--listen",
            "--session-dir", str(session),
            "--output", str(waypoint_file),
            "--request-file", str(waypoint_request),
            "--timeout", "30.0",
        ], env, session, processes)
        metadata["views"] = {
            "rviz2": bool(args.rviz or args.visualize),
            "rtabmapviz": bool(args.rtabmapviz or args.visualize),
        }
        metadata["status"] = "ready"
        (session / "mapping_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.rviz or args.visualize:
            start_process("rviz", ["rviz2", "-d", str(ROOT / "task1_rtabmap_mapping.rviz")],
                env, session, processes)
        if args.rtabmapviz or args.visualize:
            # rtabmap_viz is RTAB-Map's native 3D graph/cloud viewer. It
            # consumes the same RGB-D and truth-assisted odometry as RTAB-Map.
            start_process("rtabmapviz", [ROS2, "run", "rtabmap_viz", "rtabmap_viz",
                "--ros-args",
                "-p", "frame_id:=base_link", "-p", "odom_frame_id:=odom",
                "-p", "map_frame_id:=map", "-p", "subscribe_rgbd:=true",
                "-p", "subscribe_odom_info:=false", "-p", "subscribe_scan:=false",
                "-r", "rgbd_image:=/rtabmap/rgbd_image", "-r", "odom:=/odom",
                ], env, session, processes)
        processes["driver"].wait()
        if processes["driver"].returncode not in (0, None):
            raise RuntimeError(
                f"motion worker exited with status {processes['driver'].returncode}; "
                "see driver.log"
            )
    except KeyboardInterrupt:
        pass
    finally:
        map_saved = cleanup_mapping_session(processes, session, env, save_map=True)
        waypoint_file = session / DEFAULT_WAYPOINT_NAME
        metadata["map_saved"] = map_saved
        metadata["waypoints_recorded"] = waypoint_file.is_file()
        metadata["status"] = "succeeded" if map_saved else "stopped_without_map"
        (session / "mapping_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if metadata["map_saved"] else 1


if __name__ == "__main__":
    if "--motion-worker" in sys.argv:
        worker_args = sys.argv[1:]
        worker_args.remove("--motion-worker")
        raise SystemExit(motion_worker(worker_args))
    raise SystemExit(main())
