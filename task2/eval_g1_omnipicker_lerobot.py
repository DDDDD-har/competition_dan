"""在 OrcaLab 中运行 G1 OmniPicker OpenPI 远程策略推理。"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import traceback

import cv2
import numpy as np
from yaml import Loader, load

def _southgrid_src() -> str:
    """脚本放在 task2 目录或 SouthGrid 源码树内时都能找到本地模块。"""
    here = os.path.dirname(os.path.realpath(__file__))
    candidates = [
        here,
        os.path.abspath(os.path.join(here, "../../..")),
        os.environ.get("SOUTHGRID_SRC", ""),
    ]
    for root in candidates:
        if root and os.path.isfile(os.path.join(root, "conf", "g1_omnipicker_conf.py")):
            return root
    return candidates[0]


project_root = _southgrid_src()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.log.orca_log import OrcaLog, get_orca_logger

from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_reverse_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    BUTTON_THREE_CAM_MAP,
    omnipicker_camera_map,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/eval_g1_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
_rel_task_config = os.path.join(base_dir, "../../dataCollection/common/example.yaml")
_DEFAULT_TASK_CONFIG = (
    _rel_task_config
    if os.path.isfile(_rel_task_config)
    else os.path.join(project_root, "examples/dataCollection/common/example.yaml")
)

orca_logger = get_orca_logger(
    name="EvalG1Lerobot",
    log_file="eval_g1_omnipicker_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# State/Action：左臂位置 3、左臂四元数 4、右臂位置 3、
# 右臂四元数 4、左右夹爪归一化值各 2；四元数顺序为 xyzw。
_L_GRIP_RANGES = agent_conf.gripper_l["actuator_ranges"]
_R_GRIP_RANGES = agent_conf.gripper_r["actuator_ranges"]


def _denorm_grip(norm_val: float, grip_range: tuple[float, float]) -> float:
    """将 [0,1] 归一化值反归一化回电机量程内的绝对值。"""
    lo, hi = float(grip_range[0]), float(grip_range[1])
    return float(np.clip(norm_val, 0.0, 1.0)) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# 四元数 / 位置保护（EEFDevice 与 parse_policy_action 共用）
# ---------------------------------------------------------------------------

def _safe_quat(quat: np.ndarray, fallback: np.ndarray | None) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.size != 4 or not np.all(np.isfinite(q)):
        q = np.zeros(4, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        if fallback is not None:
            fb = np.asarray(fallback, dtype=np.float32).reshape(-1)
            fn = float(np.linalg.norm(fb))
            if fn >= 1e-6:
                return (fb / fn).astype(np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _safe_pos(pos: np.ndarray, fallback: np.ndarray | None) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    if p.size != 3 or not np.all(np.isfinite(p)) or float(np.linalg.norm(p)) < 1e-4:
        if fallback is not None:
            fb = np.asarray(fallback, dtype=np.float32).reshape(-1)
            if fb.size == 3 and np.all(np.isfinite(fb)):
                return fb.astype(np.float32)
        return np.zeros(3, dtype=np.float32)
    return p.astype(np.float32)


# ---------------------------------------------------------------------------
# EEFDevice：将策略输出的末端动作实时转发给 OSC 控制器
# ---------------------------------------------------------------------------

class EEFDevice(AbstractDevice):
    """将策略输出的 18 维 action 原样转发给 OSC 双臂与夹爪。与官方推理脚本一致，不锁左臂。"""

    def __init__(
        self,
        l_arm=None,
        r_arm=None,
        l_grip=None,
        r_grip=None,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.l_pos_b = None if l_pos_b is None else np.asarray(l_pos_b, dtype=np.float32)
        self.l_quat_b = None if l_quat_b is None else np.asarray(l_quat_b, dtype=np.float32)
        self.r_pos_b = None if r_pos_b is None else np.asarray(r_pos_b, dtype=np.float32)
        self.r_quat_b = None if r_quat_b is None else np.asarray(r_quat_b, dtype=np.float32)
        self.l_grip_ctrl = None if l_grip_ctrl is None else np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        self.r_grip_ctrl = None if r_grip_ctrl is None else np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)

    def set_target(
        self,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        if l_pos_b is not None:
            self.l_pos_b = np.asarray(l_pos_b, dtype=np.float32)
        if l_quat_b is not None:
            self.l_quat_b = np.asarray(l_quat_b, dtype=np.float32)
        if r_pos_b is not None:
            self.r_pos_b = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self.r_quat_b = np.asarray(r_quat_b, dtype=np.float32)
        if l_grip_ctrl is not None:
            self.l_grip_ctrl = np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        if r_grip_ctrl is not None:
            self.r_grip_ctrl = np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)

    def update(self):
        if self.l_arm is not None and self.l_pos_b is not None and self.l_quat_b is not None:
            self.l_arm.update_action_position(self.l_pos_b)
            self.l_arm.update_action_axisangle(self.l_quat_b)
        if self.r_arm is not None and self.r_pos_b is not None and self.r_quat_b is not None:
            self.r_arm.update_action_position(self.r_pos_b)
            self.r_arm.update_action_axisangle(self.r_quat_b)
        if self.l_grip is not None and self.l_grip_ctrl is not None:
            self.l_grip.update_ctrl(self.l_grip_ctrl)
        if self.r_grip is not None and self.r_grip_ctrl is not None:
            self.r_grip.update_ctrl(self.r_grip_ctrl)


# ---------------------------------------------------------------------------
# Action 工具
# ---------------------------------------------------------------------------

def parse_policy_action(raw_action: np.ndarray, fallback: dict | None = None) -> dict:
    """将 18 维策略输出拆分为末端位姿 + 归一化夹爪 dict。

    夹爪保留归一化 [0,1]，施加给电机时再反归一化（见 action_dict_for_apply）。
    四元数若范数为 0 / NaN，则回退到 fallback 或单位四元数。
    位置若全零 / NaN，则回退到 fallback，避免把手臂打到原点。
    """
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size < 18:
        raise ValueError(f"Expected at least 18 action dims, got {action.size}")
    fb = fallback or {}
    return {
        "l_pos_b":          _safe_pos(action[0:3], fb.get("l_pos_b")),
        "l_quat_b":         _safe_quat(action[3:7], fb.get("l_quat_b")),
        "r_pos_b":          _safe_pos(action[7:10], fb.get("r_pos_b")),
        "r_quat_b":         _safe_quat(action[10:14], fb.get("r_quat_b")),
        "l_grip_inner_norm": float(np.clip(np.nan_to_num(action[14], nan=0.5), 0.0, 1.0)),
        "l_grip_outer_norm": float(np.clip(np.nan_to_num(action[15], nan=0.5), 0.0, 1.0)),
        "r_grip_inner_norm": float(np.clip(np.nan_to_num(action[16], nan=0.5), 0.0, 1.0)),
        "r_grip_outer_norm": float(np.clip(np.nan_to_num(action[17], nan=0.5), 0.0, 1.0)),
    }


def action_dict_for_apply(action_dict: dict) -> dict:
    """把归一化 [0,1] 的夹爪值反归一化为电机绝对值，位姿原样透传。

    夹爪反归一化公式（与 G1OmniPickerLeRobotStorage.build_state 正向归一化一致）：
        val = norm * (hi - lo) + lo
    默认量程 (-1, 2)，即 val = norm * 3 - 1。
    """
    l_inner = _denorm_grip(action_dict["l_grip_inner_norm"], _L_GRIP_RANGES[0])
    l_outer = _denorm_grip(action_dict["l_grip_outer_norm"], _L_GRIP_RANGES[1])
    r_inner = _denorm_grip(action_dict["r_grip_inner_norm"], _R_GRIP_RANGES[0])
    r_outer = _denorm_grip(action_dict["r_grip_outer_norm"], _R_GRIP_RANGES[1])
    return {
        "l_pos_b":    np.asarray(action_dict["l_pos_b"],  dtype=np.float32).copy(),
        "l_quat_b":   np.asarray(action_dict["l_quat_b"], dtype=np.float32).copy(),
        "r_pos_b":    np.asarray(action_dict["r_pos_b"],  dtype=np.float32).copy(),
        "r_quat_b":   np.asarray(action_dict["r_quat_b"], dtype=np.float32).copy(),
        "l_grip_ctrl": np.array([l_inner, l_outer], dtype=np.float32),
        "r_grip_ctrl": np.array([r_inner, r_outer], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# 相机观测构建器 & 策略运行器（与青龙版本相同，策略通信协议无差异）
# ---------------------------------------------------------------------------

class CameraObservationBuilder:
    """从 WebSocket 内存流取图，与采集时 capture_frame_images 逻辑完全一致。"""

    def __init__(
        self,
        cameras: dict,
        camera_name_map: dict[str, str],
        target_hw: tuple = (480, 640),
    ):
        self.cameras = cameras
        self.camera_name_map = camera_name_map
        self.target_hw = target_hw

    def build_images(self) -> dict:
        H, W = self.target_hw
        images = {}
        for env_camera_name, policy_camera_name in self.camera_name_map.items():
            cam = self.cameras.get(env_camera_name)
            if cam is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                try:
                    frame, _ = cam.get_frame(format="rgb24")
                    if frame is None or frame.size == 0:
                        rgb = np.zeros((H, W, 3), dtype=np.uint8)
                    else:
                        if frame.shape[0] != H or frame.shape[1] != W:
                            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((H, W, 3), dtype=np.uint8)
            images[policy_camera_name] = rgb  # HWC uint8，与 OpenPI / 采集文档一致
        return images


class OpenPIPolicyRunner:
    """封装 openpi_client WebSocket 策略调用。"""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        camera_name_map: dict[str, str],
        cameras: dict,
        target_hw: tuple = (480, 640),
        use_images: bool = True,
    ):
        from openpi_client import websocket_client_policy

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.prompt = prompt
        self.use_images = use_images
        self.cam_builder = (
            CameraObservationBuilder(
                cameras=cameras,
                camera_name_map=camera_name_map,
                target_hw=target_hw,
            )
            if use_images
            else None
        )

    def build_observation(self, state: np.ndarray) -> dict:
        images = self.cam_builder.build_images() if self.use_images else {}
        return {"state": state, "images": images, "prompt": self.prompt}

    def infer_action_chunk(self, state: np.ndarray) -> np.ndarray:
        observation = self.build_observation(state)
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < 18:
            raise ValueError(f"Expected policy action dim >= 18, got {actions.shape}")
        return actions


# ---------------------------------------------------------------------------
# 相机预热
# ---------------------------------------------------------------------------

def warmup_camera_capture(manager, env, device, warmup_action: dict, warmup_steps: int = 5):
    for _ in range(max(0, warmup_steps)):
        device.set_target(**warmup_action)
        action = manager.run_controllers()
        env.step(action)
        env.render()
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# env / controller 构建工具
# ---------------------------------------------------------------------------

def build_default_joint_values() -> dict:
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


def create_arm(env, arm_conf):
    ctrl_names = [env.actuator(name) for name in arm_conf["motors_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_reverse_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85Reverse.ControllerType.DATA,
    )


def build_initial_action_from_state(state: np.ndarray) -> dict:
    """从首帧 18 维观测状态构造初始末端目标。"""
    return parse_policy_action(state)


def _head_camera_name(camera_map: dict) -> str | None:
    for env_name, (lerobot_key, _port) in camera_map.items():
        if lerobot_key == "cam_head" or "head" in env_name:
            return env_name
    return None


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned.strip("_") or "eval"


_PROMPT_BUTTON_SITES = {
    "红": "Group_Static_ElectricalCabinet_button01_site",
    "绿": "Group_Static_ElectricalCabinet_button02_site",
    "蓝": "Group_Static_ElectricalCabinet_button03_site",
    "黄": "Group_Static_ElectricalCabinet_button04_site",
}
_PROMPT_BUTTON_JOINTS = {
    "红": "Group_Static_ElectricalCabinet_button01_joint",
    "绿": "Group_Static_ElectricalCabinet_button02_joint",
    "蓝": "Group_Static_ElectricalCabinet_Button03_joint",
    "黄": "Group_Static_ElectricalCabinet_Button04_joint",
}
PRESS_DISP_MIN = 0.0005
TOUCH_BTN_Q_MIN = 0.001  # 仅用于日志：按钮关节位移 ≥1mm 记为碰到


def _button_site_for_prompt(prompt: str) -> str | None:
    for key, site in _PROMPT_BUTTON_SITES.items():
        if key in prompt:
            return site
    return None


def _button_joint_for_prompt(prompt: str) -> str | None:
    for key, joint in _PROMPT_BUTTON_JOINTS.items():
        if key in prompt:
            return joint
    return None


def _resolve_joint_name(env, key: str) -> str | None:
    if not key:
        return None
    candidates = [key]
    try:
        prefixed = env.joint(key)
    except Exception:
        prefixed = None
    if prefixed:
        candidates.append(prefixed)
    swapped = (
        key.replace("button", "Button")
        if "button" in key
        else key.replace("Button", "button")
    )
    if swapped != key:
        candidates.append(swapped)
    for name in candidates:
        try:
            data = env.query_joint_qpos([name])
            if name in data:
                return name
        except Exception:
            continue
    return None


def _joint_abs(env, name: str) -> float | None:
    try:
        q = env.query_joint_qpos([name])[name]
        return float(np.asarray(q, dtype=np.float64).reshape(-1)[0])
    except Exception:
        return None


def _resolve_site_name(env, key: str) -> str | None:
    candidates = [key, f"g1_omnipicker_{key}"]
    for name in candidates:
        try:
            data = env.query_site_pos_and_quat([name])
            if name in data:
                return name
        except Exception:
            continue
    return None


def _site_distance(env, a: str, b: str) -> float | None:
    try:
        data = env.query_site_pos_and_quat([a, b])
        return float(np.linalg.norm(
            np.asarray(data[a]["xpos"], dtype=np.float64)
            - np.asarray(data[b]["xpos"], dtype=np.float64)
        ))
    except Exception:
        return None


def _open_head_video_writer(path: str, hw: tuple[int, int], fps: float) -> cv2.VideoWriter:
    height, width = int(hw[0]), int(hw[1])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建头部相机视频: {path}")
    return writer


def _write_head_frame(writer: cv2.VideoWriter | None, cameras: dict, head_name: str | None) -> bool:
    if writer is None or head_name is None:
        return False
    cam = cameras.get(head_name)
    if cam is None:
        return False
    frame, _ = cam.get_frame(format="rgb24")
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker OpenPI 远程策略推理评估"
    )
    parser.add_argument("--task_config", type=str, default=_DEFAULT_TASK_CONFIG,
                        help="场景配置 YAML（默认 example.yaml）")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost", help="策略服务器主机")
    parser.add_argument("--port", type=int, default=8010, help="策略服务器端口")
    parser.add_argument("--prompt", type=str, default="按红色按钮",
                        help="任务语言描述（必须与训练时一致）")
    parser.add_argument("--sleep", action="store_true", help="按 real_time_step 节奏运行")
    parser.add_argument("--max_steps", type=int, default=500, help="每集最大控制步数")
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="每个推理 action 重复执行的控制步数",
    )
    parser.add_argument("--episodes", type=int, default=1, help="评估集数")
    parser.add_argument("--camera_warmup_steps", type=int, default=10,
                        help="每集推理前相机预热步数（默认 10）")
    parser.add_argument("--no_images", action="store_true",
                        help="跳过相机采图，发送空图（仅用 state 的策略）")
    parser.add_argument("--no_preview", action="store_true", help="不显示相机实时预览小窗口")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=os.path.join(base_dir, "eval_videos"),
        help="头部相机评测视频落盘目录",
    )
    parser.add_argument("--video_fps", type=float, default=20.0, help="头部相机录像帧率")
    parser.add_argument("--no_head_video", action="store_true", help="不录头部相机视频")
    parser.add_argument(
        "--enable_wrist_l",
        action="store_true",
        help="改用文档旧名 camera_wrist_l_color:7070 / camera_wrist_r_color:7080",
    )
    parser.add_argument(
        "--button_three_cam",
        action="store_true",
        help="使用按钮三路（默认已启用）：camera_head_color:7090 / camera_left_color:7080 / camera_right_color:7070",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    orca_logger.info(
        f"infer: full_chunk repeat={args.action_repeat} "
        f"max_steps={args.max_steps} episodes={args.episodes}"
    )

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    # storage 仅用于 obs_callback 与 build_state，不落盘。
    storage = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_eval_g1_scratch")

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=build_default_joint_values(),
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    manager.set_task(EmptyTask(env))
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE

    l_arm  = create_arm(env, agent_conf.l_arm)
    r_arm  = create_arm(env, agent_conf.r_arm)
    l_grip = create_gripper(env, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    camera_map = (
        omnipicker_camera_map(enable_wrist_l=True)
        if args.enable_wrist_l
        else dict(BUTTON_THREE_CAM_MAP)
    )
    # camera_name_map：env 相机传感器名 → 策略观测键名（与采集数据集一致）
    camera_name_map: dict[str, str] = {
        env_name: lerobot_key
        for env_name, (lerobot_key, _port) in camera_map.items()
    }
    orca_logger.info(
        f"推理相机: {list(camera_name_map.values())} "
        + ("（旧名 wrist_l/r 7070/7080）" if args.enable_wrist_l
           else "（按钮三路 head/left/right 7090/7080/7070）")
    )

    _need_cameras = (not args.no_images) or (not args.no_preview) or (not args.no_head_video)
    _head_cam_name = _head_camera_name(camera_map)
    _video_dir = os.path.abspath(args.video_dir)
    _shared_cameras: dict = {}
    _target_hw: tuple = DEFAULT_HW
    _preview_ready: bool = False
    _PREVIEW_W, _PREVIEW_H = 320, 240
    _PREVIEW_CAMS = list(camera_map.keys())
    policy_runner: OpenPIPolicyRunner | None = None
    device: EEFDevice | None = None

    _TPROF = {"ctrl": 0.0, "step": 0.0, "render": 0.0, "preview": 0.0, "n": 0}

    _video_started = False
    _head_writers: list[cv2.VideoWriter] = []
    try:
        episode_results: list[bool] = []
        jobs = [(args.prompt, ep) for ep in range(args.episodes)]

        for job_index, (current_prompt, episode_index) in enumerate(jobs):
            orca_logger.info(
                f"=== Episode {episode_index + 1}/{args.episodes} "
                f"prompt={current_prompt!r} ({job_index + 1}/{len(jobs)}) ==="
            )

            env.reset()
            time.sleep(0.1)

            if not manager.update_scene():
                orca_logger.error("update_scene 失败，退出")
                return

            env.set_default_joint_values(build_default_joint_values())
            env.mj_forward()
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            for controller in manager.controllers:
                controller.reset()
            env.render()
            time.sleep(0.05)

            # 从首帧观测状态初始化末端目标。
            _init_obs = storage.obs_callback(env)
            _init_state = storage.build_state(_init_obs)
            _init_action = build_initial_action_from_state(_init_state)
            _init_action_apply = action_dict_for_apply(_init_action)
            orca_logger.info(
                f"init_state dim={_init_state.size} "
                f"Lpos={_init_state[0:3]} Lquat={_init_state[3:7]} "
                f"|Lq|={float(np.linalg.norm(_init_state[3:7])):.3f} "
                f"Rpos={_init_state[7:10]} Rquat={_init_state[10:14]} "
                f"|Rq|={float(np.linalg.norm(_init_state[10:14])):.3f}"
            )

            if device is None:
                device = EEFDevice(
                    l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                    **_init_action_apply,
                )
                manager.set_device(device)
            else:
                device.set_target(**_init_action_apply)

            # 首集：场景就绪后启动相机内存流并连接策略服务器
            if job_index == 0:
                if _need_cameras:
                    try:
                        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                        env.begin_save_video(STREAM_TRIGGER_PATH)
                        _video_started = True
                        _shared_cameras = bring_up_cameras(
                            camera_map, port_timeout=30.0, frame_timeout=30.0
                        )
                        _target_hw = probe_camera_hw(_shared_cameras, camera_map)
                        orca_logger.info(
                            f"内存流相机已就绪（{len(_shared_cameras)} 路），分辨率={_target_hw}"
                        )
                        if not args.no_preview:
                            _n_cams = len(_shared_cameras)
                            cv2.namedWindow("eval-preview", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("eval-preview", _PREVIEW_W * max(_n_cams, 1), _PREVIEW_H)
                            _preview_ready = True
                            orca_logger.info("预览窗口已创建，按 q 提前结束当前 episode")
                    except Exception as _e:
                        orca_logger.warning(f"相机启动失败，策略将使用全黑图: {_e}")
                        _shared_cameras = {}

                policy_runner = OpenPIPolicyRunner(
                    host=args.host,
                    port=args.port,
                    prompt=current_prompt,
                    camera_name_map=camera_name_map,
                    cameras=_shared_cameras,
                    target_hw=_target_hw,
                    use_images=not args.no_images,
                )
                orca_logger.info(f"已连接策略服务器: {args.host}:{args.port}")
                orca_logger.info(f"策略元数据: {policy_runner.metadata}")

            if policy_runner is not None:
                policy_runner.prompt = current_prompt
            orca_logger.info(f"Prompt: {current_prompt}")

            if not args.no_images:
                warmup_camera_capture(
                    manager, env, device,
                    _init_action_apply,
                    args.camera_warmup_steps,
                )

            head_writer: cv2.VideoWriter | None = None
            head_video_path: str | None = None
            if not args.no_head_video:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_tag = _safe_filename(current_prompt)
                head_video_path = os.path.join(
                    _video_dir, f"{prompt_tag}_ep{episode_index:02d}_{stamp}_cam_head.mp4"
                )
                try:
                    if not _shared_cameras or _head_cam_name is None:
                        raise RuntimeError("头部相机未就绪，无法录像")
                    head_hw = _target_hw
                    _head_cam = _shared_cameras.get(_head_cam_name)
                    if _head_cam is not None:
                        _hf, _ = _head_cam.get_frame(format="rgb24")
                        if _hf is not None and getattr(_hf, "size", 0) > 0:
                            head_hw = (_hf.shape[0], _hf.shape[1])
                    head_writer = _open_head_video_writer(head_video_path, head_hw, args.video_fps)
                    _head_writers.append(head_writer)
                    orca_logger.info(f"头部相机录像: {head_video_path}")
                except Exception as _ve:
                    orca_logger.warning(f"头部相机录像未启动: {_ve}")
                    head_writer = None

            step = 0
            truncated = False
            head_frames = 0
            ee_site = _resolve_site_name(env, agent_conf.r_arm["ee_site_name"])
            btn_site_key = _button_site_for_prompt(current_prompt)
            btn_site = _resolve_site_name(env, btn_site_key) if btn_site_key else None
            btn_joint = _resolve_joint_name(env, _button_joint_for_prompt(current_prompt) or "")
            min_btn_dist: float | None = None
            btn_q0 = _joint_abs(env, btn_joint) if btn_joint else None
            max_btn_q = 0.0
            btn_x0 = None
            max_btn_disp = 0.0
            if ee_site and btn_site:
                orca_logger.info(f"按钮观测: ee={ee_site} btn={btn_site} joint={btn_joint}")
            else:
                orca_logger.warning(f"按钮观测未解析: ee={ee_site} btn_key={btn_site_key}")

            while step < args.max_steps and not truncated:
                state = storage.build_state(storage.obs_callback(env))
                action_chunk = policy_runner.infer_action_chunk(state)
                if step == 0:
                    orca_logger.info(f"action_chunk_len={len(action_chunk)} will_exec=all")

                for model_action in action_chunk:
                    if step >= args.max_steps or truncated:
                        break

                    if step == 0:
                        _a0 = np.asarray(model_action, dtype=np.float32).reshape(-1)
                        orca_logger.info(
                            f"policy first action dim={_a0.size} finite={bool(np.isfinite(_a0[:18]).all())} "
                            f"vals={np.array2string(_a0[:18], precision=4, suppress_small=True)}"
                        )
                        if not np.isfinite(_a0[:18]).all():
                            orca_logger.error("策略第一帧含 NaN/Inf，本步保持上一帧位姿")
                    parsed_action = parse_policy_action(
                        model_action,
                        fallback={
                            "l_pos_b": device.l_pos_b,
                            "l_quat_b": device.l_quat_b,
                            "r_pos_b": device.r_pos_b,
                            "r_quat_b": device.r_quat_b,
                        },
                    )
                    device.set_target(**action_dict_for_apply(parsed_action))

                    for _ in range(args.action_repeat):
                        if step >= args.max_steps or truncated:
                            break

                        start_time = time.time()
                        _pt0 = time.perf_counter()
                        action = manager.run_controllers()
                        _pt1 = time.perf_counter()
                        _, _, _, truncated, _ = env.step(action)
                        _pt2 = time.perf_counter()
                        env.render()
                        _pt3 = time.perf_counter()

                        _d = None
                        if ee_site and btn_site:
                            _d = _site_distance(env, ee_site, btn_site)
                            if _d is not None and (min_btn_dist is None or _d < min_btn_dist):
                                min_btn_dist = _d
                            try:
                                _btn = np.asarray(
                                    env.query_site_pos_and_quat([btn_site])[btn_site]["xpos"],
                                    dtype=np.float64,
                                )
                                if btn_x0 is None:
                                    btn_x0 = _btn.copy()
                                max_btn_disp = max(
                                    max_btn_disp, float(np.linalg.norm(_btn - btn_x0))
                                )
                            except Exception:
                                pass
                        if btn_joint is not None:
                            _q = _joint_abs(env, btn_joint)
                            if _q is not None:
                                if btn_q0 is None:
                                    btn_q0 = _q
                                max_btn_q = max(max_btn_q, abs(_q - btn_q0))

                        if _write_head_frame(head_writer, _shared_cameras, _head_cam_name):
                            head_frames += 1

                        # 实时预览（复用同一套内存流相机）
                        if _shared_cameras and _preview_ready:
                            try:
                                frames = []
                                for _cn in _PREVIEW_CAMS:
                                    _cam = _shared_cameras.get(_cn)
                                    if _cam is not None:
                                        _f, _ = _cam.get_frame(format="rgb24")
                                        if _f is not None and _f.size > 0:
                                            _f = cv2.resize(_f, (_PREVIEW_W, _PREVIEW_H))
                                            frames.append(cv2.cvtColor(_f, cv2.COLOR_RGB2BGR))
                                if frames:
                                    cv2.imshow("eval-preview", np.concatenate(frames, axis=1))
                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        truncated = True
                            except Exception:
                                pass

                        _pt4 = time.perf_counter()
                        _TPROF["ctrl"]    += _pt1 - _pt0
                        _TPROF["step"]    += _pt2 - _pt1
                        _TPROF["render"]  += _pt3 - _pt2
                        _TPROF["preview"] += _pt4 - _pt3
                        _TPROF["n"] += 1

                        if _TPROF["n"] % 50 == 0:
                            _n = _TPROF["n"]
                            _total = (
                                _TPROF["ctrl"] + _TPROF["step"]
                                + _TPROF["render"] + _TPROF["preview"]
                            )
                            orca_logger.info(
                                f"[PROF] n={_n}  "
                                f"ctrl={_TPROF['ctrl']/_n*1000:.1f}ms  "
                                f"env.step={_TPROF['step']/_n*1000:.1f}ms  "
                                f"render={_TPROF['render']/_n*1000:.1f}ms  "
                                f"preview={_TPROF['preview']/_n*1000:.1f}ms  "
                                f"| total≈{_total/_n*1000:.1f}ms"
                            )

                        _lp = device.l_pos_b if device.l_pos_b is not None else np.zeros(3)
                        _rp = device.r_pos_b if device.r_pos_b is not None else np.zeros(3)
                        _lg = device.l_grip_ctrl.tolist() if device.l_grip_ctrl is not None else [0, 0]
                        _rg = device.r_grip_ctrl.tolist() if device.r_grip_ctrl is not None else [0, 0]
                        _d_txt = f"{_d:.3f}" if _d is not None else "n/a"
                        orca_logger.info(
                            f"step={step:04d}/{args.max_steps}  "
                            f"cmd_L=[{_lp[0]:+.3f},{_lp[1]:+.3f},{_lp[2]:+.3f}]  "
                            f"cmd_R=[{_rp[0]:+.3f},{_rp[1]:+.3f},{_rp[2]:+.3f}]  "
                            f"grip_L=[{_lg[0]:.3f},{_lg[1]:.3f}]  "
                            f"grip_R=[{_rg[0]:.3f},{_rg[1]:.3f}]  "
                            f"ee_btn={_d_txt}  "
                            f"btn_disp={max_btn_disp:.5f}  "
                            f"btn_q={max_btn_q:.5f}"
                        )

                        step += 1
                        if truncated:
                            break

                        if args.sleep:
                            remain = manager.real_time_step - (time.time() - start_time)
                            if remain > 0:
                                time.sleep(remain)

            if head_writer is not None:
                head_writer.release()
                orca_logger.info(
                    f"头部相机视频已落盘: {head_video_path}  frames={head_frames}"
                )

            completed = not truncated
            episode_results.append(completed)
            dist_txt = f"{min_btn_dist:.4f}m" if min_btn_dist is not None else "n/a"
            pressed = max_btn_q >= PRESS_DISP_MIN or max_btn_disp >= PRESS_DISP_MIN
            touched = (
                max_btn_q >= TOUCH_BTN_Q_MIN
                or max_btn_disp >= TOUCH_BTN_Q_MIN
            )
            orca_logger.info(
                f"[{'done' if completed else 'stopped'}] "
                f"Episode {episode_index + 1} {current_prompt} finished: steps={step}  "
                f"truncated={truncated}  "
                f"min_ee_btn={dist_txt}  "
                f"max_btn_disp={max_btn_disp:.5f}  max_btn_q={max_btn_q:.5f}  "
                f"touched={touched}  pressed={pressed}"
            )

        done_count = sum(1 for ok in episode_results if ok)
        orca_logger.info(f"全部 {len(episode_results)} 集完成: {done_count} 集完整跑完")

    finally:
        for _w in _head_writers:
            try:
                _w.release()
            except Exception:
                pass
        if _shared_cameras:
            close_cameras(_shared_cameras)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if _video_started:
            try:
                env.stop_save_video()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    _rc = 0
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
        _rc = 130
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        _rc = 1
    finally:
        orca_logger.info("Exiting program")
        os._exit(_rc)
