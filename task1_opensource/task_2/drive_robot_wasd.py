#!/usr/bin/env python3
"""WASD drive controller for industrial_collaborative_robot_1 in OrcaLab."""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass

import grpc
import numpy as np
import mujoco
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.protos import mjc_message_pb2, mjc_message_pb2_grpc


ROBOT_NAME = "industrial_collaborative_robot_1"
TIME_STEP = 0.001
FRAME_SKIP = 20


@dataclass(frozen=True)
class Actuator:
    name: str
    index: int
    joint_name: str
    ctrl_range: tuple[float, float]
    wheel: str


class SceneKeyboard:
    """Read W/A/S/D events from the OrcaLab viewport."""

    EVENTS = {
        "keyboard_key_alphanumeric_W": "W",
        "keyboard_key_alphanumeric_A": "A",
        "keyboard_key_alphanumeric_S": "S",
        "keyboard_key_alphanumeric_D": "D",
    }

    def __init__(self, address: str) -> None:
        self.channel = grpc.insecure_channel(address)
        self.stub = mjc_message_pb2_grpc.GrpcServiceStub(self.channel)
        self.request = mjc_message_pb2.GetKeyPressedEventsRequest()

    def get_state(self) -> dict[str, int]:
        state = {key: 0 for key in self.EVENTS.values()}
        for event in self.stub.GetKeyPressedEvents(self.request, timeout=1.0).events:
            key = self.EVENTS.get(event)
            if key is not None:
                state[key] = 1
        return state

    def close(self) -> None:
        self.channel.close()


class DesktopKeyboard:
    """Read physical desktop key press/release events; Esc requests exit."""

    def __init__(self) -> None:
        from pynput import keyboard

        self.keyboard = keyboard
        # P is consumed by the RTAB-Map mapping driver as a manual strict-loop
        # trigger.  The standalone WASD driver simply ignores it.
        self.state = {key: 0 for key in "WASDP"}
        self.stop_requested = False
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.start()

    def _set(self, key, pressed: int) -> None:
        try:
            char = key.char.upper()
        except (AttributeError, TypeError):
            return
        if char in self.state:
            self.state[char] = pressed

    def _press(self, key) -> None:
        if key == self.keyboard.Key.esc:
            self.stop_requested = True
        else:
            self._set(key, 1)

    def _release(self, key) -> None:
        self._set(key, 0)

    def get_state(self) -> dict[str, int]:
        if self.stop_requested:
            raise KeyboardInterrupt
        return self.state.copy()

    def close(self) -> None:
        self.listener.stop()
        self.listener.join(timeout=1.0)


def belongs_to_robot(name: str, robot: str) -> bool:
    return name == robot or name.startswith(robot + "_")


def bind_chassis(model, robot: str) -> tuple[list[Actuator], dict[str, Actuator]]:
    """Find the two front drive and four steering actuators in the live model."""
    drives: list[Actuator] = []
    steering: dict[str, Actuator] = {}
    for actuator_name, info in model.get_actuator_dict().items():
        joint_name = str(info.get("JointName", ""))
        if not (belongs_to_robot(actuator_name, robot) or belongs_to_robot(joint_name, robot)):
            continue
        match = re.search(r"wheel_(fl|fr|bl|br)_", actuator_name.lower())
        if match is None:
            continue
        ctrl_range = tuple(float(value) for value in info.get("CtrlRange", ()))
        if len(ctrl_range) != 2 or ctrl_range[0] == ctrl_range[1]:
            continue
        item = Actuator(actuator_name, model.actuator_name2id(actuator_name),
                        joint_name, ctrl_range, match.group(1))
        if actuator_name.endswith("_joint_mctrl") and item.wheel in {"fl", "fr"}:
            drives.append(item)
        elif actuator_name.endswith("_steer_joint_pctrl"):
            steering[item.wheel] = item
    if {item.wheel for item in drives} != {"fl", "fr"} or set(steering) != {"fl", "fr", "bl", "br"}:
        raise RuntimeError(
            "Live robot does not expose the expected FL/FR drive plus four steering actuators. "
            "Check that the correct scene is playing."
        )
    return drives, steering


def scaled_value(actuator: Actuator, normalized: float) -> float:
    low, high = actuator.ctrl_range
    return (low + high) / 2 + float(np.clip(normalized, -1, 1)) * (high - low) / 2


def snapshot_live_state(address: str) -> tuple[np.ndarray, np.ndarray]:
    """Read state before OrcaGymLocalEnv performs its unavoidable initial reset."""
    channel = grpc.insecure_channel(address)
    try:
        stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)
        response = stub.QueryAllQposQvelQacc(
            mjc_message_pb2.QueryAllQposQvelQaccRequest(), timeout=3.0
        )
        return np.asarray(response.qpos, dtype=float), np.asarray(response.qvel, dtype=float)
    finally:
        channel.close()


def restore_live_state(env: OrcaGymLocalEnv, qpos: np.ndarray, qvel: np.ndarray) -> None:
    if qpos.shape != (env.model.nq,) or qvel.shape != (env.model.nv,):
        raise RuntimeError(
            f"Scene changed while connecting: saved qpos/qvel={qpos.shape}/{qvel.shape}, "
            f"compiled={(env.model.nq,)}/{(env.model.nv,)}"
        )
    env.gym._mjData.qpos[:] = qpos
    env.gym._mjData.qvel[:] = qvel
    mujoco.mj_forward(env.gym._mjModel, env.gym._mjData)
    env.loop.run_until_complete(env.gym.set_qpos(qpos))
    env.loop.run_until_complete(env.gym.set_qvel(qvel))
    env.gym.mj_forward()
    env.gym.update_data()
    env.render()


def bind_pose_holds(env: OrcaGymLocalEnv, robot: str):
    """Capture every non-chassis scalar joint for exact per-frame pose locking."""
    model, data = env.gym._mjModel, env.gym._mjData
    pose_holds = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name.startswith(robot + "_") or "wheel_" in name or name.endswith("_free_joint"):
            continue
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        pose_holds.append((qadr, int(model.jnt_dofadr[joint_id]), float(data.qpos[qadr])))
    arm_count = sum("_arm_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
                    for jid in range(model.njnt)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").startswith(robot + "_"))
    if arm_count != 14:
        raise RuntimeError(f"Expected 14 arm joints, found {arm_count}")
    return pose_holds


def lock_pose(env: OrcaGymLocalEnv, pose_holds) -> None:
    """Use the same exact qpos pinning strategy as the project navigation scripts."""
    data = env.gym._mjData
    for qadr, vadr, target in pose_holds:
        data.qpos[qadr] = target
        data.qvel[vadr] = 0.0
    env.mj_forward()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default="127.0.0.1:50051")
    parser.add_argument("--robot", default=ROBOT_NAME)
    parser.add_argument("--speed", type=float, default=0.12,
                        help="normalized drive command, recommended 0.05 to 0.25")
    parser.add_argument("--turn-speed", type=float, default=0.40,
                        help="normalized steering command, recommended 0.2 to 0.7")
    parser.add_argument("--left-sign", type=float, choices=(-1, 1), default=1)
    parser.add_argument("--right-sign", type=float, choices=(-1, 1), default=1)
    parser.add_argument("--input", choices=("desktop", "orcalab"), default="desktop")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    env = None
    keyboard = None
    pose_holds = None
    controller_started = False
    try:
        saved_qpos, saved_qvel = snapshot_live_state(args.addr)
        env = OrcaGymLocalEnv(frame_skip=FRAME_SKIP, orcagym_addr=args.addr,
                              agent_names=[args.robot], time_step=TIME_STEP)
        restore_live_state(env, saved_qpos, saved_qvel)
        drives, steering = bind_chassis(env.model, args.robot)
        pose_holds = bind_pose_holds(env, args.robot)
        print("Bound chassis actuators:")
        for actuator in drives:
            print(f"  drive {actuator.wheel}: {actuator.name}, range={actuator.ctrl_range}")
        for actuator in steering.values():
            print(f"  steer {actuator.wheel}: {actuator.name}, range={actuator.ctrl_range}")
        if args.inspect:
            return 0

        keyboard = DesktopKeyboard() if args.input == "desktop" else SceneKeyboard(args.addr)
        controller_started = True
        print("W/S: forward/reverse; A/D: steer; release keys to stop and centre steering.")
        print("Desktop input: press Esc to exit. OrcaLab input: use Ctrl+C in this terminal.")
        while True:
            start = time.monotonic()
            state = keyboard.get_state()
            forward = (state["W"] - state["S"]) * args.speed
            turn = (state["A"] - state["D"]) * args.turn_speed
            lock_pose(env, pose_holds)
            control = np.zeros(env.model.nu)
            for actuator in drives:
                sign = args.left_sign if actuator.wheel == "fl" else args.right_sign
                control[actuator.index] = scaled_value(actuator, forward * sign)
            # Front and rear steer oppositely to create a vehicle-style turn.
            for wheel, actuator in steering.items():
                phase = 1.0 if wheel in {"fl", "fr"} else -1.0
                control[actuator.index] = scaled_value(actuator, turn * phase)
            env.do_simulation(control, FRAME_SKIP)
            lock_pose(env, pose_holds)
            env.render()
            delay = TIME_STEP * FRAME_SKIP - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("Stopping controller.")
    except grpc.RpcError as error:
        print(f"Cannot connect to the playing simulation at {args.addr}: {error.code().name}")
        return 2
    finally:
        if keyboard is not None:
            keyboard.close()
        if env is not None and controller_started:
            if pose_holds is not None:
                lock_pose(env, pose_holds)
                control = np.zeros(env.model.nu)
            else:
                control = np.zeros(env.model.nu)
            env.do_simulation(control, 1)
            env.render()
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
