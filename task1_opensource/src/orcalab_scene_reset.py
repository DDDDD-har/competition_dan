#!/usr/bin/env python3
"""Shared OrcaLab scene reset helper for current Task 1 launchers."""

from __future__ import annotations

import grpc
import numpy as np
from orca_gym.protos import mjc_message_pb2, mjc_message_pb2_grpc

from task1_scene_contract import ROBOT_HOLDER_BODY, ROBOT_NAME


def _holder_xy(env, robot_name: str) -> tuple[str, list[float]]:
    bodies = env.model.get_body_dict()
    holder_name = f"{robot_name}_robot_holder1"
    if holder_name not in bodies:
        holder_name = next(
            (name for name in bodies if name.endswith("_robot_holder1")),
            "",
        )
    if not holder_name:
        raise RuntimeError("OrcaGym reset verification could not find robot holder body")
    body_id = int(bodies[holder_name]["ID"])
    return holder_name, [float(v) for v in env.gym._mjData.xpos[body_id, :2]]


def restart_current_scene(
    address: str,
    timeout: float,
    robot_name: str = ROBOT_NAME,
) -> None:
    """Reset the currently open OrcaLab scene through the remote editor API."""
    channel = grpc.insecure_channel(address)
    stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)
    env = None
    try:
        model = stub.QueryModelInfo(mjc_message_pb2.QueryModelInfoRequest(), timeout=timeout)
        stub.SetSimulationState(
            mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.PAUSED),
            timeout=timeout,
        )
        stub.LoadInitialFrame(mjc_message_pb2.LoadInitialFrameRequest(), timeout=timeout)
        reset_state = stub.QueryAllQposQvelQacc(
            mjc_message_pb2.QueryAllQposQvelQaccRequest(), timeout=timeout
        )
        qpos = np.asarray(reset_state.qpos, dtype=float)
        qvel = np.asarray(reset_state.qvel, dtype=float)

        from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

        env = OrcaGymLocalEnv(
            frame_skip=1,
            orcagym_addr=address,
            agent_names=[robot_name],
            time_step=0.001,
        )
        if env.gym.data.qpos.shape != qpos.shape or env.gym.data.qvel.shape != qvel.shape:
            raise RuntimeError("OrcaGym model dimensions differ from reset scene state")
        env.gym._mjData.qpos[:] = qpos
        env.gym._mjData.qvel[:] = qvel
        env.gym.mj_forward()
        env.gym.update_data()
        env.init_qpos_qvel()
        env.set_time_step(env.time_step)

        holder_name, verified_xy = _holder_xy(env, robot_name)
        if holder_name != ROBOT_HOLDER_BODY and robot_name == ROBOT_NAME:
            raise RuntimeError(
                f"expected holder {ROBOT_HOLDER_BODY}, found {holder_name}"
            )

        stub.SetSimulationState(
            mjc_message_pb2.SetSimulationStateRequest(state=mjc_message_pb2.RUNNING),
            timeout=timeout,
        )
        resumed = stub.QueryAllQposQvelQacc(
            mjc_message_pb2.QueryAllQposQvelQaccRequest(), timeout=timeout
        )
        env.gym._mjData.qpos[:] = np.asarray(resumed.qpos, dtype=float)
        env.gym._mjData.qvel[:] = np.asarray(resumed.qvel, dtype=float)
        env.gym.mj_forward()
        _, after_xy = _holder_xy(env, robot_name)
        if max(abs(a - b) for a, b in zip(verified_xy, after_xy)) > 1e-3:
            raise RuntimeError(
                f"scene reset state changed during resume: before={verified_xy}, after={after_xy}"
            )

        print(
            f"Current scene reset through OrcaLab LoadInitialFrame + OrcaGym verification; "
            f"robot={robot_name}, holder={holder_name}, nbody={model.nbody}, "
            f"ngeom={model.ngeom}, world_xy={verified_xy}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        channel.close()
