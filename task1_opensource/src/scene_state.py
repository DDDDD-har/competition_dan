"""Small scene-state adapter used by the Task 1 safety-map audit."""

from __future__ import annotations

import grpc
import mujoco
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.protos import mjc_message_pb2, mjc_message_pb2_grpc


def snapshot_live_state(address: str) -> tuple[np.ndarray, np.ndarray]:
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
