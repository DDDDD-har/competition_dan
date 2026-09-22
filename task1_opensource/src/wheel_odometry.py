#!/usr/bin/env python3
"""Calibrated four-wheel-steering odometry for ``service_robot_1``.

The estimator is deliberately independent from MuJoCo truth pose.  It uses
wheel hinge qpos for pose increments, wheel hinge qvel for the reported body
twist, and steering hinge qpos for rolling directions.  All are sampled after
the owning OrcaGym environment calls ``do_simulation()``.
``MujocoWheelOdomPublisher`` is a thin ROS/MuJoCo adapter around the pure
``FourWheelOdometry`` core.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


WHEELS = ("fl", "fr", "bl", "br")


@dataclass(frozen=True)
class WheelOdomSample:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    rolling_rms_m: float
    dt: float


class FourWheelOdometry:
    """Integrate SE(2) motion from four rolling and steering joints."""

    def __init__(
        self,
        radius: float,
        positions: Mapping[str, tuple[float, float] | list[float]],
        encoder_signs: Mapping[str, float],
        lateral_weight: float = 0.1,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("wheel radius must be positive")
        if not 0.0 <= lateral_weight <= 1.0:
            raise ValueError("lateral_weight must be in [0, 1]")
        if set(positions) != set(WHEELS) or set(encoder_signs) != set(WHEELS):
            raise ValueError("positions and encoder signs must define fl/fr/bl/br")
        self.radius = float(radius)
        self.positions = {wheel: np.asarray(positions[wheel], dtype=float) for wheel in WHEELS}
        self.encoder_signs = {wheel: float(encoder_signs[wheel]) for wheel in WHEELS}
        self.lateral_weight = float(lateral_weight)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._previous_wheel: dict[str, float] | None = None
        self._previous_steer: dict[str, float] | None = None
        self._previous_time: float | None = None
        self.update_count = 0
        self.reset_count = 0
        self.max_rolling_rms_m = 0.0

    def reset(
        self,
        wheel_angles: Mapping[str, float],
        steer_angles: Mapping[str, float],
        sample_time: float,
        *,
        reset_pose: bool = True,
    ) -> None:
        self._validate_sample(wheel_angles, steer_angles)
        self._previous_wheel = {wheel: float(wheel_angles[wheel]) for wheel in WHEELS}
        self._previous_steer = {wheel: float(steer_angles[wheel]) for wheel in WHEELS}
        self._previous_time = float(sample_time)
        if reset_pose:
            self.x = self.y = self.yaw = 0.0
        self.reset_count += 1

    @staticmethod
    def _validate_sample(
        wheel_angles: Mapping[str, float], steer_angles: Mapping[str, float]
    ) -> None:
        if set(wheel_angles) != set(WHEELS) or set(steer_angles) != set(WHEELS):
            raise ValueError("wheel/steer samples must define fl/fr/bl/br")
        values = [*wheel_angles.values(), *steer_angles.values()]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("wheel/steer samples must be finite")

    def update(
        self,
        wheel_angles: Mapping[str, float],
        steer_angles: Mapping[str, float],
        sample_time: float,
        wheel_velocities: Mapping[str, float] | None = None,
    ) -> WheelOdomSample | None:
        self._validate_sample(wheel_angles, steer_angles)
        if wheel_velocities is not None:
            if set(wheel_velocities) != set(WHEELS):
                raise ValueError("wheel velocity samples must define fl/fr/bl/br")
            if not all(math.isfinite(float(value)) for value in wheel_velocities.values()):
                raise ValueError("wheel velocity samples must be finite")
        now = float(sample_time)
        if self._previous_wheel is None or self._previous_steer is None or self._previous_time is None:
            self.reset(wheel_angles, steer_angles, now)
            return None
        dt = now - self._previous_time
        # A scene reset moves simulation time backwards and replaces all qpos.
        # Re-baseline instead of integrating the reset jump as robot motion.
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            self.reset(wheel_angles, steer_angles, now, reset_pose=False)
            return None

        rolling_rows: list[list[float]] = []
        rolling_rhs: list[float] = []
        lateral_rows: list[list[float]] = []
        for wheel in WHEELS:
            x_i, y_i = self.positions[wheel]
            delta_q = self.encoder_signs[wheel] * (
                float(wheel_angles[wheel]) - self._previous_wheel[wheel]
            )
            steer = 0.5 * (float(steer_angles[wheel]) + self._previous_steer[wheel])
            c, s = math.cos(steer), math.sin(steer)
            rolling_rows.append([c, s, -y_i * c + x_i * s])
            rolling_rhs.append(self.radius * delta_q)
            lateral_rows.append([-s, c, y_i * s + x_i * c])

        a_roll = np.asarray(rolling_rows, dtype=float)
        b_roll = np.asarray(rolling_rhs, dtype=float)
        if self.lateral_weight > 0.0:
            a = np.vstack((a_roll, self.lateral_weight * np.asarray(lateral_rows, dtype=float)))
            b = np.concatenate((b_roll, np.zeros(4, dtype=float)))
        else:
            a, b = a_roll, b_roll
        delta, _residuals, rank, _singular = np.linalg.lstsq(a, b, rcond=None)
        if rank < 3 or not np.all(np.isfinite(delta)):
            self.reset(wheel_angles, steer_angles, now, reset_pose=False)
            return None
        dx, dy, dtheta = map(float, delta)

        # SE(2) exponential: the least-squares delta is expressed in the body
        # frame at the beginning of the interval.
        if abs(dtheta) < 1e-9:
            local_x, local_y = dx, dy
        else:
            a_theta = math.sin(dtheta) / dtheta
            b_theta = (1.0 - math.cos(dtheta)) / dtheta
            local_x = a_theta * dx - b_theta * dy
            local_y = b_theta * dx + a_theta * dy
        c0, s0 = math.cos(self.yaw), math.sin(self.yaw)
        self.x += c0 * local_x - s0 * local_y
        self.y += s0 * local_x + c0 * local_y
        self.yaw = math.atan2(math.sin(self.yaw + dtheta), math.cos(self.yaw + dtheta))

        rolling_rms = float(np.sqrt(np.mean((a_roll @ delta - b_roll) ** 2)))
        self.max_rolling_rms_m = max(self.max_rolling_rms_m, rolling_rms)
        self.update_count += 1
        self._previous_wheel = {wheel: float(wheel_angles[wheel]) for wheel in WHEELS}
        self._previous_steer = {wheel: float(steer_angles[wheel]) for wheel in WHEELS}
        self._previous_time = now
        if wheel_velocities is None:
            vx, vy, wz = dx / dt, dy / dt, dtheta / dt
        else:
            velocity_rhs = np.asarray([
                self.radius * self.encoder_signs[wheel]
                * float(wheel_velocities[wheel])
                for wheel in WHEELS
            ], dtype=float)
            if self.lateral_weight > 0.0:
                velocity_b = np.concatenate((velocity_rhs, np.zeros(4, dtype=float)))
            else:
                velocity_b = velocity_rhs
            velocity, _v_residuals, velocity_rank, _v_singular = np.linalg.lstsq(
                a, velocity_b, rcond=None
            )
            if velocity_rank < 3 or not np.all(np.isfinite(velocity)):
                vx, vy, wz = dx / dt, dy / dt, dtheta / dt
            else:
                vx, vy, wz = map(float, velocity)
        return WheelOdomSample(
            self.x, self.y, self.yaw, vx, vy, wz, rolling_rms, dt,
        )


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("wheel_radius_m", "wheel_positions_m", "encoder_sign",
                "drive_joints", "steer_joints")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"wheel calibration is missing: {', '.join(missing)}")
    return data


class MujocoWheelOdomPublisher:
    """Read MuJoCo joints and publish the canonical ROS odometry and TF."""

    def __init__(
        self,
        node,
        mj_model,
        mj_data,
        calibration_path: Path,
        *,
        odom_topic: str = "/odom",
        odom_frame: str = "odom",
        base_frame: str = "base_link",
        publish_tf: bool = True,
        lateral_weight: float = 0.1,
    ) -> None:
        # Lazy imports keep the pure estimator unit-testable without ROS or
        # MuJoCo installed in the test interpreter.
        import mujoco
        from nav_msgs.msg import Odometry
        from tf2_ros import TransformBroadcaster

        self._mujoco = mujoco
        self._Odometry = Odometry
        self.node = node
        self.model = mj_model
        self.data = mj_data
        self.odom_topic = odom_topic
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        calibration = load_calibration(calibration_path)
        robot_name = str(calibration["robot_name"])
        self.wheel_qadr: dict[str, int] = {}
        self.wheel_dadr: dict[str, int] = {}
        self.steer_qadr: dict[str, int] = {}
        for wheel in WHEELS:
            wheel_name = str(calibration["drive_joints"][wheel])
            steer_name = str(calibration["steer_joints"][wheel])
            wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
            steer_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, steer_name)
            if wheel_id < 0 or steer_id < 0:
                raise RuntimeError(
                    f"calibrated wheel joint missing for {robot_name}/{wheel}: "
                    f"{wheel_name}, {steer_name}"
                )
            self.wheel_qadr[wheel] = int(mj_model.jnt_qposadr[wheel_id])
            self.wheel_dadr[wheel] = int(mj_model.jnt_dofadr[wheel_id])
            self.steer_qadr[wheel] = int(mj_model.jnt_qposadr[steer_id])
        self.estimator = FourWheelOdometry(
            float(calibration["wheel_radius_m"]),
            calibration["wheel_positions_m"],
            calibration["encoder_sign"],
            lateral_weight=lateral_weight,
        )
        self.publisher = node.create_publisher(Odometry, odom_topic, 20)
        self.tf_broadcaster = TransformBroadcaster(node) if publish_tf else None
        self.last_sample: WheelOdomSample | None = None
        self.last_wheel_qpos: dict[str, float] = {}
        self.last_wheel_qvel: dict[str, float] = {}
        self.last_steer_qpos: dict[str, float] = {}
        self.publish_count = 0
        self._reset_from_data()

    def _joint_state(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        wheels = {wheel: float(self.data.qpos[address])
                  for wheel, address in self.wheel_qadr.items()}
        wheel_velocities = {wheel: float(self.data.qvel[address])
                            for wheel, address in self.wheel_dadr.items()}
        steering = {wheel: float(self.data.qpos[address])
                    for wheel, address in self.steer_qadr.items()}
        self.last_wheel_qpos = wheels.copy()
        self.last_wheel_qvel = wheel_velocities.copy()
        self.last_steer_qpos = steering.copy()
        return wheels, wheel_velocities, steering

    def _reset_from_data(self) -> None:
        wheels, _wheel_velocities, steering = self._joint_state()
        self.estimator.reset(wheels, steering, float(self.data.time))

    def publish(self, stamp) -> WheelOdomSample | None:
        from geometry_msgs.msg import TransformStamped

        wheels, wheel_velocities, steering = self._joint_state()
        sample = self.estimator.update(
            wheels, steering, float(self.data.time), wheel_velocities=wheel_velocities
        )
        if sample is None:
            return None
        self.last_sample = sample
        msg = self._Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = sample.x
        msg.pose.pose.position.y = sample.y
        msg.pose.pose.orientation.z = math.sin(sample.yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(sample.yaw * 0.5)
        msg.twist.twist.linear.x = sample.vx
        msg.twist.twist.linear.y = sample.vy
        msg.twist.twist.angular.z = sample.wz
        # The calibration is accurate to roughly 1%, while the arc tests show
        # small lateral slip.  Publish realistic non-zero uncertainty.
        pose_variance = max(2.5e-4, sample.rolling_rms_m ** 2)
        twist_variance = max(1.0e-3, (sample.rolling_rms_m / sample.dt) ** 2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = pose_variance
        msg.pose.covariance[35] = max(1.0e-3, 4.0 * pose_variance)
        msg.pose.covariance[14] = msg.pose.covariance[21] = msg.pose.covariance[28] = 1.0e6
        msg.twist.covariance[0] = msg.twist.covariance[7] = twist_variance
        msg.twist.covariance[35] = max(2.0e-3, 4.0 * twist_variance)
        msg.twist.covariance[14] = msg.twist.covariance[21] = msg.twist.covariance[28] = 1.0e6
        self.publisher.publish(msg)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = sample.x
            tf.transform.translation.y = sample.y
            tf.transform.rotation.z = msg.pose.pose.orientation.z
            tf.transform.rotation.w = msg.pose.pose.orientation.w
            self.tf_broadcaster.sendTransform(tf)
        self.publish_count += 1
        return sample

    def diagnostics(self) -> dict:
        return {
            "odom_topic": self.odom_topic,
            "odom_frame": self.odom_frame,
            "base_frame": self.base_frame,
            "publish_count": self.publish_count,
            "update_count": self.estimator.update_count,
            "reset_count": self.estimator.reset_count,
            "max_rolling_rms_m": self.estimator.max_rolling_rms_m,
            "pose_xy_yaw": [self.estimator.x, self.estimator.y, self.estimator.yaw],
            "wheel_qpos": self.last_wheel_qpos,
            "wheel_qvel": self.last_wheel_qvel,
            "steer_qpos": self.last_steer_qpos,
            "qpos_used_for_pose": True,
            "qvel_used_for_twist": True,
            "truth_pose_used": False,
        }
