"""Linear S-curve limits and speed-profile audit for the chassis bridge."""

from __future__ import annotations

import math

LINEAR_ACCEL_LIMIT = 1.2
LINEAR_JERK_LIMIT = 2.0
JERK_DEADBAND_V = 0.02
JERK_ACCEPTANCE_SLACK = 1e-3


def required_speed_change_distance(
    start_speed: float,
    end_speed: float,
    accel_limit: float = LINEAR_ACCEL_LIMIT,
    jerk_limit: float = LINEAR_JERK_LIMIT,
) -> float:
    """Minimum symmetric S-curve distance with zero endpoint acceleration."""
    v0, v1 = abs(float(start_speed)), abs(float(end_speed))
    delta_v = abs(v0 - v1)
    if delta_v <= 1e-12:
        return 0.0
    accel = max(abs(float(accel_limit)), 1e-6)
    jerk = max(abs(float(jerk_limit)), 1e-6)
    ramp_delta_v = accel * accel / jerk
    if delta_v <= ramp_delta_v:
        duration = 2.0 * math.sqrt(delta_v / jerk)
    else:
        duration = 2.0 * accel / jerk + (delta_v - ramp_delta_v) / accel
    return 0.5 * (v0 + v1) * duration


def step_linear_scurve(
    velocity: float,
    accel: float,
    target_velocity: float,
    dt: float,
    accel_limit: float = LINEAR_ACCEL_LIMIT,
    jerk_limit: float = LINEAR_JERK_LIMIT,
) -> tuple[float, float]:
    """Advance one sample of a jerk-limited linear velocity command."""
    dt = max(float(dt), 1e-4)
    accel_limit = abs(float(accel_limit))
    jerk_limit = abs(float(jerk_limit))
    if jerk_limit <= 0.0:
        delta = accel_limit * dt
        next_velocity = min(max(target_velocity, velocity - delta), velocity + delta)
        next_accel = (next_velocity - velocity) / dt
        return float(next_velocity), float(next_accel)

    # Start braking when the remaining speed gap is no larger than the
    # velocity spent bringing the current accel back to zero at j_max.
    stop_dv = accel * abs(accel) / (2.0 * jerk_limit)
    remaining = target_velocity - (velocity + stop_dv)
    desired = 0.0 if abs(remaining) <= 1e-6 else math.copysign(accel_limit, remaining)
    next_accel = min(max(desired, accel - jerk_limit * dt), accel + jerk_limit * dt)
    next_accel = min(max(next_accel, -accel_limit), accel_limit)
    next_velocity = velocity + next_accel * dt
    return float(next_velocity), float(next_accel)


def summarize_speed_profile(
    samples: list[tuple[float, float]],
    *,
    accel_limit: float = LINEAR_ACCEL_LIMIT,
    jerk_limit: float = LINEAR_JERK_LIMIT,
    deadband_v: float = JERK_DEADBAND_V,
) -> dict[str, float | int | bool]:
    """Finite-difference accel/jerk of the applied linear command."""
    max_abs_accel = 0.0
    max_abs_jerk = 0.0
    counted_jerk = 0
    previous_time = None
    previous_velocity = None
    previous_accel = None
    for stamp, velocity in samples:
        if previous_time is None:
            previous_time = stamp
            previous_velocity = velocity
            continue
        dt = max(stamp - previous_time, 1e-3)
        accel = (velocity - previous_velocity) / dt
        max_abs_accel = max(max_abs_accel, abs(accel))
        if previous_accel is not None and max(abs(velocity), abs(previous_velocity)) >= deadband_v:
            jerk = (accel - previous_accel) / dt
            max_abs_jerk = max(max_abs_jerk, abs(jerk))
            counted_jerk += 1
        previous_time = stamp
        previous_velocity = velocity
        previous_accel = accel
    return {
        "sample_count": len(samples),
        "jerk_sample_count": counted_jerk,
        "max_abs_accel": float(max_abs_accel),
        "max_abs_jerk": float(max_abs_jerk),
        "accel_limit": float(accel_limit),
        "jerk_limit": float(jerk_limit),
        "jerk_deadband_v": float(deadband_v),
        "within_jerk_limit": bool(max_abs_jerk <= jerk_limit + JERK_ACCEPTANCE_SLACK),
    }


def profile_within_jerk_limit(profile: dict | None, jerk_limit: float = LINEAR_JERK_LIMIT) -> bool:
    if not profile:
        return False
    try:
        return float(profile["max_abs_jerk"]) <= float(jerk_limit) + JERK_ACCEPTANCE_SLACK
    except (KeyError, TypeError, ValueError):
        return False
