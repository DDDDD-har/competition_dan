"""Geometry-driven speed limits for an arbitrary planned Ackermann path."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from chassis_motion_profile import required_speed_change_distance
from path_safety_filter import (
    densify_path,
    footprint_clearance_at_pose,
    path_footprint_clearances,
    swept_clear,
)


def cumulative_arc_lengths(path: Iterable[tuple[float, float, float]]) -> list[float]:
    poses = list(path)
    if not poses:
        return []
    result = [0.0]
    for first, second in zip(poses[:-1], poses[1:]):
        result.append(
            result[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    return result


def _index_at_arc(arcs: list[float], origin: int, offset_m: float) -> int:
    """Return the pose index whose arc length is closest to origin+offset."""
    target = arcs[origin] + float(offset_m)
    if offset_m >= 0.0:
        index = origin
        while index + 1 < len(arcs) and arcs[index] < target:
            index += 1
        return index
    index = origin
    while index > 0 and arcs[index] > target:
        index -= 1
    return index


def local_curvatures(
    path: Iterable[tuple[float, float, float]],
    *,
    window_m: float = 0.55,
) -> list[float]:
    """Estimate signed XY curvature from a chord window, not adjacent samples.

    Hybrid-A* angle bins can flip 10–15° between 8 cm poses. Using only
    neighbors treats that quantization as a 2–3 /m turn and brakes a
    straight corridor. A ~0.55 m stencil follows the real path heading.
    """
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 3:
        return [0.0] * len(poses)
    arcs = cumulative_arc_lengths(poses)
    half = max(float(window_m) * 0.5, 1e-3)
    result = [0.0] * len(poses)
    for index in range(len(poses)):
        left = _index_at_arc(arcs, index, -half)
        right = _index_at_arc(arcs, index, half)
        if left >= index or right <= index:
            left = max(0, index - 1)
            right = min(len(poses) - 1, index + 1)
        if left == index or right == index or left == right:
            continue
        inbound = math.atan2(
            poses[index][1] - poses[left][1],
            poses[index][0] - poses[left][0],
        )
        outbound = math.atan2(
            poses[right][1] - poses[index][1],
            poses[right][0] - poses[index][0],
        )
        ds = max(
            0.5 * ((arcs[index] - arcs[left]) + (arcs[right] - arcs[index])),
            1e-6,
        )
        result[index] = math.atan2(
            math.sin(outbound - inbound),
            math.cos(outbound - inbound),
        ) / ds
    return result


def point_to_segment_distance(point, start, end) -> float:
    """XY distance from ``point`` to the closed segment ``start→end``."""
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def inside_score_circle(
    site,
    inbound_pose,
    *,
    stop_offset_m: float,
    score_radius_m: float,
) -> tuple[float, float]:
    """Place a navigation stop short of ``site`` but still inside the circle."""
    offset = max(0.0, min(float(stop_offset_m), max(0.0, float(score_radius_m) - 0.05)))
    dx = float(site[0]) - float(inbound_pose[0])
    dy = float(site[1]) - float(inbound_pose[1])
    distance = math.hypot(dx, dy)
    if offset <= 1e-9 or distance <= offset + 1e-9:
        return (float(site[0]), float(site[1]))
    scale = (distance - offset) / distance
    return (float(inbound_pose[0]) + dx * scale, float(inbound_pose[1]) + dy * scale)


def trim_path_to_score_stop(
    points: Iterable[tuple[float, float, float]],
    site,
    *,
    stop_offset_m: float,
    score_radius_m: float,
) -> list[tuple[float, float, float]]:
    """Walk back along a site-ending path; keep the stop inside the circle."""
    poses = [tuple(map(float, pose)) for pose in points]
    if len(poses) < 2:
        return poses
    offset = max(0.0, min(float(stop_offset_m), max(0.0, float(score_radius_m) - 0.05)))
    if offset <= 1e-9:
        return poses
    remaining = offset
    trimmed = list(poses)
    while len(trimmed) >= 2 and remaining > 1e-9:
        x0, y0, _yaw0 = trimmed[-2]
        x1, y1, yaw1 = trimmed[-1]
        segment = math.hypot(x1 - x0, y1 - y0)
        if segment <= 1e-9:
            trimmed.pop()
            continue
        if segment <= remaining + 1e-9:
            remaining -= segment
            trimmed.pop()
            continue
        scale = remaining / segment
        trimmed[-1] = (x1 - scale * (x1 - x0), y1 - scale * (y1 - y0), yaw1)
        remaining = 0.0
    if len(trimmed) < 2:
        return poses
    end = trimmed[-1]
    if math.hypot(end[0] - float(site[0]), end[1] - float(site[1])) >= (
        float(score_radius_m) - 0.02
    ):
        return poses
    return trimmed


def terminal_path_metrics(
    path: Iterable[tuple[float, float, float]],
    *,
    window_m: float = 2.0,
) -> dict[str, float | bool]:
    """Measure whether a terminal path can be driven straight in forwards."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return {
            "forward_only": False,
            "window_length_m": 0.0,
            "end_vs_inbound_rad": math.inf,
            "yaw_change_rad": math.inf,
            "max_abs_curvature_1pm": math.inf,
        }
    arcs = cumulative_arc_lengths(poses)
    start_arc = max(0.0, arcs[-1] - max(float(window_m), 0.1))
    start = next(
        (index for index, arc in enumerate(arcs) if arc >= start_arc),
        len(poses) - 2,
    )
    start = min(start, len(poses) - 2)
    inbound = math.atan2(
        poses[-1][1] - poses[start][1],
        poses[-1][0] - poses[start][0],
    )
    end_vs_inbound = math.atan2(
        math.sin(poses[-1][2] - inbound),
        math.cos(poses[-1][2] - inbound),
    )
    yaw_change = math.atan2(
        math.sin(poses[-1][2] - poses[start][2]),
        math.cos(poses[-1][2] - poses[start][2]),
    )
    forward_only = True
    for first, second in zip(poses[:-1], poses[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if dx * math.cos(first[2]) + dy * math.sin(first[2]) < -1e-4:
            forward_only = False
            break
    curvatures = local_curvatures(poses)
    return {
        "forward_only": forward_only,
        "window_length_m": float(arcs[-1] - arcs[start]),
        "end_vs_inbound_rad": abs(float(end_vs_inbound)),
        "yaw_change_rad": abs(float(yaw_change)),
        "max_abs_curvature_1pm": max(
            (abs(float(value)) for value in curvatures[start:]),
            default=0.0,
        ),
    }


def terminal_path_is_trackable(
    metrics: dict[str, float | bool],
    *,
    max_alignment_error: float = 0.55,
    max_yaw_change: float = 0.85,
    max_curvature: float = 3.0,
) -> bool:
    """Hard gate for a forward, non-curling final approach."""
    return bool(
        metrics["forward_only"]
        and float(metrics["end_vs_inbound_rad"]) <= max_alignment_error
        and float(metrics["yaw_change_rad"]) <= max_yaw_change
        and float(metrics["max_abs_curvature_1pm"]) <= max_curvature
    )


def _reachable_speed(
    target_speed: float,
    distance: float,
    contract_speed: float,
    accel_limit: float,
    jerk_limit: float,
) -> float:
    """Largest speed that can transition to target within distance."""
    low = max(0.0, float(target_speed))
    high = max(low, float(contract_speed))
    if required_speed_change_distance(high, low, accel_limit, jerk_limit) <= distance:
        return high
    for _ in range(36):
        middle = 0.5 * (low + high)
        if required_speed_change_distance(middle, low, accel_limit, jerk_limit) <= distance:
            low = middle
        else:
            high = middle
    return low


def _segment_heading(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    return math.atan2(
        float(end[1]) - float(start[1]),
        float(end[0]) - float(start[0]),
    )


def simplify_collinear_path(
    path: Iterable[tuple[float, float, float]],
    *,
    angle_tol_rad: float = 0.08,
    min_seg_m: float = 0.15,
) -> list[tuple[float, float, float]]:
    """Drop redundant Theta* vertices that lie on the same straight segment."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 3:
        return poses
    kept = [0]
    for index in range(1, len(poses) - 1):
        prev = poses[kept[-1]]
        current = poses[index]
        nxt = poses[index + 1]
        heading_in = _segment_heading(prev, current)
        heading_out = _segment_heading(current, nxt)
        dyaw = abs(
            math.atan2(
                math.sin(heading_out - heading_in),
                math.cos(heading_out - heading_in),
            )
        )
        if dyaw > float(angle_tol_rad):
            kept.append(index)
    kept.append(len(poses) - 1)
    simplified = [poses[index] for index in kept]
    if len(simplified) < 2:
        return poses
    merged = [simplified[0]]
    for point in simplified[1:]:
        span = math.hypot(point[0] - merged[-1][0], point[1] - merged[-1][1])
        if span >= float(min_seg_m) or point == simplified[-1]:
            merged.append(point)
    return merged if len(merged) >= 2 else poses


def max_lateral_from_chord(
    path: Iterable[tuple[float, float, float]],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> float:
    """Maximum lateral offset of path points from the chord start→end."""
    ax, ay = map(float, start_xy)
    bx, by = map(float, end_xy)
    dx, dy = bx - ax, by - ay
    span = math.hypot(dx, dy)
    if span <= 1e-9:
        return 0.0
    maximum = 0.0
    for pose in path:
        px, py = float(pose[0]), float(pose[1])
        lateral = abs(dy * px - dx * py + bx * ay - by * ax) / span
        maximum = max(maximum, lateral)
    return maximum


def snap_corridor_segment(
    path: Iterable[tuple[float, float, float]],
    site_a: tuple[float, float],
    site_b: tuple[float, float],
    *,
    axis: str = "y",
    blend_m: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Snap a corridor leg onto the official site centreline (A↔B same Y)."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return poses
    target = (
        float(site_a[1]) if axis == "y" else float(site_a[0])
    )
    ax, ay = map(float, site_a)
    bx, by = map(float, site_b)
    chord_len = math.hypot(bx - ax, by - ay)
    if chord_len <= 1e-6:
        return poses
    result: list[tuple[float, float, float]] = []
    for x, y, yaw in poses:
        along = ((float(x) - ax) * (bx - ax) + (float(y) - ay) * (by - ay)) / chord_len
        if along < -blend_m or along > chord_len + blend_m:
            result.append((x, y, yaw))
            continue
        if axis == "y":
            weight = max(0.0, 1.0 - abs(along) / max(chord_len + blend_m, 1e-6))
            snapped_y = float(y) * (1.0 - weight) + target * weight
            result.append((x, snapped_y, yaw))
        else:
            weight = max(0.0, 1.0 - abs(along) / max(chord_len + blend_m, 1e-6))
            snapped_x = float(x) * (1.0 - weight) + target * weight
            result.append((snapped_x, y, yaw))
    return result


def shortcut_visible_vertices(
    path: Iterable[tuple[float, float, float]],
    occupied_xy: np.ndarray,
    footprint,
    clearance: float = 0.05,
) -> list[tuple[float, float, float]]:
    """String-pull shortcuts that stay footprint-safe on the occupancy grid."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 3:
        return poses
    from path_safety_filter import swept_clear

    result = [poses[0]]
    anchor = 0
    probe = 2
    while probe < len(poses):
        shortcut = [poses[anchor], poses[probe]]
        safe, _ = swept_clear(shortcut, occupied_xy, footprint, clearance)
        if safe:
            probe += 1
            continue
        result.append(poses[probe - 1])
        anchor = probe - 1
        probe = anchor + 2
    if result[-1] != poses[-1]:
        result.append(poses[-1])
    return result


def refine_planned_path(
    path: Iterable[tuple[float, float, float]],
    occupied_xy: np.ndarray | None,
    footprint,
    *,
    clearance: float = 0.05,
    corridor_sites: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Simplify, corridor-snap, shortcut, and orient a global plan."""
    poses = [tuple(map(float, pose)) for pose in path]
    audit: dict = {
        "pose_count_before": len(poses),
        "length_m_before": path_length_m(poses),
    }
    if corridor_sites is None:
        refined = simplify_collinear_path(poses)
    else:
        refined = list(poses)
    if corridor_sites is not None:
        refined = snap_corridor_segment(refined, corridor_sites[0], corridor_sites[1])
    if occupied_xy is not None and len(refined) >= 3 and corridor_sites is None:
        refined = shortcut_visible_vertices(
            refined, occupied_xy, footprint, clearance,
        )
    refined = orient_path_for_ackermann(refined)
    audit.update({
        "pose_count_after": len(refined),
        "length_m_after": path_length_m(refined),
        "corridor_sites": (
            [list(corridor_sites[0]), list(corridor_sites[1])]
            if corridor_sites is not None else None
        ),
    })
    if corridor_sites is not None:
        audit["max_lateral_from_chord_m"] = max_lateral_from_chord(
            refined, corridor_sites[0], corridor_sites[1],
        )
    return refined, audit


def path_length_m(path: Iterable[tuple[float, float, float]]) -> float:
    poses = [tuple(map(float, pose)) for pose in path]
    total = 0.0
    for left, right in zip(poses, poses[1:]):
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


def build_straight_chord_path(
    start_xy: tuple[float, float],
    start_yaw: float,
    end_xy: tuple[float, float],
    *,
    end_yaw: float | None = None,
    spacing: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Dense straight segment between two XY poses for hall/corridor legs."""
    sx, sy = map(float, start_xy)
    ex, ey = map(float, end_xy)
    terminal_yaw = (
        float(end_yaw)
        if end_yaw is not None
        else math.atan2(ey - sy, ex - sx)
    )
    coarse = [
        (sx, sy, float(start_yaw)),
        (ex, ey, terminal_yaw),
    ]
    return orient_path_for_ackermann(densify_path(coarse, spacing=spacing))


def orient_path_for_ackermann(
    path: Iterable[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Assign segment-tangent headings for grid planners (e.g. Theta*)."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return poses
    oriented: list[tuple[float, float, float]] = []
    for index, (x, y, _) in enumerate(poses):
        if index + 1 < len(poses):
            nxt = poses[index + 1]
            yaw = math.atan2(nxt[1] - y, nxt[0] - x)
        else:
            yaw = oriented[-1][2] if oriented else float(poses[-1][2])
        oriented.append((x, y, yaw))
    return oriented


def segment_peak_curvature(
    path: Iterable[tuple[float, float, float]],
) -> float:
    """Maximum absolute curvature along a path segment."""
    curvatures = local_curvatures(path)
    return max((abs(float(value)) for value in curvatures), default=0.0)


def split_path_by_turn_zones(
    path: Iterable[tuple[float, float, float]],
    *,
    heading_change_rad: float = 0.40,
    window_m: float = 1.2,
    min_turn_arc_m: float = 1.0,
    min_straight_arc_m: float = 1.5,
) -> list[tuple[list[tuple[float, float, float]], bool]]:
    """Split a path into straight cruise vs turn-arc execution zones."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return [(poses, False)] if poses else []
    arcs = cumulative_arc_lengths(poses)
    turn_flags = [False] * len(poses)
    for index in range(len(poses)):
        end = index
        while end + 1 < len(poses) and arcs[end] - arcs[index] < window_m:
            end += 1
        dyaw = math.atan2(
            math.sin(poses[end][2] - poses[index][2]),
            math.cos(poses[end][2] - poses[index][2]),
        )
        if abs(dyaw) >= heading_change_rad:
            for mark in range(index, end + 1):
                turn_flags[mark] = True
    runs: list[tuple[bool, int, int]] = []
    for index, flag in enumerate(turn_flags):
        if not runs or runs[-1][0] != flag:
            runs.append((flag, index, index))
        else:
            runs[-1] = (flag, runs[-1][1], index)
    segments: list[tuple[list[tuple[float, float, float]], bool]] = []
    for is_turn, begin, end in runs:
        chunk = poses[begin : end + 1]
        run_length = arcs[end] - arcs[begin]
        minimum = min_turn_arc_m if is_turn else min_straight_arc_m
        if run_length < minimum and segments:
            prev_chunk, prev_turn = segments[-1]
            segments[-1] = (prev_chunk + chunk[1:], prev_turn)
            continue
        if segments and chunk[0] == segments[-1][0][-1]:
            chunk = chunk[1:]
        if len(chunk) >= 2:
            segments.append((chunk, is_turn))
    return segments or [(poses, False)]


def split_c_leg_execution_zones(
    path: Iterable[tuple[float, float, float]],
    *,
    heading_change_rad: float = 0.35,
    window_m: float = 1.2,
    min_turn_arc_m: float = 1.0,
    min_straight_arc_m: float = 1.5,
) -> list[tuple[list[tuple[float, float, float]], bool]]:
    """Detect hall vs bend on a C path for speed-profile bookkeeping.

    Not an execution plan. Chopping a Hybrid Dubins C path on these windows
    hands MPPI a sub-2 m corner the 2WS chassis cannot track.
    """
    return split_path_by_turn_zones(
        path,
        heading_change_rad=heading_change_rad,
        window_m=window_m,
        min_turn_arc_m=min_turn_arc_m,
        min_straight_arc_m=min_straight_arc_m,
    )


def primary_c_turn_start_arc(
    path: Iterable[tuple[float, float, float]],
    *,
    leg_start_arc: float = 0.0,
    heading_change_rad: float = 0.35,
    window_m: float = 1.2,
    min_turn_arc_m: float = 1.0,
    min_straight_arc_m: float = 1.5,
    primary_turn_curvature_1pm: float = 0.30,
) -> float | None:
    """Arc length where the main B→C bend begins (ignores B/C join artifacts)."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return None
    segments = split_c_leg_execution_zones(
        poses,
        heading_change_rad=heading_change_rad,
        window_m=window_m,
        min_turn_arc_m=min_turn_arc_m,
        min_straight_arc_m=min_straight_arc_m,
    )
    turn_chunks = [
        chunk
        for chunk, is_turn in segments
        if is_turn and segment_peak_curvature(chunk) >= primary_turn_curvature_1pm
    ]
    if not turn_chunks:
        turn_chunks = [chunk for chunk, is_turn in segments if is_turn]
    if not turn_chunks:
        return None
    main_chunk = max(
        turn_chunks,
        key=lambda chunk: (
            segment_peak_curvature(chunk),
            cumulative_arc_lengths(chunk)[-1],
        ),
    )
    start_xy = main_chunk[0][:2]
    arcs = cumulative_arc_lengths(poses)
    for index, pose in enumerate(poses):
        if math.hypot(pose[0] - start_xy[0], pose[1] - start_xy[1]) <= 0.20:
            return float(leg_start_arc) + float(arcs[index])
    return float(leg_start_arc) + float(cumulative_arc_lengths(main_chunk)[0])


def c_leg_turn_via_point(
    path: Iterable[tuple[float, float, float]],
    *,
    heading_change_rad: float = 0.35,
    window_m: float = 1.2,
    min_turn_arc_m: float = 1.0,
    min_straight_arc_m: float = 1.5,
    primary_turn_curvature_1pm: float = 0.30,
) -> tuple[float, float] | None:
    """Derive a hall→bend waypoint from the monolithic C-leg plan."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return None
    segments = split_c_leg_execution_zones(
        poses,
        heading_change_rad=heading_change_rad,
        window_m=window_m,
        min_turn_arc_m=min_turn_arc_m,
        min_straight_arc_m=min_straight_arc_m,
    )
    turn_chunks = [
        chunk
        for chunk, is_turn in segments
        if is_turn and segment_peak_curvature(chunk) >= primary_turn_curvature_1pm
    ]
    if not turn_chunks:
        turn_chunks = [chunk for chunk, is_turn in segments if is_turn]
    if not turn_chunks:
        return None
    main_chunk = max(
        turn_chunks,
        key=lambda chunk: (
            segment_peak_curvature(chunk),
            cumulative_arc_lengths(chunk)[-1],
        ),
    )
    return (float(main_chunk[0][0]), float(main_chunk[0][1]))


def last_turn_zone_start_index(
    path: Iterable[tuple[float, float, float]],
    *,
    heading_change_rad: float = 0.35,
    window_m: float = 1.2,
    min_turn_arc_m: float = 1.0,
    min_straight_arc_m: float = 1.5,
    blend_m: float = 0.40,
) -> int:
    """Index where the final heading-window turn begins, blended slightly early."""
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 3:
        return 0
    arcs = cumulative_arc_lengths(poses)
    segments = split_c_leg_execution_zones(
        poses,
        heading_change_rad=heading_change_rad,
        window_m=window_m,
        min_turn_arc_m=min_turn_arc_m,
        min_straight_arc_m=min_straight_arc_m,
    )
    turn_chunks = [chunk for chunk, is_turn in segments if is_turn]
    if turn_chunks:
        start_xy = turn_chunks[-1][0][:2]
        start_index = 0
        for index, pose in enumerate(poses):
            if math.hypot(pose[0] - start_xy[0], pose[1] - start_xy[1]) <= 0.20:
                start_index = index
                break
    else:
        target_arc = max(0.0, float(arcs[-1]) - 4.0)
        start_index = next(
            (index for index, arc in enumerate(arcs) if arc >= target_arc),
            0,
        )
    blend_arc = max(0.0, float(arcs[start_index]) - max(0.0, float(blend_m)))
    for index, arc in enumerate(arcs):
        if arc >= blend_arc - 1e-9:
            return index
    return start_index


def _nearest_occupied_xy(
    x: float, y: float, occupied_xy: np.ndarray,
) -> tuple[float, float, float] | None:
    pts = np.asarray(occupied_xy, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return None
    delta = pts - np.asarray((x, y), dtype=float)
    dist = np.hypot(delta[:, 0], delta[:, 1])
    index = int(np.argmin(dist))
    return float(pts[index, 0]), float(pts[index, 1]), float(dist[index])


def _heading_is_westbound(yaw: float, *, tolerance_rad: float = 0.30) -> bool:
    delta = math.atan2(math.sin(float(yaw) - math.pi), math.cos(float(yaw) - math.pi))
    return abs(delta) <= float(tolerance_rad)


def _smooth_path_xy(
    poses: list[tuple[float, float, float]],
    start_index: int,
    *,
    window: int = 5,
    passes: int = 2,
) -> list[tuple[float, float, float]]:
    """Average last-turn XY so per-pose obstacle pushes do not crease the polyline."""
    if len(poses) < 3 or start_index >= len(poses) - 1 or window < 3:
        return poses
    result = list(poses)
    half = max(1, window // 2)
    for _ in range(max(1, passes)):
        xs = [pose[0] for pose in result]
        ys = [pose[1] for pose in result]
        smoothed = list(result)
        for index in range(start_index + 1, len(result) - 1):
            lo = max(start_index, index - half)
            hi = min(len(result) - 1, index + half)
            count = hi - lo + 1
            smoothed[index] = (
                sum(xs[lo:hi + 1]) / count,
                sum(ys[lo:hi + 1]) / count,
                result[index][2],
            )
        result = smoothed
    return result


def widen_last_c_turn(
    path: Iterable[tuple[float, float, float]],
    occupied_xy: np.ndarray | None,
    footprint,
    *,
    target_clearance_m: float = 0.48,
    max_push_m: float = 0.16,
    step_m: float = 0.03,
    site_xy: tuple[float, float] | None = None,
    score_radius_m: float = 0.60,
    max_curvature_1pm: float = 1.30,
    path_safety_clearance_m: float = 0.05,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Push the last C turn away from occupied cells; keep the hall unchanged.

    Hybrid shortest-path Dubins hugs the cabinet inner corner (~0.37 m
    footprint clearance). 2WS then understeers into the west free space.
    Shifting only the last turn toward that free space gives the chassis an
    apex it can actually track.
    """
    poses = [tuple(map(float, pose)) for pose in path]
    audit = {
        "applied": False,
        "start_index": 0,
        "min_clearance_before_m": None,
        "min_clearance_after_m": None,
        "max_push_m": 0.0,
    }
    if (
        occupied_xy is None
        or len(poses) < 4
        or max_push_m <= 1e-9
        or target_clearance_m <= 0.0
    ):
        return poses, audit
    occupied = np.asarray(occupied_xy, dtype=float).reshape(-1, 2)
    if occupied.size == 0:
        return poses, audit
    start_index = last_turn_zone_start_index(poses)
    audit["start_index"] = int(start_index)

    def _turn_clearances(chunk) -> list[float]:
        selected = [
            pose for pose in chunk
            if not _heading_is_westbound(pose[2])
        ]
        if not selected:
            selected = list(chunk)
        return [
            value
            for value in path_footprint_clearances(selected, occupied, footprint)
            if math.isfinite(value)
        ]

    before_vals = _turn_clearances(poses[start_index:])
    audit["min_clearance_before_m"] = (
        min(before_vals) if before_vals else None
    )
    hold_end_m = 0.70 if site_xy is not None else 0.0
    eligible: list[int] = []
    direction_x = 0.0
    direction_y = 0.0
    for index in range(start_index, len(poses)):
        x, y, yaw = poses[index]
        near_site = (
            site_xy is not None
            and math.hypot(
                x - float(site_xy[0]), y - float(site_xy[1]),
            ) <= hold_end_m
        )
        if near_site or _heading_is_westbound(yaw):
            continue
        clearance = footprint_clearance_at_pose(poses[index], occupied, footprint)
        if not math.isfinite(clearance) or clearance >= target_clearance_m - 1e-9:
            continue
        nearest = _nearest_occupied_xy(x, y, occupied)
        if nearest is None or nearest[2] <= 1e-6:
            continue
        ox, oy, dist = nearest
        direction_x += (x - ox) / dist
        direction_y += (y - oy) / dist
        eligible.append(index)
    if not eligible:
        return poses, audit
    norm = math.hypot(direction_x, direction_y)
    if norm <= 1e-9:
        return poses, audit
    direction_x /= norm
    direction_y /= norm
    rank = {index: rank_i for rank_i, index in enumerate(eligible)}
    count = len(eligible)
    widened: list[tuple[float, float, float]] = []
    max_push = 0.0
    for index, pose in enumerate(poses):
        if index not in rank:
            widened.append(pose)
            continue
        taper = math.sin(math.pi * (rank[index] + 0.5) / count)
        clearance = footprint_clearance_at_pose(pose, occupied, footprint)
        need = max(
            0.0,
            target_clearance_m - (clearance if math.isfinite(clearance) else 0.0),
        )
        push = min(max_push_m, need) * taper
        if step_m > 1e-9:
            push = step_m * math.floor((push + 1e-9) / step_m)
        max_push = max(max_push, push)
        widened.append((
            pose[0] + push * direction_x,
            pose[1] + push * direction_y,
            pose[2],
        ))
    audit["max_push_m"] = float(max_push)
    audit["push_direction"] = [float(direction_x), float(direction_y)]
    if max_push <= 1e-9:
        return poses, audit
    smoothed = _smooth_path_xy(widened, start_index)
    refined = orient_path_for_ackermann(densify_path(smoothed, spacing=0.12))
    if site_xy is not None and refined:
        end = refined[-1]
        if math.hypot(
            end[0] - float(site_xy[0]), end[1] - float(site_xy[1]),
        ) >= max(0.20, float(score_radius_m) - 0.05):
            audit["rejected"] = "score_circle"
            return poses, audit
    refined_start = last_turn_zone_start_index(refined)
    peak_kappa = segment_peak_curvature(refined[refined_start:] or refined)
    original_kappa = segment_peak_curvature(poses[start_index:] or poses)
    kappa_limit = max(1.45, float(original_kappa) + 0.15)
    if peak_kappa > kappa_limit + 1e-6:
        audit["rejected"] = "curvature"
        audit["peak_curvature_1pm"] = float(peak_kappa)
        audit["curvature_limit_1pm"] = float(kappa_limit)
        return poses, audit
    safe, report = swept_clear(
        refined, occupied, footprint, path_safety_clearance_m,
    )
    if not safe:
        original_safe, _original_report = swept_clear(
            poses, occupied, footprint, path_safety_clearance_m,
        )
        if original_safe:
            audit["rejected"] = "swept"
            audit["swept_report"] = report
            return poses, audit
        audit["swept_preexisting"] = True
    after_vals = _turn_clearances(refined[refined_start:])
    min_after = min(after_vals) if after_vals else 0.0
    audit["min_clearance_after_m"] = float(min_after)
    min_before = audit["min_clearance_before_m"]
    if min_before is not None and min_after + 0.03 < float(min_before):
        audit["rejected"] = "clearance_worse"
        return poses, audit
    audit["applied"] = True
    return refined, audit


def hybrid_execution_segments(
    planned_paths: dict[str, list[tuple[float, float, float]]],
    *,
    heading_change_rad: float = 0.40,
    mppi_speed_threshold_mps: float = 10.0,
    contract_speed_mps: float = 2.0,
) -> list[tuple[str, list[tuple[float, float, float]], bool]]:
    """Per scored leg: fly-through A/B are profiled; C is one local FollowPath."""
    del heading_change_rad
    segments: list[tuple[str, list[tuple[float, float, float]], bool]] = []
    use_mppi_turns = float(contract_speed_mps) < float(mppi_speed_threshold_mps)
    for leg_name in ("A", "B", "C"):
        if leg_name not in planned_paths:
            continue
        subpath = [tuple(map(float, row)) for row in planned_paths[leg_name]]
        if len(subpath) < 2:
            continue
        if leg_name in ("A", "B") or not use_mppi_turns:
            segments.append((leg_name, subpath, False))
            continue
        segments.append((leg_name, subpath, True))
    return segments


def split_path_by_curvature(
    path: Iterable[tuple[float, float, float]],
    *,
    straight_threshold_1pm: float = 0.20,
    min_straight_run_m: float = 0.35,
    min_corner_run_m: float = 0.20,
) -> list[list[tuple[float, float, float]]]:
    """Split a path into straight/corner runs for per-subsegment profiling.

    Long straights keep peak cruise speed; corner caps only propagate within
    their short subsegment instead of braking an entire scored leg.
    """
    poses = [tuple(map(float, pose)) for pose in path]
    if len(poses) < 2:
        return [poses] if poses else []
    curvatures = local_curvatures(poses)
    arcs = cumulative_arc_lengths(poses)
    threshold = max(0.0, float(straight_threshold_1pm))
    straight_flags = [abs(curvature) < threshold for curvature in curvatures]
    runs: list[tuple[int, int, bool]] = []
    start = 0
    current = straight_flags[0]
    for index in range(1, len(poses)):
        if straight_flags[index] != current:
            runs.append((start, index - 1, current))
            start = index - 1
            current = straight_flags[index]
    runs.append((start, len(poses) - 1, current))

    def run_length(run: tuple[int, int, bool]) -> float:
        begin, end, _ = run
        return max(0.0, float(arcs[end]) - float(arcs[begin]))

    merged = list(runs)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, (begin, end, straight) in enumerate(merged):
            minimum = (
                float(min_straight_run_m) if straight else float(min_corner_run_m)
            )
            if run_length((begin, end, straight)) >= minimum:
                continue
            if index > 0:
                prev_begin, _, prev_straight = merged[index - 1]
                merged[index - 1] = (prev_begin, end, prev_straight)
                merged.pop(index)
                changed = True
                break
            if index + 1 < len(merged):
                _, next_end, next_straight = merged[index + 1]
                merged[index] = (begin, next_end, next_straight)
                merged.pop(index + 1)
                changed = True
                break

    segments: list[list[tuple[float, float, float]]] = []
    for begin, end, _ in merged:
        segment = poses[begin : end + 1]
        if segments and segment[0] != segments[-1][-1]:
            segment = [segments[-1][-1]] + segment[1:]
        elif segments:
            segment = [segments[-1][-1]] + segment
        if len(segment) >= 2:
            segments.append(segment)
    return segments


def _max_speed_after_accel(
    start_speed: float,
    distance: float,
    contract_speed: float,
    accel_limit: float,
    jerk_limit: float,
) -> float:
    """Largest speed reachable from ``start_speed`` within ``distance``."""
    if distance <= 0.0:
        return max(0.0, float(start_speed))
    low = max(0.0, float(start_speed))
    high = max(low, float(contract_speed))
    if required_speed_change_distance(high, low, accel_limit, jerk_limit) <= distance:
        return high
    for _ in range(36):
        middle = 0.5 * (low + high)
        if required_speed_change_distance(middle, low, accel_limit, jerk_limit) <= distance:
            low = middle
        else:
            high = middle
    return low


def enforce_speed_change_envelope(
    arc_lengths: list[float],
    speed_caps: list[float],
    *,
    contract_speed: float,
    accel_limit: float,
    jerk_limit: float,
) -> list[float]:
    """Apply jerk-aware braking and acceleration envelopes to local caps."""
    result = [max(0.0, min(float(contract_speed), float(value))) for value in speed_caps]
    if len(result) < 2:
        return result
    anchors = [
        (index, speed)
        for index, speed in enumerate(result)
        if speed < float(contract_speed) - 1e-9
    ]
    # Propagate every downstream restriction backward over the full available
    # distance. Doing this only between adjacent samples would incorrectly
    # restart the jerk ramp every few centimetres and suppress straight-line
    # cruise speed.
    for target, target_speed in reversed(anchors):
        for index in range(target - 1, -1, -1):
            distance = max(0.0, arc_lengths[target] - arc_lengths[index])
            result[index] = min(
                result[index],
                _reachable_speed(
                    target_speed, distance, contract_speed, accel_limit, jerk_limit
                ),
            )
    # Apply the same zero-endpoint-acceleration model while speeding up after
    # the initial stop or a geometry-induced slow zone.
    for source, source_speed in anchors:
        for index in range(source + 1, len(result)):
            distance = max(0.0, arc_lengths[index] - arc_lengths[source])
            result[index] = min(
                result[index],
                _reachable_speed(
                    source_speed, distance, contract_speed, accel_limit, jerk_limit
                ),
            )
    return result


def profile_index_at_arc(profile: list[dict], arc_m: float) -> int:
    """Return the profile sample at or just before ``arc_m``."""
    if not profile:
        return 0
    target = max(0.0, float(arc_m))
    lo = 0
    hi = len(profile) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if float(profile[mid]["arc_length_m"]) <= target + 1e-9:
            lo = mid
        else:
            hi = mid - 1
    return lo


def cross_track_speed_scale(
    cross_track_m: float,
    *,
    onset_m: float = 0.12,
    full_m: float = 0.45,
    minimum_scale: float = 0.35,
    remaining_arc_m: float | None = None,
    terminal_approach: bool = False,
) -> float:
    """Reduce speed when lateral error grows so the robot can rejoin the path."""
    if terminal_approach:
        onset_m = max(onset_m, 0.35)
        full_m = max(full_m, 1.05)
        minimum_scale = max(minimum_scale, 0.55)
        if (
            remaining_arc_m is not None
            and remaining_arc_m < 14.0
            and cross_track_m <= 1.20
        ):
            return max(minimum_scale, 0.55)
    if (
        remaining_arc_m is not None
        and remaining_arc_m < 2.0
        and cross_track_m <= 0.50
    ):
        return 1.0
    if (
        remaining_arc_m is not None
        and remaining_arc_m < 5.0
        and cross_track_m <= 0.65
    ):
        return max(minimum_scale, 0.60)
    if (
        remaining_arc_m is not None
        and remaining_arc_m < 4.0
        and cross_track_m <= 0.48
    ):
        # Inside the scored approach zone with acceptable lateral error: avoid
        # over-braking that wastes time without improving checkpoint accuracy.
        onset_m = max(onset_m, 0.20)
        full_m = max(full_m, 0.55)
        minimum_scale = max(minimum_scale, 0.60)
    if cross_track_m <= onset_m:
        return 1.0
    span = max(float(full_m) - float(onset_m), 1e-3)
    return max(
        float(minimum_scale),
        1.0 - (float(cross_track_m) - float(onset_m)) / span,
    )


def ackermann_feasible_command(
    linear: float,
    angular: float,
    *,
    max_curvature_1pm: float,
    max_angular: float,
    speed_cap_mps: float,
    min_speed_mps: float = 0.0,
) -> tuple[float, float]:
    """Reconcile a (v, w) pair with the steering base curvature envelope.

    A steered base realises ``w = v * kappa`` with ``kappa`` bounded by the
    minimum turning radius, so an angular rate requested independently of the
    linear speed is not executable: the wheels saturate and the robot stalls
    instead of turning.  The requested turn is preserved by raising the linear
    speed to ``|w| / kappa_max`` whenever the speed cap allows, and only the
    residual angular rate is trimmed once the cap is reached.
    """
    linear = float(linear)
    angular = float(np.clip(float(angular), -float(max_angular), float(max_angular)))
    cap = max(0.0, float(speed_cap_mps))
    kappa_max = max(1e-6, float(max_curvature_1pm))
    if linear < 0.0:
        # Reverse arcs obey the same envelope on the magnitude of the speed.
        forward, forward_angular = ackermann_feasible_command(
            -linear, -angular,
            max_curvature_1pm=kappa_max,
            max_angular=max_angular,
            speed_cap_mps=cap,
            min_speed_mps=min_speed_mps,
        )
        return -forward, -forward_angular
    linear = min(linear, cap)
    if abs(angular) <= 1e-9:
        return linear, 0.0
    required_speed = abs(angular) / kappa_max
    if required_speed > linear:
        linear = min(cap, required_speed)
    feasible_angular = linear * kappa_max
    if abs(angular) > feasible_angular:
        angular = math.copysign(feasible_angular, angular)
    if 0.0 < linear < float(min_speed_mps):
        linear = min(cap, float(min_speed_mps))
    return linear, angular


def coordinated_speed_cap(
    profile: list[dict],
    arc_m: float,
    cross_track_m: float,
    *,
    max_angular: float,
    lookahead_m: float,
    cross_track_onset_m: float = 0.12,
    cross_track_full_m: float = 0.45,
    remaining_arc_m: float | None = None,
    terminal_approach: bool = False,
    apply_cross_track_brake: bool = True,
) -> tuple[int, float, str]:
    """Speed cap at ``arc_m`` coordinated with downstream geometry on the profile.

    The path speed profile already propagates jerk-limited braking backward from
    corners.  The tracker consumes that cap at the projected arc position.  At a
    standstill the local sample can still be zero, so departure uses the
  lookahead target; while moving, the current arc cap plus any lower caps
    within the braking horizon govern the command.
    """
    if not profile:
        return 0, 0.0, "missing_profile"
    index = profile_index_at_arc(profile, arc_m)
    current_cap = float(profile[index]["speed_cap_mps"])
    ahead_index = profile_index_at_arc(
        profile, float(arc_m) + max(0.0, float(lookahead_m)),
    )
    ahead_cap = float(profile[ahead_index]["speed_cap_mps"])
    if current_cap < 0.20 and ahead_cap > current_cap + 0.50:
        cap = ahead_cap
        reason = str(profile[ahead_index]["limiting_factor"])
        limit_index = ahead_index
    elif current_cap < 0.05:
        cap = ahead_cap
        reason = str(profile[ahead_index]["limiting_factor"])
        limit_index = ahead_index
    else:
        cap = current_cap
        reason = str(profile[index]["limiting_factor"])
        limit_index = index
    return limit_index, max(0.0, cap), reason


def build_path_speed_profile(
    path: Iterable[tuple[float, float, float]],
    occupied_xy: np.ndarray,
    footprint,
    *,
    contract_speed: float,
    max_angular: float,
    lateral_accel_limit: float = 0.8,
    accel_limit: float = 1.2,
    jerk_limit: float = 2.0,
    clearance_hard: float = 0.08,
    clearance_soft: float = 0.45,
    minimum_motion_speed: float = 0.18,
    stop_at_end: bool = True,
    terminal_arrival_speed_mps: float = 0.0,
    terminal_zone_m: float = 1.8,
    peak_contract_speed: float | None = None,
    straight_curvature_1pm: float = 0.25,
    depart_from_rest: bool = True,
    join_speed_mps: float | None = None,
    handoff_geometry: bool = False,
    path_curvature_cap_1pm: float | None = None,
) -> list[dict[str, float | str]]:
    """Build an auditable speed cap at each path pose.

    Cruise is the contract peak. Curvature and clearance are recorded for
    audit only; they do not add extra speed knobs. The profile has no
    waypoint names, scene coordinates, or corridor-specific branches.
    """
    poses = [tuple(map(float, pose)) for pose in path]
    if not poses:
        return []
    if contract_speed <= 0.0 or max_angular <= 0.0:
        raise ValueError("contract speed and max angular speed must be positive")
    if not 0.0 <= clearance_hard < clearance_soft:
        raise ValueError("clearance thresholds must satisfy 0 <= hard < soft")
    arcs = cumulative_arc_lengths(poses)
    curvatures = local_curvatures(poses)
    if path_curvature_cap_1pm is not None:
        cap = max(0.0, float(path_curvature_cap_1pm))
        curvatures = [
            math.copysign(min(abs(value), cap), value) if abs(value) > cap else value
            for value in curvatures
        ]
    clearances = path_footprint_clearances(poses, occupied_xy, footprint)
    peak_speed = float(
        peak_contract_speed if peak_contract_speed is not None else contract_speed
    )
    peak_speed = max(float(contract_speed), peak_speed)
    raw_caps: list[float] = []
    raw_reasons: list[str] = []
    for curvature, clearance in zip(curvatures, clearances):
        absolute_curvature = abs(curvature)
        local_contract = peak_speed
        curve_reason = "peak_contract"
        if absolute_curvature > 1e-6:
            yaw_cap = max_angular / absolute_curvature
            lateral_cap = math.sqrt(
                max(float(lateral_accel_limit), 1e-6) / absolute_curvature
            )
            if min(yaw_cap, lateral_cap) < local_contract:
                curve_reason = "curvature"
        geometry_reason = curve_reason
        if math.isfinite(clearance) and clearance < clearance_soft:
            geometry_reason = "clearance"
        # Cruise at the contract peak. Curvature/clearance stay in the audit
        # fields; they must not add extra speed knobs on top of args.speed.
        cap = float(local_contract)
        reason = geometry_reason
        raw_caps.append(max(0.0, cap))
        raw_reasons.append(reason)
    if depart_from_rest:
        raw_caps[0] = 0.0
        raw_reasons[0] = "initial_acceleration"
    else:
        raw_reasons[0] = "segment_join"
    if stop_at_end:
        arrival = max(0.0, float(terminal_arrival_speed_mps))
        if arrival > 0.0:
            total_arc = float(arcs[-1]) if arcs else 0.0
            zone_m = max(float(terminal_zone_m), arrival)
            for index, arc in enumerate(arcs):
                remaining = total_arc - float(arc)
                if remaining > zone_m + 1e-9:
                    continue
                zone_frac = 1.0 - max(0.0, remaining) / zone_m
                blended = (1.0 - zone_frac) * raw_caps[index] + zone_frac * arrival
                raw_caps[index] = min(raw_caps[index], max(arrival, blended))
                if index == len(arcs) - 1:
                    raw_reasons[index] = "terminal_arrival"
                elif zone_frac > 0.05:
                    raw_reasons[index] = "terminal_zone"
        else:
            raw_caps[-1] = 0.0
            raw_reasons[-1] = "terminal_stop"
    envelope_peak = max(float(contract_speed), peak_speed)
    if handoff_geometry:
        for index, reason in enumerate(raw_reasons):
            if reason in {"curvature", "clearance"}:
                raw_caps[index] = float(envelope_peak)
                raw_reasons[index] = "segment_join"
    if handoff_geometry:
        caps = [
            max(0.0, min(float(envelope_peak), float(value)))
            for value in raw_caps
        ]
        if depart_from_rest:
            caps[0] = 0.0
    else:
        caps = enforce_speed_change_envelope(
            arcs,
            raw_caps,
            contract_speed=envelope_peak,
            accel_limit=accel_limit,
            jerk_limit=jerk_limit,
        )
        if not depart_from_rest and join_speed_mps is not None:
            entry = max(0.0, float(join_speed_mps))
            for index in range(len(caps)):
                distance = max(0.0, float(arcs[index]) - float(arcs[0]))
                caps[index] = min(
                    caps[index],
                    _max_speed_after_accel(
                        entry,
                        distance,
                        envelope_peak,
                        accel_limit,
                        jerk_limit,
                    ),
                )
    result = []
    for index, (pose, arc, curvature, clearance, raw_cap, cap) in enumerate(
        zip(poses, arcs, curvatures, clearances, raw_caps, caps)
    ):
        reason = raw_reasons[index]
        if cap + 1e-6 < raw_cap:
            reason = "braking_or_acceleration_envelope"
        result.append({
            "index": index,
            "x": pose[0],
            "y": pose[1],
            "yaw": pose[2],
            "arc_length_m": float(arc),
            "curvature_1pm": float(curvature),
            "clearance_m": float(clearance),
            "raw_speed_cap_mps": float(raw_cap),
            "speed_cap_mps": float(cap),
            "geometry_limiting_factor": raw_reasons[index],
            "limiting_factor": reason,
        })
    return result


def profile_summary(profile: list[dict]) -> dict:
    if not profile:
        return {"sample_count": 0}
    finite_clearances = [
        float(item["clearance_m"]) for item in profile
        if math.isfinite(float(item["clearance_m"]))
    ]
    reasons: dict[str, int] = {}
    for item in profile:
        reason = str(item["limiting_factor"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "sample_count": len(profile),
        "path_length_m": float(profile[-1]["arc_length_m"]),
        "minimum_speed_cap_mps": min(float(item["speed_cap_mps"]) for item in profile),
        "maximum_speed_cap_mps": max(float(item["speed_cap_mps"]) for item in profile),
        "minimum_clearance_m": min(finite_clearances, default=None),
        "maximum_abs_curvature_1pm": max(
            abs(float(item["curvature_1pm"])) for item in profile
        ),
        "limiting_factor_counts": reasons,
    }


def estimated_profile_time(
    profile: list[dict],
    *,
    minimum_speed: float = 0.18,
) -> float:
    """Estimate traversal time from the auditable local speed caps.

    Trapezoidal integration would be dominated by the intentional zero caps
    at path endpoints.  Clamp only for this planner-ranking estimate; runtime
    still consumes the original zero-to-zero jerk-aware profile.
    """
    if len(profile) < 2:
        return 0.0
    floor = max(float(minimum_speed), 1e-3)
    total = 0.0
    for first, second in zip(profile[:-1], profile[1:]):
        distance = max(
            0.0,
            float(second["arc_length_m"]) - float(first["arc_length_m"]),
        )
        average_speed = 0.5 * (
            max(floor, float(first["speed_cap_mps"]))
            + max(floor, float(second["speed_cap_mps"]))
        )
        total += distance / average_speed
    return total
