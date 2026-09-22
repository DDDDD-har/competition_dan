"""Geometry-only safety filter for Nav2 paths in the MuJoCo world frame."""
from __future__ import annotations
import math
from typing import Iterable

import numpy as np


def densify_path(path: Iterable[tuple[float, float, float]], spacing: float = 0.04):
    src = list(path); out = []
    if not src: return out
    for a, b in zip(src[:-1], src[1:]):
        d = math.hypot(b[0]-a[0], b[1]-a[1]); n = max(1, int(math.ceil(d/spacing)))
        for j in range(n):
            t=j/n; dy=math.atan2(math.sin(b[2]-a[2]), math.cos(b[2]-a[2]))
            out.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*dy))
    out.append(src[-1]); return out


def footprint_clearance_at_pose(
    pose: tuple[float, float, float],
    occupied_xy: np.ndarray,
    footprint,
) -> float:
    """Minimum XY clearance from a rectangular footprint to occupied points."""
    pts = np.asarray(occupied_xy, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return math.inf
    x, y, yaw = map(float, pose)
    c, s = math.cos(yaw), math.sin(yaw)
    local = (pts - np.asarray((x, y))) @ np.asarray(((c, -s), (s, c)))
    polygon = np.asarray(footprint, dtype=float)
    xmin, xmax = float(polygon[:, 0].min()), float(polygon[:, 0].max())
    ymin, ymax = float(polygon[:, 1].min()), float(polygon[:, 1].max())
    dx = np.maximum(np.maximum(xmin - local[:, 0], 0.0), local[:, 0] - xmax)
    dy = np.maximum(np.maximum(ymin - local[:, 1], 0.0), local[:, 1] - ymax)
    return float(np.min(np.hypot(dx, dy)))


def path_footprint_clearances(path, occupied_xy: np.ndarray, footprint) -> list[float]:
    """Return local footprint clearance for every supplied path pose."""
    return [
        footprint_clearance_at_pose(tuple(pose), occupied_xy, footprint)
        for pose in path
    ]


class OccupancySafetyIndex:
    """Reusable occupancy/footprint wrapper for repeated swept checks."""

    def __init__(self, occupied_xy: np.ndarray, footprint, clearance: float = 0.05):
        self.occupied_xy = np.asarray(occupied_xy, dtype=float).reshape(-1, 2)
        self.footprint = tuple(tuple(map(float, point)) for point in footprint)
        self.clearance = float(clearance)

    def swept_clear(self, path) -> tuple[bool, dict]:
        return swept_clear(path, self.occupied_xy, self.footprint, self.clearance)

    def path_clearances(self, path) -> list[float]:
        return path_footprint_clearances(path, self.occupied_xy, self.footprint)


def predict_straight_path(
    pose: tuple[float, float, float],
    path_yaw: float,
    linear: float,
    horizon: float,
    time_step: float = 0.10,
) -> list[tuple[float, float, float]]:
    """Integrate motion along a fixed path tangent for straight-segment audits."""
    if horizon <= 0.0 or time_step <= 0.0:
        raise ValueError("prediction horizon and time step must be positive")
    steps = max(1, int(math.ceil(horizon / time_step)))
    dt = horizon / steps
    x, y, _ = map(float, pose)
    yaw = float(path_yaw)
    result = [(x, y, yaw)]
    for _ in range(steps):
        x += linear * math.cos(yaw) * dt
        y += linear * math.sin(yaw) * dt
        result.append((x, y, yaw))
    return result


def predict_unicycle_path(
    pose: tuple[float, float, float],
    linear: float,
    angular: float,
    horizon: float,
    time_step: float = 0.10,
) -> list[tuple[float, float, float]]:
    """Integrate a short conservative path for a pending velocity command."""
    if horizon <= 0.0 or time_step <= 0.0:
        raise ValueError("prediction horizon and time step must be positive")
    steps = max(1, int(math.ceil(horizon / time_step)))
    dt = horizon / steps
    x, y, yaw = map(float, pose)
    result = [(x, y, yaw)]
    for _ in range(steps):
        next_yaw = yaw + angular * dt
        if abs(angular) <= 1e-9:
            x += linear * math.cos(yaw) * dt
            y += linear * math.sin(yaw) * dt
        else:
            radius = linear / angular
            x += radius * (math.sin(next_yaw) - math.sin(yaw))
            y -= radius * (math.cos(next_yaw) - math.cos(yaw))
        yaw = math.atan2(math.sin(next_yaw), math.cos(next_yaw))
        result.append((x, y, yaw))
    return result


def swept_clear(path, occupied_xy: np.ndarray, footprint, clearance=0.05) -> tuple[bool, dict]:
    """Check every interpolated pose against occupied world points.

    This catches collisions between sparse planner poses, including the
    rotated rear-corner sweep that the ordinary point path follower misses.
    """
    pts=np.asarray(occupied_xy,dtype=float).reshape(-1,2)
    if pts.size == 0: return True, {"checked_poses":0,"minimum_clearance_m":None}
    min_d=math.inf; bad=None
    for i,(x,y,yaw) in enumerate(densify_path(path)):
        c,s=math.cos(yaw),math.sin(yaw); q=pts-np.asarray((x,y)); local=q@np.asarray(((c,-s),(s,c)))
        poly=np.asarray(footprint,dtype=float); inside=(local[:,0]>=poly[:,0].min()-clearance)&(local[:,0]<=poly[:,0].max()+clearance)&(local[:,1]>=poly[:,1].min()-clearance)&(local[:,1]<=poly[:,1].max()+clearance)
        d=float(np.min(np.linalg.norm(q,axis=1))); min_d=min(min_d,d)
        if np.any(inside): bad={"pose_index":i,"pose":[x,y,yaw],"occupied_point":pts[np.flatnonzero(inside)[0]].tolist()}; break
    return bad is None, {"checked_poses":len(densify_path(path)),"minimum_clearance_m":min_d,"first_unsafe":bad}
