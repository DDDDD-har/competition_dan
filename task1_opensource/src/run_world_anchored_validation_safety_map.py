#!/usr/bin/env python3
"""Publish the saved RTAB-Map grid and audit scene geometry without pose/TF."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from validation_static_map_contract import (
    combine_static_and_scene_grid,
    rasterize_convex_polygon,
)

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
PROJECT = ROOT.parent
ORCALAB_SITE = Path("/home/dan/miniconda3/envs/orcalab/lib/python3.12/site-packages")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(ORCALAB_SITE) not in sys.path:
    sys.path.insert(0, str(ORCALAB_SITE))


def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    index = 0

    def token() -> bytes:
        nonlocal index
        while index < len(raw):
            if raw[index:index + 1] == b"#":
                index = raw.find(b"\n", index) + 1
            elif raw[index:index + 1].isspace():
                index += 1
            else:
                break
        end = index
        while end < len(raw) and not raw[end:end + 1].isspace():
            end += 1
        value = raw[index:end]
        index = end
        return value

    if token() != b"P5":
        raise RuntimeError(f"only binary P5 PGM maps are supported: {path}")
    width = int(token())
    height = int(token())
    maximum = int(token())
    if maximum != 255:
        raise RuntimeError(f"unsupported PGM maximum value {maximum}")
    while index < len(raw) and raw[index:index + 1].isspace():
        index += 1
    pixels = np.frombuffer(raw[index:], dtype=np.uint8)
    if pixels.size != width * height:
        raise RuntimeError(f"PGM payload size mismatch in {path}")
    return pixels.reshape(height, width)


def load_saved_map(path: Path) -> tuple[np.ndarray, dict]:
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    image_path = (path.parent / metadata["image"]).resolve()
    pixels = np.flipud(read_pgm(image_path)).astype(np.float64)
    probability = ((255.0 - pixels) / 255.0 if not int(metadata.get("negate", 0))
                   else pixels / 255.0)
    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[probability > float(metadata["occupied_thresh"])] = 100
    grid[probability < float(metadata["free_thresh"])] = 0
    return grid, metadata


def world_to_map_xy(xy: np.ndarray, calibration: dict) -> np.ndarray:
    theta = float(calibration["yaw_rad"])
    translation = np.asarray(calibration["translation_xy"], dtype=np.float64)
    rotation = np.asarray(((math.cos(theta), -math.sin(theta)),
                           (math.sin(theta), math.cos(theta))))
    return rotation @ xy + translation


def map_to_world_xy(xy: np.ndarray, calibration: dict) -> np.ndarray:
    theta = float(calibration["yaw_rad"])
    translation = np.asarray(calibration["translation_xy"], dtype=np.float64)
    rotation = np.asarray(((math.cos(theta), -math.sin(theta)),
                           (math.sin(theta), math.cos(theta))))
    return rotation.T @ (xy - translation)


def build_safety_grid(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    import mujoco
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from scene_state import restore_live_state, snapshot_live_state

    grid, map_metadata = load_saved_map(args.mapping_yaml)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    resolution = float(map_metadata["resolution"])
    origin = np.asarray(map_metadata["origin"][:2], dtype=np.float64)
    origin_yaw = float(map_metadata["origin"][2])
    if abs(origin_yaw) > 1e-9:
        raise RuntimeError("rotated occupancy-map origins are not supported")

    address = "127.0.0.1:50051"
    asyncio.set_event_loop(asyncio.new_event_loop())
    saved_qpos, saved_qvel = snapshot_live_state(address)
    env = OrcaGymLocalEnv(
        frame_skip=1, orcagym_addr=address,
        agent_names=[args.robot_name], time_step=0.001,
    )
    marked_geoms: list[dict] = []
    scene_occupied = np.zeros(grid.shape, dtype=bool)
    try:
        restore_live_state(env, saved_qpos, saved_qvel)
        model = env.gym._mjModel
        data = env.gym._mjData
        mujoco.mj_forward(model, data)
        robot_bodies = {
            body_id for body_id in range(model.nbody)
            if (model.body(body_id).name or "").startswith(args.robot_name + "_")
        }
        robot_geoms = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_bodies
        }
        ground_geoms = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
        }

        robot_z_low = math.inf
        robot_z_high = -math.inf
        for geom_id in robot_geoms:
            local_center = model.geom_aabb[geom_id, :3]
            local_extent = model.geom_aabb[geom_id, 3:]
            rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            center = np.asarray(data.geom_xpos[geom_id]) + rotation @ local_center
            extent = np.abs(rotation) @ local_extent
            robot_z_low = min(robot_z_low, float(center[2] - extent[2]))
            robot_z_high = max(robot_z_high, float(center[2] + extent[2]))

        for geom_id in range(model.ngeom):
            if geom_id in robot_geoms or geom_id in ground_geoms:
                continue
            if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
                continue
            local_center = model.geom_aabb[geom_id, :3]
            local_extent = model.geom_aabb[geom_id, 3:]
            rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            center = np.asarray(data.geom_xpos[geom_id]) + rotation @ local_center
            extent = np.abs(rotation) @ local_extent
            low = center - extent
            high = center + extent
            if high[2] < robot_z_low or low[2] > robot_z_high:
                continue
            world_corners = np.asarray([
                (low[0], low[1]), (high[0], low[1]),
                (high[0], high[1]), (low[0], high[1]),
            ])
            map_corners = np.asarray([
                world_to_map_xy(corner, calibration) for corner in world_corners
            ])
            geom_occupied = rasterize_convex_polygon(
                map_corners, origin, resolution, grid.shape
            )
            if not np.any(geom_occupied):
                continue
            scene_occupied |= geom_occupied
            marked_geoms.append({
                "id": geom_id,
                "name": model.geom(geom_id).name or "",
                "world_aabb_xy": [low[:2].tolist(), high[:2].tolist()],
                "world_z_range": [float(low[2]), float(high[2])],
            })
    finally:
        env.close()

    grid, grid_audit = combine_static_and_scene_grid(
        grid, scene_occupied, args.augment_scene_geometry
    )

    from scipy.ndimage import distance_transform_edt, label
    occupied = grid == 100
    clearance = distance_transform_edt(~occupied) * resolution
    traversable = (grid == 0) & (clearance >= args.point_clearance)
    components, _count = label(traversable, structure=np.ones((3, 3), dtype=np.int8))
    start_world = np.asarray(args.start_world_xy, dtype=np.float64)
    start_map = world_to_map_xy(start_world, calibration)
    start_col, start_row = np.rint((start_map - origin) / resolution).astype(int)
    if not (0 <= start_row < grid.shape[0] and 0 <= start_col < grid.shape[1]):
        raise RuntimeError("validation start lies outside the saved map")
    component = int(components[start_row, start_col])
    if component == 0:
        candidates = np.argwhere(traversable)
        if not len(candidates):
            raise RuntimeError("safety map has no traversable cells")
        candidate_map = origin + np.column_stack((candidates[:, 1], candidates[:, 0])) * resolution
        nearest = int(np.argmin(np.linalg.norm(candidate_map - start_map, axis=1)))
        component = int(components[tuple(candidates[nearest])])

    requested = json.loads(args.world_points_json)
    resolved_world: dict[str, list[float]] = {}
    resolved_map: dict[str, list[float]] = {}
    point_audit: dict[str, dict] = {}
    component_cells = np.argwhere(components == component)
    component_map = origin + np.column_stack(
        (component_cells[:, 1] + 0.5, component_cells[:, 0] + 0.5)
    ) * resolution
    for name, values in requested.items():
        desired_world = np.asarray(values, dtype=np.float64)
        desired_map = world_to_map_xy(desired_world, calibration)
        distances = np.linalg.norm(component_map - desired_map, axis=1)
        nearest = int(np.argmin(distances))
        resolved = component_map[nearest]
        shift = float(distances[nearest])
        if shift > args.max_point_shift:
            raise RuntimeError(
                f"no connected safe validation stop within {args.max_point_shift:.2f}m of {name}"
            )
        world_resolved = map_to_world_xy(resolved, calibration)
        row, col = component_cells[nearest]
        resolved_world[str(name)] = world_resolved.tolist()
        resolved_map[str(name)] = resolved.tolist()
        point_audit[str(name)] = {
            "requested_world_xy": desired_world.tolist(),
            "requested_map_xy": desired_map.tolist(),
            "resolved_world_xy": world_resolved.tolist(),
            "resolved_map_xy": resolved.tolist(),
            "shift_m": shift,
            "obstacle_clearance_m": float(clearance[row, col]),
        }

    report = {
        "format": "world_anchored_validation_safety_map_v1",
        "topic": args.topic,
        "frame": "map",
        "source_map": str(args.mapping_yaml.resolve()),
        "source_calibration": str(args.calibration.resolve()),
        "map_origin": map_metadata["origin"],
        "resolution": resolution,
        "size": [int(grid.shape[1]), int(grid.shape[0])],
        "robot_world_z_envelope": [robot_z_low, robot_z_high],
        "scene_collision_geom_count": len(marked_geoms),
        **grid_audit,
        "point_clearance_m": args.point_clearance,
        "route_component": component,
        "resolved_world_points": resolved_world,
        "resolved_map_points": resolved_map,
        "point_audit": point_audit,
        "pose_odom_tf_published": False,
        "truth_role": (
            "validation-route collision avoidance"
            if args.augment_scene_geometry
            else "offline scene-consistency audit only; not applied to navigation grid"
        ),
        "marked_scene_geoms": marked_geoms,
    }
    return grid, report


class SafetyMapPublisher(Node):
    def __init__(self, args: argparse.Namespace, grid: np.ndarray, report: dict) -> None:
        super().__init__("world_anchored_validation_safety_map")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(OccupancyGrid, args.topic, qos)
        self.message = OccupancyGrid()
        self.message.header.frame_id = "map"
        self.message.info.resolution = float(report["resolution"])
        self.message.info.width = int(report["size"][0])
        self.message.info.height = int(report["size"][1])
        origin = report["map_origin"]
        self.message.info.origin.position.x = float(origin[0])
        self.message.info.origin.position.y = float(origin[1])
        self.message.info.origin.orientation.z = math.sin(float(origin[2]) * 0.5)
        self.message.info.origin.orientation.w = math.cos(float(origin[2]) * 0.5)
        self.message.data = grid.ravel().tolist()
        self.create_timer(0.5, self.publish)
        self.publish()

    def publish(self) -> None:
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="service_robot_1")
    parser.add_argument("--mapping-yaml", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--world-points-json", required=True)
    parser.add_argument("--start-world-xy", nargs=2, type=float, required=True)
    parser.add_argument("--point-clearance", type=float, default=0.50)
    parser.add_argument("--max-point-shift", type=float, default=1.50)
    parser.add_argument(
        "--augment-scene-geometry", action="store_true",
        help=("explicit legacy diagnostic mode: union MuJoCo collision AABBs into "
              "the saved map; default publishes the source PGM exactly"),
    )
    parser.add_argument("--topic", default="/validation/safety_map")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    grid, report = build_safety_grid(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rclpy.init()
    node = SafetyMapPublisher(args, grid, report)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
