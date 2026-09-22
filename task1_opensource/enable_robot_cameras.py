#!/usr/bin/env python3
"""Enable color output for every robot-mounted OrcaLab camera in the live scene."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

from orcalab.actor_property import ActorEntities, ActorPropertyKey
from orcalab.path import Path
from orcalab.protos.edit_service_wrapper import EditServiceWrapper


CAMERA_COMPONENT = "{CA85BDD5-C631-4C52-853C-06D654AE7E4D}"
ROBOT_ACTOR_PATH = "/industrial_collaborative_robot_1"
# Keep this mapping independent of the editor's entity traversal order.
COLLAB_CAMERA_PORTS = {
    "camera_head": (7070, 7071),
    "camera_left": (7072, 7073),
    "camera_right": (7074, 7075),
}
HUMANOID_CAMERA_PORTS = {
    "camera_head": (7070, 7071),
    "camera_wrist_r": (7080, 7081),
    "camera_wrist_l": (7090, 7091),
}


async def walk(node, nodes) -> None:
    nodes.append(node)
    for child in node.children:
        await walk(child, nodes)


async def enable_cameras(address: str, retries: int, robot_actor: str = ROBOT_ACTOR_PATH,
                         head_depth: bool = False, wrist_enabled: bool = True,
                         restart_streams: bool = False,
                         head_rgb: bool = True, disable_all: bool = False,
                         all_depth: bool = False) -> list[dict]:
    service = EditServiceWrapper()
    service.init_grpc(address)
    try:
        for attempt in range(1, retries + 1):
            cameras_by_actor = defaultdict(set)
            for camera in await service.get_cameras():
                # Ignore the editor camera and cameras that do not belong to AgiBot G1.
                if camera.actor_path == robot_actor:
                    cameras_by_actor[camera.actor_path].add(camera.name)

            keys = []
            values_to_set = []
            configured_by_name = {}
            for actor_name in cameras_by_actor:
                camera_names = cameras_by_actor[actor_name]
                camera_ports = (HUMANOID_CAMERA_PORTS
                                if {"camera_wrist_l", "camera_wrist_r"} <= camera_names
                                else COLLAB_CAMERA_PORTS)
                actor_path = Path(actor_name)
                roots = await service.get_entity_hierarchy_batch([actor_path])
                if not roots or roots[0] is None or roots[0].entity_id == 0:
                    continue

                nodes = []
                await walk(roots[0], nodes)
                for node in nodes:
                    if node.name not in camera_ports or node.name not in camera_names:
                        continue
                    groups_result = await service.get_entity_property_groups_batch(
                        [ActorEntities(actor_path, [node.entity_id])]
                    )
                    for entity_groups in groups_result:
                        for component_groups in entity_groups:
                            for group in component_groups:
                                if group.component_type_id != CAMERA_COMPONENT:
                                    continue
                                props = {prop.name(): prop for prop in group.properties}
                                if not {"ColorCamera", "Enable", "ColorPort", "DepthPort", "VerticalFieldOfViewDeg"} <= props.keys():
                                    continue
                                expected_ports = camera_ports[node.name]
                                property_offset = len(keys)
                                is_head = node.name == "camera_head"
                                enabled = (is_head or wrist_enabled) and not disable_all
                                property_values = [("ColorCamera", bool(enabled and (not is_head or head_rgb)))]
                                if "DepthCamera" in props:
                                    property_values.append(("DepthCamera", bool(
                                        enabled and (all_depth or (head_depth and is_head))
                                    )))
                                property_values.extend([
                                    ("Enable", enabled),
                                    ("ColorPort", expected_ports[0]),
                                    ("DepthPort", expected_ports[1]),
                                ])
                                if "IsRecording" in props:
                                    property_values.append(("IsRecording", bool(enabled)))
                                if "UseNvEnc" in props:
                                    # OrcaManipulation's working camera scenes explicitly
                                    # enable the engine-side H.264 encoder.
                                    property_values.append(("UseNvEnc", bool(enabled)))
                                for property_name, value in property_values:
                                    prop = props[property_name]
                                    keys.append(
                                        ActorPropertyKey(
                                            actor_path,
                                            node.entity_id,
                                            node.entity_path,
                                            CAMERA_COMPONENT,
                                            group.component_type_index,
                                            property_name,
                                            prop.value_type(),
                                        )
                                    )
                                    values_to_set.append(value)
                                configured_by_name[node.name] = {
                                    "actor": actor_name,
                                    "name": node.name,
                                    "color_port": expected_ports[0],
                                    "depth_port": expected_ports[1],
                                    "vertical_fov": float(props["VerticalFieldOfViewDeg"].value()),
                                    "property_offset": property_offset,
                                    "property_names": [name for name, _value in property_values],
                                }

            expected_names = set(camera_ports) if cameras_by_actor else set()
            if keys and set(configured_by_name) == expected_names:
                configured = [configured_by_name[name] for name in camera_ports]
                if restart_streams:
                    disabled_values = list(values_to_set)
                    for camera in configured:
                        offset = camera["property_offset"]
                        names = camera["property_names"]
                        for prop_name in ("ColorCamera", "DepthCamera", "Enable", "IsRecording", "UseNvEnc"):
                            if prop_name in names:
                                disabled_values[offset + names.index(prop_name)] = False
                    await service.set_properties(keys, disabled_values)
                    await asyncio.sleep(1.0)
                await service.set_properties(keys, values_to_set)
                values = await service.get_properties(keys)
                for camera in configured:
                    offset = camera.pop("property_offset")
                    property_names = camera.pop("property_names")
                    readback = {
                        name: values[offset + index].value
                        for index, name in enumerate(property_names)
                    }
                    camera.update({
                        "color_camera": readback["ColorCamera"],
                        "depth_camera": readback.get("DepthCamera", "unsupported"),
                        "enabled": readback["Enable"],
                        "color_port": readback["ColorPort"],
                        "depth_port": readback["DepthPort"],
                        "is_recording": readback.get("IsRecording", "unsupported"),
                        "use_nvenc": readback.get("UseNvEnc", "unsupported"),
                    })
                    print(
                        f"{camera['actor']}:{camera['name']}: ColorCamera={camera['color_camera']}, "
                        f"DepthCamera={camera['depth_camera']}, Enable={camera['enabled']}, "
                        f"ColorPort={camera['color_port']}, "
                        f"DepthPort={camera['depth_port']}, "
                        f"IsRecording={camera['is_recording']}, UseNvEnc={camera['use_nvenc']}"
                    )
                return configured

            if configured_by_name:
                missing = sorted(expected_names - set(configured_by_name))
                print(f"Waiting for required camera components: {', '.join(missing)}")

            if attempt < retries:
                print(f"No live robot camera entity yet; retrying ({attempt}/{retries})...")
                await asyncio.sleep(1.0)
        raise RuntimeError(
            f"No camera components found on {robot_actor}. "
            "Start/play the OrcaLab scene, then run this command again."
        )
    finally:
        await service.destroy_grpc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enable all live robot color cameras")
    parser.add_argument("--edit-addr", default="127.0.0.1:50151")
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument(
        "--robot-actor", default=ROBOT_ACTOR_PATH,
        help="live OrcaLab actor path; defaults to the saved AgiBot G1 path",
    )
    parser.add_argument("--head-depth", action="store_true",
                        help="enable the existing head camera depth stream")
    parser.add_argument("--all-depth", action="store_true",
                        help="enable depth on head and both wrist/gripper cameras")
    parser.add_argument("--no-head-rgb", action="store_true",
                        help="disable head RGB while retaining optional head depth (diagnostic mode)")
    parser.add_argument("--disable-wrist-cameras", action="store_true",
                        help="disable both existing wrist cameras during task-1 navigation")
    parser.add_argument("--restart-streams", action="store_true",
                        help="disable then re-enable CameraSensor components so changed ports take effect")
    parser.add_argument("--disable-all", action="store_true",
                        help="disable every camera on the selected robot")
    args = parser.parse_args()
    asyncio.run(enable_cameras(args.edit_addr, args.retries, args.robot_actor,
                               args.head_depth, not args.disable_wrist_cameras,
                               args.restart_streams, not args.no_head_rgb, args.disable_all,
                               args.all_depth))


if __name__ == "__main__":
    main()
