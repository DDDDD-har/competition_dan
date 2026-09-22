#!/usr/bin/env python3
"""Current Task 1 scene, robot, and A/B/C world-coordinate contract.

Adaptive and non-adaptive navigation share this contract. The two schemes
only differ by ``--validation-adaptive-route-profile``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENE_FILE = ROOT / "g1_button.json"
SCENE_NAME = "g1_button.json"
ROBOT_NAME = "g1_omnipicker"
ROBOT_PATH = "/g1_omnipicker"
ROBOT_HOLDER_BODY = f"{ROBOT_NAME}_robot_holder1"
CAMERA_HEAD_BODY = f"{ROBOT_NAME}_camera_head_body2"
CHASSIS_DEVICE_SPEED_LIMIT_MPS = 30.0
STAGED_SPEED_STEP_MPS = 0.5

# Scene inspection marks (cabinet / buttons). Used by wrapup mapping only.
INSPECTION_WORLD_POINTS_XYZ: dict[str, tuple[float, float, float]] = {
    "A": (43.641, -27.776, 0.203),
    "B": (30.422, -27.776, 0.203),
    "C": (19.205, -22.952, 0.203),
}

# Optional non-scored via waypoints (names starting with "_" are never scored).
# The fast validation profile plans A→B→C directly with Nav2; no hand-placed
# geometry is injected here.
VALIDATION_VIA_WORLD_POINTS_XYZ: dict[str, tuple[float, float, float]] = {}
# Order handed to the navigator as the mission route.
VALIDATION_ROUTE_ORDER: tuple[str, ...] = ("A", "B", "C")

# Legacy fallback coordinates for tools that run without a live scene.
# Formal validation replaces them at initialization by querying
# Static_Location_A/B/C_site through OrcaGym.
WORLD_POINTS_XYZ: dict[str, tuple[float, float, float]] = {
    "A": (43.305, -27.564, 0.203),
    "B": (30.702, -27.564, 0.203),
    "C": (19.122, -22.982, 0.203),
}

# The validation navigator checks a 0.76 x 0.74 m conservative chassis
# footprint with an additional 0.03 m swept-path margin.  A 0.65 m
# centre-to-obstacle clearance keeps that inflated footprint safe at every
# heading and leaves a small map-discretisation allowance.  The C inspection
# point is close to the cabinet, so its navigation stop may be shifted within
# 0.75 m while the requested world coordinate remains recorded in the audit.
# This is centre-point connectivity clearance only; Nav2 separately checks the
# full anisotropic footprint.  Using the old 0.65 m circular proxy rejected C
# even though the scanned map contains a valid footprint-safe approach.
VALIDATION_POINT_CLEARANCE_M = 0.40
VALIDATION_MAX_POINT_SHIFT_M = 0.75
# Official point_in_region radius is 0.60 m. Internal acceptance uses the same
# circle so local validation matches orca_scorer_client checkpoint rules.
OFFICIAL_REGION_RADIUS_M = 0.60
VALIDATION_COORDINATE_TOLERANCE_M = 0.60
# Tighter A/B pass trigger so g1_omnipicker_body_link1 stays inside the
# official 0.60 m scorer circle long enough for full checkpoint credit.
VALIDATION_WAYPOINT_PASS_TOLERANCE_M = 0.36
# After Hybrid-A* reaches site C, walk back along that path — not the B→C
# chord, which would shift the stop east into the cabinet-side obstacle.
VALIDATION_SCORE_CIRCLE_STOP_OFFSET_M = 0.18
# Terminal arrival ramp is disabled for live execution: keep cruise speed
# through the scored C circle and rely on checkpoint detection only.
VALIDATION_TERMINAL_ARRIVAL_SPEED_MPS = 0.0
VALIDATION_TERMINAL_ZONE_M = 0.0
# C-leg bend is ~14 m; keep cross-track braking active until the last approach.
VALIDATION_TERMINAL_APPROACH_REMAINING_M = 5.5
# A/B hall straights and early C hall before the Nav2 bend: run at contract peak.
VALIDATION_C_LEG_STRAIGHT_CAP_MPS = 0.80
# Slow only when curvature rises on the B→C bend (lookahead along the profile arc).
VALIDATION_C_LEG_PRE_BEND_LOOKAHEAD_M = 5.0
VALIDATION_C_LEG_PRE_BEND_CURVATURE_1PM = 0.12
VALIDATION_C_LEG_PRE_BEND_CAP_MPS = 2.5
# B→C bend window: suppress cross-track braking through the committed turn.
VALIDATION_C_LEG_COMMITTED_TURN_LOOKAHEAD_M = 2.5
# Heading-window stats for C-leg speed-profile bend detection only.
# Do not chop FollowPath execution with these windows: a Hybrid Dubins C
# path is one remaining-horizon local plan, not four controller slices.
VALIDATION_C_LEG_TURN_HEADING_CHANGE_RAD = 0.35
VALIDATION_C_LEG_TURN_ZONE_WINDOW_M = 1.2
VALIDATION_C_LEG_TURN_MIN_ARC_M = 1.0
VALIDATION_C_LEG_TURN_MIN_STRAIGHT_M = 1.5
# After the global path reaches a scored goal, hold still this long, then
# local-replan the remainder from the live pose.
VALIDATION_LOCAL_REPLAN_SETTLE_S = 0.050
# Primary bend: sustained curvature above this marks the real B→C corner.
VALIDATION_C_LEG_PRIMARY_TURN_CURVATURE_1PM = 0.30
# Low-curvature legs tracked with straight_cruise (Nav2 path tangent hold).
VALIDATION_STRAIGHT_CRUISE_LEG_NAMES: tuple[str, ...] = ("A", "B")
# Relaxed cross-track braking on the C-leg hall straight before the bend.
VALIDATION_C_LEG_CROSS_TRACK_ONSET_M = 0.35
VALIDATION_C_LEG_CROSS_TRACK_FULL_M = 0.75
# Post-A B hall: allow sustained acceleration with less cross-track braking.
VALIDATION_POST_A_CROSS_TRACK_ONSET_M = 0.20
VALIDATION_POST_A_CROSS_TRACK_FULL_M = 0.55
# Target body_link1 depth inside the official 0.60 m circle for full checkpoint credit.
OFFICIAL_CHECKPOINT_DEEP_RADIUS_M = 0.45
# Cap linear speed while deepening A/B fly-through entry.
VALIDATION_CHECKPOINT_FLYTHROUGH_CAP_MPS = 0.65
# Slow C terminal capture to avoid cabinet-side environment contacts.
VALIDATION_C_TERMINAL_CAPTURE_CAP_MPS = 0.45
# Drive body_link1 deeper into the 0.60 m scorer circle before stopping capture.
OFFICIAL_CIRCLE_CAPTURE_RADIUS_SCALE = 0.55
VALIDATION_C_CHECKPOINT_PASS_M = 0.30
# A/B fly-through: body_link1 must enter the official 0.60 m circle once.
OFFICIAL_CHECKPOINT_PASS_RADIUS_SCALE = 0.95
# start_move: base_velocity_norm >= 0.05 m/s for 1.0 s on body_link1.
OFFICIAL_SCORER_START_MOVE_MIN_MPS = 0.06
OFFICIAL_SCORER_START_MOVE_HOLD_S = 1.2
# start_move needs body_link1 speed >= 0.05 m/s for 1.0 s; keep start→A > 1 s.
VALIDATION_START_TO_A_MIN_ELAPSED_S = 1.0
# Parent validation waits for this marker before starting official scoring.
ROBOT_STARTED_MARKER_NAME = "robot_started.marker"
# Departure profile: hold 0.20 m/s through start_move, then smoothstep to cruise.
VALIDATION_STARTUP_SPEED_M = 0.20
VALIDATION_STARTUP_RAMP_S = 2.0
VALIDATION_STARTUP_PLATEAU_S = 1.5
VALIDATION_STARTUP_MIN_MPS = VALIDATION_STARTUP_SPEED_M
VALIDATION_STARTUP_LINEAR_ACCEL_MPS2 = 2.0
VALIDATION_STARTUP_LINEAR_JERK_MPS3 = 1.5
# Mid-run command slew limits for steadier cruise speed.
VALIDATION_CRUISE_COMMAND_ACCEL_MPS2 = 3.0
VALIDATION_CRUISE_COMMAND_JERK_MPS3 = 1.5
VALIDATION_PROFILE_LOOKAHEAD_M = 0.65
# Scale lookahead with contract speed so 5 m/s tracking previews enough path.
VALIDATION_LOOKAHEAD_SPEED_FACTOR = 0.22
# Path-profile clearance braking: west hall / last cabinet turn have ~0.36-0.41 m
# footprint margin. Soft must sit above that so the 5 m/s hall contract tapers
# before the apex; hard stays below the C heading-search floor (~0.32 m).
VALIDATION_PROFILE_CLEARANCE_HARD_M = 0.28
VALIDATION_PROFILE_CLEARANCE_SOFT_M = 0.80
# Live C tracking: below this footprint clearance, never keep hall-cruise speed.
# 1.25 m/s is ω_max at κ≈1.0, so 2WS can still match the last cabinet arc.
VALIDATION_C_LEG_TIGHT_CLEARANCE_M = 0.55
VALIDATION_C_LEG_TIGHT_CLEARANCE_CAP_MPS = 1.25
# Smac Hybrid: keep planner penalties at the proven preflight values. Raising
# cost_penalty or global inflation blocks B→C in the narrow x≈23 corridor.
VALIDATION_PLANNER_COST_PENALTY = 2.0
VALIDATION_PLANNER_NON_STRAIGHT_PENALTY = 1.15
# Hybrid last-cabinet Dubins target. Do not write this into the global Hybrid
# YAML: session 123224 used 1.35 m on A/B/C and overshot the first C bend.
# Last-turn clearance comes from widen_last_c_turn plus north terminal headings.
VALIDATION_C_PLANNER_TURNING_RADIUS_M = 1.35
# After Hybrid returns, push only the last C turn away from occupied cells.
VALIDATION_C_TURN_TARGET_CLEARANCE_M = 0.48
VALIDATION_C_TURN_MAX_PUSH_M = 0.16
# Do not freeze the mission 1.3 m from C: official 0.60 m circle + small slack.
VALIDATION_C_STALL_GRACE_M = 0.70
# Brief zero-command hold at scored A/B fly-through points (official circle entry).
VALIDATION_AB_CHECKPOINT_DWELL_S = 0.050
# Optional hold after robot_started.marker before route motion (0 = immediate).
VALIDATION_POST_ROBOT_START_DWELL_S = 0.0
# Minimum linear command when capturing a scored checkpoint (m/s).
VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS = 0.28
# Curvature below this threshold uses peak_contract_speed on straights (1/m).
VALIDATION_STRAIGHT_CURVATURE_1PM = 0.25
# Above this contract speed, profile/chassis accel is raised for faster departure.
VALIDATION_HIGH_SPEED_THRESHOLD_MPS = 10.0
VALIDATION_HIGH_SPEED_PROFILE_ACCEL_MPS2 = 20.0
# Official scorer samples body_link1 XY; for g1_omnipicker holder navigation
# base the planar offset to body_link1 is ~0 (vertical slide only).
BODY_LINK_FORWARD_OFFSET_M = 0.0
# Adaptive speed near A/B fly-through zones (official circle radius 0.60 m).
# Official scoring only requires entering the 0.60 m circle; do not crawl.
CHECKPOINT_ZONE_RADIUS_M = OFFICIAL_REGION_RADIUS_M * 4.0
CHECKPOINT_ZONE_CAP_MPS = 8.0
CHECKPOINT_ZONE_INNER_CAP_MPS = 6.0
CHECKPOINT_ZONE_INNER_RADIUS_M = OFFICIAL_REGION_RADIUS_M * 1.5
# Preflight rejects corridor legs that deviate too far from the A→B chord.
VALIDATION_MAX_CORRIDOR_LATERAL_M = 0.25
VALIDATION_MIN_PATH_CLEARANCE_M = 0.30
# Official validation mission wall-clock limit (seconds).
VALIDATION_MISSION_TIMEOUT_S = 180.0
# Abort route/mission when wall-clock exceeds VALIDATION_MISSION_TIMEOUT_S (official 180 s).
VALIDATION_MISSION_DEADLINE_ENABLED = True
# Preflight aborts when map pose is farther than this from the localized spawn seed.
VALIDATION_START_POSE_MAX_SHIFT_M = 3.0
# SmacPlannerHybrid first configure can take >60 s on a large costmap.
VALIDATION_NAV2_STARTUP_TIMEOUT_S = 180.0
# Faster localization gate for world-seeded validation (no global loop match).
VALIDATION_STATIC_GATE_WINDOW_S = 3.0
VALIDATION_STATIC_GATE_MIN_SAMPLES = 10
# Fewer TF callbacks before the first navigation command.
VALIDATION_STARTUP_POSE_SAMPLES = 2
# Proven 20 m/s nav2_follow_path profile/MPPI accel ceiling (session 153431).
VALIDATION_PROFILE_LINEAR_ACCEL_MPS2 = 6.5
# Mission fails once the robot has started moving and then remains stopped.
VALIDATION_MOTION_STALL_SPEED_MPS = 0.05
VALIDATION_MOTION_STALL_DISPLACEMENT_M = 0.10
VALIDATION_MOTION_STALL_TIMEOUT_S = 5.0
VALIDATION_MOTION_STALL_STARTUP_GRACE_S = 8.0
# Staged formal runs resume from the current 5.0 m/s validated contract.
STAGED_SPEED_START_MPS = 2.0
# g1_omnipicker scene inventory (FL/FR x=+0.21 m, BL/BR x=-0.21 m, track y=±0.13 m).
# Drive: wheel_fl_joint_mctrl, wheel_fr_joint_mctrl.
# Steer: wheel_fl_steer_joint_pctrl, wheel_fr_steer_joint_pctrl (front 2WS only).
CHASSIS_WHEELBASE_M = 0.42
CHASSIS_TRACK_M = 0.26
CHASSIS_MAX_STEER_RAD = 0.60
# MuJoCo bridge maps cmd_vel linear to actuator scale via
# forward = clip(cmd / max_linear_speed) * speed, then speed_scale = forward / speed.
# Both max_linear_speed and speed must stay at this device ceiling; cruise limits
# come from the navigator cmd_vel only. Lowering either to validation_speed pins
# speed_scale to 1.0 and the robot runs at full actuator speed.
CHASSIS_BRIDGE_ACTUATOR_FULL_SCALE_MPS = 0.60
CHASSIS_DRIVE_WHEELS = ("fl", "fr")
CHASSIS_STEER_WHEELS = ("fl", "fr")
# 2WS Ackermann: R = L / tan(δ_max) ≈ 0.647 m. Planner uses a larger radius so
# Hybrid-A* emits a Dubins arc the chassis can track instead of a 90° L-corner.
CHASSIS_KINEMATIC_TURNING_RADIUS_M = (
    CHASSIS_WHEELBASE_M / math.tan(CHASSIS_MAX_STEER_RAD)
)
VALIDATION_TURNING_RADIUS_MARGIN = 1.25
VALIDATION_MIN_TURNING_RADIUS_M = (
    CHASSIS_KINEMATIC_TURNING_RADIUS_M * VALIDATION_TURNING_RADIUS_MARGIN
)
# The OmniPicker steers its front wheels and cannot yaw in place, so a commanded
# (v, w) pair is only executable when |w| <= v / VALIDATION_MIN_TURNING_RADIUS_M.
# Planner, speed profile, MPPI and hand-written trackers all share this single
# curvature envelope so linear and angular commands stay mutually consistent.
VALIDATION_MAX_PATH_CURVATURE_1PM = 1.0 / VALIDATION_MIN_TURNING_RADIUS_M
VALIDATION_PROFILE_LATERAL_ACCEL_MPS2 = 1.6
VALIDATION_MAX_ANGULAR_RADPS = 1.25
# Default official validation speed. Staged tests stop at their first failed
# safety acceptance and never interpret this device ceiling as an achieved
# speed.
# ω_max/κ on the tightest planned C bends is ~1.08 m/s; 0.80 m/s keeps 2WS
# inside the curvature envelope with margin for pure-pursuit lag.
VALIDATION_SPEED_M = 0.80
VALIDATION_MAX_SPEED_M = CHASSIS_DEVICE_SPEED_LIMIT_MPS
# Navigation must consume the accepted scanned occupancy map exactly. MuJoCo
# geometry is retained for collision auditing only and is never merged into
# the costmap.
VALIDATION_AUGMENT_SCENE_GEOMETRY = False

# Accepted mapping chain after the successful spawn→A→B→C long revisit.
# Adaptive and non-adaptive localization must both consume this same directory.
DEFAULT_MAPPING_DIR = (
    ROOT / "data" / "world_anchored_rtabmap_20260821T195241+0800"
)

# g1_button.json spawn recorded in mapping_manifest initial_world_pose for the
# accepted map session above (south corridor, west of the x≈47.6 wall).
SPAWN_XY = (46.325947, -27.572001)
SPAWN_YAW_RAD = 3.141592653589793

# Historical north-corridor start used only by ``--input task1_wrapup``.
WRAPUP_SPAWN_XY = (43.641, -1.272)

# Fresh-map route used by ``--input task1_wrapup``. It starts from a scene
# reset and returns over the same visual corridor so one uninterrupted
# database receives long-range loop closures without any pose reset.
WRAPUP_WAYPOINTS: tuple[tuple[str, float, float], ...] = (
    ("spawn",) + WRAPUP_SPAWN_XY,
    ("spawn_depart", WRAPUP_SPAWN_XY[0], -5.0),
    ("spawn_mid", WRAPUP_SPAWN_XY[0], -15.0),
    ("A_approach", WRAPUP_SPAWN_XY[0], -25.5),
    # Drive onto the red square, then scan the open north and west edges.
    # East/south of A are the end wall and fence; do not orbit those sides.
    ("A",) + INSPECTION_WORLD_POINTS_XYZ["A"][:2],
    ("A_edge_n", INSPECTION_WORLD_POINTS_XYZ["A"][0] - 0.20, INSPECTION_WORLD_POINTS_XYZ["A"][1] + 1.20),
    ("A_edge_w", INSPECTION_WORLD_POINTS_XYZ["A"][0] - 1.20, INSPECTION_WORLD_POINTS_XYZ["A"][1]),
    ("A_on_again",) + INSPECTION_WORLD_POINTS_XYZ["A"][:2],
    # The chassis cannot make a 90-degree forward turn against the end wall.
    # Names ending in _reverse are executed as straight reverse maneuvers.
    ("A_reverse", 43.40, -24.80),
    ("A_west", 40.50, -27.20),
    ("B",) + INSPECTION_WORLD_POINTS_XYZ["B"][:2],
    ("B_edge_n", INSPECTION_WORLD_POINTS_XYZ["B"][0], INSPECTION_WORLD_POINTS_XYZ["B"][1] + 1.20),
    ("B_edge_w", INSPECTION_WORLD_POINTS_XYZ["B"][0] - 1.20, INSPECTION_WORLD_POINTS_XYZ["B"][1]),
    ("B_edge_e", INSPECTION_WORLD_POINTS_XYZ["B"][0] + 1.20, INSPECTION_WORLD_POINTS_XYZ["B"][1]),
    ("B_on_again",) + INSPECTION_WORLD_POINTS_XYZ["B"][:2],
    ("C_hall", 19.80, INSPECTION_WORLD_POINTS_XYZ["B"][1]),
    ("C_reverse", 22.00, -28.80),
    ("C_approach", INSPECTION_WORLD_POINTS_XYZ["C"][0], -25.3),
    ("C",) + INSPECTION_WORLD_POINTS_XYZ["C"][:2],
    # Red-square edges at C. The cabinet sits north of C, so scan south/east/west.
    ("C_edge_s", INSPECTION_WORLD_POINTS_XYZ["C"][0], INSPECTION_WORLD_POINTS_XYZ["C"][1] - 1.20),
    ("C_edge_e", INSPECTION_WORLD_POINTS_XYZ["C"][0] + 1.20, INSPECTION_WORLD_POINTS_XYZ["C"][1]),
    ("C_edge_w", INSPECTION_WORLD_POINTS_XYZ["C"][0] - 1.20, INSPECTION_WORLD_POINTS_XYZ["C"][1]),
    ("C_on_again",) + INSPECTION_WORLD_POINTS_XYZ["C"][:2],
    # Back out, then orbit the combined cabinet/table footprint at 1-3 m
    # sensor range so all four sides produce occupied cells in the 2D grid.
    ("C_exit_reverse", 19.00, -26.10),
    ("outline_west_south", 17.40, -25.50),
    ("outline_west_mid", 17.40, -23.00),
    ("outline_west_north", 17.40, -20.20),
    ("outline_north_mid", 19.50, -20.20),
    ("outline_north_east", 22.20, -20.20),
    ("outline_east_mid", 22.20, -22.80),
    ("outline_east_south", 22.20, -26.10),
    ("outline_south_mid", 20.00, -26.10),
    ("outline_south_west", 17.40, -26.10),
)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def wrapup_wasd(
    x: float,
    y: float,
    yaw: float,
    target_x: float,
    target_y: float,
    arrive_m: float = 0.55,
) -> dict[str, object]:
    """Choose W/A/S/D for a world-frame waypoint."""
    dx = target_x - x
    dy = target_y - y
    dist = math.hypot(dx, dy)
    if dist <= arrive_m:
        return {"W": 0, "A": 0, "S": 0, "D": 0, "arrived": True, "dist": dist}
    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - yaw)
    state = {"W": 0, "A": 0, "S": 0, "D": 0, "arrived": False, "dist": dist}
    if err > math.radians(18.0):
        state["A"] = 1
    elif err < -math.radians(18.0):
        state["D"] = 1
    # Four-wheel steer yaws while moving. Normal 90-degree corridor corners
    # must use a forward arc; only a nearly opposite heading uses reverse.
    if abs(err) > math.radians(135.0):
        state["S"] = 1
    else:
        state["W"] = 1
    return state


def world_points_xy() -> dict[str, list[float]]:
    return {name: [xyz[0], xyz[1]] for name, xyz in WORLD_POINTS_XYZ.items()}


def order_route_points(points: dict) -> dict[str, list[float]]:
    """Order a route mapping as A -> B -> via -> C; unknown names keep their order."""
    ordered: dict[str, list[float]] = {
        name: [float(points[name][0]), float(points[name][1])]
        for name in VALIDATION_ROUTE_ORDER
        if name in points
    }
    for name, values in points.items():
        if name not in ordered:
            ordered[str(name)] = [float(values[0]), float(values[1])]
    return ordered


def scored_route_points(points: dict) -> dict[str, list[float]]:
    """Drop the non-scored via waypoints; only A/B/C are judged."""
    return {
        str(name): [float(values[0]), float(values[1])]
        for name, values in points.items()
        if not str(name).startswith("_")
    }


def validation_route_order(stop_at: str | None = None) -> tuple[str, ...]:
    """Scored checkpoints included in a validation run (default: full A→B→C)."""
    terminal = (stop_at or "C").strip().upper()
    if terminal not in VALIDATION_ROUTE_ORDER:
        raise ValueError(
            f"route stop must be one of {VALIDATION_ROUTE_ORDER}, got {stop_at!r}"
        )
    index = VALIDATION_ROUTE_ORDER.index(terminal)
    return VALIDATION_ROUTE_ORDER[: index + 1]


def validation_route_world_points(
    site_points: dict,
    *,
    stop_at: str | None = None,
) -> dict[str, list[float]]:
    """World-frame route XY for the scored sites up to ``stop_at``."""
    merged = dict(site_points)
    merged.update(VALIDATION_VIA_WORLD_POINTS_XYZ)
    ordered = order_route_points(merged)
    route_names = validation_route_order(stop_at)
    return {
        str(name): ordered[name]
        for name in route_names
        if name in ordered
    }


def checkpoint_settle_duration_s(
    name: str,
    *,
    ab_checkpoint_dwell_s: float = VALIDATION_AB_CHECKPOINT_DWELL_S,
    checkpoint_settle_s: float = 0.0,
) -> float:
    """Zero-command dwell duration after passing a scored checkpoint name."""
    if str(name).startswith("_"):
        return 0.0
    if str(name) in ("A", "B"):
        return max(0.0, float(ab_checkpoint_dwell_s))
    return max(0.0, float(checkpoint_settle_s))


def world_points_xy_json() -> str:
    return json.dumps(
        validation_route_world_points(world_points_xy()), separators=(",", ":")
    )


def contract_map_points_from_safety_report(safety_report: dict) -> dict[str, list[float]]:
    """Map-frame coverage targets: requested world points, not shifted stops."""
    audit = safety_report.get("point_audit")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("safety report is missing point_audit")
    points: dict[str, list[float]] = {}
    for name, item in audit.items():
        if not isinstance(item, dict) or "requested_map_xy" not in item:
            raise ValueError(f"{name} is missing requested_map_xy")
        values = item["requested_map_xy"]
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"{name} requested_map_xy must contain [x, y]")
        points[str(name)] = [float(values[0]), float(values[1])]
    return points


def parse_world_points(raw: str | dict) -> dict[str, list[float]]:
    """Accept ``[x, y]`` or ``[x, y, z]`` named points; return planning XY."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or not data:
        raise ValueError("world points must be a non-empty JSON object")
    points: dict[str, list[float]] = {}
    for name, values in data.items():
        if not isinstance(values, (list, tuple)) or len(values) not in {2, 3}:
            raise ValueError(f"{name} must contain [x, y] or [x, y, z]")
        points[str(name)] = [float(values[0]), float(values[1])]
    return points


def scorer_aligned_root_target(
    site_xy: tuple[float, float],
    yaw_rad: float,
    *,
    forward_offset_m: float = BODY_LINK_FORWARD_OFFSET_M,
) -> tuple[float, float]:
    """Shift the tracked root so body_link1 crosses the official site centre."""
    offset = max(0.0, float(forward_offset_m))
    yaw = float(yaw_rad)
    return (
        float(site_xy[0]) - offset * math.cos(yaw),
        float(site_xy[1]) - offset * math.sin(yaw),
    )


def scoring_speed_policy(
    *,
    phase: str,
    contract_speed_mps: float,
    speed_cap_mps: float,
    checkpoint_distance_m: float = math.inf,
    checkpoint_tol_m: float = VALIDATION_WAYPOINT_PASS_TOLERANCE_M,
    checkpoint_capture_min_mps: float = VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS,
) -> float:
    """Score-aware linear speed cap for execution phases."""
    peak = min(float(contract_speed_mps), float(speed_cap_mps))
    if (
        phase == "checkpoint_zone"
        and checkpoint_distance_m <= OFFICIAL_REGION_RADIUS_M * 1.2
        and checkpoint_distance_m > OFFICIAL_CHECKPOINT_DEEP_RADIUS_M
    ):
        peak = min(peak, VALIDATION_CHECKPOINT_FLYTHROUGH_CAP_MPS)
    del checkpoint_tol_m, checkpoint_capture_min_mps
    return peak


def startup_ramp_scale(
    elapsed_s: float | None,
    *,
    ramp_s: float = VALIDATION_STARTUP_RAMP_S,
) -> float:
    """Smoothstep 0→1 over the post-A cruise ramp window."""
    if elapsed_s is None:
        return 0.0
    if float(elapsed_s) >= float(ramp_s):
        return 1.0
    t = float(elapsed_s) / float(ramp_s)
    return t * t * (3.0 - 2.0 * t)


def start_to_a_motion_elapsed_ok(
    elapsed_s: float | None,
    *,
    min_elapsed_s: float = VALIDATION_START_TO_A_MIN_ELAPSED_S,
) -> bool:
    """True once mission motion has exceeded the start_move hold window."""
    if elapsed_s is None:
        return False
    return float(elapsed_s) > float(min_elapsed_s)


def apply_startup_ramp(
    linear: float,
    *,
    elapsed_s: float | None,
    ramp_s: float = VALIDATION_STARTUP_RAMP_S,
    min_mps: float = VALIDATION_STARTUP_MIN_MPS,
    plateau_s: float = VALIDATION_STARTUP_PLATEAU_S,
) -> float:
    """Hold ``min_mps`` for ``plateau_s``, then smoothstep to cruise."""
    if linear <= 0.0:
        return linear
    floor = min(float(min_mps), float(linear))
    if elapsed_s is None:
        return floor
    if float(plateau_s) > 0.0 and float(elapsed_s) < float(plateau_s):
        return floor
    ramp_elapsed = float(elapsed_s) - float(plateau_s)
    scale = startup_ramp_scale(ramp_elapsed, ramp_s=ramp_s)
    if scale >= 1.0:
        return float(linear)
    return floor + scale * (float(linear) - floor)


def validation_command_slew_limits(
    elapsed_s: float | None,
    *,
    base_accel_mps2: float,
    base_jerk_mps3: float,
    startup_ramp_s: float = VALIDATION_STARTUP_RAMP_S,
    startup_plateau_s: float = VALIDATION_STARTUP_PLATEAU_S,
    startup_accel_mps2: float = VALIDATION_STARTUP_LINEAR_ACCEL_MPS2,
    startup_jerk_mps3: float = VALIDATION_STARTUP_LINEAR_JERK_MPS3,
    cruise_accel_mps2: float = VALIDATION_CRUISE_COMMAND_ACCEL_MPS2,
    cruise_jerk_mps3: float = VALIDATION_CRUISE_COMMAND_JERK_MPS3,
) -> tuple[float, float]:
    """Limit command slew for gentle startup and steadier mid-run cruise."""
    jerk = max(float(base_jerk_mps3), 1e-6)
    accel = max(float(base_accel_mps2), 1e-6)
    startup_window_s = float(startup_plateau_s) + float(startup_ramp_s)
    if elapsed_s is not None and float(elapsed_s) < startup_window_s:
        accel = min(accel, float(startup_accel_mps2))
        jerk = min(jerk, float(startup_jerk_mps3))
    elif elapsed_s is not None:
        accel = min(accel, float(cruise_accel_mps2))
        jerk = min(jerk, float(cruise_jerk_mps3))
    return accel, jerk


def terminal_capture_active(
    *,
    checkpoint_distance: float,
    checkpoint_tol: float,
    remaining_arc_m: float | None,
    region_radius_m: float = OFFICIAL_REGION_RADIUS_M,
    terminal_approach_remaining_m: float = VALIDATION_TERMINAL_APPROACH_REMAINING_M,
) -> bool:
    """Only crawl into C near the path end, not while still on the B→C hall."""
    capture_dist = max(
        float(checkpoint_tol) * 1.15,
        float(region_radius_m) * 1.05,
    )
    if float(checkpoint_distance) > capture_dist:
        return False
    if remaining_arc_m is None:
        return True
    return float(remaining_arc_m) <= float(terminal_approach_remaining_m)


def scaled_profile_linear_accel(
    base_accel_mps2: float,
    peak_speed_mps: float,
) -> float:
    """Raise path-envelope accel for high contract speeds without touching jerk."""
    accel = float(base_accel_mps2)
    peak = float(peak_speed_mps)
    if peak > 2.5:
        accel = max(accel, min(peak * 0.35, 5.0))
    if peak >= VALIDATION_HIGH_SPEED_THRESHOLD_MPS:
        accel = max(
            accel,
            min(peak * 0.85, VALIDATION_HIGH_SPEED_PROFILE_ACCEL_MPS2),
        )
    return accel


def validation_profile_lookahead_m(contract_speed_mps: float) -> float:
    """Preview distance for pure pursuit; grows with the cruise contract speed."""
    return max(
        VALIDATION_PROFILE_LOOKAHEAD_M,
        float(contract_speed_mps) * VALIDATION_LOOKAHEAD_SPEED_FACTOR,
    )


def play_scene_hint() -> str:
    return f"open task_1/{SCENE_NAME} (robot {ROBOT_PATH}) and click Play"
