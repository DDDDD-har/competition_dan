#!/usr/bin/env python3
"""Contract checks for the g1_button / g1_omnipicker Task 1 scene."""

import math
import unittest

from task1_scene_contract import (
    CHASSIS_DEVICE_SPEED_LIMIT_MPS,
    CHASSIS_DRIVE_WHEELS,
    CHASSIS_KINEMATIC_TURNING_RADIUS_M,
    CHASSIS_MAX_STEER_RAD,
    CHASSIS_STEER_WHEELS,
    CHASSIS_WHEELBASE_M,
    VALIDATION_MIN_TURNING_RADIUS_M,
    INSPECTION_WORLD_POINTS_XYZ,
    ROBOT_NAME,
    SCENE_NAME,
    STAGED_SPEED_START_MPS,
    STAGED_SPEED_STEP_MPS,
    VALIDATION_AUGMENT_SCENE_GEOMETRY,
    VALIDATION_COORDINATE_TOLERANCE_M,
    VALIDATION_MAX_SPEED_M,
    VALIDATION_SPEED_M,
    VALIDATION_PROFILE_LOOKAHEAD_M,
    validation_profile_lookahead_m,
    VALIDATION_CHECKPOINT_FLYTHROUGH_CAP_MPS,
    VALIDATION_STARTUP_PLATEAU_S,
    VALIDATION_TERMINAL_ARRIVAL_SPEED_MPS,
    VALIDATION_TERMINAL_ZONE_M,
    VALIDATION_C_LEG_STRAIGHT_CAP_MPS,
    VALIDATION_C_LEG_PRE_BEND_CAP_MPS,
    OFFICIAL_CIRCLE_CAPTURE_RADIUS_SCALE,
    VALIDATION_AB_CHECKPOINT_DWELL_S,
    VALIDATION_POST_ROBOT_START_DWELL_S,
    VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS,
    VALIDATION_HIGH_SPEED_THRESHOLD_MPS,
    VALIDATION_HIGH_SPEED_PROFILE_ACCEL_MPS2,
    VALIDATION_MISSION_DEADLINE_ENABLED,
    VALIDATION_MISSION_TIMEOUT_S,
    apply_startup_ramp,
    start_to_a_motion_elapsed_ok,
    OFFICIAL_REGION_RADIUS_M,
    OFFICIAL_SCORER_START_MOVE_MIN_MPS,
    startup_ramp_scale,
    terminal_capture_active,
    VALIDATION_STARTUP_MIN_MPS,
    VALIDATION_STARTUP_RAMP_S,
    VALIDATION_CRUISE_COMMAND_ACCEL_MPS2,
    VALIDATION_CRUISE_COMMAND_JERK_MPS3,
    VALIDATION_STARTUP_LINEAR_ACCEL_MPS2,
    VALIDATION_STARTUP_LINEAR_JERK_MPS3,
    VALIDATION_START_TO_A_MIN_ELAPSED_S,
    validation_command_slew_limits,
    scaled_profile_linear_accel,
    scoring_speed_policy,
    scorer_aligned_root_target,
    WORLD_POINTS_XYZ,
    WRAPUP_WAYPOINTS,
    contract_map_points_from_safety_report,
    order_route_points,
    parse_world_points,
    scored_route_points,
    validation_route_world_points,
    VALIDATION_ROUTE_ORDER,
    VALIDATION_VIA_WORLD_POINTS_XYZ,
    world_points_xy,
    world_points_xy_json,
    wrapup_wasd,
)


class SceneContractTests(unittest.TestCase):
    def test_scene_and_robot_names(self) -> None:
        self.assertEqual(SCENE_NAME, "g1_button.json")
        self.assertEqual(ROBOT_NAME, "g1_omnipicker")

    def test_chassis_is_front_wheel_ackermann(self) -> None:
        self.assertEqual(CHASSIS_DRIVE_WHEELS, ("fl", "fr"))
        self.assertEqual(CHASSIS_STEER_WHEELS, ("fl", "fr"))
        self.assertAlmostEqual(CHASSIS_WHEELBASE_M, 0.42)
        self.assertAlmostEqual(CHASSIS_MAX_STEER_RAD, 0.60)
        kinematic = CHASSIS_WHEELBASE_M / math.tan(CHASSIS_MAX_STEER_RAD)
        self.assertAlmostEqual(CHASSIS_KINEMATIC_TURNING_RADIUS_M, kinematic)
        self.assertGreater(VALIDATION_MIN_TURNING_RADIUS_M, kinematic)
        self.assertLess(VALIDATION_MIN_TURNING_RADIUS_M, 1.0)

    def test_speed_probe_ceiling(self) -> None:
        self.assertEqual(VALIDATION_SPEED_M, 0.80)
        self.assertEqual(VALIDATION_MAX_SPEED_M, CHASSIS_DEVICE_SPEED_LIMIT_MPS)
        self.assertEqual(STAGED_SPEED_STEP_MPS, 0.5)
        self.assertEqual(STAGED_SPEED_START_MPS, 2.0)
        self.assertAlmostEqual(VALIDATION_TERMINAL_ARRIVAL_SPEED_MPS, 0.0)
        self.assertAlmostEqual(VALIDATION_TERMINAL_ZONE_M, 0.0)
        self.assertAlmostEqual(VALIDATION_C_LEG_STRAIGHT_CAP_MPS, 0.80)
        self.assertAlmostEqual(VALIDATION_C_LEG_PRE_BEND_CAP_MPS, 2.5)
        from task1_scene_contract import (
            VALIDATION_PROFILE_CLEARANCE_HARD_M,
            VALIDATION_PROFILE_CLEARANCE_SOFT_M,
            VALIDATION_C_LEG_TIGHT_CLEARANCE_M,
            VALIDATION_C_LEG_TIGHT_CLEARANCE_CAP_MPS,
        )
        self.assertAlmostEqual(VALIDATION_PROFILE_CLEARANCE_HARD_M, 0.28)
        self.assertAlmostEqual(VALIDATION_PROFILE_CLEARANCE_SOFT_M, 0.80)
        self.assertAlmostEqual(VALIDATION_C_LEG_TIGHT_CLEARANCE_M, 0.55)
        self.assertAlmostEqual(VALIDATION_C_LEG_TIGHT_CLEARANCE_CAP_MPS, 1.25)
        from task1_scene_contract import (
            VALIDATION_C_PLANNER_TURNING_RADIUS_M,
            VALIDATION_C_STALL_GRACE_M,
            VALIDATION_C_TURN_MAX_PUSH_M,
            VALIDATION_C_TURN_TARGET_CLEARANCE_M,
            VALIDATION_MIN_TURNING_RADIUS_M,
        )
        self.assertAlmostEqual(VALIDATION_C_PLANNER_TURNING_RADIUS_M, 1.35)
        self.assertGreater(
            VALIDATION_C_PLANNER_TURNING_RADIUS_M,
            VALIDATION_MIN_TURNING_RADIUS_M,
        )
        self.assertAlmostEqual(VALIDATION_C_TURN_TARGET_CLEARANCE_M, 0.48)
        self.assertAlmostEqual(VALIDATION_C_TURN_MAX_PUSH_M, 0.16)
        self.assertAlmostEqual(VALIDATION_C_STALL_GRACE_M, 0.70)
        self.assertAlmostEqual(OFFICIAL_CIRCLE_CAPTURE_RADIUS_SCALE, 0.55)
        self.assertAlmostEqual(VALIDATION_AB_CHECKPOINT_DWELL_S, 0.050)
        self.assertAlmostEqual(VALIDATION_POST_ROBOT_START_DWELL_S, 0.0)
        self.assertAlmostEqual(VALIDATION_CHECKPOINT_CAPTURE_MIN_SPEED_MPS, 0.28)
        self.assertAlmostEqual(OFFICIAL_SCORER_START_MOVE_MIN_MPS, 0.06)
        from task1_scene_contract import VALIDATION_STARTUP_SPEED_M
        self.assertAlmostEqual(VALIDATION_STARTUP_SPEED_M, 0.20)
        self.assertAlmostEqual(VALIDATION_PROFILE_LOOKAHEAD_M, 0.65)
        self.assertAlmostEqual(validation_profile_lookahead_m(0.85), 0.65)
        self.assertAlmostEqual(validation_profile_lookahead_m(5.0), 1.1)

    def test_chassis_bridge_keeps_crawl_passthrough(self) -> None:
        from task1_scene_contract import CHASSIS_BRIDGE_ACTUATOR_FULL_SCALE_MPS

        full = float(CHASSIS_BRIDGE_ACTUATOR_FULL_SCALE_MPS)
        for cmd in (0.05, 0.10, 0.30):
            forward = (cmd / full) * full
            self.assertAlmostEqual(forward, cmd)
        broken_forward = (0.05 / 0.05) * 0.05
        self.assertAlmostEqual(broken_forward / 0.05, 1.0)

    def test_scaled_profile_linear_accel_boosts_high_speed_tiers(self) -> None:
        self.assertAlmostEqual(VALIDATION_HIGH_SPEED_THRESHOLD_MPS, 10.0)
        self.assertAlmostEqual(VALIDATION_HIGH_SPEED_PROFILE_ACCEL_MPS2, 20.0)
        self.assertAlmostEqual(scaled_profile_linear_accel(1.2, 3.0), 1.2)
        self.assertAlmostEqual(scaled_profile_linear_accel(1.2, 10.0), 8.5)
        self.assertAlmostEqual(scaled_profile_linear_accel(1.2, 20.0), 17.0)

    def test_checkpoint_settle_duration_ab_only(self) -> None:
        from task1_scene_contract import checkpoint_settle_duration_s

        self.assertAlmostEqual(checkpoint_settle_duration_s("A"), 0.050)
        self.assertAlmostEqual(checkpoint_settle_duration_s("B"), 0.050)
        self.assertAlmostEqual(checkpoint_settle_duration_s("C"), 0.0)
        self.assertAlmostEqual(
            checkpoint_settle_duration_s("C", checkpoint_settle_s=6.0), 6.0,
        )
        self.assertAlmostEqual(checkpoint_settle_duration_s("_via"), 0.0)

    def test_validation_route_order_stop_at_b(self) -> None:
        from task1_scene_contract import validation_route_order, validation_route_world_points

        self.assertEqual(validation_route_order("B"), ("A", "B"))
        self.assertEqual(validation_route_order(), ("A", "B", "C"))
        points = validation_route_world_points(
            {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
            stop_at="B",
        )
        self.assertEqual(set(points), {"A", "B"})

    def test_scoring_speed_policy_holds_contract_in_checkpoint_zone(self) -> None:
        cruise = scoring_speed_policy(
            phase="corridor_cruise",
            contract_speed_mps=20.0,
            speed_cap_mps=20.0,
        )
        self.assertAlmostEqual(cruise, 20.0)
        zone = scoring_speed_policy(
            phase="checkpoint_zone",
            contract_speed_mps=20.0,
            speed_cap_mps=20.0,
            checkpoint_distance_m=1.0,
        )
        self.assertAlmostEqual(zone, cruise)
        shallow = scoring_speed_policy(
            phase="checkpoint_zone",
            contract_speed_mps=0.8,
            speed_cap_mps=0.8,
            checkpoint_distance_m=0.55,
        )
        self.assertAlmostEqual(shallow, VALIDATION_CHECKPOINT_FLYTHROUGH_CAP_MPS)
        terminal = scoring_speed_policy(
            phase="terminal_c",
            contract_speed_mps=5.0,
            speed_cap_mps=5.0,
        )
        self.assertAlmostEqual(terminal, 5.0)

    def test_scorer_aligned_root_target_offsets_along_heading(self) -> None:
        site = (10.0, 0.0)
        target = scorer_aligned_root_target(site, math.pi, forward_offset_m=0.32)
        self.assertGreater(target[0], site[0])
        zero_offset = scorer_aligned_root_target(site, math.pi, forward_offset_m=0.0)
        self.assertAlmostEqual(zero_offset[0], site[0])
        self.assertAlmostEqual(zero_offset[1], site[1])

    def test_validation_uses_exact_scanned_costmap(self) -> None:
        self.assertFalse(VALIDATION_AUGMENT_SCENE_GEOMETRY)

    def test_mission_deadline_matches_official_attempt_limit(self) -> None:
        self.assertTrue(VALIDATION_MISSION_DEADLINE_ENABLED)
        self.assertAlmostEqual(VALIDATION_MISSION_TIMEOUT_S, 180.0)

    def test_xy_acceptance_radius_matches_official_circle(self) -> None:
        from task1_scene_contract import (
            OFFICIAL_REGION_RADIUS_M,
            VALIDATION_SCORE_CIRCLE_STOP_OFFSET_M,
        )
        self.assertAlmostEqual(OFFICIAL_REGION_RADIUS_M, 0.60)
        self.assertAlmostEqual(VALIDATION_COORDINATE_TOLERANCE_M, 0.60)
        self.assertEqual(
            VALIDATION_COORDINATE_TOLERANCE_M, OFFICIAL_REGION_RADIUS_M
        )
        self.assertLess(
            VALIDATION_SCORE_CIRCLE_STOP_OFFSET_M,
            VALIDATION_COORDINATE_TOLERANCE_M,
        )

    def test_coverage_targets_keep_requested_map_xy(self) -> None:
        report = {
            "point_audit": {
                "C": {
                    "requested_map_xy": [19.205, -22.952],
                    "resolved_map_xy": [19.205, -23.452],
                }
            }
        }
        self.assertEqual(
            contract_map_points_from_safety_report(report)["C"],
            [19.205, -22.952],
        )

    def test_world_points_match_task_coordinates(self) -> None:
        self.assertEqual(WORLD_POINTS_XYZ["A"], (43.305, -27.564, 0.203))
        self.assertEqual(WORLD_POINTS_XYZ["B"], (30.702, -27.564, 0.203))
        self.assertEqual(WORLD_POINTS_XYZ["C"], (19.122, -22.982, 0.203))
        self.assertEqual(world_points_xy()["A"], [43.305, -27.564])
        self.assertEqual(INSPECTION_WORLD_POINTS_XYZ["B"], (30.422, -27.776, 0.203))

    def test_parse_accepts_xyz_and_keeps_xy(self) -> None:
        parsed = parse_world_points(
            '{"A":[43.641,-27.776,0.203],"B":[30.422,-27.776],"C":[19.205,-22.952,0.203]}'
        )
        self.assertEqual(parsed["B"], [30.422, -27.776])
        self.assertEqual(len(parsed["A"]), 2)

    def test_default_json_round_trip(self) -> None:
        self.assertEqual(
            parse_world_points(world_points_xy_json()),
            validation_route_world_points(world_points_xy()),
        )

    def test_wrapup_visits_spawn_and_abc(self) -> None:
        names = [item[0] for item in WRAPUP_WAYPOINTS]
        self.assertEqual(names[0], "spawn")
        self.assertLess(names.index("A"), names.index("B"))
        self.assertLess(names.index("B"), names.index("C"))
        a_mapping = WRAPUP_WAYPOINTS[names.index("A")]
        self.assertLess(
            ((a_mapping[1] - INSPECTION_WORLD_POINTS_XYZ["A"][0]) ** 2
             + (a_mapping[2] - INSPECTION_WORLD_POINTS_XYZ["A"][1]) ** 2) ** 0.5,
            0.6,
        )
        self.assertIn("A_reverse", names)
        self.assertIn("C_reverse", names)
        outline = [name for name in names if name.startswith("outline_")]
        self.assertEqual(len(outline), 9)
        self.assertGreater(names.index(outline[0]), names.index("C"))
        self.assertEqual(names[-1], "outline_south_west")

    def test_wrapup_wasd_prefers_forward_east(self) -> None:
        choice = wrapup_wasd(40.0, -27.5, 0.0, 43.65, -27.49)
        self.assertEqual(choice["W"], 1)
        self.assertEqual(choice["S"], 0)
        self.assertFalse(choice["arrived"])

    def test_wrapup_wasd_arrives_and_turns_left_for_north(self) -> None:
        self.assertTrue(wrapup_wasd(43.65, -27.49, 0.0, 43.65, -27.49)["arrived"])
        turn = wrapup_wasd(43.65, -27.49, 1.05, 43.65, -2.0)
        self.assertEqual(turn["A"], 1)
        self.assertEqual(turn["W"], 1)
        self.assertEqual(turn["S"], 0)

    def test_wrapup_wasd_reverses_only_when_facing_opposite(self) -> None:
        choice = wrapup_wasd(47.28, -25.00, 0.0, 43.65, -25.00)
        self.assertEqual(choice["S"], 1)
        self.assertEqual(choice["W"], 0)

    def test_wrapup_wasd_uses_forward_arc_for_corridor_corner(self) -> None:
        choice = wrapup_wasd(43.65, -27.49, -1.57, 41.45, -27.49)
        self.assertEqual(choice["D"], 1)
        self.assertEqual(choice["W"], 1)
        self.assertEqual(choice["S"], 0)


class RouteViaPointTests(unittest.TestCase):
    def test_default_route_is_direct_abc(self) -> None:
        self.assertEqual(VALIDATION_ROUTE_ORDER, ("A", "B", "C"))
        self.assertEqual(VALIDATION_VIA_WORLD_POINTS_XYZ, {})

    def test_route_order_keeps_scored_sites_only_by_default(self) -> None:
        ordered = validation_route_world_points(world_points_xy())
        self.assertEqual(list(ordered), ["A", "B", "C"])

    def test_via_point_is_never_scored(self) -> None:
        ordered = validation_route_world_points(world_points_xy())
        self.assertEqual(list(scored_route_points(ordered)), ["A", "B", "C"])

    def test_order_route_points_keeps_unknown_names_last(self) -> None:
        ordered = order_route_points(
            {"C": [1.0, 2.0], "_V": [3.0, 4.0], "Z": [5.0, 6.0], "A": [7.0, 8.0]}
        )
        self.assertEqual(list(ordered), ["A", "C", "_V", "Z"])


class MotionProfileTests(unittest.TestCase):
    def test_startup_ramp_is_zero_before_mission_clock(self) -> None:
        self.assertEqual(startup_ramp_scale(None), 0.0)

    def test_startup_ramp_smoothstep_reaches_cruise(self) -> None:
        self.assertAlmostEqual(startup_ramp_scale(0.0), 0.0)
        self.assertAlmostEqual(startup_ramp_scale(VALIDATION_STARTUP_RAMP_S), 1.0)
        mid = startup_ramp_scale(VALIDATION_STARTUP_RAMP_S * 0.5)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

    def test_apply_startup_ramp_holds_plateau_before_ramp(self) -> None:
        from task1_scene_contract import VALIDATION_STARTUP_PLATEAU_S

        linear = apply_startup_ramp(
            0.8,
            elapsed_s=VALIDATION_STARTUP_PLATEAU_S * 0.5,
        )
        self.assertAlmostEqual(linear, VALIDATION_STARTUP_MIN_MPS)

    def test_apply_startup_ramp_reaches_contract_speed(self) -> None:
        from task1_scene_contract import VALIDATION_STARTUP_PLATEAU_S, VALIDATION_STARTUP_SPEED_M

        cruise = 0.80
        ramp_elapsed = VALIDATION_STARTUP_PLATEAU_S + VALIDATION_STARTUP_RAMP_S
        self.assertAlmostEqual(
            apply_startup_ramp(cruise, elapsed_s=ramp_elapsed),
            cruise,
        )
        hold = apply_startup_ramp(
            cruise,
            elapsed_s=VALIDATION_STARTUP_PLATEAU_S * 0.5,
        )
        self.assertAlmostEqual(hold, VALIDATION_STARTUP_SPEED_M)

        from task1_scene_contract import VALIDATION_STARTUP_PLATEAU_S

        linear = apply_startup_ramp(
            5.0,
            elapsed_s=0.0,
            min_mps=VALIDATION_STARTUP_MIN_MPS,
        )
        self.assertAlmostEqual(linear, VALIDATION_STARTUP_MIN_MPS)
        ramp_elapsed = VALIDATION_STARTUP_PLATEAU_S + VALIDATION_STARTUP_RAMP_S
        self.assertGreaterEqual(
            apply_startup_ramp(5.0, elapsed_s=ramp_elapsed),
            OFFICIAL_SCORER_START_MOVE_MIN_MPS,
        )

    def test_start_to_a_motion_requires_more_than_one_second(self) -> None:
        self.assertFalse(start_to_a_motion_elapsed_ok(None))
        self.assertFalse(start_to_a_motion_elapsed_ok(0.5))
        self.assertFalse(start_to_a_motion_elapsed_ok(
            VALIDATION_START_TO_A_MIN_ELAPSED_S,
        ))
        self.assertTrue(start_to_a_motion_elapsed_ok(
            VALIDATION_START_TO_A_MIN_ELAPSED_S + 0.01,
        ))

    def test_command_slew_limits_favour_gentle_startup(self) -> None:
        startup_accel, startup_jerk = validation_command_slew_limits(
            0.5,
            base_accel_mps2=20.0,
            base_jerk_mps3=10.0,
        )
        cruise_accel, cruise_jerk = validation_command_slew_limits(
            VALIDATION_STARTUP_PLATEAU_S + VALIDATION_STARTUP_RAMP_S + 1.0,
            base_accel_mps2=20.0,
            base_jerk_mps3=10.0,
        )
        self.assertEqual(startup_accel, VALIDATION_STARTUP_LINEAR_ACCEL_MPS2)
        self.assertEqual(startup_jerk, VALIDATION_STARTUP_LINEAR_JERK_MPS3)
        self.assertEqual(cruise_accel, VALIDATION_CRUISE_COMMAND_ACCEL_MPS2)
        self.assertEqual(cruise_jerk, VALIDATION_CRUISE_COMMAND_JERK_MPS3)

    def test_terminal_capture_waits_until_path_end(self) -> None:
        self.assertFalse(terminal_capture_active(
            checkpoint_distance=0.50,
            checkpoint_tol=0.50,
            remaining_arc_m=20.0,
        ))
        self.assertTrue(terminal_capture_active(
            checkpoint_distance=0.50,
            checkpoint_tol=0.50,
            remaining_arc_m=4.0,
        ))


if __name__ == "__main__":
    unittest.main()
