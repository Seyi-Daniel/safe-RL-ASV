import math
import unittest

from simulations.environment import SingleVessel2FeatureEnv, Vessel
from simulations.hyperparameters import EnvParams, RewardParams
import numpy as np


class TestGeometryClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SingleVessel2FeatureEnv(render=False)

    def _mk(self, x: float, y: float, heading_deg: float) -> Vessel:
        return Vessel(x=x, y=y, h=math.radians(heading_deg), speed=0.0, goal_x=0.0, goal_y=0.0)

    def test_head_on(self):
        v1 = self._mk(0.0, 0.0, 0.0)
        v2 = self._mk(100.0, 0.0, 180.0)
        scenario, rb1, rb2 = self.env.classify_geometry(v1, v2)
        self.assertEqual(scenario, "head_on")
        self.assertTrue(0.0 <= rb1 < 360.0)
        self.assertTrue(0.0 <= rb2 < 360.0)

    def test_overtaking(self):
        # vessel1 sees vessel2 directly astern -> overtaking scenario by sector rule
        v1 = self._mk(0.0, 0.0, 0.0)
        v2 = self._mk(-100.0, 0.0, 0.0)
        scenario, rb1, rb2 = self.env.classify_geometry(v1, v2)
        self.assertEqual(scenario, "overtaking")
        self.assertTrue(112.5 <= rb1 <= 247.5 or 112.5 <= rb2 <= 247.5)

    def test_crossing(self):
        # vessel2 on starboard beam of vessel1, asymmetric bearings -> crossing
        v1 = self._mk(0.0, 0.0, 0.0)
        v2 = self._mk(0.0, -100.0, 90.0)
        scenario, rb1, rb2 = self.env.classify_geometry(v1, v2)
        self.assertEqual(scenario, "crossing")
        self.assertTrue(0.0 <= rb1 < 360.0)
        self.assertTrue(0.0 <= rb2 < 360.0)

    def test_always_one_of_three_for_non_identical_positions(self):
        allowed = {"head_on", "overtaking", "crossing"}
        v1 = self._mk(0.0, 0.0, 0.0)
        for x, y in [(10.0, 0.0), (0.0, 10.0), (-10.0, -5.0), (25.0, 40.0), (-30.0, 7.0)]:
            v2 = self._mk(x, y, 45.0)
            scenario, rb1, rb2 = self.env.classify_geometry(v1, v2)
            self.assertIn(scenario, allowed)
            self.assertTrue(0.0 <= rb1 < 360.0)
            self.assertTrue(0.0 <= rb2 < 360.0)


if __name__ == "__main__":
    unittest.main()


class TestRiskGate(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SingleVessel2FeatureEnv(render=False)

    def _mk(self, x: float, y: float, heading_deg: float, speed: float) -> Vessel:
        return Vessel(x=x, y=y, h=math.radians(heading_deg), speed=speed, goal_x=0.0, goal_y=0.0)

    def test_parallel_same_velocity_has_infinite_tcpa_and_no_risk(self):
        v1 = self._mk(0.0, 0.0, 0.0, 5.0)
        v2 = self._mk(100.0, 0.0, 0.0, 5.0)
        risk, tcpa, dcpa = self.env.assess_risk(v1, v2)
        self.assertFalse(risk)
        self.assertTrue(math.isinf(tcpa))
        self.assertAlmostEqual(dcpa, 100.0, places=6)

    def test_direct_collision_course_has_risk_true(self):
        v1 = self._mk(0.0, 0.0, 0.0, 5.0)
        v2 = self._mk(100.0, 0.0, 180.0, 5.0)
        risk, tcpa, dcpa = self.env.assess_risk(v1, v2)
        self.assertTrue(risk)
        self.assertTrue(0.0 <= tcpa <= self.env.envp.tcpa_risk_threshold)
        self.assertLessEqual(dcpa, self.env.envp.dcpa_risk_threshold)


class TestAssignRoles(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SingleVessel2FeatureEnv(render=False)

    def test_head_on_both_give_way(self):
        r1, r2 = self.env.assign_roles("head_on", rb_1=0.0, rb_2=0.0)
        self.assertEqual((r1, r2), ("give_way", "give_way"))

    def test_overtaking_overtaker_give_way(self):
        # rb_1 in aft sector => vessel1 sees vessel2 astern => vessel2 is overtaking
        r1, r2 = self.env.assign_roles("overtaking", rb_1=180.0, rb_2=0.0)
        self.assertEqual((r1, r2), ("stand_on", "give_way"))

    def test_crossing_vessel1_starboard_sees_vessel2(self):
        # vessel1 sees vessel2 on starboard (negative signed bearing): give-way for vessel1
        r1, r2 = self.env.assign_roles("crossing", rb_1=330.0, rb_2=30.0)
        self.assertEqual((r1, r2), ("give_way", "stand_on"))

    def test_crossing_mirrored_vessel1_stand_on(self):
        # mirror case: vessel2 sees vessel1 on starboard => vessel2 give-way
        r1, r2 = self.env.assign_roles("crossing", rb_1=30.0, rb_2=330.0)
        self.assertEqual((r1, r2), ("stand_on", "give_way"))


class TestLockStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        envp = EnvParams(
            lock_enter_persistence_steps=1,
            require_reset_viable_takeover_path=False,
            enable_no_takeover_early_done=False,
            episode_seconds=30.0,
        )
        self.env = SingleVessel2FeatureEnv(envp, RewardParams(), render=False)
        self.env.reset(seed=123)

    def _set_crossing_risk_state(self) -> None:
        # Crossing at (50, 0): vessel1 eastbound, vessel2 northbound.
        self.env.vessel1 = Vessel(x=0.0, y=0.0, h=0.0, speed=5.0, goal_x=500.0, goal_y=0.0)
        self.env.vessel2 = Vessel(x=50.0, y=-50.0, h=math.pi / 2.0, speed=5.0, goal_x=50.0, goal_y=500.0)
        self.env.vessel1_reached = False
        self.env.vessel2_reached = False

    def test_locks_when_risk_triggered(self):
        self._set_crossing_risk_state()
        _, _, _, info = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(self.env.locked)
        self.assertEqual(info["colregs_scenario"], self.env.locked_scenario)
        self.assertIn(self.env.locked_scenario, {"crossing", "head_on", "overtaking"})

    def test_locked_scenario_and_roles_do_not_change_after_geometry_drift(self):
        self._set_crossing_risk_state()
        _, _, _, info1 = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        locked_scenario = self.env.locked_scenario
        locked_role_v1 = self.env.locked_role_v1
        locked_role_v2 = self.env.locked_role_v2
        self.assertTrue(self.env.locked)

        # Force a head-on geometry layout next step, but locked scenario/roles should persist.
        self.env.vessel1.x, self.env.vessel1.y, self.env.vessel1.h = 0.0, 0.0, 0.0
        self.env.vessel2.x, self.env.vessel2.y, self.env.vessel2.h = 100.0, 0.0, math.pi
        self.env.vessel1.speed = 5.0
        self.env.vessel2.speed = 5.0

        _, _, _, info2 = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(self.env.locked)
        self.assertEqual(info2["colregs_scenario"], locked_scenario)
        self.assertEqual(info2["vessel1_role"], locked_role_v1)
        self.assertEqual(info2["vessel2_role"], locked_role_v2)


class TestRLGiveWayRouting(unittest.TestCase):
    def setUp(self) -> None:
        envp = EnvParams(
            lock_enter_persistence_steps=1,
            require_reset_viable_takeover_path=False,
            enable_no_takeover_early_done=False,
            episode_seconds=30.0,
        )
        self.env = SingleVessel2FeatureEnv(envp, RewardParams(), render=False)
        self.env.reset(seed=321)

    def _set_locked_crossing_v1_giveway(self) -> None:
        # crossing with risk: vessel1 gives way to vessel2
        self.env.vessel1 = Vessel(x=0.0, y=0.0, h=0.0, speed=5.0, goal_x=500.0, goal_y=0.0)
        self.env.vessel2 = Vessel(x=50.0, y=50.0, h=-math.pi / 2.0, speed=5.0, goal_x=50.0, goal_y=-500.0)
        self.env.vessel1_reached = False
        self.env.vessel2_reached = False
        self.env.step(np.array([0.0, 0.0], dtype=np.float32))  # lock

    def _set_locked_crossing_v2_giveway(self) -> None:
        # mirrored crossing: vessel2 gives way to vessel1
        self.env.vessel1 = Vessel(x=0.0, y=0.0, h=0.0, speed=5.0, goal_x=500.0, goal_y=0.0)
        self.env.vessel2 = Vessel(x=50.0, y=-50.0, h=math.pi / 2.0, speed=5.0, goal_x=50.0, goal_y=500.0)
        self.env.vessel1_reached = False
        self.env.vessel2_reached = False
        self.env.step(np.array([0.0, 0.0], dtype=np.float32))  # lock

    def test_rl_action_only_affects_vessel1_when_vessel1_giveway(self):
        self._set_locked_crossing_v1_giveway()
        h1_before = float(self.env.vessel1.h)
        h2_before = float(self.env.vessel2.h)

        self.env.step(np.array([1.0, 0.0], dtype=np.float32))
        h1_after = float(self.env.vessel1.h)
        h2_after = float(self.env.vessel2.h)

        self.assertTrue(self.env.vessel1_rl_active)
        self.assertFalse(self.env.vessel2_rl_active)
        self.assertNotAlmostEqual(h1_before, h1_after, places=5)
        self.assertAlmostEqual(h2_before, h2_after, places=5)

    def test_rl_action_only_affects_vessel2_when_vessel2_giveway(self):
        self._set_locked_crossing_v2_giveway()
        h1_before = float(self.env.vessel1.h)
        h2_before = float(self.env.vessel2.h)

        self.env.step(np.array([1.0, 0.0], dtype=np.float32))
        h1_after = float(self.env.vessel1.h)
        h2_after = float(self.env.vessel2.h)

        self.assertFalse(self.env.vessel1_rl_active)
        self.assertTrue(self.env.vessel2_rl_active)
        self.assertAlmostEqual(h1_before, h1_after, places=5)
        self.assertNotAlmostEqual(h2_before, h2_after, places=5)
