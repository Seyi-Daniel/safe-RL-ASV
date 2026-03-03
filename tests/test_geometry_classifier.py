import math
import unittest

from environment import SingleVessel2FeatureEnv, Vessel


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
        # vessel1 sees vessel2 on starboard: give-way for vessel1
        r1, r2 = self.env.assign_roles("crossing", rb_1=30.0, rb_2=300.0)
        self.assertEqual((r1, r2), ("give_way", "stand_on"))

    def test_crossing_mirrored_vessel1_stand_on(self):
        # mirror case: vessel2 sees vessel1 on starboard => vessel2 give-way
        r1, r2 = self.env.assign_roles("crossing", rb_1=300.0, rb_2=30.0)
        self.assertEqual((r1, r2), ("stand_on", "give_way"))
