import unittest

from jev_robotics import RobotDecisionEngine, RobotState, Skill


class RobotDecisionTests(unittest.TestCase):
    def state(self, **overrides):
        values = dict(
            end_effector_xyz=(0.0, 0.0, 0.0),
            target_xyz=(0.3, 0.0, 0.0),
            reach_radius_m=0.8,
            payload_kg=0.3,
            payload_limit_kg=2.0,
            human_distance_m=2.0,
            collision_detected=False,
            gripper_has_object=False,
        )
        values.update(overrides)
        return RobotState(**values)

    def skills(self):
        return [
            Skill("move_to_target", "approach the object", 1.0, 5.0),
            Skill("grasp", "close gripper on object", 0.4, 2.0),
            Skill("hold", "wait without motion", 0.1, 0.1),
        ]

    def test_unreachable_motion_is_masked(self):
        state = self.state(target_xyz=(2.0, 0.0, 0.0))
        result = RobotDecisionEngine(0.0, 1.1).decide(state, "What next?", self.skills())
        self.assertFalse(result.feasible["move_to_target"])
        self.assertEqual(result.probabilities["move_to_target"], 0.0)
        self.assertEqual(result.choice, "hold")

    def test_human_proximity_leaves_hold(self):
        state = self.state(human_distance_m=0.4)
        result = RobotDecisionEngine(0.0, 1.1).decide(state, "What next?", self.skills())
        self.assertEqual(result.choice, "hold")
        self.assertEqual(result.probabilities["hold"], 1.0)

    def test_probabilities_sum_to_one(self):
        result = RobotDecisionEngine(0.0, 1.1).decide(
            self.state(), "Choose a safe next skill", self.skills()
        )
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_payload_limit_masks_motion(self):
        state = self.state(payload_kg=3.0, payload_limit_kg=2.0)
        result = RobotDecisionEngine(0.0, 1.1).decide(state, "What next?", self.skills())
        self.assertFalse(result.feasible["move_to_target"])
        self.assertEqual(result.choice, "hold")

    def test_all_infeasible_requests_operator(self):
        state = self.state(collision_detected=True)
        moving_only = [Skill("move_to_target", "approach object", 1.0, 5.0)]
        result = RobotDecisionEngine().decide(state, "What next?", moving_only)
        self.assertEqual(result.choice, "request_operator")
        self.assertTrue(result.requires_operator)

    def test_place_requires_an_object(self):
        skills = [
            Skill("place", "place the carried object", 1.0, 4.0),
            Skill("hold", "wait without motion", 0.1, 0.1),
        ]
        result = RobotDecisionEngine(0.0, 1.1).decide(self.state(), "What next?", skills)
        self.assertFalse(result.feasible["place"])
        self.assertEqual(result.choice, "hold")

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.state(reach_radius_m=0.0)


if __name__ == "__main__":
    unittest.main()
