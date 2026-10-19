# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Safety-relevant checks for optional single-arm and dual-arm target smoothing."""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "a1z_g05_deploy" / "client"))
from trajectory_smoother import ArmCommandStreamer, BoundedJointInterpolator


class FakeArm:
    is_running = True
    is_estopped = False

    def __init__(self):
        self.commands = []
        self.held = False

    def command_joint_pos(self, position):
        self.commands.append(np.asarray(position).copy())
        return True

    def get_fault_status(self):
        return "none"

    def latch_last_command(self, reason):
        self.held = True


class SmoothingTests(unittest.TestCase):
    def test_targets_are_continuous_and_do_not_overshoot(self):
        smoother = BoundedJointInterpolator(np.zeros(6), 4, 20, .04)
        smoother.set_target(np.full(6, .2))
        points = np.array([smoother.advance(.004) for _ in range(100)])
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue((points >= 0).all())
        self.assertTrue((points <= .2 + 1e-9).all())
        self.assertLess(np.max(np.abs(np.diff(points, axis=0))), .0161)
        self.assertTrue(np.allclose(points[-1], .2, atol=1e-3))
        smoother.set_target(np.zeros(6))
        reverse = np.array([smoother.advance(.004) for _ in range(100)])
        self.assertTrue((reverse >= -1e-9).all())
        self.assertTrue(np.allclose(reverse[-1], 0, atol=1e-3))

    def test_stale_policy_holds_and_stops_stream(self):
        arm = FakeArm()
        stream = ArmCommandStreamer(arm, np.zeros(6), stale_timeout=.08)
        try:
            stream.start()
            stream.set_target(np.full(6, .1))
            time.sleep(.13)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                stream.check()
            self.assertTrue(arm.held)
            self.assertGreater(len(arm.commands), 10)
        finally:
            stream.stop()


class DualSmoothingTests(unittest.TestCase):
    def test_dual_client_checks_both_targets_before_sending(self):
        from dual_arm_client import Hardware

        class GripperArm(FakeArm):
            def __init__(self):
                super().__init__()
                self.grippers = []

            def command_gripper(self, value):
                self.grippers.append(value)

        class TargetStream:
            def __init__(self):
                self.targets = []

            def check(self):
                pass

            def set_targets(self, targets):
                self.targets.append([v.copy() for v in targets])

        left, right = GripperArm(), GripperArm()
        hardware = Hardware({"max_joint_step": .3}, execute=True)
        hardware.arms = [left, right]
        hardware.previous = (np.zeros(7), np.zeros(7))
        hardware.streamer = TargetStream()
        action = {"left_arm": np.full(6, .1), "left_gripper": [.4],
                  "right_arm": np.full(6, -.1), "right_gripper": [.6]}
        hardware.send(action)
        self.assertEqual(left.grippers, [.4])
        self.assertEqual(right.grippers, [.6])
        self.assertEqual(len(hardware.streamer.targets), 1)
        self.assertTrue(np.allclose(hardware.streamer.targets[0][0], .1))
        self.assertTrue(np.allclose(hardware.streamer.targets[0][1], -.1))
        self.assertEqual(left.commands + right.commands, [])
        action["right_arm"] = np.full(6, 1.0)
        with self.assertRaisesRegex(ValueError, r"right joint\d+ step"):
            hardware.send(action)
        self.assertEqual(left.grippers, [.4])
        self.assertEqual(right.grippers, [.6])
        self.assertEqual(len(hardware.streamer.targets), 1)

    def test_both_arms_hold_when_one_side_rejects_command(self):
        from trajectory_smoother import DualArmCommandStreamer

        left, right = FakeArm(), FakeArm()
        right.command_joint_pos = lambda _position: False
        stream = DualArmCommandStreamer(
            [left, right], [np.zeros(6), np.zeros(6)], stale_timeout=1.0
        )
        try:
            stream.start()
            stream.thread.join(timeout=1.0)
            with self.assertRaisesRegex(RuntimeError, "SDK refused target"):
                stream.check()
            self.assertTrue(left.held)
            self.assertTrue(right.held)
        finally:
            stream.stop()

    def test_both_arms_receive_bounded_targets_and_hold_when_stale(self):
        from trajectory_smoother import DualArmCommandStreamer

        left, right = FakeArm(), FakeArm()
        stream = DualArmCommandStreamer(
            [left, right], [np.zeros(6), np.zeros(6)],
            frequency=250, max_velocity=4, max_acceleration=20,
            stale_timeout=.08,
        )
        try:
            stream.start()
            stream.set_targets([np.full(6, .2), np.full(6, -.2)])
            time.sleep(.13)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                stream.check()
            self.assertTrue(left.held and right.held)
            self.assertGreater(len(left.commands), 10)
            self.assertEqual(len(left.commands), len(right.commands))
            self.assertTrue(all(cmd.shape == (6,) for cmd in left.commands + right.commands))
            self.assertTrue(all((cmd <= .2 + 1e-9).all() for cmd in left.commands))
            self.assertTrue(all((cmd >= -.2 - 1e-9).all() for cmd in right.commands))
        finally:
            stream.stop()


if __name__ == "__main__":
    unittest.main()
