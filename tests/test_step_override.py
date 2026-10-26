"""Temporary consecutive-target override routes through bounded interpolation."""
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1z_g05_deploy/client'))
from dual_arm_client import Hardware, validate_smoothing_config
from trajectory_smoother import BoundedJointInterpolator


def hardware(enforce=True, sent=True):
    instance = Hardware({'max_joint_step': .1, 'max_initial_joint_step': .1,
                         'enforce_joint_step': enforce}, execute=True)
    instance.previous = (np.zeros(7), np.zeros(7))
    instance.action_sent = sent
    instance.arms = [Mock(is_running=True, is_estopped=False) for _ in range(2)]
    instance.streamer = Mock()
    action = {'left_arm': np.zeros(6), 'left_gripper': [.1],
              'right_arm': np.array([0., .11909, 0., 0., 0., 0.]), 'right_gripper': [-.002]}
    return instance, action


def test_logged_jump_is_accepted_by_enabled_override_and_only_queued_for_smoothing():
    instance, action = hardware(enforce=False)
    instance.send(action)
    queued = instance.streamer.set_targets.call_args.args[0]
    assert queued[1][1] == pytest.approx(.11909)
    for arm in instance.arms:
        arm.command_joint_pos.assert_not_called()
    instance.arms[1].command_gripper.assert_called_once_with(0.)


@pytest.mark.parametrize('enforce,sent', [(True, True), (False, False)])
def test_default_consecutive_check_and_initial_check_still_reject_jump(enforce, sent):
    instance, action = hardware(enforce=enforce, sent=sent)
    with pytest.raises(ValueError, match='exceeds'):
        instance.send(action)
    instance.streamer.set_targets.assert_not_called()
    for arm in instance.arms:
        arm.command_gripper.assert_not_called()


def test_override_cannot_send_without_streamer_or_pass_nonfinite_targets():
    instance, action = hardware(enforce=False)
    instance.streamer = None
    with pytest.raises(RuntimeError, match='smoothing is required'):
        instance.send(action)
    instance.streamer = Mock()
    action['right_arm'][1] = np.nan
    with pytest.raises(ValueError, match='Invalid right_arm'):
        instance.send(action)
    instance.streamer.set_targets.assert_not_called()


def test_override_requires_explicit_boolean_and_enabled_bounded_smoothing():
    with pytest.raises(ValueError, match='requires bounded smoothing'):
        validate_smoothing_config({'enforce_joint_step': False})
    with pytest.raises(ValueError, match='boolean'):
        validate_smoothing_config({'enforce_joint_step': 'false'})


def test_logged_jump_is_spread_across_multiple_control_ticks():
    smoother = BoundedJointInterpolator(np.zeros(6), 4., 20., .04)
    smoother.set_target(np.full(6, .11909))
    points = np.array([np.zeros(6), *[smoother.advance(.004) for _ in range(100)]])
    steps = np.diff(points, axis=0)
    assert np.max(steps) <= 4. * .004 + 1e-9
    assert points[1, 0] < .001
    assert np.all(points >= 0.) and np.all(points <= .11909)
    assert np.allclose(points[-1], .11909, atol=1e-3)
