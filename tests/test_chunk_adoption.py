"""Chunk adoption helpers used by the prefetch path (pure functions, no I/O)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1z_g05_deploy/client'))
import dual_arm_client  # noqa: E402
from dual_arm_client import PARTS, validate_action  # noqa: E402

# The chunk-prefetch path is currently reverted; keep these tests ready for when
# it returns instead of failing collection.
adopt_action_chunk = getattr(dual_arm_client, 'adopt_action_chunk', None)
chunk_step = getattr(dual_arm_client, 'chunk_step', None)
pytestmark = pytest.mark.skipif(adopt_action_chunk is None or chunk_step is None,
                                reason='chunk prefetch path is not present')


def chunk(rows=24, value=0.5):
    return {name: np.full((rows, size), value) for name, size in PARTS.items()}


def test_adopts_a_well_formed_chunk_and_reports_rows():
    served, rows = adopt_action_chunk({'action_chunk': chunk(24)}, 24)
    assert rows == 24
    assert set(served) == set(PARTS)
    assert served['left_arm'].shape == (24, 6)


def test_trims_to_the_configured_action_steps():
    served, rows = adopt_action_chunk({'action_chunk': chunk(32)}, 24)
    assert rows == 24
    assert all(array.shape[0] == 24 for array in served.values())


def test_absent_chunk_keeps_step_serving():
    assert adopt_action_chunk({'action': chunk(1)}, 24) == (None, 0)
    assert adopt_action_chunk({}, 24) == (None, 0)
    assert adopt_action_chunk({'action_chunk': {}}, 24) == (None, 0)


@pytest.mark.parametrize('mutate', [
    lambda c: c.pop('right_gripper'),
    lambda c: c.__setitem__('left_arm', np.zeros((24, 5))),
    lambda c: c.__setitem__('left_arm', np.zeros(6)),
    lambda c: c.__setitem__('left_arm', np.full((24, 6), np.nan)),
    lambda c: c.__setitem__('right_arm', np.zeros((23, 6))),
])
def test_malformed_chunks_fall_back_instead_of_executing(mutate):
    served = chunk()
    mutate(served)
    assert adopt_action_chunk({'action_chunk': served}, 24) == (None, 0)


def test_each_row_passes_the_existing_action_validation():
    served, rows = adopt_action_chunk({'action_chunk': chunk(24)}, 24)
    for row in range(rows):
        left, right = validate_action(chunk_step(served, row))
        assert left.shape == (7,) and right.shape == (7,)


def test_gripper_rows_outside_zero_one_are_still_rejected():
    served = chunk()
    served['left_gripper'][:] = 5.0
    served, rows = adopt_action_chunk({'action_chunk': served}, 24)
    with pytest.raises(ValueError, match='exceeds'):
        validate_action(chunk_step(served, 0))
