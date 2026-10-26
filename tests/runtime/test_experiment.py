from g05_runtime.experiment import ExperimentExecutor
from g05_runtime.engine import SyncExecutor


def test_sync_fast_keeps_fixed_playback_and_baseline_is_unchanged():
    assert ExperimentExecutor.fixed_playback is True
    assert SyncExecutor.fixed_playback is False
