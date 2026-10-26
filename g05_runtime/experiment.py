"""Scheme 3: full-chunk synchronous inference with an uncompressed playback clock."""
from .engine import SyncExecutor


class ExperimentExecutor(SyncExecutor):
    fixed_playback=True
