"""Scheme 1: time-aligned asynchronous execution with model-level RTC guidance."""
from .engine import AsyncExecutor


class ExperimentExecutor(AsyncExecutor):
    def adopt(self,new,old,tick):
        # Prefix consistency was applied during FM sampling. Do not average the
        # generated trajectory again or re-anchor it against the current state.
        self.journal.emit('rtc_adoption',tick=tick,origin_tick=new.origin,
                          elapsed_prefix_steps=tick-new.origin,remaining_old_steps=old.remaining(tick))
        return new
