"""Scheme 2: time-aligned asynchronous chunks with a bounded overlap transition.

This is post-generation trajectory transition, not inference-time RTC.
"""
import numpy as np
from .engine import AsyncExecutor,Plan
from .wire import JOINTS,GRIPPERS


class ExperimentExecutor(AsyncExecutor):
    def adopt(self,new,old,tick):
        offset=tick-new.origin
        count=min(self.cfg.execution.transition_steps,old.remaining(tick),new.remaining(tick))
        if count<2:
            raise RuntimeError('Insufficient aligned overlap for a continuous asynchronous transition')
        result=new.values.copy()
        for i in range(count):
            fraction=i/(count-1)
            weight=6*fraction**5-15*fraction**4+10*fraction**3
            prior=old.at(tick+i)
            result[offset+i,JOINTS]=(1-weight)*prior[JOINTS]+weight*new.values[offset+i,JOINTS]
            # Never average discrete grasp/release events into partial openings.
            if i<count-1:
                result[offset+i,GRIPPERS]=prior[GRIPPERS]
        self.journal.emit('transition',tick=tick,steps=count,new_offset=offset,
                          raw_new_target=new.values[offset].tolist(),old_target=old.at(tick).tolist())
        return Plan(result,new.origin,new.observed_s,new.request_s)
