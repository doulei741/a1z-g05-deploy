"""Execution infrastructure shared by separately versioned experiment policies."""
from __future__ import annotations
from dataclasses import dataclass
import math
import queue
import threading
import time
import numpy as np
from .wire import validate_chunk


@dataclass(frozen=True)
class Plan:
    values: np.ndarray
    origin: int
    observed_s: float
    request_s: float

    def remaining(self,tick):
        return self.origin+len(self.values)-tick

    def at(self,tick):
        index=tick-self.origin
        if index<0 or index>=len(self.values):
            raise RuntimeError('Action buffer exhausted or plan is from the future')
        return self.values[index].copy()

    def prefix(self,tick):
        offset=tick-self.origin
        if not 0<=offset<len(self.values):
            raise RuntimeError('No aligned old prefix available')
        return self.values[offset:].copy()


class Job:
    """Exactly one bounded in-flight prediction; a late result never sends hardware actions."""
    def __init__(self,fn):
        self.result=queue.Queue(maxsize=1)
        def work():
            try:
                self.result.put((fn(),None))
            except BaseException as exc:
                self.result.put((None,exc))
        self.thread=threading.Thread(target=work,daemon=True,name='g05-next-plan')
        self.thread.start()

    def poll(self):
        try:
            value,error=self.result.get_nowait()
        except queue.Empty:
            return None
        if error:
            raise RuntimeError(f'Prediction failed: {error}') from error
        return value


class Executor:
    def __init__(self,cfg,hardware,predict,journal,stop):
        self.cfg=cfg;self.hardware=hardware;self.predict=predict;self.journal=journal;self.stop=stop
        self.latencies=[];self.cycles=0;self.count=0;self.last_target=None

    def request(self,origin,old=None,epoch=None):
        if self.stop.is_set():
            raise InterruptedError('Stopped before observation')
        capture_start=time.monotonic()
        observation=self.hardware.observe()
        observed=time.monotonic()
        if epoch is not None:
            origin=max(origin,math.floor((observed-epoch)*self.cfg.fps))
        prefix=old.prefix(origin) if old is not None else None
        delay=self.delay_steps()
        if prefix is not None:
            delay=min(delay,len(prefix))
        start=time.monotonic()
        response=self.predict(observation,prefix=prefix,delay_steps=delay)
        elapsed=time.monotonic()-start
        if self.stop.is_set():
            raise InterruptedError('Stopped while predicting')
        values=validate_chunk(response['actions'],tolerance=self.cfg.model.gripper_boundary_tolerance)
        # Only this producer updates the list; consumer reads a snapshot.
        self.latencies.append(elapsed);self.latencies=self.latencies[-64:]
        self.cycles+=1
        self.journal.emit('prediction',origin_tick=origin,observed_s=observed,
                          observation_read_s=observed-capture_start,request_s=elapsed,
                          delay_steps=delay,prefix_steps=0 if prefix is None else len(prefix),
                          server_timing=response.get('timing',{}),model_target=values.tolist())
        return Plan(values,origin,observed,elapsed)

    def delay_steps(self):
        recent=tuple(self.latencies)
        latency=max(recent,default=.20)+self.cfg.execution.latency_margin_s
        return max(1,math.ceil(latency*self.cfg.fps))

    def send(self,target,tick,scheduled=None):
        if self.stop.is_set():
            raise InterruptedError('Stopped before sending target')
        self.hardware.send(target)
        self.last_target=target.copy();self.count+=1
        self.journal.emit('target',tick=tick,scheduled_s=scheduled,target=target.tolist(),
                          feedback=self.hardware.feedback())

    def wait_until(self,deadline):
        self.stop.wait(max(0,deadline-time.monotonic()))


class SyncExecutor(Executor):
    """Baseline reproduces inference-inside-tick scheduling; experiment fixes opt in."""
    fixed_playback=False

    def run(self):
        end=time.monotonic()+self.cfg.duration
        tick=0;period=1/self.cfg.fps
        while not self.stop.is_set() and time.monotonic()<end:
            start=time.monotonic()
            plan=self.request(tick)
            for index in range(self.cfg.model.action_steps):
                if self.stop.is_set() or time.monotonic()>=end:
                    return
                self.send(plan.values[index],tick)
                tick+=1
                if self.fixed_playback or index>0:
                    start=time.monotonic()
                self.wait_until(min(end,start+period))
                start=time.monotonic()
            if self.cfg.model.max_cycles and self.cycles>=self.cfg.model.max_cycles:
                return


class AsyncExecutor(Executor):
    """Fixed-clock asynchronous engine; branch policy defines generation and adoption."""
    def adopt(self,new,old,tick):
        raise NotImplementedError('Experiment must implement an explicit adoption policy')

    def run(self):
        first_started=time.monotonic()
        plan=self.request(0)
        epoch=time.monotonic()
        end=first_started+self.cfg.duration
        tick=0;job=None;period=1/self.cfg.fps
        while not self.stop.is_set() and time.monotonic()<end:
            scheduled=epoch+tick*period
            self.wait_until(min(end,scheduled))
            if self.stop.is_set() or time.monotonic()>=end:
                break
            lateness=time.monotonic()-scheduled
            if lateness>self.cfg.execution.max_control_lateness_s:
                raise RuntimeError(f'Control deadline missed by {lateness:.3f}s')
            # Never burst old targets to catch up after a delayed control tick.
            actual_tick=max(tick,math.floor((time.monotonic()-epoch)/period))
            if actual_tick>tick:
                self.journal.emit('skipped_expired_ticks',from_tick=tick,to_tick=actual_tick)
                tick=actual_tick;scheduled=epoch+tick*period
            if job is not None:
                replacement=job.poll()
                if replacement is not None:
                    if replacement.remaining(tick)<=0 or replacement.origin>tick:
                        raise RuntimeError('Prediction arrived outside its executable time horizon')
                    plan=self.adopt(replacement,plan,tick)
                    self.journal.emit('adoption',tick=tick,origin_tick=plan.origin,
                                      elapsed_prefix_steps=tick-plan.origin,request_s=plan.request_s)
                    job=None
            if plan.remaining(tick)<=0:
                self.journal.emit('buffer_exhausted',tick=tick)
                raise RuntimeError('Action buffer exhausted; holding instead of replaying stale actions')
            self.send(plan.at(tick),tick,scheduled)
            remaining=plan.remaining(tick+1)
            lead=max(self.cfg.execution.lead_steps,self.delay_steps()+self.cfg.execution.transition_steps)
            if lead>=len(plan.values):
                raise RuntimeError('Measured latency leaves insufficient action horizon')
            if job is None and remaining<=lead:
                if self.cfg.model.max_cycles and self.cycles>=self.cfg.model.max_cycles:
                    return
                old=plan;origin=tick
                job=Job(lambda:self.request(origin,old,epoch))
            tick+=1


def make_executor(cfg,hardware,predict,journal,stop):
    if cfg.engine=='baseline':
        cls=SyncExecutor
    else:
        from .experiment import ExperimentExecutor
        cls=ExperimentExecutor
    return cls(cfg,hardware,predict,journal,stop)
