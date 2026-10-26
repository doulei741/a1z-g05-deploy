import json
from pathlib import Path
import threading
import time
import numpy as np
import pytest
from g05_runtime.config import RunConfig
from g05_runtime.engine import AsyncExecutor,Plan,SyncExecutor
from g05_runtime.journal import Journal
from g05_runtime.locks import Leases
from g05_runtime.wire import pack,unpack,split,vector,validate_chunk


def config(tmp_path,**kw):
    return RunConfig(bundle=tmp_path,model_python=Path('/usr/bin/python3'),task='test',duration=1.4,**kw)


class Device:
    def __init__(self): self.targets=[]
    def observe(self): return {}
    def send(self,x): self.targets.append((time.monotonic(),x.copy()))
    def feedback(self): return None


class IdentityAdoption(AsyncExecutor):
    def adopt(self,new,old,tick): return new


def predictor(stop,delays):
    calls=[]
    def predict(obs,**kwargs):
        i=len(calls);calls.append(kwargs)
        if stop.wait(delays[min(i,len(delays)-1)]): raise InterruptedError('stopped')
        a=np.zeros((32,14));a[:,[6,13]]=.5
        return {'actions':a}
    return predict,calls


def execute(tmp_path,cls,delays,duration=1.4):
    cfg=config(tmp_path);cfg.duration=duration;stop=threading.Event();device=Device();journal=Journal(tmp_path/'run')
    predict,calls=predictor(stop,delays);engine=cls(cfg,device,predict,journal,stop)
    try: engine.run()
    finally: stop.set();journal.close()
    return device,calls


def test_wire_roundtrip_and_shape():
    a=np.arange(32*14,dtype=float).reshape(32,14);a[:,[6,13]]=.5
    np.testing.assert_array_equal(vector(unpack(pack(split(a)))),a)
    validate_chunk(a)
    with pytest.raises(ValueError):validate_chunk(a[:2])
    with pytest.raises(ValueError):pack(np.array([object()]))


def test_config_cannot_silently_accept_unknown_options(tmp_path):
    with pytest.raises(ValueError):config(tmp_path,unknown=True)
    with pytest.raises(ValueError):RunConfig(bundle=tmp_path,model_python=Path('/usr/bin/python3'),task='x',engine='rtc',model={'action_head':'ar'})


def test_shared_hardware_lock_is_exclusive():
    with Leases(['test:runtime-exclusive']):
        with pytest.raises(RuntimeError):
            with Leases(['test:runtime-exclusive']): pass


def test_baseline_reproduces_compressed_first_pair(tmp_path):
    device,_=execute(tmp_path,SyncExecutor,[.12],duration=.4)
    assert device.targets[1][0]-device.targets[0][0]<.02


def test_fixed_playback_does_not_compress_first_pair(tmp_path):
    class Fixed(SyncExecutor):fixed_playback=True
    device,_=execute(tmp_path,Fixed,[.12],duration=.4)
    assert device.targets[1][0]-device.targets[0][0]>=.03


def test_async_hides_latency_without_target_bursts(tmp_path):
    device,calls=execute(tmp_path,IdentityAdoption,[.12],duration=1.7)
    intervals=np.diff([t for t,x in device.targets])
    assert len(calls)>=2
    assert min(intervals)>.015
    assert max(intervals)<.09
    assert calls[1]['prefix'] is not None


def test_unexpected_latency_underflow_fails_instead_of_replaying(tmp_path):
    with pytest.raises(RuntimeError,match='buffer exhausted'):
        execute(tmp_path,IdentityAdoption,[.01,1.2],duration=1.5)


def test_cancelled_prediction_never_sends_an_action(tmp_path):
    cfg=config(tmp_path);stop=threading.Event();device=Device();journal=Journal(tmp_path/'cancelled')
    predict,calls=predictor(stop,[.6]);engine=SyncExecutor(cfg,device,predict,journal,stop)
    timer=threading.Timer(.1,stop.set);timer.start()
    try:
        with pytest.raises(InterruptedError):engine.run()
        assert not device.targets
    finally:stop.set();timer.join();journal.close()


def test_plan_refuses_out_of_time_access():
    p=Plan(np.zeros((32,14)),10,0,0)
    with pytest.raises(RuntimeError):p.at(9)
    with pytest.raises(RuntimeError):p.at(42)


def test_model_does_not_outlive_its_controller(tmp_path):
    import os
    import subprocess
    import sys
    ready=tmp_path/'child-ready'
    child_code="from g05_runtime.process_guard import bind_parent; import os,sys,time; from pathlib import Path; bind_parent(int(sys.argv[1])); Path(sys.argv[2]).write_text('ready'); time.sleep(60)"
    parent_code="""import os,subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
deadline=time.monotonic()+5
while not Path(sys.argv[2]).exists() and time.monotonic()<deadline: time.sleep(.01)
assert Path(sys.argv[2]).exists()
print(p.pid,flush=True)
"""
    result=subprocess.run([sys.executable,'-c',parent_code,child_code,str(ready)],capture_output=True,text=True,timeout=8,check=True)
    pid=int(result.stdout.strip());deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        try:
            state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        except FileNotFoundError:
            return
        if state=='Z': return
        time.sleep(.01)
    # Only our child is cleaned up if the guard regresses.
    os.kill(pid,9)
    pytest.fail('Model child remained alive after controller exit')
