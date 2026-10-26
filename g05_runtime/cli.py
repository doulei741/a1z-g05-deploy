"""Standalone G0.5 inference entry point. Hardware is opt-in, never opened by checks/replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
from .bundle import inspect_bundle
from .config import RunConfig
from .engine import make_executor
from .journal import Journal
from .locks import Leases
from .transport import Predictor
from .version import ROOT,capabilities,identity
from .wire import split,unpack,validate_chunk


class Replay:
    def __init__(self,cfg,stub=False):
        self.cfg=cfg;self.last=None;self.stub=stub
        if stub:
            self.images={name:np.zeros((3,480,640),dtype=np.uint8) for name in ('head_rgb','left_wrist_rgb','right_wrist_rgb')}
            self.state=np.zeros(14,dtype=np.float32);self.state[[6,13]]=.8
        else:
            with np.load(cfg.bundle/'offline_observation.npz',allow_pickle=False) as sample:
                self.images={k:sample[k].copy() for k in ('head_rgb','left_wrist_rgb','right_wrist_rgb')}
                self.state=sample['state'].copy()

    def open(self):
        pass

    def observe(self):
        return {'images':self.images,'state':split(self.state),'task':self.cfg.task,
                'frequency':30,'embodiment_type':'galaxea_a1z'}

    def send(self,target):
        if not np.isfinite(target).all():
            raise ValueError('Non-finite replay target')
        self.last=target.copy()

    def feedback(self):
        return None  # Replay has no physical joint feedback.

    def hold(self):
        pass

    def close(self):
        pass


class StubPredictor:
    def __init__(self,stop,latency=.17):
        self.stop=stop;self.latency=latency;self.calls=0

    def __call__(self,obs,prefix=None,delay_steps=0):
        if self.stop.wait(self.latency):
            raise InterruptedError('Stub cancelled')
        self.calls+=1
        target=np.zeros((32,14),dtype=np.float64)
        target[:,0]=.02*np.sin(np.linspace(0,1,32)+self.calls*.1)
        target[:,[6,13]]=.8
        return {'actions':target,'timing':{'stub':True}}


def gpu_check():
    import csv
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                  '--format=csv,noheader,nounits'],text=True)
    blockers=[]
    for row in csv.reader(rows.splitlines(),skipinitialspace=True):
        if len(row)!=3:
            raise RuntimeError('Cannot validate GPU compute ownership')
        pid,name,memory=row
        try:
            exe=(Path('/proc')/pid/'exe').resolve(strict=True)
            viewer=exe==Path(name).resolve() and exe.parts[-3:]==('rerun_sdk','rerun_cli','rerun') and int(memory)<=512
        except (OSError,ValueError):
            viewer=False
        if not viewer:
            blockers.append(f'{pid} {name} {memory}MiB')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if blockers or free<18000:
        raise RuntimeError(f'GPU unavailable: other compute={blockers}, free={free}MiB')


class ModelProcess:
    def __init__(self,cfg,config_path,folder,stop):
        self.cfg=cfg;self.config_path=config_path;self.folder=folder;self.stop=stop;self.process=None;self.log=None

    def __enter__(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        self.url=f'ws://127.0.0.1:{port}'
        env=dict(os.environ)
        env.update(PYTHONPATH=os.pathsep.join([str(ROOT),str((self.cfg.model_source or ROOT/'GalaxeaVLA')/'src')]),
                   PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                   NO_PROXY='127.0.0.1,localhost',A1Z_WORKSPACE=str(ROOT),
                   A1Z_DATASET=str(ROOT/'data'),G05_OUTPUT_DIR=str(ROOT/'outputs'))
        env.pop('ALL_PROXY',None);env.pop('all_proxy',None)
        self.log=(self.folder/'server.log').open('w')
        self.process=subprocess.Popen([str(self.cfg.model_python),'-m','g05_runtime.server','--config',str(self.config_path),
                                       '--port',str(port)],cwd=self.cfg.model_source or ROOT/'GalaxeaVLA',env=env,
                                      stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        from websockets.sync.client import connect
        deadline=time.monotonic()+300
        try:
            while not self.stop.is_set():
                if self.process.poll() is not None:
                    raise RuntimeError(f'Model server exited ({self.process.returncode}); see {self.folder/"server.log"}')
                try:
                    with connect(self.url,open_timeout=.5,close_timeout=.2,proxy=None) as ws:
                        meta=unpack(ws.recv(timeout=2))
                        if meta.get('protocol_version')!=1:
                            raise ValueError('Model protocol mismatch')
                        return self
                except (OSError,TimeoutError):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Model loading timed out')
                    self.stop.wait(.25)
            raise InterruptedError('Stopped during model loading')
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=5)
        if self.log:
            self.log.close()


def event(kind,**data):
    print('G05_EVENT '+json.dumps({'type':kind,'data':data},ensure_ascii=False),flush=True)


def run(cfg,execute=False,stub=False):
    cap=capabilities()
    if cfg.engine not in cap['supported_engines']:
        raise ValueError(f"This pinned runtime supports {cap['supported_engines']}, not {cfg.engine}")
    if execute and (not cfg.safety_confirmed or cfg.hardware is None):
        raise ValueError('Hardware execution requires explicit confirmation and device selections')
    marker={} if stub else inspect_bundle(cfg.bundle,verify=True)
    stop=threading.Event()
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in previous:
        signal.signal(s,lambda *_:stop.set())
    folder=cfg.output_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+cfg.engine+'_'+uuid.uuid4().hex[:8])
    journal=Journal(folder)
    source=cfg.model_source or ROOT/'GalaxeaVLA'
    journal.emit('provenance',**cap,selected_model=identity(source),hardware_connected=execute,
                 run_kind='hardware' if execute else ('stub' if stub else 'recorded_observation_replay'),
                 checkpoint_step=marker.get('checkpoint_step'),weight=marker.get('files',{}).get('model.pt'))
    config_path=folder/'config.json';config_path.write_text(cfg.model_dump_json(indent=2))
    event('run_created',folder=str(folder),engine=cfg.engine,hardware_connected=execute)
    hardware=None;executor=None;failure=None
    resources=[]
    if not stub:
        resources.append('gpu:0')
    if execute:
        resources += ['can:'+c for c in (cfg.hardware.left_can,cfg.hardware.right_can)]
        resources += ['camera:'+str(c.serial or c.index_or_path) for c in cfg.hardware.cameras.values()]
    try:
        with Leases(resources):
            if not stub:
                gpu_check()
            from contextlib import nullcontext
            model_context=nullcontext(None) if stub else ModelProcess(cfg,config_path,folder,stop)
            with model_context as model:
                predict=StubPredictor(stop) if stub else Predictor(model.url,cfg,stop)
                warm_latency=None
                if not stub:
                    # Cold start happens against recorded observations before energizing devices.
                    sample=Replay(cfg);warm=predict(sample.observe())
                    validate_chunk(warm['actions'])
                    if cfg.engine=='rtc':
                        # Warm the guided path before energizing hardware and
                        # budget its measured cost, not the faster FM-only path.
                        for _ in range(2):
                            started=time.monotonic()
                            warm=predict(sample.observe(),prefix=warm['actions'][-16:],delay_steps=8)
                            validate_chunk(warm['actions'])
                            warm_latency=time.monotonic()-started
                        journal.emit('rtc_warmup',request_s=warm_latency,hardware_connected=False)
                    event('offline_check_passed',hardware_connected=False)
                if execute:
                    from .hardware import A1ZHardware
                    hardware=A1ZHardware(cfg,journal=journal)
                else:
                    hardware=Replay(cfg,stub=stub)
                try:
                    hardware.open()
                    event('ready',hardware_connected=execute)
                    print('G0.5 rollout connected' if execute else 'G0.5 recorded RGB/state check passed',flush=True)
                    executor=make_executor(cfg,hardware,predict,journal,stop)
                    if warm_latency is not None:
                        executor.latencies=[warm_latency]
                    executor.run()
                finally:
                    # Hold before shutting down model, journal or cameras.
                    stop.set()
                    hardware.hold()
                    hardware.close()
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        journal.emit('fault',reason=failure)
        raise
    finally:
        stop.set()
        summary={'engine':cfg.engine,'hardware_connected':execute,'task_success_verified':False,
                 'stop_reason':failure or 'completed_or_operator_stopped',
                 'targets':executor.count if executor else 0,'predictions':executor.cycles if executor else 0}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        journal.close()
        for s,handler in previous.items():
            signal.signal(s,handler)
        event('finished',**summary,folder=str(folder))
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capabilities',action='store_true',help='Print JSON without loading model/devices')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--duration',type=float)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='Validate configuration without model/device access')
    mode.add_argument('--offline',action='store_true',help='Real model with recorded RGB/state; no devices')
    mode.add_argument('--stub',action='store_true',help='CPU synthetic latency test; no GPU/devices')
    mode.add_argument('--execute',action='store_true',help='Operator-authorized hardware execution')
    args=parser.parse_args()
    if args.capabilities:
        print(json.dumps(capabilities(),ensure_ascii=False));return
    if args.config is None:
        parser.error('--config is required')
    cfg=RunConfig.model_validate_json(args.config.read_text()).resolve(args.config.parent)
    if args.duration is not None:
        cfg=RunConfig.model_validate({**cfg.model_dump(),'duration':args.duration})
    if cfg.engine not in capabilities()['supported_engines']:
        parser.error('Selected runtime does not support requested engine')
    if not any((args.offline,args.stub,args.execute)):
        inspect_bundle(cfg.bundle,verify=False)
        print('Configuration valid; model and hardware not loaded');return
    run(cfg,execute=args.execute,stub=args.stub)


if __name__=='__main__':
    main()
