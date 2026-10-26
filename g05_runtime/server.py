"""Independent model process. Complete-chunk protocol; never imports hardware or GUI."""
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import random
import sys
import time
from pathlib import Path
import numpy as np
from .config import RunConfig
from .version import ROOT
from .wire import pack,unpack,vector,validate_chunk


def load_model(cfg):
    source=cfg.model_source or ROOT/'GalaxeaVLA'
    sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('g05_runtime_upstream_server',source/'scripts/serve_policy.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import torch
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    checkpoint=cfg.bundle/'model.pt'
    processor=str(cfg.bundle/'hf_processor');tokenizer=str(cfg.bundle/'action_tokenizer.pt')
    overrides=cfg.model.overrides()+[
        'eval_embodiment=galaxea_a1z',f'datastatics_path={str(cfg.bundle/"dataset_stats.json")}',
        f'model.model_arch.hf_processor_path={processor}',
        f'model.processor.tokenizer_params.pretrained_model_name_or_path={processor}',
        f'tokenizer.vq_config.ckpt_dir={tokenizer}',f'model.tokenizer.vq_config.ckpt_dir={tokenizer}',
        f'model.model_arch.AT_CONFIG.ckpt_dir={tokenizer}']
    resolved=module.load_config_from_run_dir(module.find_run_dir(checkpoint),str(checkpoint),overrides)
    if 'embodiment_datasets' in resolved.data:
        module.filter_embodiment(resolved,'galaxea_a1z')
    policy,processor=module.setup(resolved,device='cuda')
    inferencer=module.PolicyInferencer(policy,processor,device='cuda')
    if cfg.engine=='rtc' and not hasattr(inferencer,'infer_rtc'):
        raise RuntimeError('Pinned model source does not implement inference-time RTC')
    return module,inferencer


def predict(module,inferencer,message):
    observation=message['observation']
    data=module.build_obs_dict(observation,inferencer.processor)
    start=time.monotonic()
    if message.get('prefix') is not None:
        actions,timing=inferencer.infer_rtc(data,message['prefix'],**message['rtc'])
    else:
        result,timing=inferencer.infer_with_timing([data]);actions=result[0]
    arrays={k:(v.detach().float().cpu().numpy() if hasattr(v,'detach') else np.asarray(v))
            for k,v in actions.items() if not k.startswith('_')}
    arrays={k:(v[0] if v.ndim==3 else v) for k,v in arrays.items()}
    values=validate_chunk(vector(arrays))
    return {'id':message['id'],'actions':values,'timing':{**timing,'server_total_ms':(time.monotonic()-start)*1000}}


async def serve(cfg,port):
    import websockets
    module,inferencer=load_model(cfg)
    gate=asyncio.Lock()
    async def handler(ws):
        await ws.send(pack({'protocol_version':1,'horizon':32,'rtc':hasattr(inferencer,'infer_rtc')}))
        async for raw in ws:
            try:
                message=unpack(raw)
                async with gate:
                    result=await asyncio.to_thread(predict,module,inferencer,message)
                await ws.send(pack(result))
            except Exception as exc:
                await ws.send(pack({'error':f'{type(exc).__name__}: {exc}'}))
    async with websockets.serve(handler,'127.0.0.1',port,max_size=64*1024**2,compression=None):
        print('G05_RUNTIME_MODEL_READY',flush=True)
        await asyncio.Future()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args();cfg=RunConfig.model_validate_json(args.config.read_text()).resolve(args.config.parent)
    asyncio.run(serve(cfg,args.port))


if __name__=='__main__':
    main()
