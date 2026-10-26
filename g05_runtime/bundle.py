"""Portable bundle validation, independent of the GUI model registry."""
import hashlib
import json
from pathlib import Path
import yaml
from .wire import PARTS


def inspect_bundle(root,verify=False):
    root=Path(root)
    marker=json.loads((root/'a1z_g05.json').read_text())
    cfg=yaml.safe_load((root/'.hydra/config.yaml').read_text())
    shape=cfg['data']['embodiment_datasets']['galaxea_a1z']['shape_meta']
    for group in ('action','state'):
        if {m['key']:m['raw_shape'] for m in shape[group]}!=PARTS:
            raise ValueError(f'Bundle {group} is not the A1Z 14D contract')
    if marker.get('action_semantics')!='absolute_arm_rad_gripper_opening' or marker.get('optimizer_included',False):
        raise ValueError('Expected inference-only absolute-radian/normalized-gripper bundle')
    if cfg['data']['action_size']!=32:
        raise ValueError('Experiments require a 32-step trained horizon')
    for name,row in marker['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe bundle manifest path')
        path=root/path
        if not path.is_file() or path.stat().st_size!=row['bytes']:
            raise ValueError(f'Missing or changed bundle file: {name}')
        if verify:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    digest.update(block)
            if digest.hexdigest()!=row['sha256']:
                raise ValueError(f'Bundle SHA256 mismatch: {name}')
    return marker
