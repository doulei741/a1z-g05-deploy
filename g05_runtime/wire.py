"""Shared numeric-only MessagePack protocol for complete action chunks."""
import msgpack
import numpy as np

PARTS = {'left_arm':6,'left_gripper':1,'right_arm':6,'right_gripper':1}
JOINTS = [*range(6), *range(7,13)]
GRIPPERS = [6,13]


def encode(value):
    if isinstance(value,np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise ValueError('Only numeric arrays are supported')
        return {'__ndarray__':True,'data':value.tobytes(),'dtype':value.dtype.str,'shape':value.shape}
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def decode(value):
    if value.get('__ndarray__'):
        dtype=np.dtype(value['dtype'])
        if dtype.kind not in 'biuf':
            raise ValueError('Non-numeric dtype')
        return np.frombuffer(value['data'],dtype=dtype).reshape(value['shape']).copy()
    return value


def pack(value):
    return msgpack.packb(value,default=encode,use_bin_type=True)


def unpack(value):
    if not isinstance(value,bytes):
        raise ValueError('Expected binary MessagePack')
    return msgpack.unpackb(value,object_hook=decode,raw=False,max_array_len=1_000_000)


def split(values):
    values=np.asarray(values)
    return dict(zip(PARTS,np.split(values,[6,7,13],axis=-1)))


def vector(action):
    if set(action)!=set(PARTS):
        raise ValueError('Expected four A1Z action groups')
    return np.concatenate([np.asarray(action[k]) for k in PARTS],axis=-1)


def validate_chunk(values,horizon=32,tolerance=.02):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(horizon,14) or not np.isfinite(values).all():
        raise ValueError(f'Expected finite ({horizon},14) chunk; got {values.shape}')
    if np.any(values[:,GRIPPERS]<-tolerance) or np.any(values[:,GRIPPERS]>1+tolerance):
        raise ValueError(f'Gripper output outside normalized boundary tolerance: '
                         f'min={values[:,GRIPPERS].min():.6f}, max={values[:,GRIPPERS].max():.6f}, tolerance={tolerance}')
    result=values.copy();result[:,GRIPPERS]=np.clip(result[:,GRIPPERS],0,1)
    return result
