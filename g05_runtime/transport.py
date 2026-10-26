"""Interruptible complete-chunk requests; model never owns robot device handles."""
import threading
import time
from websockets.sync.client import connect
from .wire import pack,unpack,split


class Predictor:
    def __init__(self,url,cfg,stop):
        self.url=url;self.cfg=cfg;self.stop=stop;self.sequence=0

    def __call__(self,observation,prefix=None,delay_steps=0):
        self.sequence+=1;request_id=self.sequence
        with connect(self.url,open_timeout=3,close_timeout=1,max_size=64*1024**2,compression=None,proxy=None) as ws:
            meta=unpack(ws.recv(timeout=3))
            if meta.get('protocol_version')!=1 or meta.get('horizon')!=32:
                raise ValueError('Incompatible runtime server handshake')
            if self.cfg.engine=='rtc' and not meta.get('rtc'):
                raise ValueError('Requested RTC but model server has no RTC support')
            message={'id':request_id,'observation':observation}
            if self.cfg.engine=='rtc' and prefix is not None:
                message['prefix']=split(prefix)
                message['rtc']={'delay_steps':delay_steps,'soft_steps':self.cfg.execution.rtc_soft_steps,
                                'guidance_weight':self.cfg.execution.rtc_guidance_weight}
            if self.stop.is_set():
                raise InterruptedError('Prediction cancelled')
            ws.send(pack(message));deadline=time.monotonic()+self.cfg.model.request_timeout
            while not self.stop.is_set():
                try:
                    response=unpack(ws.recv(timeout=min(.1,max(.001,deadline-time.monotonic()))))
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    if response.get('id')!=request_id:
                        raise ValueError('Stale or mismatched response ID')
                    return response
                except TimeoutError:
                    if time.monotonic()>=deadline:
                        raise TimeoutError('Model request timed out')
            raise InterruptedError('Prediction cancelled')
