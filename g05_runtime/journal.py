"""Bounded asynchronous run journal; disk/UI are never waited on by the controller."""
import json
import queue
import threading
import time


class Journal:
    def __init__(self,folder,capacity=16384):
        folder.mkdir(parents=True,exist_ok=False)
        self.folder=folder
        self.queue=queue.Queue(maxsize=capacity)
        self.error=None
        self.thread=threading.Thread(target=self._writer,daemon=True,name='g05-journal')
        self.thread.start()

    def _writer(self):
        try:
            with (self.folder/'events.jsonl').open('w') as stream:
                while True:
                    value=self.queue.get()
                    if value is None:
                        break
                    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
        except BaseException as exc:
            self.error=exc

    def emit(self,kind,**data):
        if self.error:
            raise RuntimeError(f'Run journal failed: {self.error}')
        try:
            self.queue.put_nowait({'type':kind,'monotonic_s':time.monotonic(),**data})
        except queue.Full as exc:
            raise RuntimeError('Run journal queue full; refusing unrecorded continuation') from exc

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None,timeout=2)
            self.thread.join(timeout=5)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(f'Journal did not finish: {self.error}')
