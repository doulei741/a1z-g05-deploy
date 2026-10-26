"""Shared user-scoped hardware/GPU leases across CLI, GUI and experiment directories."""
import fcntl
import hashlib
import os
from pathlib import Path


class Leases:
    def __init__(self,resources):
        self.resources=sorted(set(resources));self.files=[]

    def __enter__(self):
        root=Path(os.environ.get('XDG_RUNTIME_DIR',f'/tmp/a1z-g05-{os.getuid()}'))/'g05-leases'
        root.mkdir(parents=True,exist_ok=True,mode=0o700)
        try:
            for name in self.resources:
                path=root/(hashlib.sha256(name.encode()).hexdigest()+'.lock')
                stream=path.open('a+')
                try:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    raise RuntimeError(f'Resource is in use: {name}')
                self.files.append(stream)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        for stream in reversed(self.files):
            stream.close()
        self.files.clear()
