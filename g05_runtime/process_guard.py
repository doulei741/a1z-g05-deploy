"""Linux parent-death guard: model must not outlive a forcibly terminated controller."""
import ctypes
import os
import signal
import sys


def bind_parent(expected_pid):
    if sys.platform!='linux':
        raise RuntimeError('The A1Z process guard requires Linux')
    if os.getppid()!=expected_pid:
        raise InterruptedError('Controller exited before model startup')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(),'Cannot install model parent-death guard')
    if os.getppid()!=expected_pid:
        raise InterruptedError('Controller exited while installing model process guard')
