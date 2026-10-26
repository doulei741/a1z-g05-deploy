"""Runtime capabilities and immutable code provenance; no device imports."""
import json
import subprocess
from pathlib import Path
from . import PROTOCOL_VERSION,__version__

ROOT=Path(__file__).resolve().parents[1]


def identity(path):
    try:
        def git(*args):
            return subprocess.check_output(['git','-C',str(path),*args],text=True,stderr=subprocess.DEVNULL).strip()
        return {'path':str(path),'commit':git('rev-parse','HEAD'),
                'branch':git('branch','--show-current'),'dirty':bool(git('status','--porcelain'))}
    except (OSError,subprocess.SubprocessError):
        return {'path':str(path),'commit':None,'dirty':None}


def capabilities():
    manifest=json.loads((ROOT/'runtime.json').read_text())
    return {'protocol_version':PROTOCOL_VERSION,'package_version':__version__,
            'runtime':identity(ROOT),'model':identity(ROOT/'GalaxeaVLA'),**manifest}
