import threading
import numpy as np
import pytest
from g05_runtime.config import RunConfig
from g05_runtime.engine import Plan
from g05_runtime.experiment import ExperimentExecutor
from g05_runtime.journal import Journal


def test_adoption_uses_matching_time_and_keeps_gripper_events(tmp_path):
    cfg=RunConfig(bundle=tmp_path,model_python=tmp_path/'python',task='test',engine='async')
    old=np.zeros((32,14));new=np.ones((32,14))
    old[:,0]=np.arange(32)*.01;new[:,0]=np.arange(32)*.01+.1
    journal=Journal(tmp_path/'events')
    engine=ExperimentExecutor(cfg,None,None,journal,threading.Event())
    try:
        adopted=engine.adopt(Plan(new,20,0,.15),Plan(old,0,0,.15),26)
        assert adopted.values[6,0]==old[26,0]
        assert adopted.values[9,0]==new[9,0]
        assert adopted.values[6,6]==0 and adopted.values[8,6]==0
        assert adopted.values[9,6]==1
        np.testing.assert_array_equal(adopted.values[10:],new[10:])
    finally:journal.close()


def test_too_late_overlap_is_rejected(tmp_path):
    cfg=RunConfig(bundle=tmp_path,model_python=tmp_path/'python',task='test',engine='async')
    journal=Journal(tmp_path/'events');engine=ExperimentExecutor(cfg,None,None,journal,threading.Event())
    try:
        with pytest.raises(RuntimeError,match='overlap'):
            engine.adopt(Plan(np.zeros((32,14)),20,0,.15),Plan(np.zeros((32,14)),0,0,.15),31)
    finally:journal.close()
