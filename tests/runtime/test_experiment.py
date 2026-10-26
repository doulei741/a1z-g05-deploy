import threading
import numpy as np
from g05_runtime.config import RunConfig
from g05_runtime.engine import Plan
from g05_runtime.experiment import ExperimentExecutor
from g05_runtime.journal import Journal


def test_rtc_uses_model_conditioned_plan_without_second_blend(tmp_path):
    cfg=RunConfig(bundle=tmp_path,model_python=tmp_path/'python',task='test',engine='rtc')
    journal=Journal(tmp_path/'events');engine=ExperimentExecutor(cfg,None,None,journal,threading.Event())
    new=Plan(np.ones((32,14)),20,0,.3);old=Plan(np.zeros((32,14)),0,0,.15)
    try:assert engine.adopt(new,old,27) is new
    finally:journal.close()
