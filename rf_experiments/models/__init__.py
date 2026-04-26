from .conv1d_snn import Conv1DSNN
from .conv2d_snn import Conv2DSNN
from .lsm import LSMSNN
from .hrm_rf import HRMRFClassifier

def build_model(cfg):
    mtype = cfg["type"]
    params = cfg.get("params", {})

    if mtype == "conv1d_snn":
        return Conv1DSNN(**params)
    elif mtype == "conv2d_snn":
        return Conv2DSNN(**params)
    elif mtype == "conv2d_ann":
        return Conv2DANN(**params)
    elif mtype == "lsm":
        return LSMSNN(**params)
    elif mtype == "hrm_rf":
        return HRMRFClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {mtype}")
