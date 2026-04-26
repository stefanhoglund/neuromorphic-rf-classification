from .sigma_delta import SigmaDeltaEncoder
from .iq_grid import IQGridEncoder
from .tf_events import TFEventsEncoder
from .rate_code import RateCodeEncoder
from .latency_ttfs import LatencyTTFSEncoder
from .level_crossing_adc import LevelCrossingADCEncoder


def build_encoder(ecfg):
    enc_type = ecfg["type"]
    params = ecfg.get("params", {})

    if enc_type == "sigma_delta":
        return SigmaDeltaEncoder(**params)
    elif enc_type == "iq_grid":
        return IQGridEncoder(**params)
    elif enc_type == "tf_events":
        return TFEventsEncoder(**params)
    elif enc_type == "rate_code":
        return RateCodeEncoder(**params)
    elif enc_type == "latency_ttfs":
        return LatencyTTFSEncoder(**params)
    elif enc_type == "level_crossing_adc":
        return LevelCrossingADCEncoder(**params)
    else:
        raise ValueError(f"Unknown encoding type: {enc_type}")
