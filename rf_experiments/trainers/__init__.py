from .bptt_trainer import BPTTTrainer
from .dcll_trainer import DCLLTrainer
from .local_rule_trainer import LocalRuleTrainer


def build_trainer(cfg, model, device, run_dir, snr_values):
    tcfg = cfg["training"]
    trainer_type = tcfg["trainer"]

    if trainer_type == "bptt":
        return BPTTTrainer(cfg, model, device, run_dir, snr_values)
    elif trainer_type == "dcll":
        return DCLLTrainer(cfg, model, device, run_dir, snr_values)
    elif trainer_type == "local_rule":
        return LocalRuleTrainer(cfg, model, device, run_dir, snr_values)
    else:
        raise ValueError(f"Unknown trainer: {trainer_type}")
