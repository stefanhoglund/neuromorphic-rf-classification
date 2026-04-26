# run_experiment.py
import argparse
import os
from datetime import datetime

import yaml

from rf_experiments.runner import ExperimentRunner
from rf_experiments.logging_utils import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- load config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- derive run directory with timestamp ----
    exp_name = cfg["experiment"]["name"]
    out_root = cfg["experiment"].get("output_dir", "results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{timestamp}"         
    run_dir = os.path.join(out_root, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # ---- save the *exact* config used for this run ----
    config_out_path = os.path.join(run_dir, "config.yaml")
    with open(config_out_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # ---- logger ----
    logger = get_logger(run_dir)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Saved resolved config to: {config_out_path}")
    logger.info(f"Run directory: {run_dir}")

    # ---- run experiment ----
    runner = ExperimentRunner(cfg, run_dir, logger=logger)
    metrics = runner.run()

    logger.info(f"Finished experiment: {exp_name}")
    logger.info(f"Best val metrics: {metrics.get('best_val', {})}")
    logger.info(f"Test metrics: {metrics.get('test', {})}")


if __name__ == "__main__":
    main()
