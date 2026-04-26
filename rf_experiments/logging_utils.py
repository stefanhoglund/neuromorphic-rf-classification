# rf_experiments/logging_utils.py
import logging
import os
from typing import Optional


def get_logger(run_dir: Optional[str] = None, name: str = "rf_experiment"):
    logger = logging.getLogger(name)
    if logger.handlers:
        # already configured
        return logger

    logger.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))

    logger.addHandler(ch)

    # optional file handler
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(fh)

    return logger
