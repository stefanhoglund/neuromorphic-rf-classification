# rf_experiments/runner.py
import torch
import os
import json
import time
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from rf_experiments.models import build_model
from rf_experiments.trainers import build_trainer
from rf_experiments.encoders import build_encoder
from rf_experiments.dataset_structured import RadioML2018StructuredHDF5


class ExperimentRunner:
    def __init__(self, cfg, run_dir, logger=None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = logger

        tcfg = cfg.get("training", {})
        device_str = tcfg.get("device", "auto").lower()
        if device_str == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device_str == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "training.device is 'mps' but torch.backends.mps.is_available() is False"
                )
            self.device = torch.device("mps")
        elif device_str in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "training.device is 'cuda' but no CUDA device is available"
                )
            self.device = torch.device("cuda")
        else:
            # "cpu" or anything else -> CPU
            self.device = torch.device("cpu")

        if self.logger is not None:
            self.logger.info(f"Using device: {self.device}")
        else:
            print(f"Using device: {self.device}")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _resolve_device(self):
        tcfg = self.cfg["training"]
        requested = tcfg.get("device", "auto")

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested == "cuda" and not torch.cuda.is_available():
            self._log("device='cuda' requested but CUDA not available, falling back to CPU")
            return "cpu"

        if requested == "mps" and not torch.backends.mps.is_available():
            self._log("device='mps' requested but MPS not available, falling back to CPU")
            return "cpu"

        return requested

    def _build_dataloaders(self):
        dcfg = self.cfg["dataset"]
        ecfg = self.cfg["encoding"]

        self._log(f"Dataset config: {dcfg}")
        self._log(f"Encoding config: {ecfg}")

        encoder = build_encoder(ecfg)

        full_ds = RadioML2018StructuredHDF5(
            h5_path=dcfg["path"],
            encoder=encoder,
            frames_per_combo=dcfg.get("frames_per_combo", 256),
            mods_to_use=dcfg.get("mods_to_use"),
            snrs_to_use=dcfg.get("snrs_to_use"),
        )

        n_total = len(full_ds)
        self._log(f"Total logical dataset size: {n_total} samples")
        self._log(f"Num classes (from dataset): {full_ds.num_classes}")
        self.num_classes = full_ds.num_classes

        train_frac = dcfg.get("train_frac", 0.7)
        val_frac = dcfg.get("val_frac", 0.15)

        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        n_test = n_total - n_train - n_val

        self._log(f"Splits: train={n_train}, val={n_val}, test={n_test}")

        g = torch.Generator().manual_seed(self.cfg["experiment"].get("seed", 0))
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test], generator=g
        )

        dist = full_ds.inspect_distribution()
        classes = dist["classes"]
        counts = dist["class_counts"]
        snrs = dist["snrs"]
        snr_counts = dist["snr_counts"]

        self._log(f"Full dataset labels: {dict(zip(classes.tolist(), counts.tolist()))}")
        self._log(f"Full dataset SNRs: {dict(zip(snrs.tolist(), snr_counts.tolist()))}")



        tcfg = self.cfg["training"]
        batch_size = tcfg["batch_size"]
        num_workers = 0
        pin_memory = False

        self._log(f"Dataloader: batch_size={batch_size}, num_workers={num_workers}")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader, full_ds.snr_values

    def run(self):
        device = self._resolve_device()
        self._log(f"Using device: {device}")
        self._log(f"Torch version: {torch.__version__}")

        train_loader, val_loader, test_loader, snr_values = self._build_dataloaders()

        self.cfg["model"]["params"]["num_classes"] = self.num_classes
        
        self._log(f"Building model: {self.cfg['model']}")

        model_cfg = self.cfg["model"]
        params = model_cfg.setdefault("params", {})

        inferred = self.num_classes
        explicit = params.get("num_classes", None)

        if explicit is not None and explicit != inferred:
            self._log(
                f"WARNING: model.num_classes={explicit} but dataset has {inferred} classes. "
                f"Overriding to {inferred}."
            )

        params["num_classes"] = inferred  # enforce consistency

        self._log(f"Building model with num_classes={params['num_classes']}")
        model = build_model(model_cfg).to(device)

        self._log(f"Building trainer: {self.cfg['training']}")
        trainer = build_trainer(self.cfg, model, device, self.run_dir, snr_values)
        if hasattr(trainer, "logger"):
            trainer.logger = self.logger

        self._log("Starting training loop...")

        # --- timing ---
        t0_total = time.perf_counter()
        t0_train = time.perf_counter()


        best_metrics, best_state = trainer.train(train_loader, val_loader)   # expects dict with acc/loss

        train_time = time.perf_counter() - t0_train

        best_ckpt_path = os.path.join(self.run_dir, "best_model.pt")
        torch.save(
            {
                "model_state_dict": best_state,
                "cfg": self.cfg,                # full config for reconstruction
                "best_epoch": best_metrics["epoch"],
                "best_val_acc": best_metrics["acc"],
                "num_classes": self.num_classes,  # or wherever you keep this
            },
            best_ckpt_path,
        )
        self.logger.info(f"Saved best model checkpoint to {best_ckpt_path}")



        self._log("Evaluating on test set with best-val checkpoint...")
        
        model.load_state_dict(best_state)
        
        # test_metrics = trainer.evaluate(test_loader, split="test")
        test_metrics, y_true, y_pred = trainer.evaluate(
            test_loader, split="test", epoch=best_metrics["epoch"], return_details=True, save_details=True,
            save_tag="test_final"
        )

        drop_frac = 0.3  # 30% of time steps zeroed

        test_dropout_metrics = trainer.evaluate(
            test_loader,
            split="test",            
            epoch=None,              
            return_details=False,
            save_details=True,
            save_tag=f"test_dropout_{int(drop_frac*100)}",
            data_dropout=drop_frac,
            permute_time=False,
        )

        self.logger.info(
            f"Test metrics with {int(drop_frac*100)}% time-step dropout: "
            f"{test_dropout_metrics}"
        )


        test_shuffled_metrics = trainer.evaluate(
            test_loader,
            split="test",
            epoch=None,
            return_details=False,
            save_details=True,
            save_tag="test_shuffled_time",
            data_dropout=None,
            permute_time=True,
        )

        self.logger.info(
            f"Test metrics with shuffled time dimension: {test_shuffled_metrics}"
        )


        # Save confusion matrix image
        self._save_confusion_matrix(y_true, y_pred)

        total_time = time.perf_counter() - t0_total

        metrics = {
            "best_val": best_metrics,
            "test": test_metrics,
            "training_device": str(self.device),
            "train_time_seconds": train_time,
             "total_time_seconds":total_time
        }

        os.makedirs(self.run_dir, exist_ok=True)
        metrics_path = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # self._log(f"Saved metrics to {metrics_path}")

        if self.logger is not None:
            self.logger.info(f"Saved metrics to {metrics_path}")
            self.logger.info(f"Finished experiment: {self.cfg['experiment']['name']}")
            self.logger.info(
                f"Best val metrics: {best_metrics}"
            )
            self.logger.info(
                f"Test metrics: {test_metrics}"
            )

        return metrics

    def _save_confusion_matrix(self, y_true, y_pred):
        """
        Save a normalized confusion matrix (test set) as PNG in this run_dir.
        """
        # y_true / y_pred are plain lists of ints
        labels = list(range(self.num_classes))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # normalize per row (true label)
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Test confusion matrix (normalized)")

        fig.tight_layout()
        out_path = os.path.join(self.run_dir, "confusion_matrix.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        self._log(f"Saved confusion matrix to {out_path}")
