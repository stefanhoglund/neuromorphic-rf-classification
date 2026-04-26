import os
import csv
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np             


class BPTTTrainer:
    def __init__(self, cfg, model, device, run_dir, snr_values):
        self.cfg = cfg
        self.model = model
        # self.device = device

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.run_dir = run_dir
        self.snr_values = snr_values

        tcfg = cfg["training"]
        self.epochs = tcfg["epochs"]

        self.criterion = nn.CrossEntropyLoss()

         # patience=None disables early stopping
        self.early_stopping_patience = tcfg.get("early_stopping_patience", None)
        self.early_stopping_min_delta = float(
            tcfg.get("early_stopping_min_delta", 0.0)
        )

        # --- optimizer ---
        opt_name = tcfg.get("optimizer", "adam").lower()
        lr = tcfg.get("lr", 1e-3)
        weight_decay = float(tcfg.get("weight_decay", 0.0))
        # weight_decay = tcfg.get("weight_decay", 0.0)

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,   
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,   
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # --- LR scheduler (optional) ---
        self.scheduler = None
        
        sch_cfg = tcfg.get("lr_scheduler", None)
        if sch_cfg is not None:
            # Allow lr_scheduler: "cosine" OR lr_scheduler: {type: cosine, ...}
            if isinstance(sch_cfg, str):
                sch_type = sch_cfg.lower()
                sch_params = {}
            elif isinstance(sch_cfg, dict):
                sch_type = sch_cfg.get("type", "step").lower()
                sch_params = sch_cfg
            else:
                raise ValueError(f"Unsupported lr_scheduler config: {sch_cfg}")

            if sch_type == "step":
                step_size = sch_params.get("step_size", 10)
                gamma = sch_params.get("gamma", 0.1)
                self.scheduler = lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            elif sch_type == "cosine":
                self.scheduler = lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs
                )
            else:
                raise ValueError(f"Unknown lr_scheduler type: {sch_type}")

        self.grad_clip = tcfg.get("grad_clip", None)

        log_cfg = cfg.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 50)
        self.compute_per_snr = log_cfg.get("compute_per_snr", True)

        # CSV for per-epoch metrics
        self.metrics_csv = os.path.join(run_dir, "metrics_per_epoch.csv")
        with open(self.metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "loss", "acc"])

        # Per-SNR metrics per epoch (for val/test)
        self.metrics_snr_csv = os.path.join(run_dir, "metrics_per_snr_epoch.csv")
        with open(self.metrics_snr_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "snr", "loss", "acc", "correct", "total"])


        # logger will be injected by runner if it has one
        self.logger = None

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _run_epoch(self, loader, epoch, train=True):
        if train:
            self.model.train()
            split = "train"
        else:
            self.model.eval()
            split = "val"

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (spikes, labels, snr) in enumerate(loader):
            spikes = spikes.to(self.device)  # [B, T, C, H, W]
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(spikes)
            loss = self.criterion(logits, labels)

            if train:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += labels.size(0)

            if train and (batch_idx + 1) % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                    f"loss={loss.item():.4f}"
                    f"[epoch {epoch}] lr={current_lr:.6f}"
                )

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # with open(self.metrics_csv, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch, split, avg_loss, avg_acc])

        return avg_loss, avg_acc

    def train(self, train_loader, val_loader):
        best_val_acc = -float("inf")
        best_metrics = None
        best_state_dict = None
        last_val_metrics = None

        no_improve_epochs = 0  # for early stopping

        for epoch in range(1, self.epochs + 1):
            self._log(f"[epoch {epoch}/{self.epochs}] training...")
            train_loss, train_acc = self._run_epoch(train_loader, epoch, train=True)
            self._log(
                f"[epoch {epoch}] train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.4f}"
            )

            # --- write train metrics row
            with open(self.metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, "train", train_loss, train_acc])

            # --- validation + per-SNR logging
            val_metrics = self.evaluate(val_loader, split="val", epoch=epoch)
            last_val_metrics = val_metrics
            self._log(
                f"[epoch {epoch}] val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}"
            )

            # --- write val metrics row
            with open(self.metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, "val", val_metrics["loss"], val_metrics["acc"]]
                )

            current_val_acc = val_metrics["acc"]
            improved = current_val_acc > best_val_acc + self.early_stopping_min_delta

            if improved:
                best_val_acc = current_val_acc
                best_metrics = dict(val_metrics)
                best_metrics["epoch"] = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
                self._log(
                    f"[epoch {epoch}] new best val_acc={best_val_acc:.4f}"
                )
            else:
                no_improve_epochs += 1
                self._log(
                    f"[epoch {epoch}] no improvement in val_acc "
                    f"(streak={no_improve_epochs})"
                )

                # --- early stopping check
                if (
                    self.early_stopping_patience is not None
                    and no_improve_epochs >= self.early_stopping_patience
                ):
                    self._log(
                        f"[epoch {epoch}] early stopping triggered "
                        f"(patience={self.early_stopping_patience})"
                    )
                    break

            # LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        if best_metrics is None:
            best_metrics = dict(last_val_metrics)
            best_metrics["epoch"] = epoch
            best_state_dict = copy.deepcopy(self.model.state_dict())

        return best_metrics, best_state_dict
   

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        split: str = "test",
        epoch: int | None = None,
        return_details: bool = False,
        save_details: bool = False,
        save_tag: str | None = None,
        data_dropout: float | None = None,  # e.g. 0.3 means drop 30% of timesteps
        permute_time: bool = False,         # True = randomize time order
    ):
        """
        Evaluate model on a data loader.

        - Optional 'data_dropout': randomly zero out a fraction of time steps.
        - Optional 'permute_time': randomly permute the time dimension per batch.
        - Still computes per-SNR loss/acc.
        - Still measures latency and (approx) energy per sample if 'hardware.assumed_power_w' is set in cfg.
        - Optional NPZ saving of y_true/y_pred/snr (+ latency/energy) for plotting scripts.
        """

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        per_snr = {}
        all_labels = []
        all_preds = []
        all_snr = []

        forward_time_total = 0.0  # sum of forward() time over all batches

        # Approx hardware power for energy estimate (optional)
        hw_cfg = self.cfg.get("hardware", {})
        assumed_power_w = hw_cfg.get("assumed_power_w", None)  # e.g. 150 for GPU, 30–50 for M1/M2

        for spikes, labels, snr in loader:
            # spikes: [B, T, C, H, W] (for IQ grid)
            spikes = spikes.to(self.device)
            labels = labels.to(self.device)

            B, T = spikes.shape[0], spikes.shape[1]

            # ---------- 1) DATA DROPOUT (simulate bad transmissions) ----------
            if data_dropout is not None and data_dropout > 0.0:
                keep_prob = 1.0 - data_dropout
                # Bernoulli mask over (B, T); drop entire time-steps for each sample
                mask = (torch.rand(B, T, device=spikes.device) < keep_prob)
                # Avoid pathological case where everything is dropped
                if mask.sum() == 0:
                    # force at least one timestep per sample to be kept
                    mask[:, 0] = True

                mask = mask.view(B, T, 1, 1, 1)
                spikes = spikes * mask  # zeroed timesteps ≈ missing packets

            # ---------- 2) RANDOM PERMUTATION OF TIME (out-of-order packets) ----------
            if permute_time:
                # Same permutation for the whole batch: simulates a fixed reordering
                perm = torch.randperm(T, device=spikes.device)
                spikes = spikes[:, perm, ...]  # reorder time dimension

            # ---------- 3) FORWARD + TIMING ----------
            start_t = time.perf_counter()
            logits = self.model(spikes)
            forward_time_total += time.perf_counter() - start_t

            # Per-sample cross-entropy
            loss_per_sample = F.cross_entropy(logits, labels, reduction="none")
            loss = loss_per_sample.mean()

            batch_size = labels.size(0)
            total_loss += float(loss_per_sample.sum().item())
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

            if return_details or save_details:
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_snr.append(snr.clone())  # snr is already on CPU

            if self.compute_per_snr:
                snr_np = snr.numpy()
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                loss_np = loss_per_sample.detach().cpu().numpy()

                for s, p, y, l in zip(snr_np, preds_np, labels_np, loss_np):
                    key = int(s)
                    if key not in per_snr:
                        per_snr[key] = {
                            "correct": 0,
                            "total": 0,
                            "loss_sum": 0.0,
                        }
                    per_snr[key]["total"] += 1
                    per_snr[key]["loss_sum"] += float(l)
                    if p == y:
                        per_snr[key]["correct"] += 1

        # ---------- 4) AGGREGATE METRICS ----------
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        metrics = {"loss": avg_loss, "acc": avg_acc}

        # Latency per sample
        if total_samples > 0:
            avg_lat_s = forward_time_total / total_samples
            metrics["latency_ms_per_sample"] = avg_lat_s * 1e3
            if assumed_power_w is not None:
                metrics["energy_j_per_sample"] = assumed_power_w * avg_lat_s
            else:
                metrics["energy_j_per_sample"] = None
        else:
            metrics["latency_ms_per_sample"] = float("nan")
            metrics["energy_j_per_sample"] = None

        # Per-SNR aggregates
        if self.compute_per_snr and per_snr:
            metrics["per_snr_acc"] = {
                snr_val: v["correct"] / v["total"] for snr_val, v in per_snr.items()
            }
            metrics["per_snr_loss"] = {
                snr_val: v["loss_sum"] / v["total"] for snr_val, v in per_snr.items()
            }

            if epoch is not None:
                # CSV header is: epoch, split, snr, loss, acc, correct, total
                with open(self.metrics_snr_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    for snr_val, v in per_snr.items():
                        acc_snr = v["correct"] / v["total"]
                        loss_snr = v["loss_sum"] / v["total"]
                        writer.writerow(
                            [epoch, split, snr_val, loss_snr, acc_snr, v["correct"], v["total"]]
                        )

        # ---------- 5) OPTIONAL NPZ SAVE + RETURN DETAILS ----------
        if return_details or save_details:
            import torch as _torch

            y_true = _torch.cat(all_labels).numpy()
            y_pred = _torch.cat(all_preds).numpy()
            snr_all = _torch.cat(all_snr).numpy()

            if save_details:
                # Filename: test_predictions_<tag>.npz (if save_tag), else test_predictions.npz
                if save_tag:
                    fname = f"{split}_predictions_{save_tag}.npz"
                else:
                    fname = f"{split}_predictions.npz"

                out_path = os.path.join(self.run_dir, fname)
                npz_dict = {
                    "y_true": y_true.astype(np.int64),
                    "y_pred": y_pred.astype(np.int64),
                    "snr": snr_all,
                    "latency_ms_per_sample": metrics["latency_ms_per_sample"],
                }
                if metrics["energy_j_per_sample"] is not None:
                    npz_dict["energy_j_per_sample"] = metrics["energy_j_per_sample"]

                np.savez(out_path, **npz_dict)
                self._log(f"Saved predictions to {out_path}")

            if return_details:
                # keep API: list[int]
                return metrics, y_true.tolist(), y_pred.tolist()

        return metrics
