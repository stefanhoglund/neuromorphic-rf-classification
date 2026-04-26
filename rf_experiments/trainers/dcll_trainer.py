# rf_experiments/trainers/dcll_trainer.py

from typing import Tuple

import torch

from .bptt_trainer import BPTTTrainer


class DCLLTrainer(BPTTTrainer):
    """
    DCLL-style trainer (Deep Continuous Local Learning, simplified).

    Differences vs BPTTTrainer:
      - Expects the model to provide `forward_steps(x) -> (logits_final, logits_steps)`,
        where:
          logits_final: [B, C]
          logits_steps: [T, B, C]
      - Uses a loss over ALL time steps (local in time):
          L = mean_t CE(logits_t, y)
        instead of CE on only the final logits.

    Everything else (optimizer, logging, early stopping, metrics.json) is reused
    from BPTTTrainer.
    """

    def __init__(self, cfg, model, device, run_dir, snr_values):
        # same signature as BPTTTrainer so the runner factory still works
        super().__init__(cfg, model, device, run_dir, snr_values)
        self._log("Initialized DCLLTrainer (time-local loss over forward_steps).")

        if not hasattr(self.model, "forward_steps"):
            raise ValueError(
                "DCLLTrainer expects model.forward_steps(x) -> (logits_final, logits_steps)."
            )

    # ------------------------------------------------------------------ #
    # Override only the epoch loop; keep train() from BPTTTrainer
    # ------------------------------------------------------------------ #

    def _run_epoch(
        self,
        loader,
        epoch: int,
        train: bool = True,
    ) -> Tuple[float, float]:
        """
        Run one epoch using time-local loss.

        Returns:
          (epoch_loss, epoch_accuracy)

        Loss:
          - computed over ALL time steps:
              logits_steps: [T, B, C]
              L = CE(logits_steps[t], y) averaged over t and batch.

        Accuracy:
          - computed from `logits_final` (aggregated logits) as in standard training.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (spikes, labels, snr) in enumerate(loader):
            spikes = spikes.to(self.device)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            # model-specific forward: must return (logits_final, logits_steps)
            logits_final, logits_steps = self.model.forward_steps(spikes)
            # logits_final: [B, C]
            # logits_steps: [T, B, C]

            T, B, C = logits_steps.shape

            # --- Local-in-time loss -----------------------------------------
            # Flatten time+batch: [(T*B), C]
            logits_flat = logits_steps.view(T * B, C)
            # Repeat labels along time: [T, B] -> flatten -> [(T*B)]
            labels_flat = labels.unsqueeze(0).expand(T, B).reshape(-1)

            loss = self.criterion(logits_flat, labels_flat)
            # ---------------------------------------------------------------

            if train:
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            # Accuracy from final logits (global decision)
            with torch.no_grad():
                preds = torch.argmax(logits_final, dim=-1)  # [B]
                correct = (preds == labels).sum().item()
                batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            if train and (batch_idx + 1) % self.log_interval == 0:
                self._log(
                    f"[epoch {epoch}/{self.epochs}] "
                    f"batch {batch_idx+1}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = total_correct / max(1, total_samples)

        mode = "train" if train else "val"
        self._log(
            f"[epoch {epoch}] {mode}_loss={epoch_loss:.4f}, "
            f"{mode}_acc={epoch_acc:.4f}"
        )

        return epoch_loss, epoch_acc
