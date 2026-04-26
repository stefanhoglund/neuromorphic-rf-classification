# rf_experiments/trainers/localrule_trainer.py

from typing import Dict, Any, Tuple

import torch

from .bptt_trainer import BPTTTrainer


class LocalRuleTrainer(BPTTTrainer):
    """
    Trainer implementing a simple local-in-time training rule.

    Differences vs BPTTTrainer:
      - Calls model.forward_steps(x) -> (logits_final, logits_steps)
      - Uses a loss over *all time steps* instead of just final step:
          L = mean_t CE(logits_t, y)
      - Everything else (optimizer, logging, early stopping) is reused.
    """

    def __init__(self, cfg: Dict[str, Any], model, device, run_dir: str, snr_values):
        # Same call signature/order as BPTTTrainer and DCLLTrainer
        super().__init__(cfg, model, device, run_dir, snr_values)
        self._log("Initialized LocalRuleTrainer (time-local loss over forward_steps).")

        # Fail fast if the model doesn't support forward_steps
        if not hasattr(self.model, "forward_steps"):
            raise ValueError(
                "LocalRuleTrainer expects model.forward_steps(x) "
                "-> (logits_final, logits_steps)"
            )

    # ------------------------------------------------------------------ #
    # Epoch loop (unchanged)
    # ------------------------------------------------------------------ #
    def _run_epoch(
        self,
        loader,
        epoch: int,
        train: bool = True,
    ) -> Tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        device = self.device
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (spikes, labels, snr) in enumerate(loader):
            spikes = spikes.to(device)
            labels = labels.to(device)

            if train:
                self.optimizer.zero_grad()

            # Expect model.forward_steps to return:
            #   logits_final: [B, C]
            #   logits_steps: [T, B, C]
            logits_final, logits_steps = self.model.forward_steps(spikes)

            T, B, C = logits_steps.shape

            # Local-in-time loss over all time steps
            logits_flat = logits_steps.view(T * B, C)
            labels_flat = labels.unsqueeze(0).expand(T, B).reshape(-1)
            loss = self.criterion(logits_flat, labels_flat)

            if train:
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            # Accuracy from final logits
            with torch.no_grad():
                preds = torch.argmax(logits_final, dim=-1)
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
