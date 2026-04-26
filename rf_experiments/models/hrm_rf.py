import torch
import torch.nn as nn
import torch.nn.functional as F


class HRMRFClassifier(nn.Module):
    """
    HRM-style RF classifier:

    - Input: spikes or features of shape [B, T, C, H, W] or [B, T, C].
    - Low-level GRU runs over time conditioned on a high-level state.
    - High-level GRU runs over aggregated low-level states for a few cycles.
    - Classifier head on final low-level states (and optionally high-level state).

    This is a simplified HRM, not a full reproduction of Wang et al.'s code,
    but it captures the hierarchical fast/slow recurrent idea.
    """

    def __init__(
        self,
        input_channels: int = 1,
        grid_size: int | None = None,
        input_dim: int | None = None,
        low_dim: int = 128,
        high_dim: int = 128,
        num_cycles: int = 4,
        num_classes: int = 24,
        dropout: float = 0.1,
        agg: str = "mean",  # how to aggregate time for high-level: mean | last
    ):
        super().__init__()

        # Determine feature dimension
        if input_dim is None:
            if grid_size is None:
                raise ValueError(
                    "Either input_dim or grid_size must be provided to HRMRFClassifier."
                )
            # C*H*W
            self.input_dim = input_channels * grid_size * grid_size
        else:
            self.input_dim = input_dim

        self.low_dim = low_dim
        self.high_dim = high_dim
        self.num_cycles = num_cycles
        self.num_classes = num_classes
        self.agg = agg

        # Low-level GRU: fast recurrent module
        # Input: [batch, T, input_dim + high_dim] (conditioned on high-level state)
        self.low_gru = nn.GRU(
            input_size=self.input_dim + high_dim,
            hidden_size=low_dim,
            batch_first=True,
        )

        # High-level GRU: slow recurrent module over cycles
        # Input: aggregated low-level state per cycle
        self.high_gru = nn.GRU(
            input_size=low_dim,
            hidden_size=high_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier: can use both low- and high-level states
        self.classifier = nn.Linear(low_dim + high_dim, num_classes)

    # ------------------------------------------------------------------ #
    # Utility: reshape input
    # ------------------------------------------------------------------ #
    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] or [B, T, C]
        returns: [B, T, D]
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.view(B, T, C * H * W)
        elif x.dim() == 3:
            return x
        else:
            raise ValueError(f"Unexpected input shape for HRMRFClassifier: {x.shape}")

    # ------------------------------------------------------------------ #
    # Core HRM-style reasoning loop
    # ------------------------------------------------------------------ #
    def forward_steps(self, x: torch.Tensor):
        """
        HRM-style recurrent reasoning.

        x: [B, T, C, H, W] or [B, T, C]

        Returns:
          logits_final: [B, num_classes]
          logits_steps: [T, B, num_classes]  # time-major, for DCLL/local trainers
        """
        if not x.is_contiguous():
            x = x.contiguous()

        B, T = x.shape[0], x.shape[1]
        x_flat = self._flatten_input(x)  # [B, T, D]

        device = x.device

        # Initialize high-level hidden state h_H: [1, B, H]
        h_H = torch.zeros(1, B, self.high_dim, device=device)

        # We'll accumulate per-time-step logits across cycles and average
        logits_steps_acc = []

        for cycle in range(self.num_cycles):
            # Repeat high-level state H over all time steps
            # h_H: [1, B, H] -> [B, T, H]
            h_H_exp = h_H.expand(1, B, self.high_dim).transpose(0, 1)  # [B, 1, H]
            h_H_exp = h_H_exp.expand(B, T, self.high_dim)              # [B, T, H]

            # Low-level input: concat features and high-level context per time step
            low_input = torch.cat([x_flat, h_H_exp], dim=-1)  # [B, T, D+H]

            # Run low-level GRU over time
            low_out, h_L = self.low_gru(low_input)  # low_out: [B,T,low_dim]; h_L: [1,B,low_dim]

            # Aggregate low-level over time for high-level update
            if self.agg == "mean":
                low_agg = low_out.mean(dim=1, keepdim=True)  # [B,1,low_dim]
            elif self.agg == "last":
                low_agg = low_out[:, -1:, :]                 # [B,1,low_dim]
            else:
                raise ValueError(f"Unknown agg type {self.agg}")

            # High-level GRU over cycles (sequence length = 1 per cycle)
            # Input to high GRU must be [B, L, low_dim], L=1 here
            high_out, h_H = self.high_gru(low_agg, h_H)  # h_H updated in place

            # Classification head: combine last low_out and high state
            # Use final time-step low_out for decision
            low_last = low_out[:, -1, :]             # [B, low_dim]
            high_last = h_H[-1]                     # [B, high_dim]

            joint = torch.cat([low_last, high_last], dim=-1)
            joint = self.dropout(joint)
            logits_cycle = self.classifier(joint)   # [B, num_classes]

            # For training with DCLL/local rules, we want a per-time-step sequence
            # We simply broadcast cycle's logits to all time steps and accumulate
            logits_steps_cycle = logits_cycle.unsqueeze(0).expand(T, B, self.num_classes)
            logits_steps_acc.append(logits_steps_cycle)

        # Stack over cycles: [num_cycles, T, B, C]
        logits_steps_all = torch.stack(logits_steps_acc, dim=0)
        # Average across cycles: [T, B, C]
        logits_steps = logits_steps_all.mean(dim=0)

        # Final decision: last time step (after averaging cycles)
        logits_final = logits_steps[-1]  # [B, C]

        return logits_final, logits_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_final, _ = self.forward_steps(x)
        return logits_final
