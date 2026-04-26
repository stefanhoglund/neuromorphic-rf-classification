# rf_experiments/models/conv2d_ann.py

import torch
import torch.nn as nn


class Conv2DANN(nn.Module):
    """
    2D convolutional ANN for inputs like IQGridEncoder.

    Expected input shape: [B, T, C_in, H, W]

    - Processes the sequence over time (T steps).
    - At each time step, passes a single frame [B, C_in, H, W] through
      Conv2d + ReLU (+ optional pooling + optional dropout).
    - Global average pools spatial dims and accumulates logits over time.

    Designed to be structurally comparable to Conv2DSNN.
    """

    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        kernel_sizes: list[int],
        grid_size: int,
        num_classes: int = 24,
        dropout: float = 0.0,
        pool: bool = True,
        time_agg: str = "sum",
        # accept extra kwargs so we can reuse SNN configs without errors
        **kwargs,
    ):
        super().__init__()

        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have same length"

        self.input_channels = int(input_channels)
        self.conv_channels = [int(c) for c in conv_channels]
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self.grid_size = int(grid_size)
        self.num_classes = int(num_classes)
        self.pool_enabled = bool(pool)
        self.time_agg = time_agg

        self.num_layers = len(self.conv_channels)

        # --- convolution + activation blocks ---
        in_ch = self.input_channels
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        for out_ch, k in zip(self.conv_channels, self.kernel_sizes):
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    padding=k // 2,  # keep spatial size before pooling
                )
            )
            # simple ReLU; could switch to others if desired
            self.activations.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2) if self.pool_enabled else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # compute final spatial size after pooling (same logic as SNN)
        H = W = self.grid_size
        if self.pool_enabled:
            for _ in range(self.num_layers):
                H = (H + 1) // 2
                W = (W + 1) // 2
        self.final_H = H
        self.final_W = W

        self.readout = nn.Linear(self.conv_channels[-1], self.num_classes)

    # ----------------------------------------------------------------------
    # Core time loop
    # ----------------------------------------------------------------------
    def _forward_time_loop(self, x: torch.Tensor, return_steps: bool = False):
        """
        x: [B, T, C_in, H, W]
        return_steps: if True, also return cumulative logits per time step.

        Returns:
          logits_sum: [B, num_classes]
          logits_steps (optional): [T, B, num_classes]
        """
        assert x.dim() == 5, f"Expected [B, T, C, H, W], got {x.shape}"
        B, T, C_in, H, W = x.shape

        logits_sum = torch.zeros(B, self.num_classes, device=x.device)
        logits_steps = [] if return_steps else None

        for t in range(T):
            h = x[:, t]  # [B, C_in, H, W]

            for conv, act in zip(self.convs, self.activations):
                h = conv(h)
                h = act(h)
                if self.pool is not None:
                    h = self.pool(h)
                if self.dropout is not None:
                    h = self.dropout(h)

            # global average pool over spatial dims
            feat = h.mean(dim=(2, 3))  # [B, C_last]
            logits_t = self.readout(feat)
            logits_sum = logits_sum + logits_t

            if return_steps:
                # Keep the same semantics as Conv2DSNN: cumulative logits
                logits_steps.append(logits_sum.clone())

        if return_steps:
            logits_steps = torch.stack(logits_steps, dim=0)  # [T, B, num_classes]

        return logits_sum, logits_steps

    # ----------------------------------------------------------------------
    # Public forwards
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training / standard forward.

        x: [B, T, C_in, H, W]
        Returns: [B, num_classes] (time-summed or time-mean logits)
        """
        logits_sum, _ = self._forward_time_loop(x, return_steps=False)

        if self.time_agg == "mean":
            T = x.shape[1]
            logits_sum = logits_sum / T  # average over time instead of pure sum

        return logits_sum

    def forward_steps(self, x: torch.Tensor):
        """
        Forward with step-wise logits (for DCLL / latency analysis).

        x: [B, T, C_in, H, W]
        Returns:
          logits_sum: [B, num_classes]
          logits_steps: [T, B, num_classes] (cumulative over time)
        """
        return self._forward_time_loop(x, return_steps=True)
