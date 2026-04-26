import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class Conv2DSNN(nn.Module):
    """
    2D convolutional SNN for inputs like IQGridEncoder.

    Expected input shape: [B, T, C_in, H, W]

    - Processes the sequence over time (T steps).
    - At each time step, passes a single frame [B, C_in, H, W] through
      Conv2d + LIF (+ optional pooling + optional dropout).
    - Global average pools spatial dims and accumulates logits over time.
    """

    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        kernel_sizes: list[int],
        grid_size: int,
        lif_beta: float = 0.9,
        num_classes: int = 24,
        dropout: float = 0.0,
        pool: bool = True,
        time_agg: str = "sum",
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

        # --- convolution + LIF blocks ---
        in_ch = self.input_channels
        self.convs = nn.ModuleList()
        self.lifs = nn.ModuleList()

        for out_ch, k in zip(self.conv_channels, self.kernel_sizes):
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    padding=k // 2,  # keep spatial size before pooling
                )
            )
            self.lifs.append(
                snn.Leaky(
                    beta=lif_beta,
                    spike_grad=surrogate.fast_sigmoid(),
                )
            )
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2) if self.pool_enabled else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # compute final spatial size after pooling
        H = W = self.grid_size
        if self.pool_enabled:
            for _ in range(self.num_layers):
                H = (H + 1) // 2
                W = (W + 1) // 2
        self.final_H = H
        self.final_W = W

        self.readout = nn.Linear(self.conv_channels[-1], self.num_classes)

        # ---- spike tracking state ----
        self.track_spikes = False
        self.reset_spike_stats()

    # ----------------------------------------------------------------------
    # Spike tracking helpers
    # ----------------------------------------------------------------------
    def reset_spike_stats(self):
        # total spikes per LIF layer, across all batches & time
        self.spike_counts = [0.0 for _ in range(self.num_layers)]
        # number of neurons per layer per example (C * H * W)
        self.num_neurons = [0 for _ in range(self.num_layers)]
        # total (time * batch) samples processed
        self.total_time_samples = 0

    def enable_spike_tracking(self, enabled: bool = True):
        self.track_spikes = bool(enabled)
        if enabled:
            self.reset_spike_stats()

    def _register_time_samples(self, T: int, B: int):
        if self.track_spikes:
            self.total_time_samples += T * B

    def _accumulate_spikes(self, layer_idx: int, spk: torch.Tensor):
        if not self.track_spikes:
            return
        # spk: [B, C, H, W]
        s = float(spk.detach().sum().item())
        self.spike_counts[layer_idx] += s
        if self.num_neurons[layer_idx] == 0:
            self.num_neurons[layer_idx] = spk[0].numel()

    def get_spike_stats(self):
        total_spikes = sum(self.spike_counts)
        total_neurons = sum(self.num_neurons)
        if self.total_time_samples > 0 and total_neurons > 0:
            global_rate = total_spikes / (self.total_time_samples * total_neurons)
        else:
            global_rate = None

        layers = []
        for i in range(self.num_layers):
            if self.total_time_samples > 0 and self.num_neurons[i] > 0:
                rate = self.spike_counts[i] / (
                    self.total_time_samples * self.num_neurons[i]
                )
            else:
                rate = None
            layers.append(
                {
                    "layer_index": i,
                    "total_spikes": self.spike_counts[i],
                    "num_neurons": self.num_neurons[i],
                    "avg_rate": rate,
                }
            )

        return {
            "total_spikes": total_spikes,
            "total_time_samples": self.total_time_samples,
            "global_rate": global_rate,
            "layers": layers,
        }

    # ----------------------------------------------------------------------
    # Forward passes
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

        # register time*batch for spike-rate normalization
        self._register_time_samples(T, B)

        logits_sum = torch.zeros(B, self.num_classes, device=x.device)
        logits_steps = [] if return_steps else None

        # mem states per LIF layer, initialized lazily on first time step
        mem_states = [None for _ in range(self.num_layers)]

        for t in range(T):
            h = x[:, t]  # [B, C_in, H, W]

            for layer_idx, (conv, lif) in enumerate(zip(self.convs, self.lifs)):
                cur = conv(h)  # [B, C_out, H', W']

                if mem_states[layer_idx] is None:
                    # Initialize membrane to zeros with same shape as cur
                    mem_states[layer_idx] = torch.zeros_like(cur)

                spk, mem = lif(cur, mem_states[layer_idx])
                mem_states[layer_idx] = mem



                # spike tracking
                self._accumulate_spikes(layer_idx, spk)

                h = spk
                if self.pool is not None:
                    h = self.pool(h)
                if self.dropout is not None:
                    h = self.dropout(h)

            # global average pool over spatial dims
            feat = h.mean(dim=(2, 3))  # [B, C_last]
            logits_t = self.readout(feat)
            logits_sum = logits_sum + logits_t

            if return_steps:
                logits_steps.append(logits_sum.clone())

        if return_steps:
            logits_steps = torch.stack(logits_steps, dim=0)  # [T, B, num_classes]

        return logits_sum, logits_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward.

        x: [B, T, C_in, H, W]
        Returns: [B, num_classes] time-summed logits
        """
        logits_sum, _ = self._forward_time_loop(x, return_steps=False)

        if self.time_agg == "mean":
            T = x.shape[1]
            logits_sum = logits_sum / T  # average over time instead of pure sum

        return logits_sum

    def forward_steps(self, x: torch.Tensor):
        """
        Evaluation forward with step-wise logits (for decision latency).

        x: [B, T, C_in, H, W]
        Returns:
          logits_sum: [B, num_classes]
          logits_steps: [T, B, num_classes] (cumulative over time)
        """
        return self._forward_time_loop(x, return_steps=True)
