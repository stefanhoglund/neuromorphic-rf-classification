# rf_experiments/models/conv1d_snn.py

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class Conv1DSNN(nn.Module):
    """
    1D convolutional SNN.

    Accepts:
      - [B, T, C_in]          (sequence of feature vectors)
      - [B, T, C_in, H, W]    (e.g. IQGrid -> pooled over H,W to [B, T, C_in])

    Processes time by stepping over T and updating LIF membranes.
    """

    def __init__(
        self,
        input_channels: int,
        conv_channels=None,
        hidden_channels=None,     # alias for conv_channels for older configs
        kernel_sizes=None,
        lif_beta: float = 0.9,
        num_classes: int = 24,
        dropout: float = 0.0,
        time_agg: str = "sum",
        **kwargs,
    ):
        super().__init__()

        # allow both conv_channels and hidden_channels
        if conv_channels is None and hidden_channels is None:
            raise ValueError("Conv1DSNN needs conv_channels or hidden_channels")
        if conv_channels is None:
            conv_channels = hidden_channels

        if kernel_sizes is None:
            raise ValueError("Conv1DSNN needs kernel_sizes")

        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have same length"

        self.input_channels = int(input_channels)
        self.conv_channels = [int(c) for c in conv_channels]
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self.num_layers = len(self.conv_channels)
        self.num_classes = int(num_classes)
        self.time_agg = time_agg

        # conv + LIF blocks
        self.convs = nn.ModuleList()
        self.lifs = nn.ModuleList()

        in_ch = self.input_channels
        for out_ch, k in zip(self.conv_channels, self.kernel_sizes):
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    padding=k // 2,  # keep length
                )
            )
            self.lifs.append(
                snn.Leaky(
                    beta=lif_beta,
                    spike_grad=surrogate.fast_sigmoid(),
                )
            )
            in_ch = out_ch

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.readout = nn.Linear(in_ch, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C_in] or [B, T, C_in, H, W]
        Returns: [B, num_classes] (time-summed or time-averaged logits)
        """
        # Handle IQ-grid style input [B, T, C, H, W] by pooling over H,W
        if x.dim() == 5:
            # [B, T, C, H, W] -> [B, T, C]
            x = x.mean(dim=(3, 4))

        if x.dim() != 3:
            raise ValueError(
                f"Conv1DSNN expected [B, T, C_in] or [B, T, C_in, H, W], got {x.shape}"
            )

        B, T, C_in = x.shape
        if C_in != self.input_channels:
            raise ValueError(
                f"Conv1DSNN: input_channels={self.input_channels}, "
                f"but got C_in={C_in} from data"
            )

        # [B, T, C] -> [B, C, T] for Conv1d
        h_seq = x.permute(0, 2, 1)  # [B, C_in, T]

        logits_sum = torch.zeros(B, self.num_classes, device=x.device)

        # one membrane state per LIF layer
        mem_states = [None for _ in range(self.num_layers)]

        for t in range(T):
            h = h_seq[:, :, t : t + 1]  # [B, C_in, 1]

            for layer_idx, (conv, lif) in enumerate(zip(self.convs, self.lifs)):
                cur = conv(h)  # [B, C_out, 1]

                if mem_states[layer_idx] is None:
                    mem_states[layer_idx] = torch.zeros_like(cur)

                spk, mem = lif(cur, mem_states[layer_idx])
                mem_states[layer_idx] = mem

                h = spk
                if self.dropout is not None:
                    h = self.dropout(h)

            # h: [B, C_last, 1] -> [B, C_last]
            feat = h.mean(dim=2)
            logits_t = self.readout(feat)
            logits_sum = logits_sum + logits_t

        if self.time_agg == "mean":
            logits_sum = logits_sum / T

        return logits_sum
