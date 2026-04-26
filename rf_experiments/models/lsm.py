# rf_experiments/models/lsm.py

import torch
import torch.nn as nn


class LSMSNN(nn.Module):
    """
    Simple Liquid State Machine / Echo State Network style model.

    - Random, sparse recurrent reservoir (optionally trainable).
    - Input projection from flattened input at each time step into reservoir.
    - Nonlinear leaky update of reservoir state over time.
    - Linear readout from aggregated reservoir states to class logits.

    Shapes:
      x: [B, T, C, H, W]  or  [B, T, C]
      internal reservoir states: [T, B, R]
      logits: [B, num_classes]
      logits_steps: [T, B, num_classes]
    """

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int = 24,
        reservoir_size: int = 512,
        leak: float = 0.9,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        sparsity: float = 0.1,
        train_reservoir: bool = False,
        readout_agg: str = "mean",  # "mean" | "last" | "sum"
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.reservoir_size = reservoir_size
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.train_reservoir = train_reservoir
        self.readout_agg = readout_agg

        # Linear readout from reservoir -> class logits
        self.readout = nn.Linear(reservoir_size, num_classes)

        # If input_dim is given at construction time, initialize on CPU;
        # model.to(device) will move parameters/buffers later.
        if self.input_dim is not None:
            self._init_weights(self.input_dim, device=None)

    # ------------------------------------------------------------------ #
    # Weight initialization
    # ------------------------------------------------------------------ #

    def _init_weights(self, input_dim: int, device: torch.device | None):
        """
        Initialize random input and reservoir weights.

        If device is None, weights are created on CPU and will be moved by
        model.to(device) later. If device is given, create directly on that device.
        """
        # If W_in/W_res already exist (Parameter or buffer), don't re-init.
        if hasattr(self, "W_in") and hasattr(self, "W_res"):
            return

        self.input_dim = input_dim
        dev = device if device is not None else torch.device("cpu")

        R = self.reservoir_size

        # Input projection: [input_dim, R]
        W_in = torch.randn(input_dim, R, device=dev) * self.input_scaling

        # Recurrent reservoir: [R, R], sparse
        W_res = torch.randn(R, R, device=dev)
        mask = (torch.rand(R, R, device=dev) < self.sparsity).float()
        W_res = W_res * mask

        # Scale to desired spectral radius (approximate via power iteration)
        if self.spectral_radius is not None and self.spectral_radius > 0:
            with torch.no_grad():
                v = torch.randn(R, 1, device=dev)
                for _ in range(20):
                    v = W_res @ v
                    v = v / (v.norm() + 1e-6)
                Av = W_res @ v
                eig_approx = (v.t() @ Av).item()
                if abs(eig_approx) > 1e-6:
                    scale = self.spectral_radius / abs(eig_approx)
                    W_res = W_res * scale

        if self.train_reservoir:
            # learnable reservoir
            self.W_in = nn.Parameter(W_in)
            self.W_res = nn.Parameter(W_res)
        else:
            # fixed reservoir (buffers)
            self.register_buffer("W_in", W_in)
            self.register_buffer("W_res", W_res)

    def _maybe_init_weights(self, x: torch.Tensor):
        """
        Lazily initialize weights if input_dim was not known at __init__ time.
        """
        B, T = x.shape[0], x.shape[1]
        x_flat = x.view(B, T, -1)
        input_dim_dyn = x_flat.shape[-1]

        if self.input_dim is None:
            # create weights on the same device as x
            self._init_weights(input_dim_dyn, device=x.device)
        elif self.input_dim != input_dim_dyn:
            raise ValueError(
                f"LSMSNN expected input_dim={self.input_dim}, "
                f"but got {input_dim_dyn} from input shape {x.shape}"
            )

    # ------------------------------------------------------------------ #
    # Reservoir dynamics
    # ------------------------------------------------------------------ #

    def _run_reservoir(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the reservoir over time.

        x: [B, T, ...]  (C/H/W or C)
        Returns:
          states: [T, B, R] reservoir activations at each time step.
        """
        # Flatten all non-(B,T) dims into feature dim
        B, T = x.shape[0], x.shape[1]
        x_flat = x.view(B, T, -1)  # [B, T, D]

        self._maybe_init_weights(x)

        R = self.reservoir_size
        device = x.device

        # Reservoir state: [B, R]
        r = torch.zeros(B, R, device=device)
        states = []

        for t in range(T):
            u_t = x_flat[:, t, :]  # [B, D]

            # input + recurrent contribution
            pre = u_t @ self.W_in + r @ self.W_res  # [B, R]

            # leaky tanh update (rate-based reservoir)
            r = (1.0 - self.leak) * r + self.leak * torch.tanh(pre)

            states.append(r.unsqueeze(0))  # [1, B, R]

        # [T, B, R]
        states = torch.cat(states, dim=0)
        return states

    # ------------------------------------------------------------------ #
    # Forward APIs
    # ------------------------------------------------------------------ #

    def forward_steps(self, x: torch.Tensor):
        """
        Forward pass that returns:
          logits:       [B, num_classes] (aggregated over time)
          logits_steps: [T, B, num_classes] (per time step)

        x: [B, T, C] or [B, T, C, H, W]
        """
        if not x.is_contiguous():
            x = x.contiguous()

        states = self._run_reservoir(x)  # [T, B, R]
        T, B, R = states.shape

        # Per-time-step logits
        logits_steps = self.readout(states.view(T * B, R)).view(
            T, B, self.num_classes
        )

        # Aggregate over time for final logits
        if self.readout_agg == "last":
            logits = logits_steps[-1]  # [B, num_classes]
        elif self.readout_agg == "sum":
            logits = logits_steps.sum(dim=0)
        elif self.readout_agg == "mean":
            logits = logits_steps.mean(dim=0)
        else:
            raise ValueError(f"Unknown readout_agg: {self.readout_agg}")

        return logits, logits_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard nn.Module forward: returns only aggregated logits.

        x: [B, T, C] or [B, T, C, H, W]
        """
        logits, _ = self.forward_steps(x)
        return logits
