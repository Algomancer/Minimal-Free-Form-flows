"""
Free-Form Flows with Finite Differences
=========================================

Variant of FFF that replaces all forward/dual-mode AD with finite differences.
The Hutchinson surrogate v^T J_enc(x) . sg(J_dec(z) v) requires JVPs and VJPs.
Here we estimate both via finite differences:

  J_f(x) v  ≈  [f(x + εv) - f(x - εv)] / (2ε)

This removes the need for torch.autograd.forward_ad and torch.autograd.grad
in the training loss entirely. Parameter gradients still flow through the
reconstruction and log p_Z terms via normal backprop; the volume-change
surrogate is estimated without any explicit Jacobian computation.

Reference: Draxler et al., "Free-form flows: Make Any Architecture a Normalizing Flow"
"""

from math import sqrt, log, pi
from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
if DEVICE.type == "cuda":
    torch.cuda.init()

LOG_2PI = log(2 * pi)

# =============================================================================
# Data Generation
# =============================================================================

def sample_8gaussians(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D mixture of 8 Gaussians arranged in a circle."""
    scale = 4.0
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=torch.float32, device=device) * scale

    x = 0.5 * torch.randn(n, 2, device=device)
    center_ids = torch.randint(0, 8, (n,), device=device)
    x = (x + centers[center_ids]) / np.sqrt(2)
    return x

# =============================================================================
# Neural Network
# =============================================================================

class Net(nn.Module):
    """MLP with skip connection and SiLU activations."""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)

# =============================================================================
# Primitives
# =============================================================================

def logp_z(z: Tensor) -> Tensor:
    """Standard normal log-probability, summed over dims."""
    d = z.shape[-1]
    return -0.5 * (z ** 2).sum(dim=-1) - 0.5 * d * LOG_2PI


def fd_jvp(f: nn.Module, x: Tensor, v: Tensor, eps: float = 1e-4) -> Tensor:
    """Finite-difference JVP: (f(x + εv) - f(x - εv)) / (2ε).

    Differentiable w.r.t. f's parameters (both forward passes go through f).
    """
    return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


def fd_jvp_nograd(f: nn.Module, x: Tensor, v: Tensor, eps: float = 1e-4) -> Tensor:
    """Finite-difference JVP without gradient tracking (for stop-gradient terms)."""
    with torch.no_grad():
        return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


def full_jacobian_fd(f: nn.Module, x: Tensor, eps: float = 1e-4) -> Tensor:
    """Full Jacobian via finite differences. O(d) forward passes.
    Used only at test time for exact NLL computation."""
    dim = x.shape[-1]
    jac = torch.zeros(x.shape[0], dim, dim, device=x.device)
    with torch.no_grad():
        for i in range(dim):
            ei = torch.zeros_like(x)
            ei[:, i] = 1.0
            jac[:, :, i] = (f(x + eps * ei) - f(x - eps * ei)) / (2 * eps)
    return jac

# =============================================================================
# Probe Vectors
# =============================================================================

def probe_randn(z: Tensor) -> Tensor:
    """Random unit probe vector scaled by sqrt(dim)."""
    v = torch.randn_like(z)
    return v / v.norm(dim=-1, keepdim=True) * sqrt(z.shape[-1])


def probe_qr(z: Tensor) -> Tensor:
    """QR-orthonormalized probe vector scaled by sqrt(dim) (lower variance)."""
    dim = z.shape[-1]
    v = torch.randn(z.shape[0], dim, 1, device=z.device, dtype=z.dtype)
    q = torch.linalg.qr(v).Q.squeeze(-1)
    return q * sqrt(dim)

# =============================================================================
# Volume Change Surrogate (Finite Difference)
# =============================================================================
#
# Original surrogate: S = v^T J_enc(x) . sg(J_dec(z) v)
#
# With finite differences:
#   J_dec(z) v ≈ fd_jvp_nograd(dec, z, v)    [detached, no grad needed]
#   J_enc(x) w ≈ fd_jvp(enc, x, w)           [need param grads]
#
# where w = sg(J_dec(z) v). The encoder FD is differentiable w.r.t. encoder
# params because enc(x ± εw) are standard forward passes.
#
# Surrogate: S = v^T . fd_jvp(enc, x, w)
#

def hutchinson_surrogate_fd(
    x: Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    sample_v: Callable[[Tensor], Tensor] = probe_randn,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Finite-difference Hutchinson surrogate for log |det J_dec|.
    Returns (surrogate [batch], z [batch, dim], x_rec [batch, dim])."""
    z = encoder(x)
    x_rec = decoder(z)

    v = sample_v(z)

    # Decoder JVP via FD (stop-gradient — no param grads needed)
    jvp_dec = fd_jvp_nograd(decoder, z.detach(), v, eps=eps)

    # Encoder JVP via FD in direction w = sg(jvp_dec)
    # Differentiable w.r.t. encoder params
    w = jvp_dec.detach()
    jvp_enc = fd_jvp(encoder, x.detach(), w, eps=eps)

    # S = v^T . J_enc(x) . sg(J_dec(z) v) ≈ v^T . jvp_enc
    surrogate = (v * jvp_enc).sum(dim=-1)

    return surrogate, z, x_rec

# =============================================================================
# Exact NLL (Finite Difference)
# =============================================================================

def compute_nll(x: Tensor, encoder: nn.Module, decoder: nn.Module, eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Exact NLL via full decoder Jacobian (finite differences). Returns (nll, z, x_rec)."""
    with torch.no_grad():
        z = encoder(x)
        x_rec = decoder(z)
    jac = full_jacobian_fd(decoder, z, eps=eps)
    log_abs_det = torch.linalg.slogdet(jac).logabsdet
    nll = -logp_z(z) + log_abs_det
    return nll, z, x_rec

# =============================================================================
# Loss
# =============================================================================

def surrogate_elbo(
    x: Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    beta: float,
    sample_v: Callable[[Tensor], Tensor] = probe_randn,
    eps: float = 1e-4,
) -> Tensor:
    """Per-sample loss [batch]."""
    surrogate, z, x_rec = hutchinson_surrogate_fd(x, encoder, decoder, sample_v, eps=eps)
    recon = ((x - x_rec) ** 2).sum(dim=-1)
    nll = -logp_z(z) - surrogate
    return beta * recon + nll

# =============================================================================
# Free-Form Flow Model (Finite Difference)
# =============================================================================

class FreeFormFlow(nn.Module):
    """Free-form flow using finite-difference volume change estimation."""

    def __init__(self, dim: int = 2, hidden_dim: int = 64, beta: float = 150.0, eps: float = 1e-4):
        super().__init__()
        self.encoder = Net(dim, hidden_dim)
        self.decoder = Net(dim, hidden_dim)
        self.beta = beta
        self.dim = dim
        self.eps = eps

    @torch.compile(fullgraph=True)
    def forward(self, x: Tensor) -> Tensor:
        return surrogate_elbo(x, self.encoder, self.decoder, self.beta, eps=self.eps)

    @torch.no_grad()
    def generate(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.decoder(z)

# =============================================================================
# Visualization
# =============================================================================

def viz_2d_data(data: Tensor, filename: str = "data.jpg"):
    plt.figure()
    data = data.cpu()
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.axis("scaled")
    plt.savefig(filename, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_real_vs_gen(real: Tensor, generated: Tensor, step: int, filename: str = "samples.jpg"):
    r = real.detach().cpu().numpy()
    g = generated.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.scatter(r[:, 0], r[:, 1], s=2, alpha=0.3, c="black")
    ax1.set_title("Target (p)")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax2.scatter(g[:, 0], g[:, 1], s=2, alpha=0.3, c="tab:orange")
    ax2.set_title(f"Generated (step {step})")
    ax2.set_aspect("equal")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Training
# =============================================================================

def train(
    model: FreeFormFlow,
    data_fn: Callable[[int], Tensor],
    n_iter: int = 100_000,
    batch_size: int = 512,
    lr: float = 4e-4,
    grad_clip: float = 1.0,
    sample_every: int = 5000,
    n_samples: int = 4096,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)
    pbar = trange(n_iter)

    test_batch = data_fn(1024)

    for i in pbar:
        x = data_fn(batch_size)
        loss = model(x).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if (i + 1) % 100 == 0:
            pbar.set_description(f"loss: {loss.item():.4f}")

        if (i + 1) % sample_every == 0:
            model.eval()
            viz_real_vs_gen(data_fn(n_samples), model.generate(n_samples), i + 1)

            nll, z, x_rec = compute_nll(test_batch, model.encoder, model.decoder, eps=model.eps)
            recon = ((test_batch - x_rec) ** 2).sum(dim=-1).mean()
            pbar.write(f"  step {i+1}: recon={recon:.1e}, NLL={nll.mean():.2f}")
            model.train()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    viz_2d_data(sample_8gaussians(4096))

    model = FreeFormFlow(dim=2, beta=150.0, eps=1e-4).to(DEVICE)
    train(model, sample_8gaussians, n_iter=100_000, batch_size=512)

    model.eval()
    viz_2d_data(model.generate(4096), filename="final_samples.jpg")

    test = sample_8gaussians(2048)
    nll, z, x_rec = compute_nll(test, model.encoder, model.decoder, eps=model.eps)
    print(f"Final exact NLL: {nll.mean():.3f}")
    print(f"Final recon MSE: {((test - x_rec)**2).sum(-1).mean():.3e}")
    print("Done.")
