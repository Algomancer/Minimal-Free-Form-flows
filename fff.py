"""
Free-Form Flows from Scratch
=============================

A minimal implementation of Free-Form Flows (FFF) for 2D toy data.
FFF learns an encoder-decoder pair where latent codes z = enc(x) follow a
simple prior p_Z = N(0,I). The encoder and decoder are unconstrained --
no invertibility required. The volume change term in the change-of-variables
formula is estimated via a Hutchinson surrogate using mixed forward/backward AD.

Sampling is 1-NFE: z ~ N(0,I), x = dec(z).

Reference: Draxler et al., "Free-form flows: Make Any Architecture a Normalizing Flow"
"""

from math import sqrt, log, pi
from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
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
#
#   log p_Z(z) = -||z||^2 / 2 - (d/2) log(2pi)
#

def logp_z(z: Tensor) -> Tensor:
    """Standard normal log-probability, summed over dims."""
    d = z.shape[-1]
    return -0.5 * (z ** 2).sum(dim=-1) - 0.5 * d * LOG_2PI


def fwd_jvp(f: nn.Module, x: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Forward-mode AD: returns (f(x), J_f(x) . v)."""
    with dual_level():
        dual_x = make_dual(x, v)
        dual_y = f(dual_x)
        y, jvp = unpack_dual(dual_y)
    return y, jvp


def bwd_vjp(f_out: Tensor, f_in: Tensor, v: Tensor) -> Tensor:
    """Backward-mode AD: returns J_f^T . v (with graph for param grads)."""
    (vjp,) = grad(f_out, f_in, v, create_graph=True)
    return vjp


def full_jacobian(f: nn.Module, x: Tensor) -> Tensor:
    """Full Jacobian J[i,j] = df(x)_i / dx_j via O(d) backward passes."""
    x = x.detach().requires_grad_()
    with torch.enable_grad():
        y = f(x)
        dim = x.shape[-1]
        jac = torch.zeros(x.shape[0], dim, dim, device=x.device)
        for i in range(dim):
            ei = torch.zeros_like(y)
            ei[:, i] = 1.0
            (row,) = grad(y, x, ei, retain_graph=True)
            jac[:, i, :] = row
    return jac

# =============================================================================
# Probe Vectors
# =============================================================================
#
# Random vectors v with E[vv^T] = I, scaled by sqrt(d) to preserve trace.
#

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
# Volume Change Surrogate
# =============================================================================
#
# Exact change-of-variables needs log |det J_dec(z)|, which is O(d^3).
# The surrogate gives an unbiased gradient estimator in O(d):
#
#   S = v^T J_enc(x) . sg(J_dec(z) v)
#
# Key property: grad_theta E[S] = grad_theta E[log |det J_dec|]
#

def hutchinson_surrogate(
    x: Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    sample_v: Callable[[Tensor], Tensor] = probe_randn,
) -> tuple[Tensor, Tensor, Tensor]:
    """Hutchinson surrogate for log |det J_dec|.
    Returns (surrogate [batch], z [batch, dim], x_rec [batch, dim])."""
    x = x.detach().requires_grad_()
    z = encoder(x)
    v = sample_v(z)

    x_rec, jvp_dec = fwd_jvp(decoder, z, v)
    vjp_enc = bwd_vjp(z, x, v)

    surrogate = (vjp_enc * jvp_dec.detach()).sum(dim=-1)
    return surrogate, z, x_rec

# =============================================================================
# Exact NLL
# =============================================================================
#
# NLL = -log p_Z(enc(x)) + log |det J_dec(enc(x))|
#
# Full Jacobian of decoder, O(d^3). Only for test-time metrics.
#

def compute_nll(x: Tensor, encoder: nn.Module, decoder: nn.Module) -> tuple[Tensor, Tensor, Tensor]:
    """Exact NLL via full decoder Jacobian. Returns (nll, z, x_rec)."""
    with torch.no_grad():
        z = encoder(x)
    jac = full_jacobian(decoder, z)
    x_rec = decoder(z.detach())
    log_abs_det = torch.linalg.slogdet(jac).logabsdet
    nll = -logp_z(z) + log_abs_det
    return nll, z, x_rec.detach()

# =============================================================================
# Loss
# =============================================================================
#
# L = beta . ||x - dec(enc(x))||^2 - log p_Z(z) - S
#
# At convergence: enc ~ dec^{-1}, z = enc(x) ~ N(0,I).
#

def surrogate_elbo(
    x: Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    beta: float,
    sample_v: Callable[[Tensor], Tensor] = probe_randn,
) -> Tensor:
    """Per-sample loss [batch]."""
    surrogate, z, x_rec = hutchinson_surrogate(x, encoder, decoder, sample_v)
    recon = ((x - x_rec) ** 2).sum(dim=-1)
    nll = -logp_z(z) - surrogate
    return beta * recon + nll

# =============================================================================
# Free-Form Flow Model
# =============================================================================

class FreeFormFlow(nn.Module):
    """Free-form flow: unconstrained encoder-decoder with surrogate volume change."""

    def __init__(self, dim: int = 2, hidden_dim: int = 64, beta: float = 150.0):
        super().__init__()
        self.encoder = Net(dim, hidden_dim)
        self.decoder = Net(dim, hidden_dim)
        self.beta = beta
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return surrogate_elbo(x, self.encoder, self.decoder, self.beta)

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

            nll, z, x_rec = compute_nll(test_batch, model.encoder, model.decoder)
            recon = ((test_batch - x_rec) ** 2).sum(dim=-1).mean()
            pbar.write(f"  step {i+1}: recon={recon:.1e}, NLL={nll.mean():.2f}")
            model.train()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    viz_2d_data(sample_8gaussians(4096))

    model = FreeFormFlow(dim=2, beta=150.0).to(DEVICE)
    train(model, sample_8gaussians, n_iter=100_000, batch_size=512)

    model.eval()
    viz_2d_data(model.generate(4096), filename="final_samples.jpg")

    test = sample_8gaussians(2048)
    nll, z, x_rec = compute_nll(test, model.encoder, model.decoder)
    print(f"Final exact NLL: {nll.mean():.3f}")
    print(f"Final recon MSE: {((test - x_rec)**2).sum(-1).mean():.3e}")
    print("Done.")
