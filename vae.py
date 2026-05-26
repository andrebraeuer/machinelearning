"""
PyTorch β-VAE for multivariate standardized GARCH residuals (exposé Part 1).
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    VAL_FRAC,
)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BetaVAE(nn.Module):
    """
    Baseline architecture (exposé p. 2):
    2 × hidden (tanh) → Gaussian latent → decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (64, 64),
        latent_dim: int = 3,
        beta: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        h0, h1 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h0),
            nn.Tanh(),
            nn.Linear(h0, h1),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(h1, latent_dim)
        self.fc_logvar = nn.Linear(h1, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h0),
            nn.Tanh(),
            nn.Linear(h0, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=dev)
        return self.decode(z)


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    loss_type: Literal["mse", "tail_mse"] = "mse",
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")

    # Tail-weighted MSE: up-weight observations in the joint tail (L3).
    per_row = ((x_hat - x) ** 2).mean(dim=1)
    tail_score = x.abs().max(dim=1).values
    thresh = torch.quantile(tail_score, 0.90)
    weights = torch.where(tail_score >= thresh, 2.0, 1.0)
    return (per_row * weights).mean()


def elbo_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    loss_type: Literal["mse", "tail_mse"] = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = reconstruction_loss(x, x_hat, loss_type)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


def apply_crisis_oversample(z: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Duplicate crisis-period rows (L7): bottom 10% by ‖z‖."""
    if factor <= 1.0:
        return z
    score = np.linalg.norm(z, axis=1)
    thresh = np.quantile(score, 0.10)
    crisis = z[score <= thresh]
    n_extra = int((factor - 1.0) * len(crisis))
    if n_extra <= 0:
        return z
    idx = np.random.choice(len(crisis), size=n_extra, replace=True)
    return np.vstack([z, crisis[idx]])


def train_vae(
    model: BetaVAE,
    z: np.ndarray,
    *,
    loss_type: Literal["mse", "tail_mse"] = "mse",
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
    val_frac: float = VAL_FRAC,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    """
    Train with chronological validation split and early stopping.
    Configurations with no meaningful fit after min_epochs are flagged via `converged=False`.
    """
    device = device or choose_device()
    model = model.to(device)

    n_val = max(1, int(val_frac * z.shape[0])) if val_frac > 0 else 0
    z_tr = torch.from_numpy(z[:-n_val] if n_val else z).float().to(device)
    z_va = torch.from_numpy(z[-n_val:]).float().to(device) if n_val else None

    loader = DataLoader(TensorDataset(z_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_sum = 0.0
        n_seen = 0
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, mu, lv = model(xb)
            loss, _, _ = elbo_loss(xb, x_hat, mu, lv, model.beta, loss_type)
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tr_sum += loss.item() * bs
            n_seen += bs
        tr_loss = tr_sum / n_seen
        history["train_loss"].append(tr_loss)

        if z_va is not None:
            model.eval()
            with torch.no_grad():
                x_hat_v, mu_v, lv_v = model(z_va)
                val_loss, _, _ = elbo_loss(
                    z_va, x_hat_v, mu_v, lv_v, model.beta, loss_type
                )
                val_loss = val_loss.item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
                if epoch >= min_epochs and bad >= patience:
                    break
        elif epoch >= max_epochs:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    converged = best_epoch >= min_epochs and np.isfinite(best_val)
    if verbose:
        print(f"    VAE trained: best_epoch={best_epoch}, val_loss={best_val:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "converged": converged,
    }


@torch.no_grad()
def sample_residuals(model: BetaVAE, n: int) -> np.ndarray:
    return model.sample(n).cpu().numpy()
