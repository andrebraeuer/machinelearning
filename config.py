"""
Configuration for the VAE-GARCH expose pipeline (Fritzsch et al. 2024 template).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

SEED = 42
WINDOW = 500
N_SIM = 10_000
SCALE = 100.0
BATCH_SIZE = 64
MAX_EPOCHS = 200
MIN_EPOCHS = 30
PATIENCE = 15
LEARNING_RATE = 1e-3
VAL_FRAC = 0.15

VAR_ALPHA = 0.01
ES_ALPHA = 0.025
VAR_QUANTILE = 1.0 - VAR_ALPHA
ES_QUANTILE = 1.0 - ES_ALPHA

TRADING_DAYS_PER_YEAR = 252
RETRAIN_ANNUAL = TRADING_DAYS_PER_YEAR
RETRAIN_QUARTERLY = TRADING_DAYS_PER_YEAR // 4

DATA_PATH = "data/qrm2025_returns.csv"
STAGE1_END_FRAC = 0.55
STAGE2_START_FRAC = 0.55

COPULA_FAMILIES = (
    "gaussian",
    "t",
    "clayton",
    "gumbel",
    "frank",
    "gaussian_mixture",
    "dcc",
)
# Joe copula omitted (not in all copulae versions). Vine requires pyvinecop (optional).

SUBPERIODS = {
    "full": (0.0, 1.0),
    "pre_crisis": (0.0, 0.35),
    "financial_crisis": (0.35, 0.45),
    "covid": (0.75, 0.85),
    "post_crisis": (0.45, 0.75),
}


@dataclass(frozen=True)
class VAEConfig:
    name: str
    latent_dim: int = 3
    beta: float = 2.0
    hidden_dims: tuple[int, int] = (64, 64)
    loss: Literal["mse", "tail_mse"] = "mse"
    batch_size: int = BATCH_SIZE
    window: int = WINDOW
    retrain_every: int = RETRAIN_ANNUAL
    crisis_oversample: float = 1.0
    experiment: str = "B"


def _baseline() -> VAEConfig:
    return VAEConfig(name="B_baseline")


def oat_experiments() -> list[VAEConfig]:
    b = _baseline()
    configs: list[VAEConfig] = [b]

    for dz in (2, 3, 5):
        if dz != b.latent_dim:
            configs.append(replace(b, name=f"L1_latent_{dz}", latent_dim=dz, experiment="L1"))

    for beta in (1.0, 2.0, 4.0):
        if beta != b.beta:
            configs.append(replace(b, name=f"L2_beta_{beta}", beta=beta, experiment="L2"))

    for loss in ("mse", "tail_mse"):
        if loss != b.loss:
            configs.append(replace(b, name=f"L3_loss_{loss}", loss=loss, experiment="L3"))

    for w in (32, 64, 128):
        if w != b.hidden_dims[0]:
            configs.append(
                replace(b, name=f"L4_width_{w}", hidden_dims=(w, w), experiment="L4")
            )

    for freq, days in (("quarterly", RETRAIN_QUARTERLY), ("once", 10**9)):
        if days != b.retrain_every:
            configs.append(
                replace(b, name=f"L5_retrain_{freq}", retrain_every=days, experiment="L5")
            )

    for win in (500, 1000):
        if win != b.window:
            configs.append(replace(b, name=f"L6_window_{win}", window=win, experiment="L6"))

    for mult in (1.0, 2.0):
        if mult != b.crisis_oversample:
            configs.append(
                replace(
                    b,
                    name=f"L7_crisis_{int(mult)}x",
                    crisis_oversample=mult,
                    experiment="L7",
                )
            )

    return configs


def oat_experiments_fast() -> list[VAEConfig]:
    return [
        _baseline(),
        replace(_baseline(), name="L1_latent_2", latent_dim=2, experiment="L1"),
        replace(_baseline(), name="L2_beta_4", beta=4.0, experiment="L2"),
        replace(_baseline(), name="L3_loss_tail_mse", loss="tail_mse", experiment="L3"),
    ]
