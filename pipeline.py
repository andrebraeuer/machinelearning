"""
End-to-end pipeline implementing the exposé methodology in PyTorch.

Stages
------
1. Part 1, Stage 1: OAT VAE design study on a fixed window (no rolling).
2. Part 1, Stage 2: Rolling-window VaR/ES for selected configurations.
3. Part 2: Model risk (MAD) for VAE and copula candidate sets.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtesting import run_model_backtests
from config import (
    ES_ALPHA,
    ES_QUANTILE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    N_SIM,
    PATIENCE,
    SCALE,
    SEED,
    STAGE1_END_FRAC,
    STAGE2_START_FRAC,
    SUBPERIODS,
    VAR_ALPHA,
    VAR_QUANTILE,
    VAEConfig,
)
from copula import simulate_copula_residuals
from evaluation import summarize_stage1_metrics
from garch import build_residuals_matrix, portfolio_weights
from modelrisk import (
    calculate_dispersion,
    calculate_mad,
    filter_valid_models,
    mad_by_subperiod,
    mad_summary,
)
from vae import BetaVAE, apply_crisis_oversample, choose_device, sample_residuals, train_vae


def load_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.iloc[0].abs().sum() < 1e-8:
        df = df.iloc[1:].reset_index(drop=True)
    return df.astype(float)


def risk_from_simulation(
    z_sim: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Empirical VaR (99%) and ES (97.5%) for portfolio returns."""
    r_sim = mu + sigma * z_sim
    pf = r_sim @ weights
    var_level = np.quantile(pf, VAR_QUANTILE)
    es_level = np.quantile(pf, ES_QUANTILE)
    tail = pf[pf <= es_level]
    es = float(tail.mean()) if len(tail) else float(es_level)
    return float(var_level), float(es)


def _make_vae(cfg: VAEConfig, input_dim: int, device: torch.device) -> BetaVAE:
    m = BetaVAE(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
        beta=cfg.beta,
    )
    return m.to(device)


def fit_vae_on_residuals(
    resid: np.ndarray,
    cfg: VAEConfig,
    device: torch.device,
    *,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
) -> tuple[BetaVAE, dict]:
    z_train = apply_crisis_oversample(resid.copy(), cfg.crisis_oversample)
    model = _make_vae(cfg, resid.shape[1], device)
    info = train_vae(
        model,
        z_train,
        loss_type=cfg.loss,
        batch_size=cfg.batch_size,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=PATIENCE,
        device=device,
        verbose=False,
    )
    return model, info


def stage1_oat_study(
    returns: pd.DataFrame,
    configs: list[VAEConfig],
    *,
    end_frac: float = STAGE1_END_FRAC,
    n_sim: int = N_SIM,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
) -> pd.DataFrame:
    """
    Fixed-window OAT sensitivity (exposé: e.g. 2001–2012, no rolling).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = choose_device()
    weights = portfolio_weights(returns.columns)
    n_end = int(len(returns) * end_frac)
    results = []

    for cfg in configs:
        win = min(cfg.window, n_end - 1)
        window = returns.iloc[n_end - win : n_end]
        resid, mu, sigma = build_residuals_matrix(window, scale=SCALE)
        if len(resid) < 50:
            continue

        model, train_info = fit_vae_on_residuals(
            resid, cfg, device, max_epochs=max_epochs, min_epochs=min_epochs
        )
        if not train_info.get("converged", True) and train_info.get("best_epoch", 0) < min_epochs:
            continue

        z_sim = sample_residuals(model, n_sim)
        var_f, es_f = risk_from_simulation(z_sim, mu, sigma, weights)
        metrics = summarize_stage1_metrics(resid, z_sim)

        realized_win = window.values[-1] @ weights
        results.append(
            {
                "config": cfg.name,
                "experiment": cfg.experiment,
                "latent_dim": cfg.latent_dim,
                "beta": cfg.beta,
                "loss": cfg.loss,
                "hidden": cfg.hidden_dims[0],
                "window": cfg.window,
                "val_loss": train_info["best_val_loss"],
                "converged": train_info["converged"],
                "VaR_99": var_f,
                "ES_975": es_f,
                "corr_frobenius": metrics["frobenius"],
                "corr_max_abs": metrics["max_abs"],
                "tail_joint_emp": metrics["tail_joint_empirical"],
                "tail_joint_sim": metrics["tail_joint_simulated"],
                "realized_last": realized_win,
            }
        )

    return pd.DataFrame(results)


def rolling_forecasts_vae(
    returns: pd.DataFrame,
    cfg: VAEConfig,
    *,
    start_idx: int,
    n_forecast: int,
    n_sim: int = N_SIM,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling one-step-ahead VaR and ES (500-day window, annual VAE retrain by default).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = choose_device()
    weights = portfolio_weights(returns.columns)
    d = returns.shape[1]
    win = cfg.window

    dates = returns.index[start_idx : start_idx + n_forecast]
    var_s = pd.Series(index=dates, dtype=float, name=cfg.name)
    es_s = pd.Series(index=dates, dtype=float, name=cfg.name)

    vae_model: BetaVAE | None = None
    last_refit = -10**9

    for t, date in enumerate(dates):
        i = start_idx + t
        window = returns.iloc[i - win : i]
        resid, mu, sigma = build_residuals_matrix(window, scale=SCALE)

        if t - last_refit >= cfg.retrain_every or vae_model is None:
            vae_model, info = fit_vae_on_residuals(
                resid, cfg, device, max_epochs=max_epochs, min_epochs=min_epochs
            )
            if not info.get("converged", True):
                continue
            last_refit = t

        z_sim = sample_residuals(vae_model, n_sim)
        var_f, es_f = risk_from_simulation(z_sim, mu, sigma, weights)
        var_s.loc[date] = var_f
        es_s.loc[date] = es_f

    return var_s.to_frame(), es_s.to_frame()


def rolling_forecasts_copula(
    returns: pd.DataFrame,
    family: str,
    *,
    start_idx: int,
    n_forecast: int,
    window: int = 500,
    n_sim: int = N_SIM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling copula-GARCH VaR/ES for one copula family."""
    np.random.seed(SEED)
    weights = portfolio_weights(returns.columns)
    dates = returns.index[start_idx : start_idx + n_forecast]
    var_s = pd.Series(index=dates, dtype=float, name=family)
    es_s = pd.Series(index=dates, dtype=float, name=family)

    for t, date in enumerate(dates):
        i = start_idx + t
        wdata = returns.iloc[i - window : i]
        resid, mu, sigma = build_residuals_matrix(wdata, scale=SCALE)
        try:
            z_sim = simulate_copula_residuals(resid, family, n_sim)
        except Exception:
            continue
        var_f, es_f = risk_from_simulation(z_sim, mu, sigma, weights)
        var_s.loc[date] = var_f
        es_s.loc[date] = es_f

    return var_s.to_frame(), es_s.to_frame()


def select_stage2_configs(
    stage1_df: pd.DataFrame,
    top_k: int = 5,
) -> list[str]:
    """Pick best OAT configs by correlation fit for Stage 2 rolling."""
    if stage1_df.empty:
        return []
    ranked = stage1_df.sort_values("corr_frobenius").head(top_k)
    return ranked["config"].tolist()


def run_stage2_and_modelrisk(
    returns: pd.DataFrame,
    vae_configs: list[VAEConfig],
    copula_families: list[str],
    *,
    start_frac: float = STAGE2_START_FRAC,
    n_forecast: int | None = None,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
    output_dir: str = "results",
) -> dict:
    """
    Part 2: rolling forecasts, backtests, MAD for VAE and copula sets.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_idx = int(len(returns) * start_frac)
    if n_forecast is None:
        n_forecast = len(returns) - start_idx
    n_forecast = min(n_forecast, len(returns) - start_idx)
    realized = (returns.values[start_idx : start_idx + n_forecast] @ portfolio_weights(returns.columns))
    realized_s = pd.Series(realized, index=returns.index[start_idx : start_idx + n_forecast])

    var_vae = pd.DataFrame(index=realized_s.index)
    es_vae = pd.DataFrame(index=realized_s.index)

    for cfg in vae_configs:
        vdf, edf = rolling_forecasts_vae(
            returns,
            cfg,
            start_idx=start_idx,
            n_forecast=n_forecast,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
        )
        var_vae = var_vae.join(vdf, how="outer")
        es_vae = es_vae.join(edf, how="outer")

    var_cop = pd.DataFrame(index=realized_s.index)
    es_cop = pd.DataFrame(index=realized_s.index)
    for fam in copula_families:
        vdf, edf = rolling_forecasts_copula(
            returns,
            fam,
            start_idx=start_idx,
            n_forecast=n_forecast,
        )
        var_cop = var_cop.join(vdf, how="outer")
        es_cop = es_cop.join(edf, how="outer")

    var_vae = var_vae.dropna(how="all")
    es_vae = es_vae.reindex(var_vae.index)
    var_cop = var_cop.dropna(how="all")
    es_cop = es_cop.reindex(var_cop.index)

    bt_vae = run_model_backtests(realized_s.reindex(var_vae.index), var_vae, es_vae)
    bt_cop = run_model_backtests(
        realized_s.reindex(var_cop.index), var_cop, es_cop
    )

    valid_vae = filter_valid_models(bt_vae, var_vae)
    valid_cop = filter_valid_models(bt_cop, var_cop)

    var_vae_v = var_vae[valid_vae] if valid_vae else var_vae
    var_cop_v = var_cop[valid_cop] if valid_cop else var_cop

    mad_vae = calculate_mad(var_vae_v) if var_vae_v.shape[1] >= 2 else pd.Series(0.0, index=var_vae_v.index)
    mad_cop = calculate_mad(var_cop_v) if var_cop_v.shape[1] >= 2 else pd.Series(0.0, index=var_cop_v.index)

    var_vae.to_csv(out / "var_vae.csv")
    var_cop.to_csv(out / "var_copula.csv")
    mad_vae.to_csv(out / "mad_vae.csv")
    mad_cop.to_csv(out / "mad_copula.csv")
    bt_vae.to_csv(out / "backtest_vae.csv")
    bt_cop.to_csv(out / "backtest_copula.csv")

    summary = {
        "mad_vae_summary": mad_summary(mad_vae),
        "mad_cop_summary": mad_summary(mad_cop),
        "mad_vae_subperiods": mad_by_subperiod(mad_vae, SUBPERIODS),
        "mad_cop_subperiods": mad_by_subperiod(mad_cop, SUBPERIODS),
        "dispersion_vae": calculate_dispersion(var_vae_v),
        "dispersion_cop": calculate_dispersion(var_cop_v),
        "backtest_vae": bt_vae,
        "backtest_copula": bt_cop,
        "n_valid_vae": len(valid_vae),
        "n_valid_cop": len(valid_cop),
        "mean_mad_vae": float(mad_vae.mean()),
        "mean_mad_cop": float(mad_cop.mean()),
    }
    pd.DataFrame(
        {
            "MAD_VAE": mad_vae,
            "MAD_COP": mad_cop.reindex(mad_vae.index),
        }
    ).to_csv(out / "mad_comparison.csv")

    return summary
