"""
Backtests for VaR and ES forecasts (exposé Part 1 evaluation).

- Christoffersen & Pelletier (2004) duration-based VaR test at 99%
- Conditional ES calibration (Nolde & Ziegel, 2017 style simplified test) at 97.5%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def violations(realized: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Hit sequence: 1 if realized return < VaR forecast (loss exceeds VaR)."""
    return (np.asarray(realized) < np.asarray(var_forecast)).astype(int)


def christoffersen_pelletier_duration_test(
    viol: np.ndarray, alpha: float
) -> dict:
    """
    Duration-based independence test (Christoffersen & Pelletier, 2004).
    H0: durations between VaR violations are Exponential (memoryless).
    """
    v = np.asarray(viol).astype(int)
    n = len(v)
    n_viol = int(v.sum())
    if n_viol < 2:
        return {
            "LR": np.nan,
            "p_value": np.nan,
            "b_hat": np.nan,
            "n_violations": n_viol,
            "reject_h0": False,
        }

    hit_idx = np.where(v == 1)[0]
    durations = np.diff(hit_idx).astype(float)
    d_first = float(hit_idx[0] + 1)
    d_last = float(n - 1 - hit_idx[-1])
    d_obs = durations
    d_cens = np.array(
        [d for d in (d_first, d_last) if d > 0],
        dtype=float,
    )

    def neg_ll_weibull(params, d_obs, d_cens):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        ll_obs = np.sum(
            np.log(a)
            + np.log(b)
            + (b - 1) * np.log(a * d_obs)
            - (a * d_obs) ** b
        )
        ll_cens = -np.sum((a * d_cens) ** b) if len(d_cens) else 0.0
        return -(ll_obs + ll_cens)

    def neg_ll_exp(a, d_obs, d_cens):
        if a <= 0:
            return 1e10
        ll_obs = np.sum(np.log(a) - a * d_obs)
        ll_cens = -a * np.sum(d_cens) if len(d_cens) else 0.0
        return -(ll_obs + ll_cens)

    total_time = float(d_obs.sum() + d_cens.sum())
    if total_time <= 0:
        return {
            "LR": np.nan,
            "p_value": np.nan,
            "b_hat": np.nan,
            "n_violations": n_viol,
            "reject_h0": False,
        }

    a_hat_h0 = len(d_obs) / total_time
    ll_h0 = -neg_ll_exp(a_hat_h0, d_obs, d_cens)

    res = optimize.minimize(
        neg_ll_weibull,
        x0=[a_hat_h0, 1.0],
        args=(d_obs, d_cens),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    if not res.success:
        return {
            "LR": np.nan,
            "p_value": np.nan,
            "b_hat": np.nan,
            "n_violations": n_viol,
            "reject_h0": False,
        }

    a_hat_h1, b_hat = res.x
    ll_h1 = -res.fun
    LR = max(0.0, -2.0 * (ll_h0 - ll_h1))
    p_value = 1.0 - stats.chi2.cdf(LR, df=1)
    return {
        "LR": float(LR),
        "p_value": float(p_value),
        "b_hat": float(b_hat),
        "n_violations": n_viol,
        "reject_h0": bool(p_value < 0.05),
    }


def es_conditional_calibration_test(
    realized: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    alpha_es: float = 0.025,
) -> dict:
    """
    Simplified conditional ES backtest (Nolde & Ziegel, 2017 spirit).

    On VaR-exceedance days, tests whether the average realized loss exceeds
    the ES forecast (ES should be at least as conservative as realized tail loss).
    Uses a one-sided t-test on standardized shortfall residuals.
    """
    r = np.asarray(realized, dtype=float)
    var_f = np.asarray(var_forecast, dtype=float)
    es_f = np.asarray(es_forecast, dtype=float)

    hits = r < var_f
    n_hit = int(hits.sum())
    if n_hit < 5:
        return {
            "n_exceedances": n_hit,
            "statistic": np.nan,
            "p_value": np.nan,
            "reject_h0": False,
        }

    # Losses are negative returns; VaR/ES stored as negative quantiles in pipeline
    loss_real = -r[hits]
    loss_es = -es_f[hits]
    shortfall = loss_real - loss_es
    stat = float(shortfall.mean() / (shortfall.std(ddof=1) / np.sqrt(n_hit) + 1e-12))
    p_value = float(1.0 - stats.t.cdf(stat, df=n_hit - 1))
    return {
        "n_exceedances": n_hit,
        "statistic": stat,
        "p_value": p_value,
        "reject_h0": bool(p_value < 0.05),
    }


def run_model_backtests(
    realized: pd.Series,
    var_df: pd.DataFrame,
    es_df: pd.DataFrame,
    *,
    var_alpha: float = 0.01,
    es_alpha: float = 0.025,
) -> pd.DataFrame:
    """Run duration (VaR) and ES calibration tests for each model column."""
    rows = []
    for model in var_df.columns:
        var_s = var_df[model].astype(float)
        es_s = es_df[model].astype(float)
        viol = violations(realized.values, var_s.values)
        dur = christoffersen_pelletier_duration_test(viol, var_alpha)
        es_t = es_conditional_calibration_test(
            realized.values, var_s.values, es_s.values, es_alpha
        )
        rows.append(
            {
                "model": model,
                "var_violations": int(viol.sum()),
                "var_violation_rate": float(viol.mean()),
                "duration_LR": dur["LR"],
                "duration_p": dur["p_value"],
                "duration_pass": not dur["reject_h0"],
                "es_statistic": es_t["statistic"],
                "es_p": es_t["p_value"],
                "es_pass": not es_t["reject_h0"],
                "both_pass": (not dur["reject_h0"]) and (not es_t["reject_h0"]),
            }
        )
    return pd.DataFrame(rows).set_index("model")
