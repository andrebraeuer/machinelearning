"""
Model risk metrics (Fritzsch et al., 2024): MAD across valid models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_mad(var_df: pd.DataFrame) -> pd.Series:
    """
    Mean absolute deviation of daily VaR forecasts across models.

    MAD_t = (1/|M_t|) sum_j |VaR_{j,t} - VaR_t|
    where VaR_t is the cross-sectional mean at time t.
    """
    row_mean = var_df.mean(axis=1)
    return var_df.sub(row_mean, axis=0).abs().mean(axis=1)


def calculate_dispersion(var_df: pd.DataFrame) -> pd.DataFrame:
    """Robustness: standard deviation and IQR of VaR forecasts per day."""
    return pd.DataFrame(
        {
            "std": var_df.std(axis=1),
            "iqr": var_df.quantile(0.75, axis=1) - var_df.quantile(0.25, axis=1),
        }
    )


def mad_summary(
    mad: pd.Series,
    *,
    portfolio_value: float = 100_000,
    horizon_days: int = 10,
) -> pd.DataFrame:
    stats = {
        "Min": mad.min(),
        "Median": mad.median(),
        "Mean": mad.mean(),
        "Max": mad.max(),
        "SD": mad.std(),
    }
    scale = portfolio_value * np.sqrt(horizon_days)
    return pd.DataFrame(
        {
            "MAD (%)": pd.Series(stats) * 100,
            "MAD ($)": pd.Series(stats) * scale,
        }
    )


def filter_valid_models(
    backtest_df: pd.DataFrame,
    var_df: pd.DataFrame,
) -> list[str]:
    """Models that pass both VaR duration and ES backtests (M_t)."""
    if "both_pass" not in backtest_df.columns:
        return list(var_df.columns)
    return backtest_df.index[backtest_df["both_pass"]].tolist()


def mad_by_subperiod(
    mad: pd.Series,
    fractions: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    n = len(mad)
    rows = {}
    for name, (a, b) in fractions.items():
        i0 = int(a * n)
        i1 = int(b * n)
        if i1 <= i0:
            continue
        rows[name] = mad.iloc[i0:i1].mean()
    return pd.DataFrame.from_dict(rows, orient="index", columns=["mean_MAD"])
