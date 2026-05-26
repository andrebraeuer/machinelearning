"""
Fitting the marginals using an ARMA(1,1)-GARCH(1,1) model (exposé: skewed-t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from config import SCALE


# Helper functions
# _-Präfix: signalisiert, dass diese Funtkionen privat sind und nicht direkt von außen aufgerufen werden sollen.
# fit_marginals() bleibt die einzig öffentliche Schnittstelle
def _extract_forecast(res, horizon, scale):
    """
    Extract mu and sigma forecast from fitted arch model.
    """
    fc = res.forecast(horizon=horizon)
    return {
        "mu":    fc.mean.iloc[-1, 0] / scale,
        "sigma": np.sqrt(fc.variance.values[-1, 0]) / scale,
    }


def _extract_dist_params(params, dist):
    """
    Extract distribution parameters from fitted arch model params.
    """
    return {
        "nu":     params.get("nu")     if dist == "t"     else
                  params.get("eta")    if dist == "skewt" else None,
        "lambda": params.get("lambda") if dist == "skewt" else None,
        }


# ARMA-GARCH models 
def fit_marginals(returns, garch_order=(1, 1), lags=1, dist="t", horizon=1, scale=100):
    """
    Fit an AR(lags)-GARCH(p,q) model to a single return series.

    Parameters
    ----------
    returns : pd.Series
        Raw (unscaled) return series for one asset.
    garch_order : tuple of int, optional
        (p, q) order of the GARCH volatility process. Default is (1, 1).
    lags : int, optional
        Number of autoregressive lags in the mean equation. Default is 1.
    dist : str, optional
        Innovation distribution. One of 'normal', 't', or 'skewt'. Default is 't'.
    horizon : int, optional
        Forecast horizon in periods. Default is 1.
    scale : float, optional
        Multiplicative scaling applied before fitting for numerical stability.
        All forecasts are rescaled back to the original units.

    Returns
    -------
    dict with keys:
        residuals_std : pd.Series   -- standardized residuals; first row is NaN
        mu            : float       -- one-step-ahead mean forecast
        sigma         : float       -- one-step-ahead volatility forecast
        nu            : float|None  -- degrees of freedom, else None
        lambda        : float|None  -- skewness parameter in (-1, 1) for 'skewt', else None
    """
    p, q = garch_order
    res = arch_model(returns*scale, mean="AR", lags=lags, vol="Garch", p=p, q=q, dist=dist).fit(disp="off", show_warning=False)

    return {
        "residuals_std": res.std_resid,
        **_extract_forecast(res, horizon, scale),
        **_extract_dist_params(res.params, dist),
    }


def build_residuals_matrix(
    returns_window: pd.DataFrame,
    *,
    garch_order=(1, 1),
    lags: int = 1,
    dist: str = "skewt",
    scale: float = SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit AR(1)-GARCH(1,1) per asset on a rolling window.

    Returns
    -------
    resid : (T_eff, d) standardized residuals (NaN rows dropped)
    mu    : (d,) one-step-ahead mean forecasts
    sigma : (d,) one-step-ahead volatility forecasts
    """
    fits = {
        c: fit_marginals(
            returns_window[c],
            garch_order=garch_order,
            lags=lags,
            dist=dist,
            horizon=1,
            scale=scale,
        )
        for c in returns_window.columns
    }

    resid = np.column_stack([f["residuals_std"].values for f in fits.values()])
    mask = ~np.isnan(resid).any(axis=1)
    resid = resid[mask]

    mu = np.array([f["mu"] for f in fits.values()])
    sigma = np.array([f["sigma"] for f in fits.values()])
    return resid, mu, sigma


def portfolio_weights(columns: list[str] | pd.Index) -> np.ndarray:
    """
    Exposé weights: 40% equities, 40% bonds, 10% commodities, 10% real estate.
    Missing RE index: allocate its 10% to the bond leg.
    """
    cols = list(columns)
    w = np.zeros(len(cols))
    eq_share = 0.40 / 3.0
    for i, c in enumerate(cols):
        cu = c.upper()
        if any(k in cu for k in ("STOXX", "DOW", "MSCI", "EQUITY", "S&P 500")):
            w[i] = eq_share
        elif "TREASURY" in cu or "BOND" in cu or "SOVEREIGN" in cu:
            w[i] = 0.50
        elif "GSCI" in cu or "COMMOD" in cu:
            w[i] = 0.10
        elif "REAL" in cu or "ESTATE" in cu:
            w[i] = 0.10
    if w.sum() <= 0:
        w[:] = 1.0 / len(cols)
    else:
        w /= w.sum()
    return w
