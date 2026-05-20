"""
Fitting the marginals using an ARMA-GARCH model.
"""

import numpy as np
from arch import arch_model


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
        **_extract_dist_params(res.params, dist)
    }
