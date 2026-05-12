"""
ARMA-GARCH(1,1) with t-distributed innovations.
"""


def fit_garch(returns, scale=100):
    """
    Fit ARMA(1,1)-GARCH(1,1) model with Student-t innovations using the `arch` package.
    Parameters:
        - returns: Array-like, the return series to be modeled.
        - scale: Float, a scaling factor for the returns to improve convergence (default is 100).
    Returns:
        - residuals_std: Array, standardized residuals (innovations) from the fitted model.
        - sigma1: Float, forecasted conditional volatility for the next period.
        - mu1: Float, forecasted mean for the next period.
    """
    from arch import arch_model 

    model = arch_model(returns*scale, mean="AR", lags=1, vol="Garch", p=1, q=1, dist="t") # scale returns for better convergence

    res = model.fit(disp="off", show_warning=False) # disp="off" to suppress output, show_warning=False to ignore convergence warnings

    residuals_std = res.std_resid # standardized residuals (innovations)

    sigma1 = res.forecast(horizon=1).variance.values[-1, 0] ** 0.5 / scale # forecasted volatility for the next period; scale back to original units

    mu1 = res.forecast(horizon=1).mean.iloc[-1, 0] / 100 # forecasted mean for the next period; scale back to original units

    return residuals_std, sigma1, mu1