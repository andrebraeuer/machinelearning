"""
GARCH(1,1) models with t-distributed innovations.
"""

def fit_garch_constmean(returns, garch_order=(1, 1), dist="t", horizon=1, scale=100):
    """
    Fit GARCH(1,1) model with Student-t innovations using the `arch` packages.
    Parameters:
        - returns: Array-like, the return series to be modeled.
        - garch_order: Tuple, the order of the GARCH model (default is (1, 1)).
        - dist: String, the distribution of the innovations (default is "t" for Student-t).
        - horizon: Int, the number of periods to forecast (default is 1).
        - scale: Float, a scaling factor for the returns to improve convergence (default is 100).
    Returns:
        - A dictionary containing:
            - residuals_std: Array, standardized residuals (innovations) from the fitted model.
            - mu: Float, forecasted mean for the next period(s).
            - sigma: Float, forecasted conditional volatility for the next period(s).
            - nu: Float or None, degrees of freedom for the t-distribution (if dist is "t"), otherwise None.
    """
    from arch import arch_model
    import numpy as np

    p, q = garch_order
    res_garch = arch_model(returns*scale, mean="Constant", vol="Garch", p=p, q=q, dist=dist).fit(disp="off", show_warning=False) # mean="Zero" because the mean process is already captured by the ARMA model
    nu = res_garch.params["nu"] if dist == "t" else None # degrees of freedom for the t-distribution
    residuals_std = res_garch.std_resid # standardized residuals (innovations)

    sigma1 = np.sqrt(res_garch.forecast(horizon=horizon).variance.values[-1, 0]) / scale # forecasted volatility from GARCH model; scale back to original units
    mu1 = res_garch.forecast(horizon=horizon).mean.iloc[-1, 0] / scale # forecasted mean return from the ARMA model; scale back to original units

    return {"residuals_std": residuals_std, "mu": mu1, "sigma": sigma1, "nu": nu}

def fit_ar_garch(returns, garch_order=(1, 1), lags=1, dist="t", horizon=1, scale=100):
    """
    Fit AR(1)-GARCH(1,1) model with Student-t innovations using the `arch` packages.
    Parameters:
        - returns: Array-like, the return series to be modeled.
        - garch_order: Tuple, the order of the GARCH model (default is (1, 1)).
        - lags: Int, the number of lags for the AR model (default is 1).
        - dist: String, the distribution of the innovations (default is "t" for Student-t).
        - horizon: Int, the number of periods to forecast (default is 1).
        - scale: Float, a scaling factor for the returns to improve convergence (default is 100).
    Returns:
        - A dictionary containing:
            - residuals_std: Array, standardized residuals (innovations) from the fitted model.
            - mu: Float, forecasted mean for the next period(s).
            - sigma: Float, forecasted conditional volatility for the next period(s).
            - nu: Float or None, degrees of freedom for the t-distribution (if dist is "t"), otherwise None.
    """
    from arch import arch_model
    import numpy as np

    p, q = garch_order
    res_garch = arch_model(returns*scale, mean="AR", lags=lags, vol="Garch", p=p, q=q, dist=dist).fit(disp="off", show_warning=False) # mean="Zero" because the mean process is already captured by the ARMA model
    nu = res_garch.params["nu"] if dist == "t" else None # degrees of freedom for the t-distribution
    residuals_std = res_garch.std_resid # standardized residuals (innovations)

    sigma1 = np.sqrt(res_garch.forecast(horizon=horizon).variance.values[-1, 0]) / scale # forecasted volatility from GARCH model; scale back to original units
    mu1 = res_garch.forecast(horizon=horizon).mean.iloc[-1, 0] / scale # forecasted mean return from the ARMA model; scale back to original units

    return {"residuals_std": residuals_std, "mu": mu1, "sigma": sigma1, "nu": nu}

def fit_arma_garch(returns, arma_order=(1, 0, 1), garch_order=(1, 1), dist="t", horizon=1, scale=100):
    """
    Fit ARMA(1,0,1)-GARCH(1,1) model with Student-t innovations using the `statsmodels` and `arch` packages.
    Parameters:
        - returns: Array-like, the return series to be modeled.
        - arma_order: Tuple, the order of the ARMA model (default is (1, 0, 1)).
        - garch_order: Tuple, the order of the GARCH model (default is (1, 1)).
        - dist: String, the distribution of the innovations (default is "t" for Student-t).
        - horizon: Int, the number of periods to forecast (default is 1).
        - scale: Float, a scaling factor for the returns to improve convergence (default is 100).
    Returns:
        - A dictionary containing:
            - residuals_std: Array, standardized residuals (innovations) from the fitted model.
            - mu: Float, forecasted mean for the next period(s).
            - sigma: Float, forecasted conditional volatility for the next period(s).
            - nu: Float or None, degrees of freedom for the t-distribution (if dist is "t"), otherwise None.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    import numpy as np

    # 1. Fit ARMA(1,0,1) model to the returns
    res_arma = ARIMA(returns*scale, order=arma_order).fit() # scale returns for better convergence
    residuals_arma = res_arma.resid # ARMA residuals

    # 2. Fit GARCH(1,1) model to the ARMA residuals
    p, q = garch_order
    res_garch = arch_model(residuals_arma, mean="Zero", vol="Garch", p=p, q=q, dist=dist).fit(disp="off", show_warning=False) # mean="Zero" because the mean process is already captured by the ARMA model
    nu = res_garch.params["nu"] if dist == "t" else None # degrees of freedom for the t-distribution
    residuals_std = res_garch.std_resid # standardized residuals (innovations)

    # 3. Forecasting the next period's volatility and mean
    sigma1 = np.sqrt(res_garch.forecast(horizon=horizon).variance.values[-1, 0]) / scale # forecasted volatility from GARCH model; scale back to original units
    mu1 = res_arma.forecast(steps=horizon).iloc[0] / scale # forecasted mean return from the ARMA model; scale back to original units

    return {"residuals_std": residuals_std, "mu": mu1, "sigma": sigma1, "nu": nu}





