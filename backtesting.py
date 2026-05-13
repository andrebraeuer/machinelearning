"""
Backtests for VaR and ES forecasts.
 
Implements two backtests for risk measure forecasts:
* Unconditional coverage test (Kupiec test) for VaR forecasts.
* Independence test (Christoffersen test) for VaR forecasts.
"""

def get_exceedances(returns, var_forecast):
    """
    Get binary array of exceedances (1 if return < VaR forecast, else 0).
    """
    return (returns < var_forecast).astype(int)

