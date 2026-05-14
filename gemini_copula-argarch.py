import numpy as np
import pandas as pd
from arch import arch_model
from copulae import GaussianCopula
from scipy.stats import skewt
import matplotlib.pyplot as plt

def run_risk_forecast(returns_df, window_size=500, n_sim=10000, alpha=0.05):
    """
    Implementierung nach Fritzsch et al. (2024) mit ARMA-GARCH Marginals 
    und Gaussian Copula Abhängigkeit.
    """
    n_assets = returns_df.shape[1]
    n_days = len(returns_df)
    
    results = []

    # Rolling Window Loop
    for start in range(0, n_days - window_size):
        end = start + window_size
        window_data = returns_df.iloc[start:end]
        
        std_residuals = np.zeros_like(window_data)
        u_data = np.zeros_like(window_data)
        marginal_params = []
        forecasted_vol = []
        forecasted_mu = []

        # 1. Marginals: ARMA(1,1)-GARCH(1,1) mit Skewed-t
        for i in range(n_assets):
            # Modell-Spezifikation (festgehalten nach Fritzsch et al.)
            model = arch_model(window_data.iloc[:, i], vol='Garch', p=1, q=1, 
                               dist='skewt', mean='AR', lags=1)
            res = model.fit(disp='off')
            
            # Standardisierte Residuen extrahieren
            z = res.resid / res.conditional_volatility
            std_residuals[:, i] = z
            
            # Transformation in den Copula-Raum [0, 1] via Skewed-t CDF
            # Wir nutzen die Parameter aus dem Fit: df (nu) und skew (lambda)
            nu, lam = res.params['nu'], res.params['eta'] 
            # Hinweis: 'arch' nutzt oft andere Benennungen, wir nehmen die Fitted Dist
            u_data[:, i] = res.model.distribution.cdf(z.dropna(), res.params[-2:])
            
            # Ein-Schritt-Prognose für Volatilität und Mean
            forecast = res.forecast(horizon=1)
            forecasted_vol.append(np.sqrt(forecast.variance.values[-1, 0]))
            forecasted_mu.append(forecast.mean.values[-1, 0])
            marginal_params.append(res.params[-2:]) # nu und lambda speichern

        # 2. Abhängigkeit: Gaussian Copula
        # Fit der Copula auf die Pseudo-Observations u_data
        g_cop = GaussianCopula(dim=n_assets)
        g_cop.fit(u_data)
        
        # Simulation von 10.000 Vektoren aus der Copula
        sim_u = g_cop.random(n_sim)
        
        # 3. Rücktransformation in den Return-Space
        sim_returns = np.zeros((n_sim, n_assets))
        for i in range(n_assets):
            # Zurück zu standardisierten Residuen via Inverse Skewed-t CDF
            nu, lam = marginal_params[i]
            # Wir nutzen das Distribution-Objekt des fitted models für den PPF
            model_dist = arch_model(window_data.iloc[:, i], dist='skewt').model.distribution
            z_sim = model_dist.ppf(sim_u[:, i], marginal_params[i])
            
            # Skalierung mit GARCH Forecast (Standardisierte Residuen -> Returns)
            sim_returns[:, i] = forecasted_mu[i] + z_sim * forecasted_vol[i]
        
        # 4. Aggregation zum Portfolio (hier: Gleichgewichtet 1/N)
        portfolio_sim = sim_returns.mean(axis=1)
        
        # 5. Risikomaße: VaR (empirisches Quantil) und ES
        var_val = np.percentile(portfolio_sim, alpha * 100)
        es_val = portfolio_sim[portfolio_sim <= var_val].mean()
        
        results.append({
            'Date': returns_df.index[end],
            'Actual': returns_df.iloc[end].mean(),
            'VaR': var_val,
            'ES': es_val
        })
        
        if end % 10 == 0:
            print(f"Tag {end}/{n_days} berechnet...")

    return pd.DataFrame(results).set_index('Date')

# Beispielaufruf (Dummy Daten)
# df = pd.read_csv('your_returns.csv', index_col=0, parse_dates=True)
# forecast_results = run_risk_forecast(df)