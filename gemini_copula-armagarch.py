import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from copulae import GaussianCopula

def run_proper_arma_garch_risk(returns_df, window_size=500, n_sim=10000, alpha=0.05):
    n_assets = returns_df.shape[1]
    n_days = len(returns_df)
    results = []

    # Wir folgen Fritzsch et al.: Modell wird einmal gewählt und konstant gehalten.
    # Hier: Rolling Window mit Neuschätzung der Parameter (Spezifikation bleibt fix).
    for start in range(0, n_days - window_size):
        end = start + window_size
        window_data = returns_df.iloc[start:end]
        
        u_data = np.zeros((window_size, n_assets))
        forecasts_mu = []
        forecasts_sigma = []
        dist_params = []
        model_distributions = []

        for i in range(n_assets):
            asset_data = window_data.iloc[:, i]
            
            # --- SCHRITT 1: ARMA(1,1) für den Mittelwert ---
            # Wir nutzen ARIMA(1,0,1) für den Mean-Teil
            arma_model = ARIMA(asset_data, order=(1, 0, 1))
            arma_res = arma_model.fit()
            
            # Residuen des ARMA-Modells
            arma_resid = arma_res.resid
            
            # Prognose für den Mittelwert (T+1)
            next_mu = arma_res.forecast(steps=1).values[0]

            # --- SCHRITT 2: GARCH(1,1) auf die ARMA-Residuen ---
            # Hier nutzen wir die Skewed-t Verteilung (Hansen, 1994)
            garch_model = arch_model(arma_resid, vol='Garch', p=1, q=1, 
                                     dist='skewt', mean='Zero')
            garch_res = garch_model.fit(disp='off')
            
            # Standardisierte Residuen (z = epsilon / sigma)
            # Wichtig: Diese gehen in die Copula!
            z = garch_res.resid / garch_res.conditional_volatility
            
            # Transformation in den U-Raum [0,1] via Skewed-t CDF
            # Parameter: nu (df) und eta (skew)
            p_dist = garch_res.params[-2:] # nu und eta
            u_data[:, i] = garch_res.model.distribution.cdf(z, p_dist)
            
            # Prognose für die Volatilität (T+1)
            garch_forecast = garch_res.forecast(horizon=1)
            next_sigma = np.sqrt(garch_forecast.variance.values[-1, 0])
            
            # Speichern für Simulation
            forecasts_mu.append(next_mu)
            forecasts_sigma.append(next_sigma)
            dist_params.append(p_dist)
            model_distributions.append(garch_res.model.distribution)

        # --- SCHRITT 3: Abhängigkeit (Gaussian Copula) ---
        g_cop = GaussianCopula(dim=n_assets)
        g_cop.fit(u_data)
        sim_u = g_cop.random(n_sim)
        
        # --- SCHRITT 4: Simulation der Portfolio-Returns ---
        sim_returns = np.zeros((n_sim, n_assets))
        for i in range(n_assets):
            # Inverse CDF (PPF) der Skewed-t
            z_sim = model_distributions[i].ppf(sim_u[:, i], dist_params[i])
            # Rekonstruktion: Return = ARMA_mu + z * GARCH_sigma
            sim_returns[:, i] = forecasts_mu[i] + z_sim * forecasts_sigma[i]
            
        portfolio_sim = sim_returns.mean(axis=1)
        
        # --- SCHRITT 5: Risiko-Metriken ---
        var_val = np.percentile(portfolio_sim, alpha * 100)
        es_val = portfolio_sim[portfolio_sim <= var_val].mean()
        
        results.append({
            'Date': returns_df.index[end],
            'Actual': returns_df.iloc[end].mean(),
            'VaR': var_val,
            'ES': es_val
        })
        
        if end % 20 == 0:
            print(f"Tag {end} prozessiert...")

    return pd.DataFrame(results).set_index('Date')