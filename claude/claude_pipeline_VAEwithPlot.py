"""
VAE-GARCH Value-at-Risk Forecasting Pipeline
============================================

End-to-end walk-forward pipeline for one-step-ahead portfolio VaR forecasting:

    returns  -->  AR(1)-GARCH(1,1)-t (per asset)
             -->  standardized residuals
             -->  VAE (refit every K steps)
             -->  Monte-Carlo sampling of innovations
             -->  simulated portfolio returns
             -->  VaR at chosen confidence levels

The VAE refit cadence is fully configurable (`vae_refit_every`):
    1   -> refit every trading day
    10  -> refit every 10 days
    252 -> refit once per trading year
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Hyperparameter defaults (override at call site or in your config module)
# ---------------------------------------------------------------------------
HIDDEN_DIMS    = (64, 64)
LATENT_DIM     = 3
BETA           = 2.0
BATCH_SIZE     = 64
MAX_EPOCHS     = 200
LEARNING_RATE  = 1e-3
WINDOW         = 1000
N_FORECAST     = 500
SCALE          = 100
TRADING_DAYS_PER_YEAR = 252


# ===========================================================================
# 1. AR(1)-GARCH(1,1) per asset
# ===========================================================================
def fit_ar_garch(returns, garch_order=(1, 1), lags=1, dist="t",
                 horizon=1, scale=100):
    """
    Fit AR(1)-GARCH(1,1) with Student-t innovations.

    Note: AR(1) leaves a NaN in the first row of the standardized residuals
    because there is no lag for the first observation.

    Returns
    -------
    dict with:
        residuals_std : standardized residuals (innovations)
        mu            : one-step-ahead mean forecast (original scale)
        sigma         : one-step-ahead vol forecast (original scale)
        nu            : Student-t degrees of freedom (or None)
    """
    from arch import arch_model

    p, q = garch_order
    res = arch_model(returns * scale, mean="AR", lags=lags,
                     vol="Garch", p=p, q=q, dist=dist
                     ).fit(disp="off", show_warning=False)

    nu            = res.params["nu"] if dist == "t" else None
    residuals_std = res.std_resid

    fc    = res.forecast(horizon=horizon)
    sigma = np.sqrt(fc.variance.values[-1, 0]) / scale
    mu    = fc.mean.iloc[-1, 0] / scale

    return {"residuals_std": residuals_std,
            "mu": mu, "sigma": sigma, "nu": nu}


def build_residuals_matrix(returns_window, scale=SCALE):
    """
    Fit AR(1)-GARCH(1,1)-t per asset on the rolling window.

    Returns
    -------
    resid : (T-1, n_assets) standardized residuals (NaN first row dropped)
    mu    : (n_assets,)     one-step-ahead mean forecasts
    sigma : (n_assets,)     one-step-ahead vol forecasts
    fits  : dict            per-asset fit dicts (for inspection)
    """
    fits = {c: fit_ar_garch(returns_window[c], garch_order=(1, 1),
                            dist="t", horizon=1, scale=scale)
            for c in returns_window.columns}

    resid = np.array([f["residuals_std"] for f in fits.values()]).T
    mask  = ~np.isnan(resid).any(axis=1)
    resid = resid[mask]

    mu    = np.array([f["mu"]    for f in fits.values()])
    sigma = np.array([f["sigma"] for f in fits.values()])
    return resid, mu, sigma, fits


# ===========================================================================
# 2. VAE model
# ===========================================================================
class VAE_model_complex(nn.Module):
    """
    Baseline VAE architecture:
        - 2 hidden layers x 64 with Tanh activations
        - latent dim = 3
        - beta-VAE loss (beta = 2)
        - MSE reconstruction
    """

    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS,
                 latent_dim=LATENT_DIM, beta=BETA):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        # Encoder: x -> hidden -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
        )
        # Latent heads: hidden -> (mu, logvar)
        self.encoder_mu     = nn.Linear(hidden_dims[1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder mirrors encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        z = mu + sigma * eps,  eps ~ N(0, I)
        Reparameterization trick: keeps the random draw outside the
        differentiable path so gradients flow through mu and sigma.
        """
        sigma = torch.exp(0.5 * logvar)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sampling(self, n):
        """
        Generate n synthetic residual vectors:
        z ~ N(0, I), then x_hat = decoder(z).
        Uses self.latent_dim (robust if multiple instances differ).
        """
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


# ===========================================================================
# 3. VAE loss and training
# ===========================================================================
def vae_loss_components(x, x_hat, mu, lv, beta=1.0):
    """
    Return (total, recon, kl) so we can monitor them separately.
    Posterior collapse shows up as kl -> 0 while recon keeps improving.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    total = recon + beta * kl
    return total, recon, kl


def VAE_train(model, z, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
              lr=LEARNING_RATE, val_frac=0.15, verbose=True):
    """
    Train the VAE with mini-batch SGD.

    Chronological train/val split BEFORE shuffling into batches
    (no future information leaks into training). Within the train
    portion we DO shuffle batches each epoch.

    Logs reconstruction and KL separately on both train and val so
    posterior collapse is visible.

    Parameters
    ----------
    val_frac : float in [0, 1)
        Fraction of the (chronologically last) data used for validation.
        Set to 0.0 to train on all data without a validation split.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    n_val = int(val_frac * z.shape[0])
    if val_frac > 0.0:
        n_val = max(1, n_val)
        z_tr  = torch.from_numpy(z[:-n_val]).float()
        z_va  = torch.from_numpy(z[-n_val:]).float()
    else:
        z_tr  = torch.from_numpy(z).float()
        z_va  = None

    loader = DataLoader(TensorDataset(z_tr),
                        batch_size=batch_size, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_total": [], "train_recon": [], "train_kl": []}
    if z_va is not None:
        history.update({"val_total": [], "val_recon": [], "val_kl": []})

    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        sum_tot = sum_rec = sum_kl = 0.0
        n_seen  = 0
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, mu, lv = model(xb)
            total, recon, kl = vae_loss_components(xb, x_hat, mu, lv,
                                                   beta=model.beta)
            total.backward()
            opt.step()

            bs = xb.size(0)
            sum_tot += total.item() * bs
            sum_rec += recon.item() * bs
            sum_kl  += kl.item()    * bs
            n_seen  += bs

        tr_tot = sum_tot / n_seen
        tr_rec = sum_rec / n_seen
        tr_kl  = sum_kl  / n_seen
        history["train_total"].append(tr_tot)
        history["train_recon"].append(tr_rec)
        history["train_kl"].append(tr_kl)

        # ---- Validate ----
        if z_va is not None:
            model.eval()
            with torch.no_grad():
                x_hat_v, mu_v, lv_v = model(z_va)
                va_tot, va_rec, va_kl = vae_loss_components(
                    z_va, x_hat_v, mu_v, lv_v, beta=model.beta)
                va_tot = va_tot.item()
                va_rec = va_rec.item()
                va_kl  = va_kl.item()
            history["val_total"].append(va_tot)
            history["val_recon"].append(va_rec)
            history["val_kl"].append(va_kl)

            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:>3}   "
                      f"train: tot={tr_tot:.4f} rec={tr_rec:.4f} kl={tr_kl:.4f}   "
                      f"val: tot={va_tot:.4f} rec={va_rec:.4f} kl={va_kl:.4f}")
        else:
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:>3}   "
                      f"train: tot={tr_tot:.4f} rec={tr_rec:.4f} kl={tr_kl:.4f}   "
                      f"(no validation set)")

    return history


# ===========================================================================
# 4. VaR utilities
# ===========================================================================
def sample_from_vae(model, n_samples):
    """
    Sample n_samples synthetic standardized residual vectors from the VAE.
    Uses the model's built-in sampling: z ~ N(0, I), then decode(z).
    """
    x_hat = model.sampling(n_samples)
    return x_hat.cpu().numpy()


def compute_var(z_sim, mu, sigma, weights, alpha=(5, 1)):
    """
    Given simulated standardized residuals and one-step-ahead forecasts,
    compute portfolio VaR at the given percentiles (e.g. (5, 1) -> 95%, 99%).
    """
    r_sim  = mu + sigma * z_sim
    pf_sim = r_sim @ np.asarray(weights, dtype=float)
    return tuple(np.percentile(pf_sim, a) for a in alpha)


# ===========================================================================
# 5. Rolling-window pipeline
# ===========================================================================
def rolling_var_pipeline(returns, weights,
                         window=WINDOW,
                         n_forecast=N_FORECAST,
                         n_sim=10_000,
                         vae_refit_every=TRADING_DAYS_PER_YEAR,
                         vae_factory=None,
                         vae_train_kwargs=None,
                         alpha=(5, 1),
                         scale=SCALE,
                         verbose=True):
    """
    Walk-forward rolling-window VaR forecasting pipeline.

    For each t in 0..n_forecast-1:
        1. Take the rolling window returns[t : t+window].
        2. Fit AR(1)-GARCH(1,1)-t per asset -> residuals, mu, sigma forecasts.
        3. If t % vae_refit_every == 0: train a fresh VAE on current residuals.
           Otherwise: reuse the most recently trained VAE.
        4. Sample n_sim standardized innovations from the VAE.
        5. Reconstruct simulated returns r = mu + sigma * z_sim, build portfolio
           returns, compute VaR at the requested alpha levels.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, shape (T, n_assets). Index = trading days.
    weights : array-like, shape (n_assets,)
        Portfolio weights.
    window : int
        Rolling window length (in trading days) used for GARCH and VAE fitting.
    n_forecast : int
        Number of one-step-ahead VaR forecasts to produce.
    n_sim : int
        Monte Carlo samples per forecast.
    vae_refit_every : int, >= 1
        Refit cadence:
            1   -> refit every day
            10  -> refit every 10 days
            252 -> refit once per trading year (default)
    vae_factory : callable () -> nn.Module
        Returns a freshly initialized VAE. Required.
    vae_train_kwargs : dict
        Keyword arguments forwarded to VAE_train.
    alpha : tuple of percentiles
        VaR levels as percentiles, e.g. (5, 1) for 95% and 99% VaR.
    scale : float
        Return scaling passed to fit_ar_garch.

    Returns
    -------
    dict with:
        var             : DataFrame (forecast_date -> VaR levels)
        mu_path         : ndarray (n_forecast, n_assets)
        sigma_path      : ndarray (n_forecast, n_assets)
        vae_refit_steps : list of t indices where the VAE was refit
        vae_histories   : list of training histories (one per refit)
    """
    if vae_factory is None:
        raise ValueError("vae_factory is required (callable returning a fresh VAE).")
    if not isinstance(vae_refit_every, int) or vae_refit_every < 1:
        raise ValueError(
            f"vae_refit_every must be a positive int, got {vae_refit_every}")

    vae_train_kwargs = vae_train_kwargs or {}
    weights          = np.asarray(weights, dtype=float)
    n_assets         = returns.shape[1]

    forecast_dates = returns.index[window : window + n_forecast]
    var_cols       = [f"VaR_{100 - a}" for a in alpha]
    var_df         = pd.DataFrame(index=forecast_dates,
                                  columns=var_cols, dtype=float)

    mu_path    = np.full((n_forecast, n_assets), np.nan)
    sigma_path = np.full((n_forecast, n_assets), np.nan)

    vae_model       = None
    vae_refit_steps = []
    vae_histories   = []

    # only log every single refit when refits are rare enough to be interesting
    log_each_refit = verbose and vae_refit_every >= 10

    for t in range(n_forecast):
        window_returns = returns.iloc[t : t + window]

        # --- 1. GARCH per asset --------------------------------------------
        resid, mu_vec, sigma_vec, _ = build_residuals_matrix(
            window_returns, scale=scale)
        mu_path[t]    = mu_vec
        sigma_path[t] = sigma_vec

        # --- 2. VAE refit ---------------------------------------------------
        if t % vae_refit_every == 0:
            if log_each_refit:
                print(f"[t={t:>4}] refitting VAE on {resid.shape[0]} residuals")
            vae_model = vae_factory()
            history = VAE_train(vae_model, resid, verbose=False,
                                **vae_train_kwargs)
            vae_refit_steps.append(t)
            vae_histories.append(history)

        # --- 3. Sample innovations from VAE --------------------------------
        z_sim = sample_from_vae(vae_model, n_sim)

        # --- 4. Simulate portfolio returns & 5. compute VaR ----------------
        var_values = compute_var(z_sim, mu_vec, sigma_vec, weights, alpha=alpha)
        for col, v in zip(var_cols, var_values):
            var_df.iloc[t][col] = v

        if verbose and t % 50 == 0:
            print(f"  t={t:>4}  {forecast_dates[t].date()}  "
                  + "  ".join(f"{c}={var_df.iloc[t][c]:.4f}" for c in var_cols))

    if verbose:
        print(f"Done. VAE was refit {len(vae_refit_steps)} times "
              f"(every {vae_refit_every} step(s)).")

    return {
        "var":             var_df,
        "mu_path":         mu_path,
        "sigma_path":      sigma_path,
        "vae_refit_steps": vae_refit_steps,
        "vae_histories":   vae_histories,
    }


# ===========================================================================
# 6. Plotting: VaR vs realised portfolio returns
# ===========================================================================
def realised_portfolio_returns(returns, weights, var_index):
    """
    Build the realised portfolio return series aligned to the VaR forecast
    dates. var_index is the DatetimeIndex from rolling_var_pipeline's var_df,
    which contains the day each one-step-ahead VaR refers to.
    """
    weights = np.asarray(weights, dtype=float)
    pf      = returns.loc[var_index] @ weights
    return pf


def plot_var_vs_returns(var_df, returns, weights,
                        title="VaR forecasts vs realised portfolio returns",
                        figsize=(13, 5),
                        breach_markers=True,
                        ax=None):
    """
    Plot realised portfolio returns against one or more VaR forecast series.

    A breach is a day where the realised return falls *below* the VaR
    (VaR is a negative number for a long portfolio). Breaches are
    highlighted as red dots when `breach_markers=True`.

    Parameters
    ----------
    var_df : pd.DataFrame
        Output of rolling_var_pipeline()['var'].
    returns : pd.DataFrame
        Original asset returns (same as fed to the pipeline).
    weights : array-like
        Portfolio weights used in the pipeline.
    title, figsize, breach_markers, ax : plotting controls.

    Returns
    -------
    (fig, ax, breach_counts)
        breach_counts : dict mapping each VaR column to its breach count
                        and empirical breach rate.
    """
    pf_realised = realised_portfolio_returns(returns, weights, var_df.index)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # realised returns
    ax.plot(pf_realised.index, pf_realised.values,
            color="black", linewidth=0.8, label="Realised portfolio return")

    # VaR lines + breach markers
    breach_counts = {}
    color_cycle   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, col in enumerate(var_df.columns):
        var_series = var_df[col].astype(float)
        ax.plot(var_series.index, var_series.values,
                color=color_cycle[i % len(color_cycle)],
                linewidth=1.2, label=col)

        breaches = pf_realised < var_series
        n_breach = int(breaches.sum())
        n_total  = int(breaches.notna().sum())
        rate     = n_breach / n_total if n_total else float("nan")
        breach_counts[col] = {"breaches": n_breach,
                              "total":    n_total,
                              "rate":     rate}

        if breach_markers and n_breach > 0:
            ax.scatter(pf_realised.index[breaches],
                       pf_realised.values[breaches],
                       color=color_cycle[i % len(color_cycle)],
                       edgecolor="red", s=22, zorder=5,
                       label=f"{col} breach ({n_breach})")

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # summary line
    summary = "  |  ".join(
        f"{col}: {d['breaches']}/{d['total']} breaches "
        f"({d['rate']:.2%}, expected {int(col.split('_')[1]) / 100:.0%} tail)"
        for col, d in breach_counts.items()
    )
    print(summary)

    return fig, ax, breach_counts


# ===========================================================================
# 7. Example usage
# ===========================================================================
if __name__ == "__main__":
    # --- expects `returns` (pd.DataFrame) and `weights` (array) to be defined ---
    #
    # def make_vae():
    #     return VAE_model_complex(
    #         input_dim=returns.shape[1],
    #         hidden_dims=HIDDEN_DIMS,
    #         latent_dim=LATENT_DIM,
    #         beta=BETA,
    #     )
    #
    # results = rolling_var_pipeline(
    #     returns=returns,
    #     weights=weights,
    #     window=WINDOW,
    #     n_forecast=N_FORECAST,
    #     n_sim=10_000,
    #     vae_refit_every=252,         # 1 = daily, 10 = every 10 days, 252 = yearly
    #     vae_factory=make_vae,
    #     vae_train_kwargs=dict(
    #         max_epochs=MAX_EPOCHS,
    #         batch_size=BATCH_SIZE,
    #         lr=LEARNING_RATE,
    #         val_frac=0.15,
    #     ),
    #     alpha=(5, 1),                # 95% and 99% VaR
    #     scale=SCALE,
    # )
    #
    # var_series = results["var"]
    #
    # # --- Backtest plot ---
    # fig, ax, breaches = plot_var_vs_returns(
    #     var_df=results["var"],
    #     returns=returns,
    #     weights=weights,
    # )
    # plt.show()
    pass
