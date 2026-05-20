"""
03_complex_vae_50day.py
-----------------------
Extension of script 02. Two new things:

  1. A MORE COMPLEX VAE
     We move from the simple VAE in script 02 (1 hidden layer of 16,
     latent_dim=2, beta=1) toward the EXPOSÉ BASELINE configuration:

                                script 02         this script
                                ---------         -----------
       hidden layers              1                 2
       hidden units               16                64
       latent dimension           2                 3
       beta                       1.0               2.0
       activation                 tanh              tanh
       batch size                 full-batch        32       <-- user requested
       early stopping             no                yes

     This matches the exposé's baseline configuration (page 2) EXCEPT
     for the batch size, which we set to 32 as the user requested
     (the exposé itself uses 64).

  2. A 50-DAY ROLLING FORECAST LOOP
     Script 02 computed ONE one-day-ahead forecast. This script computes
     50 of them -- one for each of the most recent 50 trading days.

     For each forecast day t:
         estimation_window = days [t-500, t-1]   <-- never includes day t
         refit GARCH marginals on that window
         use the VAE (trained once at the start)
         sample 10,000 synthetic residuals
         transform to returns, compute portfolio 99% VaR and 97.5% ES

     The result is a TIME SERIES of 50 daily VaR forecasts.

WHY ONLY ONE VAE TRAINING?
--------------------------
The exposé prescribes ANNUAL retraining of the VAE (page 2). One
trading year is ~252 days. Our forecast horizon is 50 days. So one
training pass is what the exposé would do here -- we train on the first
window, then hold the dependence model fixed for all 50 forecasts. The
GARCH marginals, by contrast, are refit EVERY day (also per exposé).

WHY ROLL THE WINDOW INSTEAD OF FORECASTING 50 DAYS AHEAD AT ONCE?
-----------------------------------------------------------------
Because that's not how risk forecasting works in practice. A bank
computes its VaR every morning using yesterday's data. The thesis (and
this script) mirrors that: a *one-day-ahead* forecast, but produced
once per day for 50 consecutive days. So you get a TIME SERIES of
one-day forecasts, not a multi-step-ahead forecast for day 50.

WHAT TO LOOK AT IN THE OUTPUT
-----------------------------
  * Per-day printed table of VaR and ES.
  * Summary stats (mean VaR, exceedance count: how often the realized
    portfolio return broke through the VaR threshold).
  * A plot saved to vae_50day_var.png showing the VaR/ES series with
    realized portfolio returns overlaid -- this is the standard
    visualization you would see in a Basel III backtest report.
  * A CSV saved to vae_50day_var.csv.

Run:
    python 03_complex_vae_50day.py
"""
import copy
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from arch import arch_model
import matplotlib
matplotlib.use("Agg")            # no GUI needed; we save to file
import matplotlib.pyplot as plt

# Silence the harmless "rescale to percent" warning from arch.
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# Configuration
# ============================================================
DATA           = "/mnt/user-data/uploads/qrm2025_returns.csv"
WINDOW_SIZE    = 500               # exposé: 500-day rolling window
N_FORECAST_DAYS = 50               # produce 50 daily forecasts
N_SIM          = 10_000            # exposé: 10,000 Monte-Carlo samples
VAR_LEVEL      = 0.99              # Basel III 99% VaR
ES_LEVEL       = 0.975             # Basel III 97.5% ES

# VAE hyperparameters (the exposé baseline, with batch size 32 as
# explicitly requested by the user).
LATENT_DIM     = 3
HIDDEN_DIMS    = (64, 64)
BETA           = 2.0
BATCH_SIZE     = 32                # <-- USER REQUESTED
MAX_EPOCHS     = 200
PATIENCE       = 20                # early-stopping patience
LEARNING_RATE  = 1e-3

# Reporting
PORTFOLIO_VALUE = 100_000
HOLDING_DAYS    = 10


# ============================================================
# Helper 1: fit AR(1)-GARCH(1,1)-skewed-t marginals on a window
# ============================================================
# Identical to the corresponding block in script 02. We package it
# as a function so the rolling loop can call it daily.
def fit_marginals(R_window):
    """
    R_window: (T, d) array of daily geometric returns.

    Returns
    -------
    mu_next     : (d,) one-step-ahead conditional means
    sigma_next  : (d,) one-step-ahead conditional std devs
    Z           : (T-1, d) matrix of in-sample standardized residuals
                  (T-1 because AR(1) drops the first row -- no lag).
    """
    d = R_window.shape[1]
    mu_next, sigma_next, Z_cols = [], [], []
    for j in range(d):
        am  = arch_model(R_window[:, j] * 100.0,
                         mean="ARX", lags=1,
                         vol="GARCH", p=1, q=1, dist="skewt")
        res = am.fit(disp="off", show_warning=False)
        fc  = res.forecast(horizon=1, reindex=False)
        mu_next.append(    float(fc.mean.iloc[-1, 0])               / 100.0)
        sigma_next.append( float(np.sqrt(fc.variance.iloc[-1, 0]))  / 100.0)
        # in-sample standardized residuals
        resid = np.asarray(res.resid)
        sigma = np.asarray(res.conditional_volatility)
        mask  = ~(np.isnan(resid) | np.isnan(sigma))
        Z_cols.append(resid[mask] / sigma[mask])
    T_min = min(len(z) for z in Z_cols)
    Z     = np.column_stack([z[-T_min:] for z in Z_cols])
    return np.array(mu_next), np.array(sigma_next), Z


# ============================================================
# Helper 2: the more complex VAE
# ============================================================
class ComplexVAE(nn.Module):
    """
    Exposé baseline architecture (page 2): 2 hidden x 64 tanh, latent=3,
    beta=2, MSE reconstruction. Same probabilistic machinery as the
    simple VAE in script 02 -- just a deeper / wider network.
    """
    def __init__(self, input_dim,
                 hidden_dims=HIDDEN_DIMS,
                 latent_dim=LATENT_DIM,
                 beta=BETA):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        # Encoder trunk: input -> 64 -> 64
        self.enc_trunk = nn.Sequential(
            nn.Linear(input_dim,     hidden_dims[0]), nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Tanh(),
        )
        # Two heads for mu and log_var (same idea as in script 02)
        self.enc_mu      = nn.Linear(hidden_dims[1], latent_dim)
        self.enc_log_var = nn.Linear(hidden_dims[1], latent_dim)
        # Decoder mirrors encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,     hidden_dims[1]), nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def encode(self, x):
        h = self.enc_trunk(x)
        # clamp log_var to prevent numerical explosions early in training
        return self.enc_mu(h), self.enc_log_var(h).clamp(-10, 10)

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


def vae_loss(x, x_hat, mu, log_var, beta):
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + beta * kl


# ============================================================
# Helper 3: training loop with early stopping and BATCH_SIZE=32
# ============================================================
def train_vae(model, Z, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
              patience=PATIENCE, lr=LEARNING_RATE, verbose=True):
    """
    Train the VAE with mini-batch SGD and early stopping.

    Crucially: chronological 85/15 train/val split BEFORE shuffling
    into batches (never let future information leak into training).
    Within the training portion we DO shuffle batches each epoch.
    """
    # chronological split
    n_val = max(1, int(0.15 * Z.shape[0]))
    Z_tr  = torch.from_numpy(Z[:-n_val]).float()
    Z_va  = torch.from_numpy(Z[-n_val:]).float()

    # mini-batch loader (shuffle=True is fine here, within train set)
    loader = DataLoader(TensorDataset(Z_tr),
                        batch_size=batch_size, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    best_val   = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad        = 0

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            x_hat, mu, lv = model(xb)
            loss = vae_loss(xb, x_hat, mu, lv, beta=model.beta)
            loss.backward()
            opt.step()
        # ---- validate ----
        model.eval()
        with torch.no_grad():
            x_hat_v, mu_v, lv_v = model(Z_va)
            val = vae_loss(Z_va, x_hat_v, mu_v, lv_v, beta=model.beta).item()
        # ---- early stopping ----
        if val < best_val - 1e-7:
            best_val   = val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best epoch {best_epoch}, val={best_val:.4f})")
                break
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:>3}   val_loss = {val:.4f}")

    model.load_state_dict(best_state)
    return best_epoch, best_val


# ============================================================
# Helper 4: VaR / ES from simulated residuals
# ============================================================
def compute_var_es(Z_sim, mu_next, sigma_next, weights):
    """
    Given simulated standardized residuals and the GARCH one-step
    forecasts, produce the portfolio 99% VaR and 97.5% ES.
    """
    # residual -> per-asset return forecast
    R_sim = mu_next + Z_sim * sigma_next            # (N_SIM, d)
    # aggregate to portfolio
    r_P   = R_sim @ weights                          # (N_SIM,)
    var   = -float(np.quantile(r_P, 1 - VAR_LEVEL))
    cut   = -float(np.quantile(r_P, 1 - ES_LEVEL))
    es    = -float(r_P[r_P <= -cut].mean())
    return var, es


# ============================================================
# Main pipeline
# ============================================================
def main():
    # ---- 1. Data ----
    df = pd.read_csv(DATA).iloc[1:].reset_index(drop=True)   # drop base-zero row
    R_all = df.values.astype(np.float64)
    T, d = R_all.shape
    print(f"Loaded {T} daily return vectors, {d} assets: {df.columns.tolist()}\n")

    weights = np.full(d, 1.0 / d)
    print(f"Equal weights portfolio:   w = {weights.round(3)}\n")

    # ---- 2. Pick the 50 forecast days = the most recent 50 days in the data.
    #        For day t we use the window R_all[t - WINDOW_SIZE : t].
    forecast_days = list(range(T - N_FORECAST_DAYS, T))   # length = 50
    first_day     = forecast_days[0]
    print(f"Will forecast days {first_day} ... {forecast_days[-1]} "
          f"(the most recent {N_FORECAST_DAYS} days)\n")

    # ---- 3. ONE VAE TRAINING on the very first window.
    print("=" * 64)
    print(f"VAE training (ONCE, on the window ending day {first_day - 1})")
    print("=" * 64)
    print(f"  architecture: 2 x {HIDDEN_DIMS[0]} tanh   "
          f"latent_dim = {LATENT_DIM}   beta = {BETA}")
    print(f"  batch_size  = {BATCH_SIZE}   "
          f"max_epochs = {MAX_EPOCHS}   patience = {PATIENCE}")
    print()

    R_train = R_all[first_day - WINDOW_SIZE : first_day]
    t0 = time.time()
    mu_next, sigma_next, Z_train = fit_marginals(R_train)
    print(f"  initial GARCH fit:  {time.time() - t0:.1f}s")
    print(f"  Z_train shape:      {Z_train.shape}  "
          f"(std per col ≈ {Z_train.std(axis=0).round(2)})")

    vae = ComplexVAE(input_dim=d)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE parameters:     {n_params}")
    t0 = time.time()
    best_epoch, best_val = train_vae(vae, Z_train, verbose=True)
    print(f"  training time:      {time.time() - t0:.1f}s")
    print(f"  best validation:    epoch {best_epoch}, loss = {best_val:.4f}\n")

    # ---- 4. ROLLING FORECAST LOOP -----------------------------------
    print("=" * 64)
    print(f"Rolling one-step forecast for {N_FORECAST_DAYS} days")
    print("=" * 64)
    print(f"  (refit GARCH every day; VAE stays trained from above)\n")

    records = []
    t_loop_start = time.time()

    for k, t in enumerate(forecast_days):
        # window: days BEFORE day t (so we never peek)
        R_win = R_all[t - WINDOW_SIZE : t]

        # 1) refit GARCH marginals on the rolling window
        mu_next, sigma_next, _ = fit_marginals(R_win)
        # NOTE: we do NOT retrain the VAE here. The exposé prescribes
        # ANNUAL retraining (1 year ≈ 252 days). 50 days << 252 days, so
        # one training pass is what the exposé would do. The GARCH
        # marginals are refit daily, which IS what the exposé says.

        # 2) sample N_SIM residuals from the trained VAE
        Z_sim = vae.sample(N_SIM).numpy()

        # 3) compute the portfolio VaR/ES
        var, es = compute_var_es(Z_sim, mu_next, sigma_next, weights)

        # 4) record. We also store the REALIZED portfolio return on
        #    day t (= R_all[t] @ weights). The realized return is what
        #    *actually* happened that day; comparing it to VaR is the
        #    basis of every regulatory backtest.
        realized = float(R_all[t] @ weights)
        records.append({
            "day_idx":  t,
            "VaR_99":   var,
            "ES_975":   es,
            "realized": realized,
            "exceeded": realized < -var,    # True = realized loss > VaR
        })

        if (k + 1) % 10 == 0 or k == len(forecast_days) - 1:
            elapsed = time.time() - t_loop_start
            print(f"  day {k+1:>2}/{N_FORECAST_DAYS}  t={t:>4}  "
                  f"VaR={var*100:5.3f}%  ES={es*100:5.3f}%  "
                  f"realized={realized*100:+6.3f}%  "
                  f"exceeded={records[-1]['exceeded']!s:>5}   "
                  f"[elapsed {elapsed:5.1f}s]")

    print()
    fc = pd.DataFrame(records)

    # ---- 5. SUMMARY ------------------------------------------------
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Mean VaR(99%)  over the 50 days: {fc['VaR_99'].mean()*100:.3f}%")
    print(f"  Mean ES(97.5%) over the 50 days: {fc['ES_975'].mean()*100:.3f}%")
    print(f"  Min  VaR(99%) : {fc['VaR_99'].min()*100:.3f}%   "
          f"Max VaR(99%) : {fc['VaR_99'].max()*100:.3f}%")
    print()
    n_exc = int(fc["exceeded"].sum())
    exp_exc = (1 - VAR_LEVEL) * N_FORECAST_DAYS
    print(f"  VaR exceedances: {n_exc} out of {N_FORECAST_DAYS} days")
    print(f"  Expected at 99% confidence: {exp_exc:.1f}")
    print()
    if n_exc <= exp_exc + 1:
        print(f"  -> The number of exceedances is consistent with a")
        print(f"     well-calibrated 99% VaR model on this sample.")
    elif n_exc > exp_exc + 3:
        print(f"  -> MANY MORE exceedances than expected. The VaR is")
        print(f"     UNDERESTIMATING risk (= the variance-shrinkage")
        print(f"     pathology we discussed in script 02's explanation).")
    else:
        print(f"  -> Slightly more exceedances than expected; not yet")
        print(f"     a formal rejection but worth watching. The thesis")
        print(f"     uses the Christoffersen-Pelletier duration test")
        print(f"     to make this judgment rigorously.")

    # Dollar terms for context
    var_dol  = fc["VaR_99"].mean() * PORTFOLIO_VALUE * np.sqrt(HOLDING_DAYS)
    print()
    print(f"  In dollar terms (portfolio ${PORTFOLIO_VALUE:,}, 10-day sqrt-t):")
    print(f"    mean 10-day VaR   ≈ ${var_dol:,.0f}")

    # ---- 6. SAVE OUTPUTS -------------------------------------------
    csv_path = "vae_50day_var.csv"
    fc.to_csv(csv_path, index=False)
    print(f"\n  saved per-day table to:  {csv_path}")

    # ---- 7. PLOT ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    days = np.arange(len(fc))
    # negative VaR/ES so they sit on the LOSS side (below zero)
    ax.plot(days, -fc["VaR_99"] * 100, label="-(99% VaR)",
            color="tab:red",   lw=1.4)
    ax.plot(days, -fc["ES_975"] * 100, label="-(97.5% ES)",
            color="tab:orange", lw=1.0, alpha=0.8)
    ax.bar( days, fc["realized"] * 100, label="realized portfolio return",
            width=0.6, color="tab:blue", alpha=0.55)
    # mark VaR exceedances
    exc_idx = days[fc["exceeded"].values]
    if len(exc_idx):
        ax.scatter(exc_idx, fc.loc[fc["exceeded"], "realized"] * 100,
                   color="black", s=40, zorder=5, label="VaR breach")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("day index within the 50-day forecast window")
    ax.set_ylabel("return / -loss (%)")
    ax.set_title(f"ComplexVAE one-day-ahead VaR over {N_FORECAST_DAYS} days  "
                 f"(beta={BETA}, latent={LATENT_DIM}, batch={BATCH_SIZE})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = "vae_50day_var.png"
    fig.savefig(plot_path, dpi=130)
    print(f"  saved plot to:           {plot_path}")

    # ---- 8. Closing note --------------------------------------------
    print()
    print("=" * 64)
    print("WHAT THIS SCRIPT IS THE INNER LOOP OF")
    print("=" * 64)
    print("This 50-day rolling loop is exactly the structure of the")
    print("thesis Part 2 analysis. To scale it up you would:")
    print("  * Run for ALL forecast days, not just the last 50.")
    print("  * Retrain the VAE annually (every ~252 days).")
    print("  * Run MULTIPLE VAE configurations side-by-side (L1-L7")
    print("    sensitivity study), producing a cross-model MAD per day.")
    print("  * Add backtests (Christoffersen-Pelletier, Nolde-Ziegel)")
    print("    that filter out misspecified models on each day.")
    print("  * Compare against parametric copula benchmarks.")
    print("Everything else is in place. This is the working core.")


if __name__ == "__main__":
    main()
