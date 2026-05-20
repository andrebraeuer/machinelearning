"""
02_simple_vae.py
----------------
A SIMPLE Variational Autoencoder (VAE) following the exposé pipeline.

WHAT IS A VAE?
--------------
A VAE is a kind of neural network that LEARNS A DISTRIBUTION over a
high-dimensional space. After training, you can SAMPLE new data points
that look like the training data.

That's the key difference from the MLP in script 01:

  MLP (script 01):  takes x, outputs a single prediction y_hat.
                    Answers "what is the most likely y for this x?"

  VAE (this script): takes x, learns the distribution p(x). After
                    training you can ask "give me 10,000 NEW x's that
                    look like the training data".

For multivariate risk forecasting (the exposé's goal) we need the
second one. We can't compute "the 99th percentile loss" without first
having a way to generate many plausible future return vectors.

THE EXPOSÉ PIPELINE (page 1, "General Framework")
-------------------------------------------------
We do not put raw returns directly into the VAE. We follow the standard
Copula-GARCH two-step approach (Sklar's theorem):

  Step 1.  Fit a univariate ARMA-GARCH model to each asset's return
           series. This explains the time-varying volatility.

  Step 2.  Extract STANDARDIZED RESIDUALS z_hat = (r - mu) / sigma.
           These should look like i.i.d. shocks with zero mean and unit
           variance per asset, BUT they still have CROSS-SECTIONAL
           dependence between assets.

  Step 3.  The DEPENDENCE MODEL (the VAE here) takes those standardized
           residuals as input. Its only job: learn the joint distribution
           of the 5-dim residual vector, so we can sample from it.

  Step 4.  At forecast time: sample 10,000 synthetic residual vectors
           from the VAE, transform back to return units using the
           GARCH-forecasted mu and sigma for tomorrow, aggregate to the
           portfolio, take the empirical 1st percentile -> 99% VaR.

This file implements all four steps for a single 500-day window.

THIS VAE HAS THE FOLLOWING HYPERPARAMETERS
-------------------------------------------
  encoder:    5  ->  16 (hidden)  ->  (mu in R^2, log_var in R^2)
  decoder:    2  ->  16 (hidden)  ->  5
  activation: tanh
  beta:       1.0  (the textbook VAE; the exposé baseline uses 2.0
                    but we start with 1.0 here for clarity)
  latent_dim: 2    (small enough to be easily described; the exposé
                    baseline uses 3)

The exposé's "baseline configuration" (page 2) is the more elaborate
version: 2x64 hidden, latent_dim=3, beta=2. We are deliberately simpler
here -- this file is for understanding, not for the final thesis.

Run:
    python 02_simple_vae.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import arch_model

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = "/mnt/user-data/uploads/qrm2025_returns.csv"

# ============================================================
# Step 1.  Load returns. Use the most recent 500-day window
#          (the exposé's prescribed rolling-window length).
# ============================================================
df = pd.read_csv(DATA).iloc[1:].reset_index(drop=True)
R_all = df.values.astype(np.float64)
print(f"Total sample: {R_all.shape[0]} daily returns, "
      f"{R_all.shape[1]} assets ({df.columns.tolist()})")

WINDOW = 500
R = R_all[-WINDOW:]                       # the estimation window
print(f"Estimation window: most recent {WINDOW} days\n")

# ============================================================
# Step 2.  Fit AR(1)-GARCH(1,1) with skewed-t innovations to
#          each asset separately.
# ============================================================
# Why "scale to percent"? The `arch` library is more numerically stable
# when returns are around magnitude 1 instead of 0.01.
# Why AR(1) and not ARMA(1,1)? Python's `arch` library does not implement
# ARMA mean; AR(1) is the closest available approximation. The MA(1) term
# in daily returns is typically not statistically distinguishable from
# zero anyway, so this is harmless in practice but flagged here.
print("Step 2.  Fitting per-asset AR(1)-GARCH(1,1)-skewed-t marginals ...")
fits        = []        # statsmodels result objects
mu_next     = []        # one-step-ahead conditional mean per asset
sigma_next  = []        # one-step-ahead conditional std per asset
Z_residuals = []        # standardized residuals per asset (list of 1D arrays)

for j in range(R.shape[1]):
    am  = arch_model(R[:, j] * 100.0,            # rescale for stability
                     mean="ARX", lags=1,
                     vol="GARCH", p=1, q=1,
                     dist="skewt")
    res = am.fit(disp="off", show_warning=False)
    fits.append(res)

    # One-step-ahead forecast for tomorrow
    fc = res.forecast(horizon=1, reindex=False)
    mu_next.append(    float(fc.mean.iloc[-1, 0])       / 100.0)
    sigma_next.append( float(np.sqrt(fc.variance.iloc[-1, 0])) / 100.0)

    # Standardized in-sample residuals z_hat_t = (r - mu_t)/sigma_t.
    # AR(1) leaves a NaN in the first row (no lag to use); drop it.
    resid = np.asarray(res.resid)
    sigma = np.asarray(res.conditional_volatility)
    mask  = ~(np.isnan(resid) | np.isnan(sigma))
    Z_residuals.append(resid[mask] / sigma[mask])

mu_next     = np.array(mu_next)                   # shape (5,)
sigma_next  = np.array(sigma_next)                # shape (5,)
# Align column lengths and stack into a (T-1, 5) matrix.
T_min = min(len(z) for z in Z_residuals)
Z     = np.column_stack([z[-T_min:] for z in Z_residuals])
print(f"  standardized residuals Z: shape {Z.shape}")
print(f"  per-column mean ≈ {Z.mean(axis=0).round(2)}  "
      f"(target: 0)")
print(f"  per-column std  ≈ {Z.std(axis=0).round(2)}  "
      f"(target: 1)\n")

# ============================================================
# Step 3.  Build the VAE.
# ============================================================
# Encoder produces TWO outputs from x: mu(x) and log_var(x).
# Decoder takes a 2-dim latent z and reconstructs a 5-dim vector.
#
# Why log_var, not var? A variance must be positive, but a neural network
# output is unconstrained. We let the network output log_var (any real
# number), then recover sigma = exp(0.5 * log_var) > 0. Standard trick.
INPUT_DIM  = Z.shape[1]    # 5 (the number of assets)
HIDDEN_DIM = 16
LATENT_DIM = 2
BETA       = 1.0

class SimpleVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder trunk: x -> hidden
        self.enc_trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )
        # Two heads off the hidden layer: mu and log_var
        self.enc_mu      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.enc_log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        # Decoder: z -> hidden -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.enc_mu(h), self.enc_log_var(h)

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick.

        We want to sample z ~ N(mu, sigma^2). Naively writing
            z = torch.normal(mu, sigma)
        breaks autograd: there is no gradient through a random draw.
        Instead we write
            z = mu + sigma * eps,   eps ~ N(0, 1)
        which IS differentiable in mu and sigma because the randomness
        is now external (in eps).
        """
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
        """
        Generate n synthetic residual vectors. The exposé's recipe:
        draw z ~ N(0, I), pass through the decoder.
        """
        z = torch.randn(n, LATENT_DIM)
        return self.decode(z)


model = SimpleVAE()
n_params = sum(p.numel() for p in model.parameters())
print(f"Step 3.  Built VAE  (input_dim={INPUT_DIM}, hidden={HIDDEN_DIM}, "
      f"latent_dim={LATENT_DIM}, beta={BETA})")
print(f"         trainable parameters: {n_params}\n")


# ============================================================
# Step 4.  Define the loss = reconstruction + beta * KL.
# ============================================================
# The "evidence lower bound" (ELBO) for a VAE consists of two parts:
#
#   reconstruction term:  encourage the decoded x_hat to be close to x.
#                         With a Gaussian likelihood and fixed variance,
#                         this reduces to mean squared error.
#
#   KL divergence term:   encourage q(z|x) = N(mu, sigma^2) to look
#                         like the standard normal prior N(0, I).
#                         For diagonal Gaussians there is a closed form:
#       KL = 0.5 * sum( mu^2 + sigma^2 - log(sigma^2) - 1 )
#
# The beta coefficient weights these. beta = 1 is the textbook VAE.
# beta > 1 makes the latent "more Gaussian" at the cost of reconstruction
# fidelity. The exposé baseline uses beta = 2.
def vae_loss(x, x_hat, mu, log_var, beta=BETA):
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + beta * kl, recon, kl


# ============================================================
# Step 5.  Train.
# ============================================================
Z_tensor = torch.from_numpy(Z).float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Step 5.  Training (full-batch, 500 epochs):")
for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    x_hat, mu, log_var = model(Z_tensor)
    loss, recon, kl    = vae_loss(Z_tensor, x_hat, mu, log_var)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"  epoch {epoch:>3}   total = {loss.item():.4f}   "
              f"recon = {recon.item():.4f}   KL = {kl.item():.4f}")
print()


# ============================================================
# Step 6.  Sample 10,000 synthetic residual vectors and check
#          whether they resemble the real data.
# ============================================================
N_SIM = 10_000
model.eval()
Z_sim = model.sample(N_SIM).numpy()

print("Step 6.  Diagnostics for the SAMPLED residuals")
print(f"  empirical mean:     {Z.mean(axis=0).round(2)}")
print(f"  simulated mean:     {Z_sim.mean(axis=0).round(2)}")
print(f"  empirical std:      {Z.std(axis=0).round(2)}")
print(f"  simulated std:      {Z_sim.std(axis=0).round(2)}   "
      f"<-- watch this")
print()
print("  empirical correlation matrix:")
print(np.corrcoef(Z.T).round(2))
print("  simulated correlation matrix:")
print(np.corrcoef(Z_sim.T).round(2))
print()

# Tail coverage. A well-calibrated generative model should produce about
# 1% of samples below the empirical 1st percentile and 1% above the 99th
# percentile, FOR EACH ASSET separately.
print("  Tail-coverage check (each row is one asset).")
print("    Target: ~1.0% in each tail.")
for j, col_name in enumerate(df.columns):
    p01 = np.quantile(Z[:, j], 0.01)
    p99 = np.quantile(Z[:, j], 0.99)
    lo  = float(np.mean(Z_sim[:, j] < p01)) * 100
    hi  = float(np.mean(Z_sim[:, j] > p99)) * 100
    print(f"      {col_name:>22}:  below 1%: {lo:5.2f}%   above 99%: {hi:5.2f}%")
print()


# ============================================================
# Step 7.  The PAYOFF: compute one-step VaR and ES.
# ============================================================
# Convert each simulated residual back to a return-scale forecast:
#     r_{i, T+1} = mu_{i, T+1} + sigma_{i, T+1} * z_{i, T+1}
# Aggregate to the portfolio (equal weights):
#     r_P = sum_i w_i * r_{i, T+1}
# Then:
#     99% VaR    = -quantile(r_P, 0.01)
#     97.5% ES   = -mean(r_P | r_P <= -quantile(r_P, 0.025))
R_sim   = mu_next + Z_sim * sigma_next       # shape (N_SIM, 5)
weights = np.full(R.shape[1], 1.0 / R.shape[1])
r_P     = R_sim @ weights                     # shape (N_SIM,)

VaR_level = 0.99      # Basel III
ES_level  = 0.975     # Basel III

VaR = -float(np.quantile(r_P, 1 - VaR_level))
cut = -float(np.quantile(r_P, 1 - ES_level))
ES  = -float(r_P[r_P <= -cut].mean())

# Convert to dollar terms using the standard sqrt-of-time rule.
PORTFOLIO_VALUE = 100_000
HOLDING_DAYS    = 10
scale = PORTFOLIO_VALUE * np.sqrt(HOLDING_DAYS)

print("Step 7.  One-step VaR and ES forecast")
print(f"  99% VaR  (1-day):   {VaR*100:.3f}%   "
      f"==>  ${VaR*scale:,.0f}  (10-day, sqrt-t scaled)")
print(f"  97.5% ES (1-day):   {ES *100:.3f}%   "
      f"==>  ${ES *scale:,.0f}")
print()


# ============================================================
# Step 8.  What you should take away.
# ============================================================
print("=" * 64)
print("WHAT TO TAKE AWAY")
print("=" * 64)
print()
print("1. The VAE actually GENERATES new data. Unlike the MLP in")
print("   script 01, this network does not predict a single number;")
print("   it produces 10,000 plausible 5-dim residual vectors that")
print("   we can use to compute portfolio quantiles.")
print()
print("2. The generated samples are imperfect. Look at the tail-")
print("   coverage check above: the VAE typically produces FEWER")
print("   samples in the extreme tails than reality. This is called")
print("   'variance shrinkage' and is a well-known pathology of MSE-")
print("   loss VAEs. It causes the VaR forecast to UNDERESTIMATE the")
print("   true 99% loss -- exactly the risk you most want to measure.")
print()
print("3. The thesis (see expose_bsp_Fritzsch.pdf) asks how badly this")
print("   matters in practice, and how to fix it by tuning beta, the")
print("   latent dimension, and the loss function. This script is the")
print("   smallest worked example of the framework the thesis builds")
print("   on.")
print()
print("Now read explanations.md for the full story.")
