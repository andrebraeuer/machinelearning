# Variational Autoencoders for Multivariate Portfolio Risk (Exposé Implementation)

PyTorch implementation of the master thesis exposé *Variational Autoencoders as Dependence Models for Multivariate Portfolio Risk Forecasting* (Fritzsch et al., 2024 framework).

## Methodology

1. **Marginals (fixed):** AR(1)-GARCH(1,1) with skewed-t innovations per asset.
2. **Dependence:** β-VAE on standardized GARCH residuals, or parametric copulas (benchmark).
3. **Risk:** 10,000 simulated residual vectors → portfolio returns → empirical **VaR (99%)** and **ES (97.5%)**.
4. **Part 1:** One-at-a-time (OAT) VAE design study (L1–L7) on a fixed window, then rolling evaluation for selected configs.
5. **Part 2:** Model risk as **MAD** across backtest-valid models; comparison VAE vs. copula.

## Project layout

| Module | Role |
|--------|------|
| `config.py` | Baseline hyperparameters and OAT experiment grid |
| `garch.py` | AR-GARCH marginals and portfolio weights |
| `vae.py` | PyTorch β-VAE, tail-weighted loss, training with early stopping |
| `copula.py` | Gaussian, t, Archimedean, mixture, DCC copulas |
| `backtesting.py` | Christoffersen–Pelletier duration test; ES conditional calibration |
| `evaluation.py` | Correlation and tail co-movement diagnostics |
| `modelrisk.py` | MAD, dispersion, sub-period summaries |
| `pipeline.py` | Stage 1 OAT, Stage 2 rolling, Part 2 MAD |
| `run_expose.py` | CLI entry point |

## Quick start

```bash
pip install -r requirements.txt
python run_expose.py --fast
```

Full study (long-running; GPU recommended):

```bash
python run_expose.py
```

Stage 1 only (OAT on fixed window):

```bash
python run_expose.py --stage1-only
```

Results are written to `results/` (CSV tables for OAT, VaR paths, MAD, backtests).

## Baseline VAE (exposé p. 2)

- 2 hidden layers × 64 units, `tanh`
- Latent dimension \(d_z = 3\), \(\beta = 2\)
- MSE reconstruction (optional tail-weighted MSE in L3)
- Training window 500 days, batch size 64, early stopping
- Annual VAE retraining in rolling Stage 2 (configurable in L5)

## Data

Place daily geometric returns in `data/qrm2025_returns.csv` (assets as columns). The bundled file contains five indices used for development; extend with Datastream indices per the exposé for production runs.
