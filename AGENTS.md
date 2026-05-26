# AGENTS.md

## Cursor Cloud specific instructions

This is a **Python research codebase** for a master's thesis on model risk in portfolio Value-at-Risk (VaR) forecasting. It implements a VAE-GARCH pipeline and Copula-GARCH benchmark for multivariate risk modeling. There are no web services, databases, or Docker containers — only Python scripts and Jupyter notebooks.

### Project layout

- `garch.py` — ARMA-GARCH marginal fitting (production module)
- `claude/claude_pipeline_VAEwithPlot.py` — Main VAE-GARCH rolling-window VaR pipeline (with plotting)
- `claude/claude_pipeline_VAE.py` — Same pipeline without plotting
- `old/` — Legacy copula, backtesting, and model-risk modules
- `SANDBOX/` — Jupyter notebooks for interactive analysis
- `data/qrm2025_returns.csv` — 5-asset daily return series (no date column; assign a business-day index starting ~2015-01-02 when loading)

### Running scripts

All scripts are run directly with `python3 <script.py>` from the repo root. The main pipeline files (`claude/claude_pipeline_VAEwithPlot.py`, `claude/claude_pipeline_VAE.py`) have their `__main__` blocks commented out; to run them, load data and call `rolling_var_pipeline()` programmatically.

### Key caveats

- **No `requirements.txt` or `pyproject.toml` exists.** Dependencies are installed via pip directly. The update script handles this.
- **Data has no date column.** The CSV `data/qrm2025_returns.csv` contains only numeric return columns. When loading, assign a DatetimeIndex (e.g. `pd.bdate_range(start='2015-01-02', periods=len(df), freq='B')`) so pipeline date-formatting works.
- **PyTorch CPU-only is sufficient.** Install via `pip install torch --index-url https://download.pytorch.org/whl/cpu` to save disk and install time.
- **`arch` package is on PyPI as `arch` (not `arch-py`).** Use `pip install arch`.
- **No linter, test framework, or CI is configured.** Validation is done by running the pipeline scripts and checking output.
- **`copulae` requires Python <3.13.** The current environment uses Python 3.12, which is compatible.
