"""

Copula Dependence Models for Multivariate Portfolio Risk

Architecture
------------

garch.py        - Functions to fit GARCH models.
copula.py       - Functions to fit copulas and simulate from the fitted models.
pipeline.py     - Main pipeline to fit GARCH models, copula, and simulate portfolio returns for risk estimation.

------------

"""
import numpy as np

from garch.py import fit_garch
from copula.py import fit_copula_gaussian, fit_copula_t, simulate_copula


# Configuration
SEED = 42
WINDOW = 250
FORECAST = 100
N_SIM = 10000
ALPHA = 0.05

SCALE = 100 # Scaling factor for returns in GARCH fitting to improve convergence; scale back when interpreting results

DEVICE = "cpu"

np.random.seed(SEED) # Set random seed for reproducibility

# ──────────────────────────────────────────────────────────────────────────────
# Rolling risk forecasts
# ──────────────────────────────────────────────────────────────────────────────

