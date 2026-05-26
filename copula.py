"""
Parametric copula dependence models (exposé Part 2 benchmark, M_COP).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _pseudo_obs(u: np.ndarray) -> np.ndarray:
    n, d = u.shape
    out = np.empty_like(u, dtype=float)
    for j in range(d):
        ranks = stats.rankdata(u[:, j], method="average")
        out[:, j] = ranks / (n + 1)
    return out


def _to_z(u_sim: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    d = residuals.shape[1]
    z = np.empty_like(u_sim)
    for j in range(d):
        sorted_r = np.sort(residuals[:, j])
        idx = np.clip(
            (u_sim[:, j] * len(sorted_r)).astype(int),
            0,
            len(sorted_r) - 1,
        )
        z[:, j] = sorted_r[idx]
    return z


def fit_copula(residuals: np.ndarray, family: str):
    u = _pseudo_obs(residuals)
    d = u.shape[1]
    family = family.lower()

    if family == "gaussian":
        from copulae import GaussianCopula

        cop = GaussianCopula(dim=d)
        cop.fit(u)
        return lambda n: _to_z(cop.random(n), residuals)

    if family == "t":
        from copulae import TCopula

        cop = TCopula(dim=d)
        cop.fit(u)
        return lambda n: _to_z(cop.random(n), residuals)

    if family == "clayton":
        from copulae.archimedean import ClaytonCopula

        cop = ClaytonCopula(dim=d)
        cop.fit(u)
        return lambda n: _to_z(cop.random(n), residuals)

    if family == "gumbel":
        from copulae.archimedean import GumbelCopula

        cop = GumbelCopula(dim=d)
        cop.fit(u)
        return lambda n: _to_z(cop.random(n), residuals)

    if family == "frank":
        from copulae.archimedean import FrankCopula

        cop = FrankCopula(dim=d)
        cop.fit(u)
        return lambda n: _to_z(cop.random(n), residuals)

    if family in ("vine", "gaussian_mixture"):
        return _fit_gaussian_mixture(u, residuals)

    if family == "dcc":
        return _fit_dcc(residuals)

    raise ValueError(f"Unknown copula family: {family}")


def _fit_gaussian_mixture(u: np.ndarray, residuals: np.ndarray):
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gm.fit(u)

    def simulate(n: int) -> np.ndarray:
        u_sim, _ = gm.sample(n)
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)
        return _to_z(u_sim, residuals)

    return simulate


def _fit_dcc(residuals: np.ndarray):
    corr = np.corrcoef(residuals.T)
    std = residuals.std(axis=0, ddof=1)

    def simulate(n: int) -> np.ndarray:
        L = np.linalg.cholesky(corr + 1e-8 * np.eye(corr.shape[0]))
        eps = np.random.randn(n, corr.shape[0]) @ L.T
        return eps * std

    return simulate


def simulate_copula_residuals(
    residuals: np.ndarray, family: str, n_sim: int = 10_000
) -> np.ndarray:
    sim_fn = fit_copula(residuals, family)
    return sim_fn(n_sim)
