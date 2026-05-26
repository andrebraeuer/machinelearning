"""
Part 1 descriptive evaluation: correlation fit and tail co-movement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(z: np.ndarray) -> np.ndarray:
    return np.corrcoef(z.T)


def correlation_reproduction_error(
    empirical: np.ndarray, simulated: np.ndarray
) -> dict:
    """Frobenius norm and max absolute entry-wise correlation error."""
    c_emp = correlation_matrix(empirical)
    c_sim = correlation_matrix(simulated)
    diff = c_emp - c_sim
    mask = ~np.eye(diff.shape[0], dtype=bool)
    return {
        "frobenius": float(np.linalg.norm(diff, ord="fro")),
        "max_abs": float(np.abs(diff[mask]).max()) if mask.any() else 0.0,
        "mean_abs": float(np.abs(diff[mask]).mean()) if mask.any() else 0.0,
    }


def tail_comovement_rate(
    z: np.ndarray,
    *,
    tail_quantile: float = 0.05,
) -> float:
    """
    Share of days where all assets are jointly in their left tail.
    Qualitative check for crisis co-movement.
    """
    thresh = np.quantile(z, tail_quantile, axis=0)
    joint = np.all(z <= thresh, axis=1)
    return float(joint.mean())


def summarize_stage1_metrics(
    empirical_resid: np.ndarray,
    simulated_resid: np.ndarray,
) -> dict:
    corr = correlation_reproduction_error(empirical_resid, simulated_resid)
    return {
        **corr,
        "tail_joint_empirical": tail_comovement_rate(empirical_resid),
        "tail_joint_simulated": tail_comovement_rate(simulated_resid),
    }
