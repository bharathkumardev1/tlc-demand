"""Forecast reconciliation for a zone / borough / city hierarchy.

Node order is [city, boroughs..., zones...]. `summing_matrix` maps the bottom level (zones)
to every node. Reconciliation projects a matrix of base forecasts (rows are time points,
columns are nodes) onto the set of coherent forecasts, i.e. those where every parent equals
the sum of its children:

    reconciled = base @ G.T @ S.T,   G = (S' W^-1 S)^-1 S' W^-1

W = I gives OLS, W = diag(residual variance) gives variance-scaled WLS, and the shrunk
residual covariance gives MinT (Wickramasuriya et al., 2019)."""
from __future__ import annotations

import numpy as np


def summing_matrix(borough_of_zone: list[int]) -> tuple[np.ndarray, list[int]]:
    """S for a three-level hierarchy. Returns (S, borough ids in row order)."""
    boroughs = sorted(set(borough_of_zone))
    n_zone = len(borough_of_zone)
    S = np.zeros((1 + len(boroughs) + n_zone, n_zone))
    S[0, :] = 1.0
    for z, b in enumerate(borough_of_zone):
        S[1 + boroughs.index(b), z] = 1.0
    S[1 + len(boroughs):, :] = np.eye(n_zone)
    return S, boroughs


def shrunk_covariance(resid: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance of residuals (rows are time) shrunk toward its diagonal.

    The intensity follows Schafer and Strimmer (2005): the ratio of the sampling variance of the
    off-diagonal correlations to their squared size, clipped to [0, 1]. Returns (matrix, intensity)."""
    t, n = resid.shape
    x = resid - resid.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    z = x / sd
    corr = z.T @ z / (t - 1)
    # variance of each sample correlation, from the products of standardized residuals
    # sum_t (w_t - mean(w))^2 with w_t = z_ti * z_tj, without building the t x n x n array
    w_mean = z.T @ z / t
    var_r = t / (t - 1) ** 3 * ((z ** 2).T @ (z ** 2) - t * w_mean ** 2)
    off = ~np.eye(n, dtype=bool)
    lam = float(np.clip(var_r[off].sum() / (corr[off] ** 2).sum(), 0.0, 1.0)) if (corr[off] ** 2).sum() > 0 else 1.0
    corr_s = (1 - lam) * corr
    np.fill_diagonal(corr_s, 1.0)
    return corr_s * np.outer(sd, sd), lam


def reconcile(base: np.ndarray, S: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """Project base forecasts (T x n_nodes) onto the coherent subspace. W is the n_nodes x n_nodes
    error covariance (or any positive definite weight); None means OLS."""
    if W is None:
        A = S.T @ S
        G = np.linalg.solve(A, S.T)
    else:
        Wi_S = np.linalg.solve(W, S)
        G = np.linalg.solve(S.T @ Wi_S, Wi_S.T)
    return base @ G.T @ S.T
