import numpy as np
import pytest

from tlc_demand.hierarchy import reconcile, shrunk_covariance, summing_matrix

RNG = np.random.default_rng(3)
BOROUGH = [7, 7, 7, 2, 2, 9]  # 6 zones in 3 boroughs


def coherent(bottom):
    S, _ = summing_matrix(BOROUGH)
    return bottom @ S.T, S


def test_summing_matrix_structure():
    S, boroughs = summing_matrix(BOROUGH)
    assert boroughs == [2, 7, 9] and S.shape == (1 + 3 + 6, 6)
    assert S[0].sum() == 6
    assert S[1 + boroughs.index(7)].tolist() == [1, 1, 1, 0, 0, 0]
    assert (S[4:] == np.eye(6)).all()


@pytest.mark.parametrize("use_w", [False, True])
def test_reconciled_forecasts_are_coherent(use_w):
    S, _ = summing_matrix(BOROUGH)
    base = RNG.normal(10, 3, size=(50, S.shape[0]))
    W = np.diag(RNG.uniform(0.5, 4, S.shape[0])) if use_w else None
    out = reconcile(base, S, W)
    bottom = out[:, 4:]
    assert np.allclose(out, bottom @ S.T, atol=1e-9)


def test_already_coherent_forecasts_are_unchanged():
    bottom = RNG.uniform(1, 20, size=(30, 6))
    base, S = coherent(bottom)
    W = np.diag(RNG.uniform(0.5, 4, S.shape[0]))
    assert np.allclose(reconcile(base, S), base) and np.allclose(reconcile(base, S, W), base)


def test_projection_never_increases_weighted_error_when_truth_is_coherent():
    """Orthogonal projection in the W-norm is a contraction toward any coherent point."""
    truth, S = coherent(RNG.uniform(1, 20, size=(200, 6)))
    A = RNG.normal(size=(S.shape[0], S.shape[0]))
    W = A @ A.T + np.eye(S.shape[0])
    base = truth + RNG.normal(0, 2, size=truth.shape)
    rec = reconcile(base, S, W)
    Wi = np.linalg.inv(W)
    err = lambda e: np.einsum("ti,ij,tj->t", e, Wi, e)  # noqa: E731
    assert (err(rec - truth) <= err(base - truth) + 1e-9).all()


def test_reconciliation_helps_the_top_when_only_the_top_is_noisy():
    truth, S = coherent(RNG.uniform(50, 100, size=(500, 6)))
    base = truth.copy()
    base[:, 0] += RNG.normal(0, 30, size=500)  # city forecast is bad, zone forecasts are exact
    W = np.diag([900.0] + [1e-3] * (S.shape[0] - 1))
    rec = reconcile(base, S, W)
    assert np.abs(rec[:, 0] - truth[:, 0]).mean() < 0.05 * np.abs(base[:, 0] - truth[:, 0]).mean()


def test_shrunk_covariance_properties():
    t, n = 400, 8
    resid = RNG.normal(size=(t, n)) * np.arange(1, n + 1)
    cov, lam = shrunk_covariance(resid)
    assert 0 <= lam <= 1 and lam > 0.5  # independent columns: heavy shrinkage toward the diagonal
    assert np.all(np.linalg.eigvalsh(cov) > 0) and np.allclose(np.diag(cov), resid.var(axis=0, ddof=1))
    shared = RNG.normal(size=(t, 1))
    corr_resid = shared @ np.ones((1, n)) * 3 + RNG.normal(size=(t, n))
    _, lam2 = shrunk_covariance(corr_resid)
    assert lam2 < lam  # strong common factor: less shrinkage
    few = RNG.normal(size=(12, 30))
    cov3, _ = shrunk_covariance(few)
    assert np.all(np.linalg.eigvalsh(cov3) > 0)  # usable even with fewer rows than nodes
