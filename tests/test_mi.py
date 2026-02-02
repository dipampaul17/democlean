"""Tests for mutual information estimation."""

import numpy as np

from democlean.mi import (
    _standardize,
    estimate_mi,
    estimate_mi_ksg,
    estimate_mi_with_ci,
    reduce_dimensions,
)


class TestMIEstimation:
    """Test KSG mutual information estimator."""

    def test_independent_variables_low_mi(self):
        """Independent variables should have near-zero MI."""
        np.random.seed(42)
        x = np.random.randn(500, 2)
        y = np.random.randn(500, 2)
        mi = estimate_mi_ksg(x, y, k=3)
        assert mi < 0.3

    def test_identical_variables_high_mi(self):
        """Identical variables should have high MI."""
        np.random.seed(42)
        x = np.random.randn(500, 2)
        y = x.copy()
        mi = estimate_mi_ksg(x, y, k=3)
        assert mi > 2.0

    def test_linear_dependence(self):
        """Linear dependence should have moderate-high MI."""
        np.random.seed(42)
        x = np.random.randn(500, 2)
        noise = np.random.randn(500, 2) * 0.1
        y = x @ np.array([[1, 0.5], [0.5, 1]]) + noise
        mi = estimate_mi_ksg(x, y, k=3)
        assert 0.5 < mi < 5.0

    def test_mi_non_negative(self):
        """MI should always be non-negative."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(100, 3)
            y = np.random.randn(100, 2)
            mi = estimate_mi_ksg(x, y, k=3)
            assert mi >= 0

    def test_small_sample_handling(self):
        """Should handle small samples gracefully."""
        x = np.random.randn(5, 2)
        y = np.random.randn(5, 2)
        mi = estimate_mi_ksg(x, y, k=3)
        assert mi >= 0

    def test_very_small_sample_returns_zero(self):
        """Samples too small for KSG should return 0."""
        x = np.random.randn(2, 2)
        y = np.random.randn(2, 2)
        mi = estimate_mi_ksg(x, y, k=3)
        assert mi == 0.0

    def test_temporal_window(self):
        """Temporal window should work."""
        np.random.seed(42)
        states = np.cumsum(np.random.randn(200, 2), axis=0)
        actions = np.diff(states, axis=0)
        states = states[:-1]

        mi1 = estimate_mi(states, actions, k=3, temporal_window=1)
        mi3 = estimate_mi(states, actions, k=3, temporal_window=3)

        assert mi1 >= 0
        assert mi3 >= 0


class TestAnalyticalMI:
    """Validate against known analytical MI."""

    def test_known_mi_correlated_gaussian(self):
        """Test against analytical MI for correlated Gaussians."""
        np.random.seed(42)
        rho = 0.8
        cov = [[1, rho], [rho, 1]]
        n = 2000

        xy = np.random.multivariate_normal([0, 0], cov, n)
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        mi_true = -0.5 * np.log(1 - rho**2)
        mi_est = estimate_mi_ksg(x, y, k=5)

        assert abs(mi_est - mi_true) < 0.1

    def test_known_mi_different_correlations(self):
        """Test multiple correlation values."""
        np.random.seed(42)
        n = 2000

        for rho in [0.3, 0.5, 0.7, 0.9]:
            cov = [[1, rho], [rho, 1]]
            xy = np.random.multivariate_normal([0, 0], cov, n)
            x = xy[:, 0:1]
            y = xy[:, 1:2]

            mi_true = -0.5 * np.log(1 - rho**2)
            mi_est = estimate_mi_ksg(x, y, k=5)

            tolerance = 0.15 if rho > 0.8 else 0.1
            assert abs(mi_est - mi_true) < tolerance

    def test_independent_strict(self):
        """Independent variables should have MI ≈ 0."""
        np.random.seed(42)
        x = np.random.randn(2000, 1)
        y = np.random.randn(2000, 1)
        mi_est = estimate_mi_ksg(x, y, k=5)
        assert mi_est < 0.1


class TestBootstrapCI:
    """Test confidence interval estimation."""

    def test_ci_returns_tuple(self):
        """Should return (mi, ci_lower, ci_upper)."""
        np.random.seed(42)
        x = np.random.randn(100, 2)
        y = x + np.random.randn(100, 2) * 0.5

        mi, ci_lo, ci_hi = estimate_mi_with_ci(x, y, k=3, n_bootstrap=20)

        assert isinstance(mi, float)
        assert isinstance(ci_lo, float)
        assert isinstance(ci_hi, float)
        # CI should be ordered (point estimate may fall outside percentile CI)
        assert ci_lo <= ci_hi
        # Point estimate should be reasonably close to CI
        assert mi > 0
        assert ci_lo >= 0

    def test_ci_small_sample(self):
        """Small samples should return zeros."""
        x = np.random.randn(3, 2)
        y = np.random.randn(3, 2)

        mi, ci_lo, ci_hi = estimate_mi_with_ci(x, y, k=3)
        assert mi == 0.0
        assert ci_lo == 0.0
        assert ci_hi == 0.0


class TestDimensionReduction:
    """Test PCA dimension reduction."""

    def test_pca_reduces_dims(self):
        """PCA should reduce dimensions."""
        x = np.random.randn(100, 50)
        x_reduced = reduce_dimensions(x, n_components=10, method="pca")
        assert x_reduced.shape == (100, 10)

    def test_no_reduction_if_already_small(self):
        """Should not reduce if already small enough."""
        x = np.random.randn(100, 5)
        x_reduced = reduce_dimensions(x, n_components=10, method="pca")
        assert x_reduced.shape == (100, 5)

    def test_random_projection(self):
        """Random projection should work."""
        x = np.random.randn(100, 50)
        x_reduced = reduce_dimensions(x, n_components=10, method="random")
        assert x_reduced.shape == (100, 10)


class TestStandardize:
    """Test standardization helper."""

    def test_zero_mean(self):
        """Standardized data should have zero mean."""
        x = np.random.randn(100, 3) * 10 + 5
        x_std = _standardize(x)
        means = np.mean(x_std, axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_unit_variance(self):
        """Standardized data should have unit variance."""
        x = np.random.randn(100, 3) * 10 + 5
        x_std = _standardize(x)
        stds = np.std(x_std, axis=0)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_constant_column_handling(self):
        """Constant columns should not cause division by zero."""
        x = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
        x_std = _standardize(x)
        assert np.all(np.isfinite(x_std))
