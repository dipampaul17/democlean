"""Mutual information estimation for state-action predictability.

Uses the Kraskov-Stögbauer-Grassberger (KSG) estimator for continuous variables.
Reference: Kraskov et al. (2004) "Estimating mutual information" PRE 69, 066138
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def estimate_mi_ksg(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    k: int = 3,
) -> float:
    """Estimate mutual information using KSG estimator.

    Args:
        x: Features, shape (n_samples, n_dims)
        y: Features, shape (n_samples, n_dims)
        k: Number of nearest neighbors

    Returns:
        Estimated MI in nats
    """
    n = x.shape[0]
    if n < k + 1:
        return 0.0

    x = _standardize(x)
    y = _standardize(y)
    xy = np.hstack([x, y])

    # k-th neighbor distances in joint space
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(xy)
    distances, _ = nn_joint.kneighbors(xy)
    epsilon = distances[:, k]

    # Count neighbors in marginal spaces
    nn_x = NearestNeighbors(metric="chebyshev").fit(x)
    nn_y = NearestNeighbors(metric="chebyshev").fit(y)

    n_x = np.array(
        [
            len(nn_x.radius_neighbors([x[i]], epsilon[i], return_distance=False)[0]) - 1
            for i in range(n)
        ]
    )
    n_y = np.array(
        [
            len(nn_y.radius_neighbors([y[i]], epsilon[i], return_distance=False)[0]) - 1
            for i in range(n)
        ]
    )

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(0.0, mi)


def estimate_mi(
    states: NDArray[np.floating],
    actions: NDArray[np.floating],
    k: int = 3,
    temporal_window: int = 1,
) -> float:
    """Estimate MI with optional temporal context.

    Args:
        states: Shape (T, state_dim)
        actions: Shape (T, action_dim)
        k: KSG neighbors
        temporal_window: Consecutive states to concatenate

    Returns:
        MI estimate
    """
    if temporal_window > 1:
        states = _add_temporal_context(states, temporal_window)
        actions = actions[temporal_window - 1 :]

    return estimate_mi_ksg(states, actions, k=k)


def estimate_mi_with_ci(
    states: NDArray[np.floating],
    actions: NDArray[np.floating],
    k: int = 3,
    n_bootstrap: int = 100,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Estimate MI with bootstrap confidence interval.

    Args:
        states: Shape (T, state_dim)
        actions: Shape (T, action_dim)
        k: KSG neighbors
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 95%)

    Returns:
        (mi_estimate, ci_lower, ci_upper)
    """
    n = len(states)
    if n < k + 5:
        return 0.0, 0.0, 0.0

    # Point estimate
    mi = estimate_mi_ksg(states, actions, k=k)

    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_mis = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_mi = estimate_mi_ksg(states[idx], actions[idx], k=k)
        bootstrap_mis.append(boot_mi)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_mis, 100 * alpha)
    ci_upper = np.percentile(bootstrap_mis, 100 * (1 - alpha))

    return mi, ci_lower, ci_upper


def reduce_dimensions(
    x: NDArray[np.floating],
    n_components: int = 10,
    method: str = "pca",
) -> NDArray[np.floating]:
    """Reduce dimensionality for high-dim states.

    Args:
        x: Shape (n_samples, n_dims)
        n_components: Target dimensions
        method: "pca" or "random"

    Returns:
        Reduced array, shape (n_samples, n_components)
    """
    import warnings

    if x.shape[1] <= n_components:
        return x

    if method == "pca":
        from sklearn.decomposition import PCA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = PCA(n_components=n_components, random_state=42)
            return pca.fit_transform(x)
    elif method == "random":
        from sklearn.random_projection import GaussianRandomProjection

        proj = GaussianRandomProjection(n_components=n_components, random_state=42)
        return proj.fit_transform(x)
    else:
        raise ValueError(f"Unknown method: {method}")


def _standardize(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Standardize to zero mean, unit variance."""
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    return (x - np.mean(x, axis=0, keepdims=True)) / std


def _add_temporal_context(x: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    """Stack consecutive frames."""
    n, d = x.shape
    if window >= n:
        return x
    result = np.zeros((n - window + 1, window * d))
    for i in range(window):
        result[:, i * d : (i + 1) * d] = x[i : n - window + 1 + i]
    return result
