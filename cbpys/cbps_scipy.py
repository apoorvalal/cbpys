"""Covariate Balancing Propensity Score estimation for synthetic control weights."""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def lbw(
    X0: np.array,
    X1: np.array,
) -> np.array:
    """Estimates Linear balancing weights.

    Args:
        X0 (np.array): Covariate matrix for source population
        X1 (np.array): Moments of target distribution
        y (np.array): outcome vector (optional)
        w (np.array): treatment vector (optional)

    Returns:
        np.array: weights
    """
    H00 = np.linalg.pinv(X0.T @ X0) @ X0.T
    wt = X1 @ H00
    return wt


def cbps_att(
    X, W, intercept=True, theta_init=None, method="L-BFGS-B", control={}, lam=None
):
    """Synthetic control weights using Covariate Balancing Propensity Score estimation.

    Synth control weights are the solution to the following optimization problem:

    Args:
        X (np.array): Covariate matrix (n x p)
        W (np.array): Treatment dummies (n x 1)
        intercept (bool, optional): Intercept in model? Defaults to True.
        theta_init (np.array, optional): Initial values for theta. Defaults to None.
        method (str, optional): Optimization method. Defaults to "L-BFGS-B".
        control (dict, optional): Control parameters for optimization. Defaults to {}.
        lam (np.array, optional): Regularization parameters. Defaults to None.

    Raises:
        ValueError: Estimand not supported

    Returns:
        dict: Dictionary of results
    """
    if not np.all(np.isin(W, [0, 1])):
        raise ValueError("W should be a binary vector.")
    if (
        not isinstance(X, np.ndarray)
        or X.shape[0] != len(W)
        or X.ndim != 2
        or np.any(np.isnan(X))
    ):
        raise ValueError("X should be a numeric matrix with nrows = length(W).")

    n, p = X.shape

    def objective(theta, X, W0_idx, W1_idx, lam):
        X_theta = X @ theta
        imb = (np.sum(np.exp(X_theta[W0_idx])) - np.sum(X_theta[W1_idx])) / len(W1_idx)
        regu = np.sum(lam * theta**2)
        return imb + regu

    def objective_gradient(theta, X0, X_sum1, W0_idx, lam):
        X_theta = X @ theta
        return (
            np.sum(X0 * np.exp(X_theta[W0_idx])[:, np.newaxis], axis=0) - X_sum1
        ) / n + 2 * lam * theta

    if lam is None:
        lam = np.zeros(X.shape[1])
    if intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        lam = np.hstack((0, lam))

    W1_idx = np.where(W == 1)[0]
    W0_idx = np.where(W == 0)[0]

    if theta_init is None:
        idx_small = np.hstack(
            (W1_idx, np.random.choice(W0_idx, size=len(W1_idx), replace=False))
        )
        glm = LogisticRegression().fit(X[idx_small], W[idx_small])
        theta_init = glm.coef_[0]
        if intercept:
            rho = np.mean(W)
            theta_init[0] = theta_init[0] - np.log((1 - rho) / rho) * len(
                idx_small
            ) / np.sum(W)

    X0 = X[W0_idx]
    X_sum1 = np.sum(X[W1_idx], axis=0)

    res = minimize(
        fun=lambda z: objective(z, X, W0_idx, W1_idx, lam),
        x0=theta_init,
        jac=lambda z: objective_gradient(z, X0, X_sum1, W0_idx, lam),
        method=method,
        options=control,
    )

    theta_hat = res.x
    weights_0 = np.exp(X @ theta_hat)
    LHS = np.sum(
        (1 - W)[:, np.newaxis] * X * weights_0[:, np.newaxis], axis=0
    ) / np.sum(W == 1)
    RHS = np.sum(W[:, np.newaxis] * X, axis=0) / np.sum(W == 1)
    sd_W1 = np.std(X[W1_idx], axis=0)
    sd_W1[sd_W1 == 0] = 1
    sd_W = np.std(X, axis=0)
    sd_W[sd_W == 0] = 1
    mean_diff = np.mean(X[W1_idx], axis=0) - np.apply_along_axis(
        lambda x: np.average(x, weights=weights_0[W0_idx]),
        axis=0,
        arr=X[W0_idx],
    )
    balance_std = mean_diff / sd_W1
    balance_std_pre = (np.mean(X[W1_idx], axis=0) - np.mean(X[W0_idx], axis=0)) / sd_W1
    balance_std_all = mean_diff / sd_W
    balance_std_pre_all = (
        np.mean(X[W1_idx], axis=0) - np.mean(X[W0_idx], axis=0)
    ) / sd_W

    return {
        "theta_hat": theta_hat,
        "weights_0": weights_0,
        "convergence": res.success,
        "balance_condition": np.column_stack((LHS, RHS)),
        "balance_std": balance_std[1:] if intercept else balance_std,
        "balance_std_pre": balance_std_pre[1:] if intercept else balance_std_pre,
        "balance_std_all": balance_std_all[1:] if intercept else balance_std_all,
        "balance_std_pre_all": (
            balance_std_pre_all[1:] if intercept else balance_std_pre_all
        ),
    }
