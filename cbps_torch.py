# %%
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler


# %%
def cbps_torch(
    X: np.array,
    w: np.array,
    estimand: str = "ATT",
    intercept: bool = True,
    noi: bool = False,
    niter: int = 1000,
    lr: float = 0.01,
):
    """Synthetic control weights using Covariate Balancing Propensity Score estimation

    Args:
        X (np.array): Covariate matrix (n x p)
        w (np.array): Treatment dummies (n x 1)
        estimand (str, optional): Estimand to target. "ATT" reweights control units to equalize moments with treatment, "ATE" reweights both to look like each other. Defaults to "ATT".
        intercept (bool, optional): Intercept in model? Defaults to True.
        noi (bool, optional): Report optimization details. Defaults to False.
        niter (int, optional): Iterations. Defaults to 1000.
        lr (float, optional): Learning rate for optimizer. Defaults to 0.01.

    Raises:
        ValueError: Estimand not supported

    Returns:
        np.array: Synthetic control weights
    """
    n, p = X.shape
    if intercept:
        X = np.c_[(np.ones((n, 1)), X)]
    # preprocessing for numpy to pytorch
    X, w = torch.from_numpy(X).float(), torch.from_numpy(w).float()
    # Parameters theta to be optimized
    theta = Variable(torch.randn(p + 1), requires_grad=True)

    # loss function depends on estimand
    if estimand == "ATT":

        def loss_function(theta, X, W):
            Xb = torch.matmul(X, theta)
            w0piece = torch.mul((1 - W), torch.exp(Xb))
            w1piece = torch.mul(W, Xb)
            loss = (torch.sum(w0piece) - torch.sum(w1piece)) / torch.sum(W)
            return loss

    elif estimand == "ATE":
        # Define the loss function ℓ(θ) as per equation (7.10)
        def loss_function(theta, X, W):
            Xb = torch.matmul(X, theta)
            loss = torch.mean(W * torch.exp(-Xb) + (1 - W) * Xb)
            return loss

    else:
        raise ValueError("estimand must be 'ATT' or 'ATE'")
    # Use an optimizer from PyTorch's optimization module, e.g., Adam
    optimizer = torch.optim.Adam([theta], lr=lr)
    # Iteratively apply gradient descent to minimize the loss function
    for iteration in range(niter):
        optimizer.zero_grad()  # Clear previous gradients
        loss = loss_function(theta, X, w)  # Calculate loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update theta
        if noi and iteration % 100 == 0:  # Print loss every 100 iterations
            print(f"Iteration {iteration}: loss = {loss.item()}")
    # Final theta value after optimization
    theta_final = theta.data
    if estimand == "ATT":
        return np.exp(X @ theta_final)[w == 0]
    elif estimand == "ATE":
        return np.exp(X @ theta_final)


# %%
def cbps_retarget(
    X0: np.array,
    X1: np.array,
    intercept: bool = True,
    noi: bool = False,
    niter: bool = 1000,
    lr: bool = 0.01,
):
    """Synthetic control weights for aggregated data
    Estimate balancing scores when only aggregated data for target population is available. Conceptually very similar to the standard case, just requires tweaked loss function.

    Args:
        X0 (np.array): covariate matrix for source distribution: n X p
        X1 (np.array): target moments for target distribution: p X 1
        intercept (bool, optional): Intercept. Defaults to True.
        noi (bool, optional): Report optimizer loss. Defaults to False.
        niter (bool, optional): Iterations. Defaults to 1000.
        lr (bool, optional): Learning rate. Defaults to 0.01.

    Returns:
        np.array: weight vector
    """
    n, p = X0.shape
    if intercept:
        X0 = np.c_[(np.ones((n, 1)), X0)]
        X1 = np.r_[1, X1]
    # preprocessing for numpy to pytorch
    X0, X1 = torch.from_numpy(X0).float(), torch.from_numpy(X1).float()
    theta = Variable(torch.randn(p + 1), requires_grad=True)

    # retargeting loss (same as ATT with individual level data)
    def loss_function(theta, X0, X1):
        loss = torch.mean(torch.exp(torch.matmul(X0, theta))) - torch.matmul(
            X1, theta
        )  # target moments
        return loss

    optimizer = torch.optim.Adam([theta], lr=lr)
    # Iteratively apply gradient descent to minimize the loss function
    for iteration in range(niter):
        optimizer.zero_grad()  # Clear previous gradients
        loss = loss_function(theta, X0, X1)  # Calculate loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update theta
        if noi and iteration % 100 == 0:  # Print loss every 100 iterations
            print(f"Iteration {iteration}: loss = {loss.item()}")
    theta_final = theta.data
    wgt = np.exp(X0 @ theta_final)
    return wgt


# %% testing
df = pd.read_csv("../df/lalonde_psid.csv")
w, y = df.treat.values, df.re78.values
# naive estimate is garbage - true effect is ~ 1800
y[w == 1].mean() - y[w == 0].mean()
# -15204.775555988717
# %% stabilize by scaling covariates
X = df.drop(columns=["treat", "re78"]).values
X = MinMaxScaler().fit_transform(X)
# %% cbps estimate of ATT - comes pretty close to true effect
cbps_wt = cbps_torch(X, w, niter=10_000, lr=0.001, noi=False)
y[w == 1].mean() - np.average(y[w == 0], weights=cbps_wt)
# 1671.9811205765518
# %% test aggregated data - same loss as above
X0, X1 = X[w == 0], X[w == 1].mean(axis=0)
wgt = cbps_retarget(X0, X1)
y[w == 1].mean() - np.average(y[w == 0], weights=wgt)
# 1951.5708059253102
# %%
