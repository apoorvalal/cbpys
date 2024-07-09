"""Retarget estimation for synthetic control weights in PyTorch."""

import numpy as np
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cbps_retarget(
    X0: np.array,
    X1: np.array,
    intercept: bool = True,
    noi: bool = False,
    niter: bool = 1000,
    lr: bool = 0.01,
):
    """Synthetic control weights for aggregated data.

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
    X0 = torch.from_numpy(X0).float().to(device)
    X1 = torch.from_numpy(X1).float().to(device)
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
    theta_final = theta.data.numpy()
    wgt = np.exp(X0 @ theta_final)

    return wgt
