# %%
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler


# %%
def cbps_att_torch(X, w, intercept=True, noi=False, niter=1000, lr=0.01):
    n, p = X.shape
    if intercept:
        X = np.hstack((np.ones((n, 1)), X))
    # preprocessing for numpy to pytorch
    X, w = torch.from_numpy(X).float(), torch.from_numpy(w).float()
    X, w = Variable(X), Variable(w)
    # Parameters theta to be optimized
    theta = Variable(torch.randn(p + 1), requires_grad=True)

    # Define the loss function ℓ(θ) as per equation (7.10)
    def loss_function(theta, X, W):
        Xb = torch.matmul(X, theta)
        loss = torch.mean(W * torch.exp(-Xb) + (1 - W) * Xb)
        return loss

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
    return np.exp(X @ theta_final)[w == 0]


# %%
df = pd.read_csv("../df/lalonde_psid.csv")
df.head()
# %%
w, y = df.treat.values, df.re78.values
# naive estimate is garbage - true effect is ~ 1800
y[w == 1].mean() - y[w == 0].mean()
# -15204.775555988717
# %% stabilize by scaling covariates
X = df.drop(columns=["treat", "re78"]).values
X = MinMaxScaler().fit_transform(X)
# %% cbps estimate - comes pretty close to true effect
cbps_wt = cbps_att_torch(X, w, niter=5_000, lr=0.001, noi=False)
y[w == 1].mean() - np.average(y[w == 0], weights=cbps_wt)
# 1946.715423006407
# %%
