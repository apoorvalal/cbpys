"""Covariate Balancing Propensity Score estimation in PyTorch."""

from functools import cached_property
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .diagnostics.differences import standarized_diffs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBPS:
    """Covariate Balancing Propensity Score estimation in PyTorch."""

    def __init__(
        self,
        X: Union[np.array, pd.DataFrame],
        W: Union[np.array, str],
        estimand: str = "ATT",
        intercept: bool = True,
        noi: bool = False,
        niter: int = 1000,
        lr: float = 0.01,
        svd: int = None,
        reg: float = None,
    ):
        """Covariate Balancing Propensity Score estimation in PyTorch.

        This class partially implements the covariate balancing propensity score
        calculation for two different estimands: ATT and ATE. The class uses PyTorch
        to optimize a taylored loss function and calculate iteratively optimal weights
        to balance groups.

        Some attempts at regularization are done during the optimization process.
        First, we add a L1 regularization term to the loss function that penalizes
        the loss using the absolute value of the weights. Second, we use SVD to reduce
        the dimensionality of the input matrix. This is useful to reduce sensitivity
        to noise and multicollinearity (especially when the number of covariates is
        large).

        Args:
            X (np.array): covariate matrix
            W (np.array): treatment dummy
            estimand (str, optional): ATT or ATE. Defaults to "ATT".
            intercept (bool, optional): Include intercept in model. Defaults to True.
            noi (bool, optional): Print number of iterations. Defaults to False.
            niter (int, optional): Number of iterations. Defaults to 1000.
            lr (float, optional): Learning rate. Defaults to 0.01.
            svd (int, optional): Dimensionality reduction using SVD. Defaults to None.
            reg (float, optional): Regularization parameter. Defaults to None.
        """
        self.estimand = estimand
        self.intercept = intercept
        self.noi = noi
        self.niter = niter
        self.lr = lr
        self.reg = reg
        self.svd = svd

        if self.estimand not in ["ATT", "ATE"]:
            raise NotImplementedError("Estimand in ['ATT', 'ATE'] supported")

        # Parse varaibles if a pandas dataframe is passed
        if isinstance(W, str):
            # We asumme here if a string is passed, we expect a dataframe
            if not isinstance(X, pd.DataFrame):
                raise ValueError("If W is a string, X should be a dataframe")

            self.treat_var = W
            W = X[self.treat_var].values

        if isinstance(X, pd.DataFrame):
            self.df = X.copy()
            X = X.drop(columns=[self.treat_var]).values

            # Normalize covariates if passed as a dataframe
            # maybe this should be an option
            X = MinMaxScaler().fit_transform(X)

        # Define parameters for calculation
        self.n, self.p = X.shape
        self.W = torch.from_numpy(W).float().to(device)

        # SVD for dimensionality reduction
        if self.svd is not None:
            X = torch.from_numpy(X).float().to(device)
            U, S, V = torch.linalg.svd(X)
            U_r = U[:, : self.svd]
            S_r = torch.diag(S[: self.svd])
            V_r = V[:, : self.svd].t()

            self.X = U_r @ S_r @ V_r
            print(f"Reduced dimensionality to {self.X.shape[1]} using SVD.")
        else:
            self.X = torch.from_numpy(X).float().to(device)

        # Store losses when fit is called
        self.loss = []

        if self.intercept:
            self.X = torch.concat((torch.ones((self.n, 1)).to(device), self.X), 1)

        # Parameters theta to be optimized
        self.theta = torch.randn(self.p + 1, requires_grad=True, device=device)

    def __repr__(self):
        """Representation of the class."""
        return "Esimating CBPS with PyTorch using {device}"

    def _covariate_differences(self, metric):
        """Calculate standarized differences without weights."""
        try:
            std_diffs_weight = standarized_diffs(
                df_vars=self.df,
                treat_var=self.treat_var,
                metric=metric,
                weights=self.weights,
            )
            std_diffs_unweight = standarized_diffs(
                df_vars=self.df, treat_var=self.treat_var, metric=metric
            )

        except AttributeError:
            std_diffs_weight = standarized_diffs(
                df_vars=self.X,
                treat_var=self.W,
                metric=metric,
                weights=self.weights,
            )
            std_diffs_unweight = standarized_diffs(
                df_vars=self.X, treat_var=self.W, metric=metric
            )

        return (std_diffs_unweight, std_diffs_weight)

    def diagnose(self, method, ax=None, scatter=False):
        """Plot standarized differences.

        Show plot of standarized differences for the balancing covariates to show
        the final balance achieved by the CBPS method.

        Args:
            method (str): method to calculate standarized differences
            ax (matplotlib.axes, optional): axes to plot. Defaults to None.
            scatter (bool, optional): Scatter plot of unweighted and weighted balance. Defaults to False.

        Returns:
            matplotlib.axes
        """
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))


        if not scatter:
            std_diff_uw, std_diff_w = self._covariate_differences(method)
            ax.plot(
                std_diff_w.std_diffs,
                std_diff_w.index,
                "o",
                label="Weighted",
            )
            ax.plot(std_diff_uw.std_diffs, std_diff_uw.index, "o", label="Unweighted")
            ax.axvline(0, color="black", linestyle="--")
            # Add labels to axis
            ax.set_ylabel("Variables")
            ax.set_xlabel("Standardized Differences")
            # Add legend to the plot
            ax.legend()
        else:
            std_diff_uw, std_diff_w = self._covariate_differences('asmd')
            ax.scatter(
                x = std_diff_uw.std_diffs,
                y = std_diff_w.std_diffs,
            )
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_ylabel("Weighted Standardized Differences")
            ax.set_xlabel("Unweighted Standardized Differences")
        return ax

    @staticmethod
    def loss_function_att(theta, X, W):
        """Loss function for ATT balancing loss.

        Args:
            theta (torch.tensor): parameter vector
            X (torch.tensor): covariate matrix
            W (torch.tensor): treatment dummy

        Returns:
            torch.tensor: loss
        """
        Xb = torch.matmul(X, theta)
        w0piece = torch.mul((1 - W), torch.exp(Xb))
        w1piece = torch.mul(W, Xb)
        loss = (torch.sum(w0piece) - torch.sum(w1piece)) / torch.sum(W)

        return loss

    @staticmethod
    def loss_function(theta, X, W):
        """Loss function for ATE balancing loss.

        Args:
            theta (torch.tensor): parameter vector
            X (torch.tensor): covariate matrix
            W (torch.tensor): treatment dummy

        Returns:
            torch.tensor: loss
        """
        Xb = torch.matmul(X, theta)
        loss = torch.mean(W * torch.exp(-Xb) + (1 - W) * Xb)

        return loss

    @cached_property
    def fit(self):
        """Estimate theta via SGD using target estimand loss function."""
        optimizer = torch.optim.Adam([self.theta], lr=self.lr)

        for iteration in tqdm(range(self.niter), desc="Optimizing CBPS..."):
            optimizer.zero_grad()

            if self.estimand == "ATT":
                loss = self.loss_function_att(self.theta, self.X, self.W)
            elif self.estimand == "ATE":
                loss = self.loss_function(self.theta, self.X, self.W)

            if self.reg:
                loss += self.reg * torch.linalg.vector_norm(self.theta, 1)

            loss.backward()
            optimizer.step()

            self.loss.append(loss.item())

            if self.noi and iteration % 100 == 0:
                print(f"Iteration {iteration}: loss = {loss.item()}")

        return self.theta

    @property
    def weights(self):
        """Return final weights after optimization.

        Compute weights for the estimand of interest.

        Returns:
            np.array: weights
        """
        weights = self.fit
        if self.estimand == "ATT":
            out = torch.exp(self.X @ weights)[self.W == 0]
        elif self.estimand == "ATE":
            out = torch.exp(self.X @ weights)

        return out
