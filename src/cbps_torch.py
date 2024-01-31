"""Covariate Balancing Propensity Score estimation in PyTorch."""
import numpy as np
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBPS:
    """Covariate Balancing Propensity Score estimation in PyTorch."""

    def __init__(
        self,
        X,
        W,
        estimand="ATT",
        intercept=True,
        noi=False,
        niter=1000,
        lr=0.01,
        reg=None,
    ):
        """Covariate Balancing Propensity Score estimation in PyTorch."""
        self.estimand = estimand
        self.intercept = intercept
        self.noi = noi
        self.niter = niter
        self.lr = lr
        self.reg = reg

        if self.estimand not in ["ATT", "ATE"]:
            raise NotImplementedError("Estimand in ['ATT', 'ATE'] supported")

        # Define parameters for calculation
        self.n, self.p = X.shape

        self.X = torch.from_numpy(X).float().to(device)
        self.W = torch.from_numpy(W).float().to(device)

        if self.intercept:
            self.X = np.c_[(np.ones((self.n, 1)), self.X)]

        # Parameters theta to be optimized
        self.theta = list(Variable(torch.randn(self.p + 1), requires_grad=True))

    def __repr__(self):
        return "Esimating CBPS with PyTorch using {device}"

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

    def fit(self):
        """Estimate weights optimizing loss function."""
        optimizer = torch.optim.Adam(self.theta, lr=self.lr)

        # Iteratively and update theta vector
        for iteration in range(self.niter):
            optimizer.zero_grad()
            loss = self.loss_function(self.theta, self.X_t, self.w_t)

            if self.reg:
                loss += self.reg * torch.linalg.vector_norm(self.theta, 1)

            loss.backward()
            optimizer.step()

            if self.noi and iteration % 100 == 0:
                print(f"Iteration {iteration}: loss = {loss.item()}")

        return None

    def parameters(self):
        """Return final parameters after optimization."""
        if self.estimand == "ATT":
            out = np.exp(self.X_t @ self.theta)[self.w_t == 0]
        elif self.estimand == "ATE":
            out = np.exp(self.X @ self.theta)

        return out
