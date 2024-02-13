"""Covariate Balancing Propensity Score estimation in PyTorch."""

import torch
from tqdm import tqdm
from functools import cached_property

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
            self.X = torch.concat((torch.ones((self.n, 1)).to(device), self.X), 1)

        # Parameters theta to be optimized
        self.theta = torch.randn(self.p + 1, requires_grad=True, device=device)

    def __repr__(self):
        """Representation of the class."""
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

            if self.noi and iteration % 100 == 0:
                print(f"Iteration {iteration}: loss = {loss.item()}")

        return self.theta

    def weights(self):
        """Return final weights after optimization."""
        weights = self.fit
        if self.estimand == "ATT":
            out = torch.exp(self.X @ weights)[self.W == 0]
        elif self.estimand == "ATE":
            out = torch.exp(self.X @ weights)

        return out
