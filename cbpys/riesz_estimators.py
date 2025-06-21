import cvxpy as cp
import numpy as np

from .baseclass import ATTWeightEstimator

######################################################################


class PenalizedFormEstimator(ATTWeightEstimator):
    """Estimates ATT by solving for balancing weights using a penalized form.

    The objective function is:
    Minimize_w: || (X_control.T @ w) / n_control - mean(X_treat) ||^2 + delta * ||w||^2 / n_control
    where w are the weights for control units.
    """

    def fit(
        self,
        X_treat: np.ndarray,
        X_control: np.ndarray,
        y_treat: np.ndarray,
        y_control: np.ndarray,
        delta: float = 1.0,
        **kwargs,
    ):
        """Fits the PenalizedFormEstimator.

        Args:
            X_treat: Covariates for the treated group.
            X_control: Covariates for the control group.
            y_treat: Outcomes for the treated group.
            y_control: Outcomes for the control group.
            delta (float): Regularization parameter for the L2 penalty on weights. Defaults to 1.0.
            **kwargs: Consumes any additional keyword arguments passed from the base class.
        """
        super().__init__()  # Initialize base attributes
        n_control = X_control.shape[0]
        if n_control == 0:
            raise ValueError("Control group cannot be empty.")

        phi_q_mean = np.mean(
            X_treat, axis=0
        )  # E_1[phi(X)], target mean feature vector
        phi_p = X_control  # Features of control units

        w_var = cp.Variable(n_control)

        # Imbalance term: || (1/n_control) * X_control.T @ w - mean(X_treat) ||^2
        # This measures the squared Euclidean distance between the weighted average of control features
        # and the simple average of treated features.
        # The weights w are scaled by 1/n_control in this term if we think of w as "pseudo-counts".
        # However, the original formulation in the paper seems to be || X_control.T @ w / sum(w) - mean(X_treat) ||^2 if sum(w) is not fixed.
        # Or, if sum(w) is implicitly n_control (e.g. w_i approx 1), then || X_control.T @ w / n_control - mean(X_treat) ||^2.
        # The provided code `imbalance = cp.sum_squares(w @ phi_p / n_control - phi_q)` where `w` is `w_var`
        # suggests `w_var @ phi_p` results in a feature vector. This implies `phi_p` is `(n_control, k_features)`.
        # Then `w_var @ phi_p / n_control` is `(1/n_control) * sum_i w_i * phi_p_i_k` for each feature k.
        # This is correct if w_var is (n_control).
        # So, imbalance = cp.sum_squares( (phi_p.T @ w_var) / n_control - phi_q_mean ) is more standard if w_var is column vector.
        # If w_var is a row vector (1, n_control), then w_var @ phi_p is (1, k_features).
        # cvxpy Variables are column vectors by default. So (phi_p.T @ w_var) is (k_features, 1).
        # (w_var.T @ phi_p / n_control - phi_q_mean)

        # Let's re-verify the original paper's formulation for penalized form.
        # Bruns-Smith et al. (A.1) use: min_w (1/n_0) sum_{i in I_0} (w_i - 1)^2 + lambda * || (1/n_0) sum_{i in I_0} w_i phi(X_i) - (1/n_1) sum_{j in I_1} phi(X_j) ||^2
        # This is slightly different, as it penalizes deviation of w_i from 1.
        # The user's initial code was:
        # imbalance = cp.sum_squares(w @ phi_p / n_control - phi_q) -> this is sum_k ( ( (w @ X_control)_k / n_control ) - mean(X_treat_k) )^2
        # penalty = cp.sum_squares(w) / n_control -> this is sum_i (w_i^2) / n_control

        # If w is cp.Variable(n_control), it's a column vector.
        # X_control is (n_control, n_features).
        # mean_weighted_control_features = (X_control.T @ w_var) / n_control (if we assume sum(w) approx n_control, or normalize later)
        # Or, if w are direct weights for sum: mean_weighted_control_features = (X_control.T @ w_var) / cp.sum(w_var) - this makes it non-convex.
        # The typical approach for this form is to not divide by sum(w_var) inside the optimization.
        # The division by n_control in the penalty implies w_i are of order 1.

        # Sticking to the user's original formulation structure:
        # w @ phi_p should be phi_p.T @ w if w is a column vector, or w.T @ phi_p if w is a row vector.
        # Given cp.Variable is a column vector:
        # weighted_control_features_sum = phi_p.T @ w_var
        # weighted_control_features_mean = weighted_control_features_sum / n_control (this assumes sum w_i = n_control for mean interpretation)

        # Let's use the direct interpretation from user code: w is a row vector effectively.
        # To make w_var (column vector) act as a row vector in matrix multiplication: w_var.T @ phi_p

        imbalance_vector = (
            w_var.T @ phi_p
        ) / n_control - phi_q_mean  # This is a row vector (1, k_features)
        imbalance = cp.sum_squares(imbalance_vector)
        penalty = (
            cp.sum_squares(w_var) / n_control
        )  # Sum of squares of elements of w_var, divided by n_control

        objective = cp.Minimize(imbalance + delta * penalty)
        problem = cp.Problem(objective)
        problem.solve()

        if w_var.value is None:
            raise RuntimeError(
                f"CVXPY optimization failed for PenalizedFormEstimator. Problem status: {problem.status}"
            )

        self.weights = w_var.value
        self.y_treat_mean = np.mean(y_treat)
        self.y_control = y_control
        self.sum_weights = np.sum(self.weights)
        if abs(self.sum_weights) < 1e-9:  # Check for very small sum of weights
            print(
                "Warning: Sum of weights is close to zero in PenalizedFormEstimator."
            )
            # Decide if this should be an error or allow it.
            # If sum_weights is effectively zero, ATT estimate might be unstable.


class ConstrainedFormEstimator(ATTWeightEstimator):
    """Solves for balancing weights using the constrained form.

    Minimize_w: ||w||^2
    Subject to: ||X_control.T @ w - mean(X_treat)||^2 <= delta_constraint
    where w are the weights for control units.
    """

    def fit(
        self,
        X_treat: np.ndarray,
        X_control: np.ndarray,
        y_treat: np.ndarray,
        y_control: np.ndarray,
        delta_constraint: float = 0.1,
        **kwargs,
    ):
        """Fits the ConstrainedFormEstimator.

        Args:
            X_treat: Covariates for the treated group.
            X_control: Covariates for the control group.
            y_treat: Outcomes for the treated group.
            y_control: Outcomes for the control group.
            delta_constraint (float): The upper bound on the squared imbalance. Defaults to 0.1.
                                     Note: name changed from 'delta' to avoid clash with regularization 'delta'.
            **kwargs: Consumes any additional keyword arguments passed from the base class.
        """
        super().__init__()
        n_control = X_control.shape[0]
        if n_control == 0:
            raise ValueError("Control group cannot be empty.")

        phi_q_sum = np.sum(
            X_treat, axis=0
        )  # Target sum of features for control group using weights
        # Original paper seems to use sum_w_i phi_p_i to approximate sum_phi_q_j
        # So constraint is || X_control.T @ w - sum(X_treat_j) ||^2 <= delta
        # Or if we want to match means: || X_control.T @ w / N_treat - mean(X_treat) ||^2 <= delta
        # Let's stick to matching the sum (equivalent to matching mean if weights sum to N_treat)
        # Or matching the mean of X_treat: phi_q = np.mean(X_treat, axis=0)
        # Constraint: || X_control.T @ w - N_treat * phi_q ||^2 <= delta (if w are like counts)
        # If w are normalized weights (sum(w)=1) then || X_control.T @ w - phi_q ||^2 <= delta

        # The provided code's constraint was: cp.sum_squares(w @ phi_p - phi_q) <= delta
        # where phi_q = np.mean(X_treat, axis=0) and phi_p = X_control.
        # This means: sum_k ( (sum_i w_i * X_control_ik) - mean(X_treat_k) )^2 <= delta
        # This seems like a reasonable target for imbalance.

        phi_q_mean = np.mean(X_treat, axis=0)
        phi_p = X_control  # Control features
        w_var = cp.Variable(n_control)  # Weights for control units

        objective = cp.Minimize(
            cp.sum_squares(w_var)
        )  # Minimize L2 norm of weights: ||w||^2

        # Imbalance constraint: || X_control.T @ w - mean(X_treat) ||^2 <= delta_constraint
        # This means the squared Euclidean distance between the weighted sum of control features
        # (where each feature vector X_control_i is weighted by w_i) and the mean of treated features
        # must be less than or equal to delta_constraint.
        imbalance_vector = phi_p.T @ w_var - phi_q_mean
        constraint = [cp.sum_squares(imbalance_vector) <= delta_constraint]

        problem = cp.Problem(objective, constraint)
        problem.solve(
            solver=cp.ECOS
        )  # Added a specific solver, can be SCS as well

        if w_var.value is None or problem.status in [
            cp.INFEASIBLE,
            cp.UNBOUNDED,
            cp.INFEASIBLE_INACCURATE,
        ]:
            # This can happen if delta_constraint is too small (infeasible)
            print(
                f"Warning: CVXPY optimization failed or found no solution for ConstrainedFormEstimator. "
                f"Problem status: {problem.status}. Delta constraint: {delta_constraint}. Weights set to zeros."
            )
            self.weights = np.zeros(n_control)  # Fallback to zero weights
        else:
            self.weights = w_var.value

        self.y_treat_mean = np.mean(y_treat)
        self.y_control = y_control
        self.sum_weights = np.sum(self.weights)
        if abs(self.sum_weights) < 1e-9 and w_var.value is not None:
            print(
                "Warning: Sum of weights is close to zero in ConstrainedFormEstimator."
            )


class RieszRegressionFormEstimator(ATTWeightEstimator):
    """Estimates ATT by solving for the Riesz representer `theta` in feature space, then constructing weights for control units as `w = X_control @ theta`.

    The `theta` is found by solving the regularized linear system:
    `(X_control.T @ X_control / n_control + delta * I) @ theta = mean(X_treat, axis=0)`
    This formulation is based on Proposition 3.1 of Bruns-Smith et al. (arXiv:2311.16948v1)
    for finding a Riesz Representer for ATT with an L2 penalty.
    """

    def fit(
        self,
        X_treat: np.ndarray,
        X_control: np.ndarray,
        y_treat: np.ndarray,
        y_control: np.ndarray,
        delta: float = 1.0,
        **kwargs,
    ):
        """Fits the RieszRegressionFormEstimator.

        Args:
            X_treat: Covariates for the treated group.
            X_control: Covariates for the control group.
            y_treat: Outcomes for the treated group.
            y_control: Outcomes for the control group.
            delta (float): Regularization parameter for the L2 penalty on `theta`. Defaults to 1.0.
            **kwargs: Consumes any additional keyword arguments passed from the base class.
        """
        super().__init__()
        n_control = X_control.shape[0]
        n_features = X_control.shape[1]

        if n_control == 0:
            raise ValueError("Control group cannot be empty.")

        phi_q_mean = np.mean(X_treat, axis=0)  # E_1[phi(X)]
        phi_p = X_control  # X_c

        # System: (phi_p.T @ phi_p / n_control + delta * I) @ theta = phi_q_mean
        # A = phi_p.T @ phi_p / n_control + delta * np.eye(n_features)
        # b = phi_q_mean
        # Solve A @ theta = b for theta.

        # Using cvxpy to be consistent, though np.linalg.solve is direct.
        theta_var = cp.Variable(n_features)
        # Objective based on (A theta - b)^T (A theta - b) or direct formulation if convex.
        # The direct solution for theta in (E_0[phi phi^T] + lambda I) theta = E_1[phi]
        # is theta = (E_0[phi phi^T] + lambda I)^-1 E_1[phi].
        # The weights are then w_i = <phi_i, theta>.

        # Let's use the objective from the original code, but ensure correctness of interpretation.
        # The original objective was:
        # cp.quad_form(theta, phi_p.T @ phi_p) / n_control
        # - 2 * theta @ (phi_p.T @ np.ones(n_control) / n_control * phi_q_mean) # This term was the issue
        # + delta * cp.sum_squares(theta)
        # The term (phi_p.T @ np.ones(n_control) / n_control) is mean(phi_p over control). Let's call it E0_phi_p
        # So it was: theta.T E0[phi_p phi_p.T] theta - 2 * theta.T @ E0_phi_p * phi_q_mean + delta * ||theta||^2
        # If phi_q_mean is a scalar target for E0_phi_p.T @ theta, this looks like a regression of E0_phi_p on theta with target phi_q_mean.
        # This is not standard.

        # Let's use the formulation from Bruns-Smith et al. Prop 3.1 directly.
        # (X_c.T @ X_c / n_c + reg * I) theta = X_t_mean
        # where X_c is X_control (phi_p), X_t_mean is np.mean(X_treat, axis=0) (phi_q_mean)

        E0_phi_phi_T = phi_p.T @ phi_p / n_control
        A_matrix = E0_phi_phi_T + delta * np.eye(n_features)

        try:
            # Solve (A_matrix) theta = phi_q_mean
            theta_value = np.linalg.solve(A_matrix, phi_q_mean)
        except np.linalg.LinAlgError:
            print(
                f"Warning: numpy.linalg.solve failed for RieszRegressionFormEstimator. Using pseudo-inverse. Delta: {delta}"
            )
            theta_value = np.linalg.pinv(A_matrix) @ phi_q_mean

        self.weights = phi_p @ theta_value
        self.y_treat_mean = np.mean(y_treat)
        self.y_control = y_control
        self.sum_weights = np.sum(
            self.weights
        )  # These weights are not constrained to sum to anything specific by default.
        # Normalization happens in estimate_att.
        if abs(self.sum_weights) < 1e-9:
            print(
                "Warning: Sum of weights is close to zero in RieszRegressionFormEstimator."
            )
