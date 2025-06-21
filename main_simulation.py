import numpy as np
import sklearn.preprocessing as sp
import sklearn.kernel_approximation as ska

# Assuming att_estimators.py is in the same directory or accessible via PYTHONPATH
from att_estimators import PenalizedFormEstimator, ConstrainedFormEstimator, RieszRegressionFormEstimator, KernelRieszATT
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

class kang_schafer:
    """Kang-Schafer simulation with built-in feature expansion methods.
    Attributes:
      size: The total number of treated and control units.
      covariates: Raw covariates (4 dimensions), i.i.d. standard normal.
      treatment: Unit-level treatment assignments (0 or 1).
      outcome: Unit-level outcomes.
    """

    def __init__(self, size: int = 2000, seed: int = 42):
        self.size = size
        self._rng = np.random.default_rng(seed)
        self.covariates = self._rng.standard_normal((size, 4))
        propensity_score = 1.0 / (
            1.0 + np.exp(-np.dot(self.covariates, np.array([-1.0, 0.5, -0.25, -0.1])))
        )
        self.treatment = self._rng.binomial(1, propensity_score)
        self.outcome = (
            210.0
            + np.dot(self.covariates, np.array([27.4, 13.7, 13.7, 13.7]))
            + self._rng.standard_normal(size)
        )

    @property
    def transformed_covariates(self) -> np.ndarray:
        x1, x2, x3, x4 = np.hsplit(self.covariates, 4)
        return np.hstack(
            [
                np.exp(x1 / 2.0),
                x2 / (1 + np.exp(x1)) + 10.0,
                np.power(x1 * x3 / 25 + 0.6, 3),
                np.square(x2 + x4 + 20.0),
            ]
        )

    def expand_polynomial_features(self, degree: int = 2, **kwargs) -> np.ndarray:
        poly = sp.PolynomialFeatures(degree=degree, include_bias=False, **kwargs)
        return poly.fit_transform(self.covariates)

    def expand_rbf_features(
        self, n_components: int = 100, gamma: float = 1.0, **kwargs
    ) -> np.ndarray:
        random_state = kwargs.pop("random_state", self._rng.integers(1e6))
        rbf_sampler = ska.RBFSampler(
            n_components=n_components, gamma=gamma, random_state=random_state, **kwargs
        )
        return rbf_sampler.fit_transform(self.covariates)


if __name__ == "__main__":
    simulation = kang_schafer(size=2000, seed=42)

    print("Generating Polynomial features (degree 2)...")
    features_expanded = simulation.expand_polynomial_features(degree=2)
    # To use raw covariates for KernelRieszATT:
    # features_for_kernel_method = simulation.covariates
    # Or RBF features from simulation.expand_rbf_features
    features_for_kernel_method = features_expanded # Using polynomial for initial test

    print(f"Expanded feature shape (for linear methods): {features_expanded.shape}")
    print(f"Feature shape for KernelRieszATT: {features_for_kernel_method.shape}")


    is_treated = simulation.treatment == 1
    is_control = ~is_treated

    X_treat_expanded = features_expanded[is_treated]
    X_control_expanded = features_expanded[is_control]

    # Data for KernelRieszATT (can be different from linearly expanded features)
    X_treat_kernel = features_for_kernel_method[is_treated]
    X_control_kernel = features_for_kernel_method[is_control]

    y_treat = simulation.outcome[is_treated]
    y_control = simulation.outcome[is_control]

    n_treat = X_treat_expanded.shape[0]
    n_control = X_control_expanded.shape[0]

    print(f"Number of treated units: {n_treat}")
    print(f"Number of control units: {n_control}")

    if n_treat == 0 or n_control == 0:
        print("Not enough treated or control units to proceed.")
        exit()

    results = {"True ATT": 0.0}

    # --- Penalized Form ---
    print("\n--- Penalized Form Estimator ---")
    estimator_penalized = PenalizedFormEstimator()
    try:
        estimator_penalized.fit(X_treat_expanded, X_control_expanded, y_treat, y_control, delta=1.0)
        results["ATT (Penalized)"] = estimator_penalized.estimate_att()
    except Exception as e:
        print(f"Error in PenalizedFormEstimator: {e}")
        results["ATT (Penalized)"] = float('nan')

    # --- Constrained Form ---
    print("\n--- Constrained Form Estimator ---")
    estimator_constrained = ConstrainedFormEstimator()
    try:
        estimator_constrained.fit(X_treat_expanded, X_control_expanded, y_treat, y_control, delta=0.5)
        results["ATT (Constrained)"] = estimator_constrained.estimate_att()
    except Exception as e:
        print(f"Error in ConstrainedFormEstimator: {e}")
        results["ATT (Constrained)"] = float('nan')

    # --- Riesz Regression Form ---
    print("\n--- Riesz Regression Form Estimator ---")
    estimator_riesz_reg = RieszRegressionFormEstimator()
    try:
        estimator_riesz_reg.fit(X_treat_expanded, X_control_expanded, y_treat, y_control, delta=1.0)
        results["ATT (Riesz Reg.)"] = estimator_riesz_reg.estimate_att()
    except Exception as e:
        print(f"Error in RieszRegressionFormEstimator: {e}")
        results["ATT (Riesz Reg.)"] = float('nan')

    # --- Kernel Riesz ATT Estimator ---
    print("\n--- Kernel Riesz ATT Estimator (RBF Kernel on Polynomial Features) ---")
    # Using polynomial features as input to KernelRieszATT, with an RBF kernel on top of these features
    # This means K(poly(X1), poly(X2)).
    # One might typically use raw covariates X directly with KernelRieszATT.
    # Test without CV first
    print("\n--- Kernel Riesz ATT Estimator (RBF Kernel on Polynomial Features, No CV) ---")
    kernel_att_estimator_no_cv = KernelRieszATT(
        kernel_func=rbf_kernel,
        default_kernel_args={'gamma': 0.1}, # Gamma might need tuning
        default_reg_param=0.01
    )
    try:
        kernel_att_estimator_no_cv.fit(X_treat_kernel, X_control_kernel, y_treat, y_control)
        results["ATT (Kernel Riesz RBF on Poly, No CV)"] = kernel_att_estimator_no_cv.estimate_att()
        print(f"  Used params: reg={kernel_att_estimator_no_cv.best_reg_param_}, kernel={kernel_att_estimator_no_cv.best_kernel_args_}")
    except Exception as e:
        print(f"Error in KernelRieszATT (No CV): {e}")
        results["ATT (Kernel Riesz RBF on Poly, No CV)"] = float('nan')

    # Test with CV
    print("\n--- Kernel Riesz ATT Estimator (RBF Kernel on Polynomial Features, With CV) ---")
    rbf_param_grid = {'gamma': [0.001, 0.01, 0.1, 1.0]}
    reg_param_search_grid = [0.001, 0.01, 0.1]

    kernel_att_estimator_cv = KernelRieszATT(
        kernel_func=rbf_kernel,
        kernel_param_grid=rbf_param_grid,
        reg_param_grid=reg_param_search_grid,
        cv_folds=3, # Smaller folds for faster example run (used for hyperparam CV)
        cv_scoring_metric='mmd',
        default_kernel_args={'gamma': 0.1}, # Fallback if CV fails badly
        default_reg_param=0.01,
        # DML parameters
        use_dml=False # First run with CV but no DML
    )
    try:
        kernel_att_estimator_cv.fit(X_treat_kernel, X_control_kernel, y_treat, y_control)
        results["ATT (Kernel Riesz RBF on Poly, CV no DML)"] = kernel_att_estimator_cv.estimate_att()
        print(f"  Best params from CV: reg={kernel_att_estimator_cv.best_reg_param_}, kernel={kernel_att_estimator_cv.best_kernel_args_}")
    except Exception as e:
        print(f"Error in KernelRieszATT (CV no DML): {e}")
        results["ATT (Kernel Riesz RBF on Poly, CV no DML)"] = float('nan')

    # Test with CV and DML
    print("\n--- Kernel Riesz ATT Estimator (RBF Kernel on Polynomial Features, With CV and DML) ---")
    kernel_att_estimator_cv_dml = KernelRieszATT(
        kernel_func=rbf_kernel,
        kernel_param_grid=rbf_param_grid, # Same grid for hyperparams
        reg_param_grid=reg_param_search_grid,
        cv_folds=3, # For hyperparameter CV
        cv_scoring_metric='mmd',
        default_kernel_args={'gamma': 0.1},
        default_reg_param=0.01,
        # DML parameters
        use_dml=True,
        dml_folds=3 # Can be same or different from cv_folds for hyperparams
    )
    try:
        kernel_att_estimator_cv_dml.fit(X_treat_kernel, X_control_kernel, y_treat, y_control)
        results["ATT (Kernel Riesz RBF on Poly, CV+DML)"] = kernel_att_estimator_cv_dml.estimate_att()
        print(f"  Best params from CV (used for DML folds): reg={kernel_att_estimator_cv_dml.best_reg_param_}, kernel={kernel_att_estimator_cv_dml.best_kernel_args_}")
    except Exception as e:
        print(f"Error in KernelRieszATT (CV+DML): {e}")
        results["ATT (Kernel Riesz RBF on Poly, CV+DML)"] = float('nan')


    # --- Kernel Riesz ATT Estimator (Linear Kernel on Polynomial Features) ---
    # For linear kernel, there are no kernel_args typically, so kernel_param_grid can be None or {}.
    # We can still search for the best regularization parameter.
    print("\n--- Kernel Riesz ATT Estimator (Linear Kernel on Polynomial Features, CV for reg_param) ---")
    kernel_att_linear_estimator_cv = KernelRieszATT(
        kernel_func=linear_kernel,
        kernel_param_grid=None, # No hyperparameters for linear_kernel in sklearn by default
        reg_param_grid=[0.0001, 0.001, 0.01, 0.1, 1.0], # Search reg_param
        cv_folds=3,
        default_kernel_args={},
        default_reg_param=0.01
    )
    try:
        kernel_att_linear_estimator_cv.fit(X_treat_kernel, X_control_kernel, y_treat, y_control)
        results["ATT (Kernel Riesz Linear on Poly, CV reg)"] = kernel_att_linear_estimator_cv.estimate_att()
        print(f"  Best params from CV: reg={kernel_att_linear_estimator_cv.best_reg_param_}, kernel={kernel_att_linear_estimator_cv.best_kernel_args_}")
    except Exception as e:
        print(f"Error in KernelRieszATT (Linear Kernel): {e}")
        results["ATT (Kernel Riesz Linear on Poly)"] = float('nan')


    # --- Results ---
    print("\n--- Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    att_unadjusted = np.mean(y_treat) - np.mean(y_control)
    print(f"Estimated ATT (Unadjusted): {att_unadjusted:.4f}")

    # --- Sanity Check: KernelRieszATT with raw covariates and RBF kernel ---
    print("\n--- Kernel Riesz ATT Estimator (RBF Kernel on Raw Covariates) ---")
    X_treat_raw = simulation.covariates[is_treated]
    X_control_raw = simulation.covariates[is_control]

    if X_treat_raw.shape[0] > 0 and X_control_raw.shape[0] > 0:
        kernel_att_raw_rbf_estimator = KernelRieszATT(
            kernel_func=rbf_kernel,
            kernel_args={'gamma': 0.1}, # Gamma is sensitive to feature scaling. Raw covariates are N(0,1).
            reg_param=0.01
        )
        try:
            kernel_att_raw_rbf_estimator.fit(X_treat_raw, X_control_raw, y_treat, y_control)
            att_kernel_raw_rbf = kernel_att_raw_rbf_estimator.estimate_att()
            print(f"ATT (Kernel Riesz RBF on Raw): {att_kernel_raw_rbf:.4f}")
        except Exception as e:
            print(f"Error in KernelRieszATT (RBF on Raw): {e}")
            print("ATT (Kernel Riesz RBF on Raw): nan")
    else:
        print("Not enough data for Kernel Riesz RBF on Raw test.")

```
