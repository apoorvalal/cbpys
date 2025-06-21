import numpy as np
from sklearn.metrics.pairwise import (
    rbf_kernel,
)

from .baseclass import ATTWeightEstimator


class KernelRieszATT(ATTWeightEstimator):
    """Estimates ATT using Riesz representer weights learned in a kernel-induced RKHS. The weights aim to balance the mean embedding of treated and control covariates in the RKHS.

    Features include:
    - Support for various kernel functions.
    - Automatic hyperparameter selection for kernel and regularization parameters via cross-validation (CV).
    - Optional Double Machine Learning (DML) for robust ATT estimation using cross-fitting.
    """

    def __init__(
        self,
        kernel_func=rbf_kernel,
        kernel_param_grid=None,
        reg_param_grid=None,
        cv_folds=5,
        cv_scoring_metric="mmd",
        default_kernel_args=None,
        default_reg_param=0.1,
        use_dml=False,
        dml_folds=None,
        **kwargs,
    ):
        """Initializes the KernelRieszATT estimator.

        Args:
            kernel_func (callable): Function to compute kernel matrix (e.g., `sklearn.metrics.pairwise.rbf_kernel`).
                                    Expected signature: `kernel_func(X1, X2, **kernel_args) -> np.ndarray`.
            kernel_param_grid (dict, optional): Grid of kernel parameters to search via CV.
                                                Example: `{'gamma': [0.1, 1], 'degree': [2, 3]}`.
                                                If None, `default_kernel_args` are used without CV for kernel params.
            reg_param_grid (list, optional): List of regularization parameters (lambda) to search via CV.
                                             If None, `default_reg_param` is used without CV for regularization.
            cv_folds (int): Number of folds for cross-validation of hyperparameters. Default is 5.
            cv_scoring_metric (str): Metric for hyperparameter CV. Currently supports 'mmd'. Default is 'mmd'.
            default_kernel_args (dict, optional): Default kernel arguments if `kernel_param_grid` is None or CV is skipped.
                                                  Example: `{'gamma': 0.1}`.
            default_reg_param (float): Default regularization parameter if `reg_param_grid` is None or CV is skipped. Default is 0.1.
            use_dml (bool): If True, uses Double Machine Learning (cross-fitting) for the final ATT estimation.
                            Hyperparameters are still chosen using CV on the full data first. Default is False.
            dml_folds (int, optional): Number of folds for DML cross-fitting. If None, defaults to `cv_folds`.
            **kwargs: Consumes any additional keyword arguments. This ensures robustness against changes in the base class
                      or future extensions, though currently prints a warning if unexpected arguments are received.

        Attributes:
            best_kernel_args_ (dict): The kernel arguments selected by CV, or the defaults if CV was not performed for kernel parameters.
            best_reg_param_ (float): The regularization parameter selected by CV, or the default if CV was not performed for it.
            _dml_att_estimate (float | None): Stores the final ATT estimate if DML was used. Otherwise, None.
            weights (np.ndarray | None): Control group weights if DML is not used. None if DML is used.
            y_treat_mean (float | None): Mean of treated outcomes if DML is not used. None if DML is used.
            y_control (np.ndarray | None): Control outcomes if DML is not used. None if DML is used.
            sum_weights (float | None): Sum of control weights if DML is not used. None if DML is used.
        """
        super().__init__()
        self.kernel_func = kernel_func
        self.kernel_param_grid = kernel_param_grid
        # Ensure reg_param_grid is a list, using default_reg_param if grid is None
        self.reg_param_grid = (
            reg_param_grid
            if reg_param_grid is not None
            else [default_reg_param]
        )

        self.cv_folds = cv_folds
        self.cv_scoring_metric = cv_scoring_metric

        self.default_kernel_args = (
            default_kernel_args if default_kernel_args is not None else {}
        )
        self.default_reg_param = default_reg_param

        self.use_dml = use_dml
        self.dml_folds = dml_folds if dml_folds is not None else self.cv_folds
        # Important: self.use_dml is now correctly initialized from args for the whole instance
        # super().use_dml = (
        #     self.use_dml
        # )  # Update base class attribute if needed/exists

        if kwargs:
            print(
                f"Warning: KernelRieszATT received unused keyword arguments: {list(kwargs.keys())}"
            )
            # Or raise TypeError if strictness is desired:
            # raise TypeError(f"KernelRieszATT got unexpected keyword arguments: {list(kwargs.keys())}")

        self.best_kernel_args_ = self.default_kernel_args
        self.best_reg_param_ = self.default_reg_param
        self._theta_rie_reg_form = None
        self._dml_att_estimate = None

    def _calculate_mmd_squared(
        self, X_treat_val, X_control_val, weights_c_val, current_kernel_args
    ):
        """Calculates squared MMD between X_treat_val and weighted X_control_val.

        This is used as a scoring metric in cross-validation to select hyperparameters.

        The MMD^2 is computed as:
        || (1/n_t) sum_j phi(X_t_j) - (1/sum_w) sum_i w_i phi(X_c_i) ||^2_H
        = (1/n_t^2) sum K(X_t,X_t')
          - 2/(n_t sum_w) sum w_i K(X_t, X_c_i)
          + (1/(sum_w)^2) sum w_i w_k K(X_c_i, X_c_k)

        Args:
            X_treat_val (np.ndarray): Covariates for treated units in the validation set.
            X_control_val (np.ndarray): Covariates for control units in the validation set.
            weights_c_val (np.ndarray): Weights for control validation samples, intended to balance X_treat_val.
            current_kernel_args (dict): Kernel arguments for the current hyperparameter set.

        Returns:
            float: The calculated squared MMD. Returns np.inf if inputs are invalid or weights sum to zero.
        """
        n_t_val = X_treat_val.shape[0]
        n_c_val = X_control_val.shape[0]

        if n_t_val == 0 or n_c_val == 0:
            return (
                np.inf
            )  # Should not happen with proper CV splits if data is sufficient

        sum_w = np.sum(weights_c_val)
        if abs(sum_w) < 1e-9:
            return np.inf

        K_tt_val = self.kernel_func(
            X_treat_val, X_treat_val, **current_kernel_args
        )
        K_cc_val = self.kernel_func(
            X_control_val, X_control_val, **current_kernel_args
        )
        K_tc_val = self.kernel_func(
            X_treat_val, X_control_val, **current_kernel_args
        )  # (n_t_val, n_c_val)

        term1 = np.sum(K_tt_val) / (n_t_val * n_t_val)
        term2 = -2 * np.sum(K_tc_val @ weights_c_val) / (n_t_val * sum_w)
        term3 = (weights_c_val.T @ K_cc_val @ weights_c_val) / (sum_w * sum_w)

        mmd2 = term1 + term2 + term3
        return max(0, mmd2)

    def _fit_single_config(
        self, X_treat_fit, X_control_fit, current_kernel_args, current_reg_param
    ):
        """Helper to fit weights for control units for a single configuration of kernel and regularization.

        Solves the system: (K_cc + n_control * lambda * I) w = k_target_vector
        where k_target_vector_i = mean_j(K(X_control_fit_i, X_treat_fit_j)).

        Args:
            X_treat_fit (np.ndarray): Covariates for treated units for this specific fit.
            X_control_fit (np.ndarray): Covariates for control units for this specific fit.
            current_kernel_args (dict): Kernel arguments for this fit.
            current_reg_param (float): Regularization parameter for this fit.

        Returns:
            np.ndarray: Computed weights for X_control_fit. Returns empty array if fitting fails or inputs are invalid.
        """
        n_control_fit = X_control_fit.shape[0]
        n_treat_fit = X_treat_fit.shape[0]

        if (
            n_control_fit == 0 or n_treat_fit == 0
        ):  # Should be handled by CV split checks
            return np.array([])  # Or raise error

        K_cc_fit = self.kernel_func(
            X_control_fit, X_control_fit, **current_kernel_args
        )
        K_xc_xt_fit = self.kernel_func(
            X_control_fit, X_treat_fit, **current_kernel_args
        )
        k_target_vector_fit = np.mean(K_xc_xt_fit, axis=1)

        A_matrix_fit = K_cc_fit + n_control_fit * current_reg_param * np.eye(
            n_control_fit
        )

        try:
            weights = np.linalg.solve(A_matrix_fit, k_target_vector_fit)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            print(
                f"Warning (_fit_single_config): Using pseudo-inverse. Reg: {current_reg_param}, Kernel: {current_kernel_args}"
            )
            weights = np.linalg.pinv(A_matrix_fit) @ k_target_vector_fit
        return weights

    def fit(
        self,
        X_treat: np.ndarray,
        X_control: np.ndarray,
        y_treat: np.ndarray,
        y_control: np.ndarray,
        delta: float = None,
        **kwargs_fit,
    ):
        """Fits the KernelRieszATT estimator.

        This method handles:
        1. Hyperparameter selection via cross-validation (if `kernel_param_grid` or `reg_param_grid` are set for search).
           The best kernel arguments and regularization parameter are stored in `self.best_kernel_args_` and `self.best_reg_param_`.
        2. Final ATT estimation, either:
           a. Using Double Machine Learning (DML) cross-fitting if `self.use_dml` is True. The result is stored in `self._dml_att_estimate`.
           b. Fitting on the full data using the best/default parameters if DML is not used. Results are stored in `self.weights`, etc.

        The `delta` argument here serves as a direct override for the regularization parameter for this specific `fit` call,
        bypassing CV for `reg_param` if provided.

        Args:
            X_treat (np.ndarray): Covariates for the treated group.
            X_control (np.ndarray): Covariates for the control group.
            y_treat (np.ndarray): Outcomes for the treated group.
            y_control (np.ndarray): Outcomes for the control group.
            delta (float, optional): If provided, this value is used as the regularization parameter,
                                     overriding `default_reg_param` and any CV search for `reg_param`.
            **kwargs_fit: Consumes any additional keyword arguments.
        """
        from sklearn.model_selection import (  # Keep import here for clarity on usage
            KFold,
            ParameterGrid,
        )

        # Initialize/Reset some attributes at the start of fit
        self._dml_att_estimate = None
        self.weights = None
        self.y_treat_mean = None
        self.y_control = None
        self.sum_weights = None
        # Base class use_dml is set in __init__, specific to instance config
        # self.use_dml = self.use_dml # This is already set by init

        n_control_orig = X_control.shape[0]
        n_treat_orig = X_treat.shape[0]

        if n_control_orig == 0 or n_treat_orig == 0:
            raise ValueError(
                "Treated and control groups must not be empty for fitting."
            )

        # Determine if CV is needed
        run_cv = False
        if self.kernel_param_grid:  # If grid for kernel params exists
            # Create a list of actual parameter dictionaries from the grid
            param_list = list(ParameterGrid(self.kernel_param_grid))
            if len(param_list) > 1 or (
                len(param_list) == 1
                and param_list[0] != self.default_kernel_args
            ):
                run_cv = True
        if len(self.reg_param_grid) > 1:
            run_cv = True
        if (
            delta is not None
        ):  # User override for reg_param, CV for reg_param is skipped.
            current_reg_params_to_search = [delta]
            if (
                len(self.reg_param_grid) > 1
            ):  # if delta overrides a grid search for reg_param
                run_cv = bool(
                    self.kernel_param_grid
                )  # CV only if kernel_param_grid is active
        else:
            current_reg_params_to_search = self.reg_param_grid

        if (
            not run_cv
            and not self.kernel_param_grid
            and len(current_reg_params_to_search) == 1
        ):
            # No CV, use default/provided single parameters
            self.best_kernel_args_ = self.default_kernel_args
            self.best_reg_param_ = current_reg_params_to_search[0]
            # Proceed to fit with these params directly (logic further down)

        elif self.cv_folds < 2:  # Not enough folds for CV
            print(
                "Warning: cv_folds < 2, skipping CV. Using default parameters."
            )
            self.best_kernel_args_ = self.default_kernel_args
            self.best_reg_param_ = current_reg_params_to_search[
                0
            ]  # Use first from list if grid was given
            # Proceed to fit with these params directly
            run_cv = False

        if run_cv:
            # Prepare kernel parameter sets to iterate over
            kernel_configs = (
                list(ParameterGrid(self.kernel_param_grid))
                if self.kernel_param_grid
                else [self.default_kernel_args]
            )

            best_score = np.inf

            # Ensure indices are for full X_treat and X_control passed to fit()
            # CV splits will be on the indices of X_treat and X_control respectively.
            kf_treat = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=42
            )
            kf_control = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=43
            )  # Different seed

            for reg_param_val in current_reg_params_to_search:
                for kernel_args_val in kernel_configs:
                    fold_scores = []

                    # CV over indices of X_treat and X_control
                    # Note: This aligns folds for treated and control by iteration, not by shared indices.
                    # This is typical for two-sample problems.
                    treat_splits = list(kf_treat.split(X_treat))
                    control_splits = list(kf_control.split(X_control))

                    for i in range(self.cv_folds):
                        _, val_idx_t = treat_splits[i]
                        _, val_idx_c = control_splits[i]

                        X_t_val_fold = X_treat[val_idx_t]
                        X_c_val_fold = X_control[val_idx_c]

                        if (
                            X_t_val_fold.shape[0] == 0
                            or X_c_val_fold.shape[0] == 0
                        ):
                            # This can happen if a fold ends up empty for one group, especially with small N
                            # Or if cv_folds > N_group. KFold should handle this for N < n_splits by making fewer splits.
                            # However, if one group is much smaller than cv_folds, it's an issue.
                            print(
                                f"Warning: Empty validation fold for T({X_t_val_fold.shape[0]}) or C({X_c_val_fold.shape[0]}). Skipping fold."
                            )
                            # Consider how to handle this score. Could be np.inf or skip.
                            # If KFold itself produces empty val_idx, that's an issue with KFold usage / data size.
                            # For now, if it happens, this fold score is bad.
                            fold_scores.append(np.inf)
                            continue

                        # Fit weights FOR THE VALIDATION FOLD using current params
                        # These weights are specific to X_c_val_fold to balance X_t_val_fold
                        weights_val_fold = self._fit_single_config(
                            X_t_val_fold,
                            X_c_val_fold,
                            kernel_args_val,
                            reg_param_val,
                        )

                        if (
                            weights_val_fold.size == 0
                            or np.sum(np.isnan(weights_val_fold)) > 0
                        ):  # check if _fit_single_config failed
                            fold_scores.append(np.inf)
                            continue

                        if self.cv_scoring_metric == "mmd":
                            score = self._calculate_mmd_squared(
                                X_t_val_fold,
                                X_c_val_fold,
                                weights_val_fold,
                                kernel_args_val,
                            )
                            fold_scores.append(score)
                        else:
                            raise ValueError(
                                f"Unsupported cv_scoring_metric: {self.cv_scoring_metric}"
                            )

                    current_mean_score = (
                        np.mean([s for s in fold_scores if np.isfinite(s)])
                        if any(np.isfinite(s) for s in fold_scores)
                        else np.inf
                    )

                    if current_mean_score < best_score:
                        best_score = current_mean_score
                        self.best_kernel_args_ = kernel_args_val
                        self.best_reg_param_ = reg_param_val

            if np.isinf(best_score):
                print(
                    "Warning: CV did not find valid parameters. Falling back to defaults."
                )
                self.best_kernel_args_ = self.default_kernel_args
                self.best_reg_param_ = (
                    self.default_reg_param if delta is None else delta
                )

        # --- Final parameter determination complete (either by CV or defaults) ---
        final_reg_param = delta if delta is not None else self.best_reg_param_
        final_kernel_args = self.best_kernel_args_

        if self.use_dml:
            if self.dml_folds < 2:
                print(
                    "Warning: dml_folds < 2, DML cannot be performed. Fitting on full data instead."
                )
                self.use_dml = False  # Fallback to non-DML fit

        if self.use_dml:
            # DML Cross-fitting for ATT estimation
            # Ensure dml_folds is not too large for the data
            min_samples_t = n_treat_orig // self.dml_folds
            min_samples_c = n_control_orig // self.dml_folds
            if (
                min_samples_t < 1 or min_samples_c < 1
            ):  # Crude check, KFold might handle n_splits > n_samples better
                print(
                    f"Warning: dml_folds ({self.dml_folds}) may be too large for sample sizes (T:{n_treat_orig}, C:{n_control_orig}). Reducing dml_folds."
                )
                self.dml_folds = min(
                    n_treat_orig, n_control_orig, self.dml_folds
                )
                if self.dml_folds < 2:
                    print(
                        "Fallback: Not enough data for DML. Fitting on full data."
                    )
                    self.use_dml = False  # Fallback if still too few folds

        if self.use_dml:
            print(
                f"Info: Performing DML with {self.dml_folds} folds using reg={final_reg_param}, kernel={final_kernel_args}"
            )
            kf_treat_dml = KFold(
                n_splits=self.dml_folds, shuffle=True, random_state=123
            )
            kf_control_dml = KFold(
                n_splits=self.dml_folds, shuffle=True, random_state=124
            )

            fold_att_estimates = []

            treat_dml_splits = list(kf_treat_dml.split(X_treat))
            control_dml_splits = list(kf_control_dml.split(X_control))

            for i in range(self.dml_folds):
                # For DML, we fit on one part and estimate on the other.
                # Standard DML for ATT: estimate nuisance (weights) on OOF, then form ATT on IF.
                # Simplified cross-fitting: estimate ATT per fold and average.
                # Here, "fold k" is the estimation set.
                _, est_idx_t = treat_dml_splits[i]
                _, est_idx_c = control_dml_splits[i]

                X_t_k, y_t_k = X_treat[est_idx_t], y_treat[est_idx_t]
                X_c_k, y_c_k = X_control[est_idx_c], y_control[est_idx_c]

                if X_t_k.shape[0] == 0 or X_c_k.shape[0] == 0:
                    print(
                        f"Warning (DML fold {i}): Empty data split for T({X_t_k.shape[0]}) or C({X_c_k.shape[0]}). Skipping fold."
                    )
                    continue

                weights_k = self._fit_single_config(
                    X_t_k, X_c_k, final_kernel_args, final_reg_param
                )

                if weights_k.size == 0 or np.sum(np.isnan(weights_k)) > 0:
                    print(
                        f"Warning (DML fold {i}): Weight computation failed. Skipping fold."
                    )
                    continue

                sum_weights_k = np.sum(weights_k)
                if abs(sum_weights_k) < 1e-9:
                    print(
                        f"Warning (DML fold {i}): Sum of weights is zero. Skipping ATT for this fold."
                    )
                    # This fold won't contribute to the mean ATT. Or could be np.nan.
                    continue

                att_k = (
                    np.mean(y_t_k) - np.sum(weights_k * y_c_k) / sum_weights_k
                )
                fold_att_estimates.append(att_k)

            if not fold_att_estimates:  # All folds failed
                print(
                    "Error: DML estimation failed for all folds. No ATT estimate produced."
                )
                self._dml_att_estimate = np.nan
            else:
                self._dml_att_estimate = np.mean(fold_att_estimates)

            # For DML, we don't store overall self.weights, y_treat_mean, etc.
            # The estimate is self._dml_att_estimate.
            self.weights = (
                None  # Indicate that single overall weights are not applicable.
            )

        else:  # Not using DML, fit on full data
            self.weights = self._fit_single_config(
                X_treat, X_control, final_kernel_args, final_reg_param
            )

            if self.weights.size == 0:
                raise RuntimeError(
                    f"Final weight computation failed with params: {final_kernel_args}, reg: {final_reg_param}"
                )

            self.y_treat_mean = np.mean(y_treat)
            self.y_control = y_control
            self.sum_weights = np.sum(self.weights)

            if abs(self.sum_weights) < 1e-9:
                print(
                    f"Warning: Sum of weights is close to zero in KernelRieszATT (final fit). Sum: {self.sum_weights}, Reg: {final_reg_param}, Kernel: {final_kernel_args}"
                )

    def estimate_att(self) -> float:
        """Estimates the ATT.

        If DML was used in fit(), returns the DML-aggregated ATT.
        Otherwise, computes ATT from the single fit on the full data.
        """
        if self._dml_att_estimate is not None:
            return self._dml_att_estimate

        # Standard non-DML estimation
        if (
            self.weights is None
            or self.y_treat_mean is None
            or self.y_control is None
            or self.sum_weights is None
        ):
            # This can happen if fit was called with use_dml=True but DML failed and self.weights was set to None
            if self.use_dml:  # and DML failed to produce an estimate
                print(
                    "Error: DML was used but failed to produce an estimate, and no fallback weights available."
                )
                return np.nan
            raise RuntimeError(
                "Estimator has not been fitted correctly (non-DML path)."
            )

        if (
            abs(self.sum_weights) < 1e-9
        ):  # Check for sum_weights being effectively zero
            print(
                "Warning (estimate_att): Sum of weights is close to zero. ATT estimate might be unstable or NaN."
            )
            # Depending on y_treat_mean, this could be y_treat_mean or NaN.
            # If sum_weights is zero, weighted_control_outcome is undefined.
            return (
                np.nan if self.sum_weights == 0 else self.y_treat_mean
            )  # if sum_weights is tiny but non-zero.
            # np.nan is safer if sum_weights is zero.

        weighted_control_outcome = (
            np.sum(self.weights * self.y_control) / self.sum_weights
        )
        return self.y_treat_mean - weighted_control_outcome

    # def fit(
    #     self,
    #     X_treat: np.ndarray,
    #     X_control: np.ndarray,
    #     y_treat: np.ndarray,
    #     y_control: np.ndarray,
    #     delta: float = None,
    # ):
    #     """
    #     # This is a placeholder for the diff tool, the actual method is above.
    #     # The diff tool requires the original method signature line to be present in the SEARCH block.
    #     # However, the SEARCH block was for the end of the previous fit method.
    #     # This line is just to satisfy the tool's parsing if it looks for method definition.
    #     # The actual changes are applied to the KernelRieszATT's fit and estimate_att methods.
    #     pass
    #     # Placeholder for diff utility
    #     K_tt_val: Kernel matrix for treated validation samples (n_t_val x n_t_val)
    #     K_cc_val: Kernel matrix for control validation samples (n_c_val x n_c_val)
    #     K_tc_val: Kernel matrix between treated (rows) and control (cols) validation samples (n_t_val x n_c_val)
    #     weights_c_val: Weights for control validation samples (n_c_val,)
    #     """
    #     if n_t_val == 0 or n_c_val == 0:
    #         return np.inf

    #     sum_w = np.sum(weights_c_val)
    #     if (
    #         abs(sum_w) < 1e-9
    #     ):  # Avoid division by zero if weights sum to nothing
    #         return np.inf

    #     term1 = np.sum(K_tt_val) / (n_t_val * n_t_val)

    #     # term2: -2 * E_t[<phi(X_t), sum_i w_i phi(X_c_i)/sum_w>] = -2/(n_t * sum_w) * sum_j sum_i w_i K(X_t_j, X_c_i)
    #     # K_tc_val is (n_t_val, n_c_val). K_tc_val[j,i] is K(X_t_j, X_c_i)
    #     # sum_i w_i K(X_t_j, X_c_i) is K_tc_val[j,:] @ weights_c_val
    #     # sum_j (K_tc_val[j,:] @ weights_c_val) is np.sum(K_tc_val @ weights_c_val)
    #     term2 = -2 * np.sum(K_tc_val @ weights_c_val) / (n_t_val * sum_w)

    #     # term3: E_wc[<sum_i w_i phi(X_c_i)/sum_w, sum_k w_k phi(X_c_k)/sum_w>] = 1/(sum_w)^2 * w.T @ K_cc_val @ w
    #     term3 = (weights_c_val.T @ K_cc_val @ weights_c_val) / (sum_w * sum_w)

    #     mmd2 = term1 + term2 + term3
    #     return max(0, mmd2)  # MMD^2 should be non-negative

    # def _fit_single_config(
    #     self,
    #     X_treat,
    #     X_control,
    #     current_kernel_args,
    #     current_reg_param,
    #     n_treat_fit,
    #     n_control_fit,
    # ):
    #     """Helper to fit weights for a single configuration of kernel_args and reg_param."""
    #     K_cc_fit = self.kernel_func(X_control, X_control, **current_kernel_args)
    #     K_xc_xt_fit = self.kernel_func(
    #         X_control, X_treat, **current_kernel_args
    #     )
    #     k_target_vector_fit = np.mean(K_xc_xt_fit, axis=1)

    #     # Using n_control_fit * lambda for regularization
    #     A_matrix_fit = K_cc_fit + n_control_fit * current_reg_param * np.eye(
    #         n_control_fit
    #     )

    #     try:
    #         weights = np.linalg.solve(A_matrix_fit, k_target_vector_fit)
    #     except np.linalg.LinAlgError:
    #         weights = np.linalg.pinv(A_matrix_fit) @ k_target_vector_fit
    #     return weights

    # def fit(
    #     self,
    #     X_treat: np.ndarray,
    #     X_control: np.ndarray,
    #     y_treat: np.ndarray,
    #     y_control: np.ndarray,
    #     delta: float = None,
    # ):
    #     """
    #     Fits the estimator by finding weights for control units.

    #     The weights 'w' are found by solving a kernel ridge regression type problem:
    #     (K_cc + n_control * lambda * I) w = K_ct_mean_col
    #     where:
    #     - K_cc is the kernel matrix of X_control vs X_control.
    #     - K_ct_mean_col is a vector representing the mean kernel evaluation between
    #       each control unit and all treated units: K_ct_mean_col_i = mean_j(K(X_control_i, X_treat_j)).
    #     - lambda is the regularization parameter (self.reg_param or overridden by delta).

    #     Args:
    #         X_treat: Covariates for the treated group.
    #         X_control: Covariates for the control group.
    #         y_treat: Outcomes for the treated group.
    #         y_control: Outcomes for the control group.
    #         delta: If provided, overrides the instance's reg_param for this fit.
    #     """
    #     n_control = X_control.shape[0]
    #     n_treat = X_treat.shape[0]

    #     if n_control == 0 or n_treat == 0:
    #         raise ValueError("Treated and control groups must not be empty.")

    #     current_reg_param = delta if delta is not None else self.reg_param

    #     # K_cc: Kernel matrix between control samples and control samples (n_control x n_control)
    #     K_cc = self.kernel_func(X_control, X_control, **self.kernel_args)

    #     # K_xc_xt: Kernel matrix between control samples (rows) and treated samples (cols) (n_control x n_treat)
    #     K_xc_xt = self.kernel_func(X_control, X_treat, **self.kernel_args)

    #     # K_ct_mean_col: Average kernel value between each control point and all treated points (n_control x 1)
    #     # This is mean_j K(X_control_i, X_treat_j) for each i.
    #     k_target_vector = np.mean(K_xc_xt, axis=1)  # Shape (n_control,)

    #     # System: (K_cc + n_control * lambda * I) w = k_target_vector
    #     # Using n_control * lambda for regularization makes lambda scale-invariant to n_control.
    #     # Some literature uses lambda*I, others n_control*lambda*I. We'll use n_control * lambda for now.
    #     A_matrix = K_cc + n_control * current_reg_param * np.eye(n_control)

    #     try:
    #         weights = np.linalg.solve(A_matrix, k_target_vector)
    #     except np.linalg.LinAlgError:
    #         print(
    #             f"Warning: numpy.linalg.solve failed for KernelRieszATT. Using pseudo-inverse. Reg param: {current_reg_param}"
    #         )
    #         weights = np.linalg.pinv(A_matrix) @ k_target_vector

    #     self.weights = weights
    #     self.y_treat_mean = np.mean(y_treat)
    #     self.y_control = y_control
    #     self.sum_weights = np.sum(self.weights)

    #     if abs(self.sum_weights) < 1e-9:
    #         print(
    #             f"Warning: Sum of weights is close to zero in KernelRieszATT. Sum: {self.sum_weights}, Reg param: {current_reg_param}"
    #         )
    #         # This might indicate that k_target_vector was near zero or reg_param is very high.
    #         # Or K_cc is nearly singular and heavily regularized.
    #         # If sum_weights is zero, ATT estimate will be problematic due to division by sum_weights.
