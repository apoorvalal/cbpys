import numpy as np
from abc import ABC, abstractmethod


class ATTWeightEstimator(ABC):
    """Abstract Base Class for Average Treatment Effect on the Treated (ATT) Weight Estimators.

    Subclasses are expected to implement the `fit` method to learn weights
    for control units, which can then be used by the `estimate_att` method
    to calculate the ATT.
    """

    def __init__(self):
        """Initializes common attributes for estimators."""
        self.weights: np.ndarray | None = None  # Weights for control units
        self.y_treat_mean: float | None = None  # Mean outcome of treated units
        self.y_control: np.ndarray | None = None  # Outcomes of control units
        self.sum_weights: float | None = None  # Sum of control unit weights
        self.use_dml: bool = False  # Flag indicating if DML was used in fit

    @abstractmethod
    def fit(
        self,
        X_treat: np.ndarray,
        X_control: np.ndarray,
        y_treat: np.ndarray,
        y_control: np.ndarray,
        **kwargs,
    ):
        """Fits the estimator to the data, computing weights or preparing for ATT estimation.

        Args:
            X_treat: Covariates for the treated group. Shape (n_treated, n_features).
            X_control: Covariates for the control group. Shape (n_control, n_features).
            y_treat: Outcomes for the treated group. Shape (n_treated,).
            y_control: Outcomes for the control group. Shape (n_control,).
            **kwargs: Additional estimator-specific parameters (e.g., regularization `delta`).
        """
        pass

    def estimate_att(self) -> float:
        """Estimates the ATT using the computed weights and outcome data.

        This base implementation assumes that `fit` has populated `self.weights`,
        `self.y_treat_mean`, `self.y_control`, and `self.sum_weights` if not using DML,
        or that DML-specific results are handled by overridden methods in subclasses
        (though KernelRieszATT handles DML internally and this method still works).

        Returns:
            The estimated ATT.

        Raises:
            RuntimeError: If the estimator has not been fitted correctly (non-DML path).
            UserWarning: If sum of weights is close to zero, which may lead to unstable estimates.
        """
        # KernelRieszATT handles DML by storing _dml_att_estimate and returning it here.
        # So, if _dml_att_estimate exists, it's returned by subclass's estimate_att.
        # This base method is for non-DML or when subclass doesn't override for DML.

        if (
            self.weights is None
            or self.y_treat_mean is None
            or self.y_control is None
            or self.sum_weights is None
        ):
            # Check if DML was intended and failed, leading to these being None
            if (
                hasattr(self, "_dml_att_estimate")
                and self._dml_att_estimate is not None
            ):
                # This case should be handled by KernelRieszATT's estimate_att
                pass  # Let subclass logic proceed if it has _dml_att_estimate
            elif self.use_dml:  # DML was used, but these attributes are None (implies DML path in fit didn't set them)
                # This state indicates DML was used but perhaps failed to set _dml_att_estimate,
                # and also didn't set the non-DML attributes.
                print(
                    "Warning: DML was used, but attributes for non-DML ATT calculation are missing and no DML estimate available."
                )
                return np.nan
            raise RuntimeError(
                "Estimator has not been fitted correctly for non-DML ATT calculation."
            )

        if abs(self.sum_weights) < 1e-9:
            import warnings

            warnings.warn(
                "Sum of weights is close to zero. ATT estimate might be unstable or NaN.",
                UserWarning,
            )
            return np.nan if self.sum_weights == 0 else self.y_treat_mean

        weighted_control_outcome = (
            np.sum(self.weights * self.y_control) / self.sum_weights
        )
        return self.y_treat_mean - weighted_control_outcome
