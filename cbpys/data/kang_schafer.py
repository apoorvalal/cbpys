import numpy as np


class kang_schafer:
    """Kang-Schafer simulation.

    The simulation can be used to illustrate selection bias of outcome under
    informative nonresponse. Due to the selection bias, the outcome mean for the
    treated group (200) is lower than the control group (220), but the difference
    is not attributed to the treatment. In fact, given the covariates, the outcome
    is generated independent of treatment, i.e., the true average treatment effect
    on the treated (ATT) is zero.

    #### Examples

    ```python
    simulation = kang_schafer.Simulation(200)
    ```

    Attributes:
      size: The total number of treated and control units.
      covariates: Raw covariates, generated as i.i.d. standard normal samples.
      treatment: Unit-level treatment assignments, generated from a logistic
        regression model of the covariates.
      outcome: Unit-level outcomes, generated from a linear regression model of
        the covariates.
    """

    def __init__(self, size: int = 2000):
        """Constructs a `Simulation` instance.

        Args:
          size: The total number of treated and control units.
        """
        self.size = size

        self.covariates = np.random.randn(size, 4)

        propensity_score = 1.0 / (
            1.0 + np.exp(-np.dot(self.covariates, np.array([-1.0, 0.5, -0.25, -0.1])))
        )
        self.treatment = np.random.binomial(1, propensity_score)

        self.outcome = (
            210.0
            + np.dot(self.covariates, np.array([27.4, 13.7, 13.7, 13.7]))
            + np.random.randn(size)
        )  # pyformat: disable

    @property
    def transformed_covariates(self) -> np.ndarray:
        """Returns nonlinear transformations of the raw covariates.

        When the transformed covariates are observed in place of the true
        covariates, both propensity score and outcome regression models become
        misspecified.
        """
        x1, x2, x3, x4 = np.hsplit(self.covariates, 4)
        return np.hstack(
            [
                np.exp(x1 / 2.0),
                x2 / (1 + np.exp(x1)) + 10.0,
                np.power(x1 * x3 / 25 + 0.6, 3),
                np.square(x2 + x4 + 20.0),
            ]
        )
