"""Functions to compute differences between treatment and control groups."""

import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standarized_diffs(df_vars, treat_var, metric="smd", exclude=None, weights=None):
    """Compute the standardized differences between the treatment and control groups.

    To compare the quality of the weighting, we can check the standarized differences
    between groups in our sample across the covariates used. This function calculates
    to types of metrics: the standarized mean differences (SMD) and the absolute
    standarized mean differences (ASMD). The function will return a dataframe with the
    differences for each variable.

    Args:
        df_vars (pd.DataFrame): dataframe with the covariates
        treat_var (str): name of the treatment variable
        metric (str, optional): metric to use. Defaults to "smd".
        exclude (list, optional): list of variables to exclude. Defaults to None.
        weights (np.array, optional): weights to use in the calculation. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with the standarized differences
    """
    # Split dataframe by treatment status

    # Define metrics dict for conventions
    metrics = {
        "smd": "Standarized Mean Differences",
        "asmd": "Absolute Standarized Mean Differences",
    }

    # Make the function work even if numpy arrays are passed directly rather than
    # just a DataFrame
    if isinstance(df_vars, pd.DataFrame):
        X = df_vars.values
        treat = df_vars[treat_var].values
        index = df_vars.columns
    else:
        X = df_vars
        treat = treat_var
        index = [f"V{n}" for n in range(X.shape[1])]

    # Transform to torch if not already
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float().to(device)
        treat = torch.from_numpy(treat).float().to(device)

    if weights is not None:
        mean_treat = X[treat == 1, :].mean(axis=0)
        mean_control = (X[treat == 0, :].T @ weights) / weights.sum()

    else:
        mean_treat = X[treat == 1, :].mean(axis=0)
        mean_control = X[treat == 0, :].mean(axis=0)

    # Following Imbens and Rubin (2015) we use the pooled standard deviation
    # across both groups
    std_vars = torch.sqrt(
        (torch.var(X[treat == 1, :]) + torch.var(X[treat == 0, :])) / 2
    )

    # Standardized differences
    if metric == "smd":
        std_diffs = (mean_treat - mean_control) / std_vars
    elif metric == "asmd":
        if isinstance(weights, np.ndarray):
            std_diffs = np.abs(mean_treat - mean_control) / std_vars
        else:
            std_diffs = torch.abs(mean_treat - mean_control) / std_vars
    else:
        raise NotImplementedError(
            f"Metric {metric} not recognized. Choose between {metrics.keys()}"
        )

    # Return as a dataframe with the variable names
    if isinstance(std_diffs, torch.Tensor):
        std_diffs = std_diffs.cpu().detach().numpy()

    std_diffs = pd.DataFrame(std_diffs, index=index, columns=["std_diffs"])

    # Drop columns if we don't want them!
    if exclude is not None:
        std_diffs.drop(exclude, axis=0, inplace=True)

    return std_diffs
