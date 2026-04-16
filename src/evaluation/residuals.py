import numpy as np
from src.visualization.plots import (
    plot_residuals_vs_fitted,
    plot_residual_histogram,
    plot_qq
)


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute residuals as the difference between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.ndarray
        Residuals (y_true - y_pred).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true - y_pred


def run_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Run a full residual diagnostic analysis.

    Generates three plots:
    - Residuals vs fitted values (linearity and homoscedasticity check)
    - Histogram of residuals (normality check)
    - Q-Q plot of residuals (normality check)

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    """
    residuals = compute_residuals(y_true, y_pred)
    plot_residuals_vs_fitted(y_pred, residuals)
    plot_residual_histogram(residuals)
    plot_qq(residuals)