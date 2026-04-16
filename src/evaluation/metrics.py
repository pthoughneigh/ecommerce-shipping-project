import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        Mean squared error between y_true and y_pred.
    """
    y_true = np.array(y_true.flatten())
    y_pred = np.array(y_pred.flatten())

    n = y_true.shape[0]
    mse = float(1/n * ((y_true - y_pred)**2).sum())

    logger.info(f"MSE: {mse:.4f}")
    return mse


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error between y_true and y_pred.
    """
    y_true = np.array(y_true.flatten())
    y_pred = np.array(y_pred.flatten())

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    logger.info(f"RMSE: {rmse:.4f}")
    return rmse


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        Mean absolute error between y_true and y_pred.
    """
    y_true = np.array(y_true.flatten())
    y_pred = np.array(y_pred.flatten())

    n = y_true.shape[0]
    mae = float(1 / n * (np.abs((y_true - y_pred))).sum())

    logger.info(f"MAE: {mae:.4f}")
    return mae


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Measures the proportion of variance in y_true explained by the model.
    A score of 1.0 means perfect predictions, 0.0 means the model performs
    no better than predicting the mean.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        R² score between y_true and y_pred.
    """
    y_true = np.array(y_true.flatten())
    y_pred = np.array(y_pred.flatten())

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - np.mean(y_true)) ** 2).sum()

    r2 = float(1 - ss_res / ss_tot)

    logger.info(f"R²: {r2:.4f}")
    return r2

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Test"
) -> None:
    """Print all evaluation metrics for a given dataset."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    logger.info(f"Metrics [{dataset_name}] -> RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    print(f"\n=== MODEL EVALUATION: {dataset_name} ===")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

