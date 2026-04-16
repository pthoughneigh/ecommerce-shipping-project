from src.logger import get_logger
import numpy as np

logger = get_logger(__name__)

class LinearRegression:
    """
    Ordinary least squares linear regression using the normal equation.

    Fits a linear model by computing the closed-form solution:
        β = (XᵀX)⁻¹Xᵀy

    Attributes
    ----------
    coefficients : np.ndarray or None
        Feature weights β₁, β₂, ... fitted during training.
    intercept : float or None
        Bias term β₀ fitted during training.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using the normal equation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.
        """
        logger.info(f"Fitting LinearRegression on {X.shape[0]} samples, {X.shape[1]} features.")

        X = np.array(X)
        y = np.array(y)

        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])

        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept = beta[0]
        self.coefficients = beta[1:]

        logger.info(f"Model fitted. Intercept: {self.intercept:.4f}, Coefficients: {self.coefficients}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        RuntimeError
            If predict is called before fit.
        """
        if self.coefficients is None or self.intercept is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

        X = np.array(X)

        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])

        return X_b @ np.concatenate([[self.intercept], self.coefficients])