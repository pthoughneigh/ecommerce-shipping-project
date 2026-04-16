from matplotlib import pyplot as plt
import numpy as np
from src.config import FIGURES_DIR

def plot_residuals_vs_fitted(y_pred: np.ndarray, residuals: np.ndarray) -> None:
    """
    Plot residuals against fitted values.

    Used to detect heteroscedasticity and non-linearity.
    Residuals should be randomly scattered around zero with no visible pattern.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted target values.
    residuals : np.ndarray
        Residuals (y_true - y_pred).
    """
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.set_title("Residuals vs fitted values")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.axhline(y=0, color='red', linestyle='--')

    plt.savefig(FIGURES_DIR / "residuals_vs_fitted.png")
    plt.close()


def plot_residual_histogram(residuals: np.ndarray) -> None:
    """
    Plot a histogram of residuals.

    Used to visually assess whether residuals are approximately normally distributed.
    A symmetric bell-shaped histogram suggests normality.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (y_true - y_pred).
    """
    fig, ax = plt.subplots()
    ax.hist(residuals)
    ax.set_title("Histogram of residuals")
    ax.set_xlabel("Residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Count")

    plt.savefig(FIGURES_DIR / "residual_histogram.png")
    plt.close()


def _standard_normal_pdf(x: float) -> float:
    # Gaussian curve — density of the normal distribution at point x
    # High value = many data points around that point
    # Formula: (1 / sqrt(2π)) * e^(-x²/2)
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def _standard_normal_cdf(x: float) -> float:
    # What percentage of data falls below value x
    # CDF(0) = 0.5   → 50% of data is below the mean
    # CDF(2) = 0.977 → 97.7% of data is below 2
    #
    # The true integral has no closed form so we use a tanh approximation
    # 0.044715 is an empirical coefficient that improves approximation accuracy
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _ppf(p: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    # Inverse CDF — given a percentile p, what value x corresponds to it?
    # PPF(0.5)   = 0    → value below which 50% of data falls is the mean
    # PPF(0.975) = 1.96 → value below which 97.5% of data falls is 1.96
    #
    # No closed form exists so we use Newton-Raphson
    # We search for x such that CDF(x) - p = 0

    x = 0.0  # starting point — begin at the mean

    for _ in range(max_iter):
        fx = _standard_normal_cdf(x) - p    # how far we are from the target
        fpx = _standard_normal_pdf(x)        # slope at current point (derivative of CDF = PDF)

        # Newton-Raphson step — move x toward zero
        # if fx is negative (CDF too small) → x moves right
        # if fx is positive (CDF too large) → x moves left
        x_new = x - fx / fpx

        # converged — change in x is negligible
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    # if we did not converge within max_iter, return best approximation
    return x


def plot_qq(residuals: np.ndarray) -> None:
    """
    Plot a Q-Q (quantile-quantile) plot of residuals against a normal distribution.

    Compares the distribution of residuals to the theoretical normal distribution.
    Points lying on the reference line indicate normally distributed residuals.
    Deviations in the tails suggest heavy-tailed or skewed distributions.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (y_true - y_pred).
    """
    residuals = np.sort(residuals)
    n = len(residuals)

    # percentile for each residual — subtracting 0.5 avoids 0 and 1 at the boundaries
    # because PPF(0) = -inf and PPF(1) = +inf
    p = (np.arange(1, n + 1) - 0.5) / n

    # for each percentile compute the corresponding theoretical quantile of the normal distribution
    theoretical_quantiles = np.array([_ppf(pi) for pi in p])

    fig, ax = plt.subplots()

    # points — theoretical quantiles on x-axis, actual residuals on y-axis
    # if they lie on a straight line → residuals are normally distributed
    ax.scatter(theoretical_quantiles, residuals, color='blue', s=5)

    # reference line through the first and last point
    # deviation from this line = deviation from normality
    ax.axline(
        (theoretical_quantiles[0], residuals[0]),
        (theoretical_quantiles[-1], residuals[-1]),
        color='red',
        linestyle='--'
    )

    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Ordered values')
    ax.set_title('Probability Plot')

    plt.savefig(FIGURES_DIR / "q_q_residuals.png")
    plt.close()
