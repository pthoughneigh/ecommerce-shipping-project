import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from src.config import FIGURES_DIR
from src.logger import get_logger

logger = get_logger(__name__)


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
    logger.info("Plotting residuals vs fitted values.")

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.set_title("Residuals vs fitted values")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.axhline(y=0, color='red', linestyle='--')

    fig.savefig(FIGURES_DIR / "residuals_vs_fitted.png")
    plt.close(fig)
    logger.info("Residuals vs fitted plot saved.")


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
    logger.info("Plotting residual histogram.")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(residuals)
    ax.set_title("Histogram of residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Count")

    fig.savefig(FIGURES_DIR / "residual_histogram.png")
    plt.close(fig)
    logger.info("Residual histogram saved.")


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
    logger.info("Plotting Q-Q plot.")

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

    fig.savefig(FIGURES_DIR / "q_q_residuals.png")
    plt.close(fig)
    logger.info("Q-Q plot saved.")


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    """
    logger.info(f"Plotting target distribution for '{target_col}'.")

    fig, ax = plt.subplots()
    ax.hist(df[target_col], bins=50)
    ax.axvline(df[target_col].mean(), color='red',
               linestyle='--', label=f"Mean: {df[target_col].mean():.1f}")

    ax.set_title("Distribution of target values")
    ax.set_xlabel("Target values")
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close(fig)
    logger.info("Target distribution plot saved.")


def plot_numerical_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """
    Plot histogram distributions for each numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list[str]
        List of numerical column names to plot.
    """
    logger.info(f"Plotting numerical distributions for {list(numeric_cols)}.")

    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=50)
        ax.set_title(col)
        ax.set_xlabel("Values")
        ax.set_ylabel("Count")
        fig.savefig(FIGURES_DIR / f"{col}_distribution.png")
        plt.close(fig)

    logger.info("Numerical distribution plots saved.")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str
) -> None:
    """
    Plot a correlation heatmap for numerical columns including the target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list[str]
        List of numerical column names.
    target_col : str
        Name of the target column.
    """
    logger.info("Plotting correlation heatmap.")

    cols = list(numeric_cols) + [target_col]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha='center', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{target_col}_heatmap.png")
    plt.close(fig)
    logger.info("Correlation heatmap saved.")


def plot_scatter_vs_target(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: str
) -> None:
    """
    Plot scatter plots of each numerical feature against the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_cols : list[str]
        List of numerical column names to plot.
    target_col : str
        Name of the target column.
    """
    logger.info("Plotting scatter plots vs target.")

    n_cols = 3
    n_rows = math.ceil(len(numerical_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        axes[i].scatter(df[col], df[target_col], s=5, alpha=0.5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{col} vs {target_col}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scatter_vs_target.png")
    plt.close(fig)
    logger.info("Scatter plots saved.")


def plot_categorical_vs_target(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str
) -> None:
    """
    Plot boxplots of the target variable grouped by each categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_cols : list[str]
        List of categorical column names to plot.
    target_col : str
        Name of the target column.
    """
    logger.info("Plotting categorical vs target boxplots.")

    n_cols = 3
    n_rows = math.ceil(len(categorical_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        if col == 'product_importance':
            order = ['low', 'medium', 'high']
        else:
            order = sorted(df[col].unique(), key=str)

        groups = [df[df[col] == cat][target_col].values for cat in order]

        axes[i].boxplot(groups, labels=order)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{col} vs {target_col}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_vs_target.png")
    plt.close(fig)
    logger.info("Categorical vs target plots saved.")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot actual vs predicted values.

    A perfect model would show all points on the diagonal.
    Deviations indicate systematic over- or under-prediction.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    """
    logger.info("Plotting actual vs predicted.")

    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, s=5, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=1, label='Perfect prediction')

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png")
    plt.close(fig)
    logger.info("Actual vs predicted plot saved.")


def plot_coefficients(
    feature_names: list[str],
    coefficients: np.ndarray
) -> None:
    """
    Plot model coefficients as a horizontal bar chart sorted by absolute magnitude.

    Parameters
    ----------
    feature_names : list[str]
        Names of features corresponding to each coefficient.
    coefficients : np.ndarray
        Coefficient values from the fitted model.
    """
    logger.info("Plotting coefficient chart.")

    sorted_idx = np.argsort(np.abs(coefficients))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = coefficients[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['salmon' if c > 0 else 'steelblue' for c in sorted_coefs]
    ax.barh(sorted_names, sorted_coefs, color=colors)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Model coefficients")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "coefficients.png")
    plt.close(fig)
    logger.info("Coefficient plot saved.")


def run_plots(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    coefficients: np.ndarray
) -> None:
    """
    Run all visualization plots for model evaluation and data exploration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    numeric_cols : list[str]
        List of numerical column names.
    categorical_cols : list[str]
        List of categorical column names.
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    feature_names : list[str]
        Feature names corresponding to model coefficients.
    coefficients : np.ndarray
        Model coefficient values.
    """
    logger.info("Starting full plot pipeline.")

    plot_target_distribution(df, target_col)
    plot_numerical_distributions(df, numeric_cols)
    plot_correlation_heatmap(df, numeric_cols, target_col)
    plot_scatter_vs_target(df, numeric_cols, target_col)
    plot_categorical_vs_target(df, categorical_cols, target_col)
    plot_actual_vs_predicted(y_true, y_pred)
    plot_coefficients(feature_names, coefficients)

    logger.info("Full plot pipeline complete.")