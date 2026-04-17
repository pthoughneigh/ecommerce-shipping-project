"""
main.py
-------
Entry point for the e-commerce shipping linear regression project.
Runs the full pipeline: load → clean → EDA → preprocess → split
→ train → evaluate → residuals → plots.
"""

from src.logger import get_logger
from src.data.loader import load_raw_data
from src.data.cleaning import clean_data
from src.analysis.eda import run_eda
from src.features.preprocessing import build_design_matrix
from src.features.splitting import split_data
from src.models.linear_regression import LinearRegression
from src.evaluation.metrics import evaluate_model
from src.evaluation.residuals import run_residual_analysis
from src.visualization.plots import run_plots
from src.config import (
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS
)

logger = get_logger(__name__)


def main() -> None:
    """Run the full modelling pipeline end to end."""

    logger.info("Pipeline started.")

    # ── 1. Load
    logger.info("Step 1: Loading data.")
    df_raw = load_raw_data()

    # ── 2. Clean
    logger.info("Step 2: Cleaning data.")
    df_clean = clean_data(df_raw)

    # ── 3. EDA
    logger.info("Step 3: Running EDA.")
    run_eda(
        df=df_clean,
        numeric_cols=NUMERIC_COLUMNS,
        categorical_cols=CATEGORICAL_COLUMNS,
        target_col=TARGET_COLUMN
    )

    # ── 4. Build design matrix
    logger.info("Step 4: Building design matrix.")
    df_processed = build_design_matrix(
        df=df_clean,
        unnecessary_cols=COLUMNS_TO_DROP,
        ordinal_columns=ORDINAL_COLUMNS,
        nominal_cols=NOMINAL_COLUMNS
    )

    # ── 5. Split + scale
    logger.info("Step 5: Splitting and scaling data.")
    X_train, X_test, y_train, y_test = split_data(
        df=df_processed,
        numerical_cols=NUMERIC_COLUMNS,
        target_col=TARGET_COLUMN
    )

    # ── 6. Train
    logger.info("Step 6: Training model.")
    model = LinearRegression()
    model.fit(X_train.values, y_train.values)

    # ── 7. Evaluate
    logger.info("Step 7: Evaluating model.")
    train_preds = model.predict(X_train.values)
    test_preds  = model.predict(X_test.values)

    evaluate_model(y_train.values, train_preds, "Train")
    evaluate_model(y_test.values,  test_preds,  "Test")

    # ── 8. Residual analysis
    logger.info("Step 8: Running residual analysis.")
    run_residual_analysis(y_test.values, test_preds)

    # ── 9. Plots
    logger.info("Step 9: Generating plots.")
    run_plots(
        df=df_clean,
        target_col=TARGET_COLUMN,
        numeric_cols=NUMERIC_COLUMNS,
        categorical_cols=CATEGORICAL_COLUMNS,
        y_true=y_test.values,
        y_pred=test_preds,
        feature_names=X_train.columns.tolist(),
        coefficients=model.coefficients
    )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()