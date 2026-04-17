# E-Commerce Shipping — Linear Regression from Scratch

A end-to-end Linear Regression project built using only **NumPy**, **Pandas**, and **Matplotlib** — no scikit-learn or modelling libraries. The goal is to predict the cost of a product based on shipment and customer data, while understanding the mathematics behind every step.

Built as part of a personal machine learning portfolio, working toward implementing neural networks from scratch.

---

## Project Goals

- Implement Linear Regression using the **Normal Equation** from scratch
- Build a complete, modular data pipeline with proper separation of concerns
- Perform residual diagnostics to validate model assumptions
- Understand when and why a model underperforms — not just that it does
- Practice professional Python: type hints, docstrings, logging, error handling

---

## Dataset

**E-Commerce Shipping Data** — [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics)

An international e-commerce company dataset with ~11,000 shipment records.

| Column | Type | Description |
|---|---|---|
| `warehouse_block` | Nominal | Warehouse section (A–F) |
| `mode_of_shipment` | Nominal | Flight / Ship / Road |
| `customer_care_calls` | Numerical | Number of support calls |
| `customer_rating` | Ordinal | Rating 1–5 |
| `cost_of_the_product` | Numerical | **Target** — product cost in USD |
| `prior_purchases` | Numerical | Number of previous orders |
| `product_importance` | Ordinal | Low / Medium / High |
| `gender` | Nominal | M / F |
| `discount_offered` | Numerical | Discount percentage |
| `weight_in_gms` | Numerical | Product weight in grams |
| `reached_on_time` | Binary | Delivery on time (dropped — not a valid predictor of cost) |

---

## Project Structure

```
ecommerce-shipping-project/
├── data/
│   ├── raw/                        ← original CSV (not tracked by Git)
│   └── processed/
├── outputs/
│   ├── figures/                    ← all generated plots
│   ├── logs/                       ← project.log
│   └── reports/
├── src/
│   ├── analysis/
│   │   └── eda.py                  ← numerical EDA: distributions, correlations, skewness
│   ├── data/
│   │   ├── loader.py               ← loads raw CSV with error handling
│   │   └── cleaning.py             ← column renaming, deduplication
│   ├── evaluation/
│   │   ├── metrics.py              ← MSE, RMSE, MAE, R² from scratch
│   │   └── residuals.py            ← residual plots, Q-Q plot with Newton-Raphson PPF
│   ├── features/
│   │   ├── preprocessing.py        ← ordinal encoding, one-hot encoding, design matrix
│   │   └── splitting.py            ← shuffle, train/test split, standardisation
│   ├── models/
│   │   └── linear_regression.py    ← Normal Equation + lstsq, predict()
│   ├── visualization/
│   │   └── plots.py                ← EDA plots, coefficient chart, actual vs predicted
│   ├── config.py                   ← centralised paths and column definitions
│   └── logger.py                   ← dual file/console logger
├── main.py                         ← full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/pthoughneigh/ecommerce-shipping-project.git
cd ecommerce-shipping-project
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics) and place the CSV at:
```
data/raw/ecommerce_shipping.csv
```

### 5. Run the pipeline
```bash
python main.py
```

Logs are written to `outputs/logs/project.log`. Figures are saved to `outputs/figures/`.

---

## What's Built From Scratch

| Component | Implementation |
|---|---|
| Linear Regression | Normal Equation $(X^TX)^{-1}X^Ty$ and `lstsq` |
| Pearson Correlation | Manual covariance / std computation |
| Skewness | Third standardised moment formula |
| Standardisation | Fit on train only, applied to train + test |
| Q-Q Plot | Newton-Raphson inverse normal CDF |
| All metrics | MSE, RMSE, MAE, R² without any ML libraries |

---

## Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.126 | 0.129 |
| RMSE | 44.79 | 45.43 |
| MAE | 37.60 | 37.98 |

**Key finding:** Train and test performance are nearly identical — the model is **underfitting**, not overfitting. This is consistent with the weak linear correlations found in EDA (strongest predictor: `customer_care_calls` at Pearson r = 0.32). The dataset does not contain strong linear signal for predicting product cost.

Residual analysis confirms:
- Systematic bias — model hedges toward the mean, underpredicting expensive products
- Approximately normal residuals with a heavy left tail
- Roughly constant variance (homoscedasticity largely satisfied)

---

## Key Design Decisions

**No data leakage** — standardisation is fitted on training data only and applied separately to test data using stored means and standard deviations.

**Dropped features:**
- `customer_rating` — Pearson r = 0.009, essentially no linear relationship with cost
- `reached_on_time` — delivery outcome happens after cost is set; logically invalid as a predictor

**Encoding strategy:**
- Nominal features → one-hot encoding with `drop_first=True` (avoids dummy variable trap)
- Ordinal features (`product_importance`) → label encoding: low=0, medium=1, high=2

---

## Requirements

```
pandas
numpy
matplotlib
```

Python 3.13+

---

## Background

This is the second project in a series building toward implementing a self-playing agent using neural networks. Previous project: German Credit Risk with Logistic Regression.

The focus is on understanding the mathematics of each algorithm — not on achieving the best possible score on any given dataset.