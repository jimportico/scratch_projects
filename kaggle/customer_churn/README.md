# Customer Churn (Kaggle Playground Series S6E3)

Binary classification task: predict whether a telecom customer churns (`Churn` = 1) or not (`Churn` = 0), based on account and service usage attributes. Synthetic dataset generated from the classic Telco Customer Churn dataset.

## Data

Downloaded via the Kaggle CLI as `data/playground-series-s6e3.zip` and extracted into `data/`. The zip is kept alongside the extracted CSVs; `data/` is gitignored so none of this is tracked.

| File | Rows | Description |
|---|---|---|
| `data/train.csv` | 594,194 | Labeled training data, includes `Churn` target |
| `data/test.csv` | 254,655 | Unlabeled data to generate predictions for |
| `data/sample_submission.csv` | 254,655 | Submission format template (`id`, `Churn`) |

### Columns (`train.csv`)

| Column | Description |
|---|---|
| `id` | Row identifier |
| `gender` | Male / Female |
| `SeniorCitizen` | 0 / 1 |
| `Partner` | Has a partner (Yes/No) |
| `Dependents` | Has dependents (Yes/No) |
| `tenure` | Months with the company |
| `PhoneService` | Has phone service (Yes/No) |
| `MultipleLines` | Multiple phone lines (Yes/No/No phone service) |
| `InternetService` | DSL / Fiber optic / No |
| `OnlineSecurity` | Yes/No/No internet service |
| `OnlineBackup` | Yes/No/No internet service |
| `DeviceProtection` | Yes/No/No internet service |
| `TechSupport` | Yes/No/No internet service |
| `StreamingTV` | Yes/No/No internet service |
| `StreamingMovies` | Yes/No/No internet service |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Yes/No |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| `MonthlyCharges` | Current monthly charge |
| `TotalCharges` | Total charges to date |
| `Churn` | Target (Yes/No) — training data only |

`test.csv` has the same columns minus `Churn`.

## Quick start

```python
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

train.shape, test.shape
```

Or with polars/duckdb, already available in the parent environment:

```python
import polars as pl

train = pl.read_csv("data/train.csv")
```

## Submission

Predictions must match `sample_submission.csv`'s format: `id`, `Churn` (0/1).
