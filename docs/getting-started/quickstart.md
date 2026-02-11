# Quickstart

Get up and running in 60 seconds.

## Generate a sample project

```bash
timefence quickstart churn-example
cd churn-example
```

This creates a self-contained directory with:

- `timefence.yaml` — Project configuration
- `features.py` — 4 feature definitions
- `data/` — Synthetic parquet files (users, transactions, labels)
- `data/train_LEAKY.parquet` — Pre-built dataset with planted leakage
- `README.md` — Next-step instructions

## Audit the leaky dataset

```bash
timefence audit data/train_LEAKY.parquet
```

```
TEMPORAL AUDIT REPORT
Scanned 5,000 rows

WARNING  LEAKAGE DETECTED in 3 of 4 features

  LEAK  rolling_spend_30d
        1,520 rows (30.4%) use feature data from the future
        Severity: HIGH

  LEAK  days_since_login
        4,909 rows (98.2%) use feature data from the future
        Severity: HIGH

  OK    user_country - clean (5,000 rows)
  OK    account_age_days - clean (5,000 rows)
```

## Build a clean dataset

```bash
timefence build -o train_CLEAN.parquet
```

```
Building training set...

  Labels     5,000 rows from data/labels.parquet
  Features   4 features

  Joining with point-in-time correctness (feature_time < label_time):

  OK  user_country         5,000 / 5,000 matched
  OK  account_age_days     5,000 / 5,000 matched
  OK  rolling_spend_30d    5,000 / 5,000 matched
  OK  days_since_login     5,000 / 5,000 matched

  Written   train_CLEAN.parquet (5,000 rows, 7 cols)
```

## Verify it's clean

```bash
timefence audit train_CLEAN.parquet
# ALL CLEAN - no temporal leakage detected
```

## Next steps

- [Audit your own data](../guides/audit.md) without changing your pipeline
- [Build from scratch](../guides/build.md) with Python API
- [Add to CI](../guides/ci.md) to prevent leakage in production
