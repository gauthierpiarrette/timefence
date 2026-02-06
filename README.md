# Timefence

**Temporal correctness layer for ML training data.**

Timefence guarantees no future data leakage, audits existing pipelines, and builds point-in-time correct datasets — locally, with zero infrastructure, in seconds.

From `pip install` to "I found leakage in my pipeline" in under 3 minutes.

## Install

```bash
pip install timefence
```

Three runtime dependencies: `duckdb`, `click`, `rich`. Python 3.9+.

## Quick Start

```bash
timefence quickstart churn-example
cd churn-example
```

This generates a self-contained project with synthetic data and planted leakage:

```
churn-example/
  timefence.yaml              # Project config
  features.py              # 4 feature definitions
  data/
    users.parquet           # Synthetic user data (10K users, 30K rows)
    transactions.parquet    # Synthetic transactions (200K rows)
    labels.parquet          # Churn labels (5K rows)
    train_LEAKY.parquet     # Pre-built dataset WITH planted leakage
  README.md
```

**Step 1: Find the leakage.**

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

**Step 2: Fix it.**

```bash
timefence build --labels data/labels.parquet --features features.py --output data/train_CLEAN.parquet
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

  Written   data/train_CLEAN.parquet (5,000 rows, 7 cols)
  Manifest  .timefence/builds/20260205T143022Z/build.json
```

**Step 3: Verify.**

```bash
timefence audit data/train_CLEAN.parquet
```

```
ALL CLEAN - no temporal leakage detected
```

---

## Core Concepts

Timefence has **6 user-facing concepts**:

| Concept | Definition |
|---------|-----------|
| **Source** | A table of historical data with timestamps |
| **Feature** | A named column derived from a source |
| **Labels** | Prediction targets with entity keys and event times |
| **Build** | Constructing a point-in-time correct dataset |
| **Audit** | Checking any dataset for temporal leakage |
| **Store** | A local directory that tracks builds and manifests (optional) |

### The Core Invariant

For every row in a Timefence-built training set:

```
feature_time < label_time - embargo
```

Strict less-than. No feature value used in training may have been recorded at or after the label event minus its embargo.

---

## Python API

### Source

Declare where historical data lives and how to interpret it temporally.

```python
import timefence

users = timefence.Source(
    path="data/users.parquet",
    keys=["user_id"],
    timestamp="updated_at",
)

# CSV source
events = timefence.CSVSource(
    path="data/events.csv",
    keys=["user_id"],
    timestamp="event_time",
    delimiter="|",
)

# SQL source
txns = timefence.SQLSource(
    query="SELECT * FROM transactions WHERE amount > 0",
    keys=["user_id"],
    timestamp="created_at",
    name="transactions",
)
```

Keys and timestamp are always required. Timefence never infers them.

### Feature

One class. Three modes. Exactly one of `columns`, `sql`, or `transform` must be provided.

**Mode 1: Column Selection** (~70% of features)

```python
user_country = timefence.Feature(
    source=users,
    columns=["country"],
)

# Multiple columns
user_profile = timefence.Feature(
    source=users,
    columns=["country", "signup_platform", "account_tier"],
)

# Column rename (source_col -> feature_col)
user_region = timefence.Feature(
    source=users,
    columns={"region_code": "region"},
)
```

**Mode 2: SQL** (~25% of features)

```python
rolling_spend = timefence.Feature(
    source=transactions,
    sql="""
        SELECT
            user_id,
            created_at AS feature_time,
            SUM(amount) OVER (
                PARTITION BY user_id
                ORDER BY created_at
                RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
            ) AS spend_30d
        FROM {source}
    """,
    name="rolling_spend_30d",
    embargo="1d",
)

# Or from a .sql file (recommended for production)
rolling_spend = timefence.Feature(
    source=transactions,
    sql=Path("features/rolling_spend.sql"),
    embargo="1d",
)
```

**Mode 3: Python Transform** (~5% of features)

```python
def compute_complex_feature(conn, source_table):
    conn.create_function("my_udf", lambda x: x * 2.5, [float], float)
    return conn.sql(f"""
        SELECT user_id, created_at AS feature_time,
               my_udf(raw_score) AS adjusted_score
        FROM {source_table}
    """)

complex_feature = timefence.Feature(
    source=transactions,
    transform=compute_complex_feature,
)
```

**Feature options** (apply to all modes):

```python
timefence.Feature(
    source=...,
    columns=... | sql=... | transform=...,
    name="rolling_spend_30d",        # Auto-derived when possible
    embargo="1d",                    # Computation lag buffer (default: "0d")
    key_mapping={"user_id": "customer_id"},  # When source uses different key names
    on_duplicate="error",            # "error" (default) or "keep_any"
)
```

### Labels

```python
labels = timefence.Labels(
    path="data/labels.parquet",
    keys=["user_id"],
    label_time="label_time",
    target=["churned"],
)

# From a DataFrame already in memory
labels = timefence.Labels(
    df=my_dataframe,
    keys=["user_id"],
    label_time="label_time",
    target=["churned"],
)
```

### Build

```python
result = timefence.build(
    labels=labels,
    features=[user_country, rolling_spend, complex_feature],
    output="train.parquet",

    # Temporal controls
    max_lookback="365d",       # Ignore features older than this
    max_staleness="30d",       # If best feature is older, treat as missing
    join="strict",             # "strict" (default, <) or "inclusive" (<=)
    on_missing="null",         # "null" (keep row) or "skip" (drop row)

    # Time-based splits
    splits={
        "train": ("2023-01-01", "2024-01-01"),
        "valid": ("2024-01-01", "2024-07-01"),
        "test":  ("2024-07-01", "2025-01-01"),
    },

    # Reproducibility
    store=timefence.Store(".timefence"),
)

# Inspect the result
print(result)               # Pretty summary
result.output_path           # "train.parquet"
result.manifest              # Full build manifest (dict)
result.stats                 # Row counts, feature stats, timing
result.splits                # {"train": Path, "valid": Path, "test": Path}
result.sql                   # The exact SQL executed
result.validate()            # Re-check audit passed
```

### Audit

```python
# Rebuild-and-compare mode (full audit)
report = timefence.audit(
    data="existing_training.parquet",
    features=[user_country, rolling_spend],
    keys=["user_id"],
    label_time="label_time",
)

# Temporal check mode (lightweight, no source data needed)
report = timefence.audit.temporal(
    data="existing_training.parquet",
    feature_time_columns={
        "spend_30d": "spend_computed_at",
        "country": "country_updated_at",
    },
    label_time="label_time",
)

# Use the report
report.has_leakage           # bool
report.clean_features        # ["user_country"]
report.leaky_features        # ["rolling_spend_30d"]
report["rolling_spend_30d"]  # FeatureAuditDetail

# Export
report.to_json("report.json")
report.to_html("report.html")

# CI integration
report.assert_clean()        # Raises TimefenceLeakageError if leakage found
```

### Explain

Preview join logic without executing:

```python
plan = timefence.explain(
    labels=labels,
    features=[user_country, rolling_spend],
)
print(plan)
```

Every query is copy-pasteable for manual verification.

### Diff

Compare two training datasets:

```python
diff = timefence.diff(
    old="train_v1.parquet",
    new="train_v2.parquet",
    keys=["user_id"],
    label_time="label_time",
    atol=1e-10,   # Absolute tolerance for numeric comparison
    rtol=1e-7,    # Relative tolerance for numeric comparison
)
print(diff)
```

### FeatureSet

Group features for reuse:

```python
user_features = timefence.FeatureSet(
    name="user_features",
    features=[user_country, account_age, user_tier],
)

result = timefence.build(
    labels=labels,
    features=[user_features, rolling_spend],  # Mix FeatureSets and Features
    output="train.parquet",
)
```

### Store

Track builds for reproducibility:

```python
store = timefence.Store(".timefence")
result = timefence.build(labels=labels, features=features, output="train.parquet", store=store)

# Later
builds = store.list_builds()          # All builds, newest first
manifest = store.get_build(build_id)  # Specific build manifest
```

---

## CLI Reference

### `timefence quickstart`

Generate a self-contained example project.

```bash
timefence quickstart [project-name]    # default: churn-example
timefence quickstart myproject --minimal
```

### `timefence inspect`

Suggest keys and timestamps for a data file.

```bash
timefence inspect data/users.parquet
```

### `timefence audit`

Audit any dataset for temporal leakage.

```bash
# With timefence.yaml config (flags inferred)
timefence audit data/train.parquet

# Explicit flags
timefence audit data/train.parquet \
  --features features.py \
  --keys user_id \
  --label-time label_time

# CI mode (exit 1 if leakage)
timefence audit data/train.parquet --strict

# Export
timefence audit data/train.parquet --json
timefence audit data/train.parquet --html report.html
```

### `timefence build`

Build a point-in-time correct training set.

```bash
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  --output train.parquet

# With options
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  --output train.parquet \
  --max-lookback 365d \
  --max-staleness 30d \
  --on-missing null \
  --join-mode strict

# Time-based splits
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  --output train.parquet \
  --split train:2023-01-01:2024-01-01 \
  --split test:2024-01-01:2025-01-01

# Dry run (show plan only)
timefence build --labels data/labels.parquet --features features.py --output train.parquet --dry-run
```

### `timefence explain`

Preview join logic without executing.

```bash
timefence explain --labels data/labels.parquet --features features.py

# Single feature
timefence explain --features features.py:rolling_spend_30d
```

### `timefence diff`

Compare two training datasets.

```bash
timefence diff train_v1.parquet train_v2.parquet --keys user_id --label-time label_time

# Custom numeric tolerance
timefence diff v1.parquet v2.parquet --keys user_id --label-time label_time --atol 0.01 --rtol 0.001
```

### `timefence catalog`

List all features defined in the project.

```bash
timefence catalog --features features.py
```

### `timefence doctor`

Diagnose project setup and common issues.

```bash
timefence doctor
```

### `timefence init`

Initialize a project with a `timefence.yaml` config file.

```bash
timefence init
```

---

## Configuration

`timefence.yaml` is optional. Every setting can be passed via CLI flags or the Python API.

```yaml
name: churn-model
version: "1.0"

features:
  - features.py

labels:
  path: data/labels.parquet
  keys: [user_id]
  label_time: label_time
  target: [churned]

defaults:
  max_lookback: 365d
  join: strict
  on_missing: "null"

store: .timefence/

output:
  dir: artifacts/
```

**Precedence:** CLI flags > Python API arguments > `timefence.yaml` > built-in defaults.

---

## The Join Algebra

Given a label row `(K, T)` and a feature with embargo `E`, max lookback `L`, and optional max staleness `S`:

```
candidate_rows = { f : f.key = K  AND  f.feature_time in [T - L,  T - E) }
selected       = latest feature_time from candidate_rows
if S is set and selected.feature_time < T - S: treat as missing
```

```
T - L                    T - S                    T - E          T
 |                        |                        |             |
 |     stale (miss)       |     fresh (usable)     |  embargo    |  future
 |                        |                        |  (blocked)  |  (blocked)
```

**Parameter constraints:** `L > E`, and if `S` is set: `L >= S > E`.

---

## CI/CD Integration

```yaml
# GitHub Actions
- name: Audit training data
  run: |
    pip install timefence
    timefence audit data/train.parquet \
      --features features.py \
      --strict    # Exit code 1 if leakage found
```

---

## Error Messages

Timefence errors follow a consistent structure: **what** happened, **why** it matters, **where** (specific data), and **how** to fix it.

```
TimefenceSchemaError: Feature 'clicks_7d' is missing required key column 'user_id'.

  Point-in-time joins require matching keys between labels and features.

  Expected keys: ['user_id']
  Actual columns: ['customer_id', 'feature_time', 'clicks_7d']
                   ^^^^^^^^^^^^ similar to 'user_id' - possible rename?

  Fix: Add key_mapping to your feature definition:
    timefence.Feature(..., key_mapping={"user_id": "customer_id"})
```

---

## What Timefence Is NOT

| Not This | Why | Use Instead |
|----------|-----|-------------|
| Feature store platform | No server, no online serving | Tecton, Feast |
| Data orchestrator | No scheduling | Airflow, Dagster |
| Data quality framework | Temporal correctness only | Great Expectations |
| DataFrame library | Not general-purpose | Polars, Pandas, DuckDB |
| ML pipeline framework | Produces training data only | MLflow, Metaflow |

Timefence is a **single-purpose tool**: temporal correctness for ML training data.

---

## License

MIT
