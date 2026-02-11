# Errors

All Timefence errors inherit from `TimefenceError` and follow a consistent format:

- **WHAT** happened
- **WHY** it matters
- **WHERE** in the data
- **HOW TO FIX** it

## Error hierarchy

```
TimefenceError (base)
├── TimefenceSchemaError
├── TimefenceDuplicateError
├── TimefenceTimezoneError
├── TimefenceConfigError
├── TimefenceLeakageError
└── TimefenceValidationError
```

## TimefenceSchemaError

Raised when expected columns are missing or types are mismatched between source, labels, and feature definitions.

```
TimefenceSchemaError: Feature 'rolling_spend' is missing required key column(s): ['user_id'].

  Point-in-time joins require matching keys between labels and features.
  Without ['user_id'], Timefence can't determine which feature rows belong to which entity.

  Expected keys: ['user_id']
  Actual columns: ['customer_id', 'created_at', 'amount']
  'customer_id' is similar to 'user_id' — possible rename?

  Fix: Add key_mapping to your feature definition:
    key_mapping={'user_id': 'customer_id'}
```

**Common causes:** Column renamed between sources, typos in key names, using the wrong file.

**Fix:** Correct the column name, or use `key_mapping` on the Feature to map label keys to source keys.

## TimefenceDuplicateError

Raised when duplicate `(key, timestamp)` pairs exist in a source and `on_duplicate="error"` (default).

```
TimefenceDuplicateError: Feature 'user_country' has 42 duplicate (key, feature_time) pairs.

  When multiple feature rows have the same key and timestamp, the
  point-in-time join becomes non-deterministic. Timefence cannot guarantee
  which row would be selected.

  Example duplicates (showing first 3):
    {'"user_id"': 101, '"updated_at"': Timestamp('2024-03-15 10:00:00'), 'cnt': 3}
    {'"user_id"': 205, '"updated_at"': Timestamp('2024-03-16 14:30:00'), 'cnt': 2}

  Fix (pick one):
    1. Deduplicate in your source data or SQL
    2. Set: timefence.Feature(..., on_duplicate="keep_any")
```

**Common causes:** Multiple events at the same timestamp, duplicated source data, ETL issues.

**Fix:** Deduplicate your source, or set `on_duplicate="keep_any"` if you don't care which row is selected.

## TimefenceTimezoneError

Raised when mixing timezone-aware and timezone-naive timestamps across sources and labels.

```
TimefenceTimezoneError: Mixed timezones between labels and feature 'rolling_spend'.

  Labels 'label_time' is timezone-aware (UTC).
  Feature 'rolling_spend' timestamp is timezone-naive.

  Comparing these directly could shift joins by hours.

  Sample values:
    label_time:   2024-03-15 10:00:00+00:00
    feature_time: 2024-03-15 10:00:00
```

**Common causes:** One data source uses UTC-aware timestamps, another uses naive timestamps.

**Fix:** Normalize all timestamps to the same type — either all timezone-aware or all timezone-naive.

## TimefenceConfigError

Raised for invalid parameter combinations.

```
TimefenceConfigError: embargo (400d) must be less than max_lookback (365d).

  When embargo equals or exceeds max_lookback, the join window is empty —
  no feature can ever match. This is almost certainly a misconfiguration.

  Current: max_lookback=365d, embargo=400d → empty window
  Likely intent: max_lookback=365d, embargo=1d

  Fix: Increase max_lookback or decrease embargo.
```

Other examples:

- `join` must be `"strict"` or `"inclusive"`
- `on_missing` must be `"null"` or `"skip"`
- `on_duplicate` must be `"error"` or `"keep_any"`
- Feature requires exactly one of `columns`, `sql`, or `transform`
- Duplicate or colliding feature names

## TimefenceLeakageError

Raised by `report.assert_clean()` when temporal leakage is detected.

```
TimefenceLeakageError: Temporal leakage detected in features: rolling_spend_30d, days_since_login
```

**Where it appears:** Only from `AuditReport.assert_clean()`. The `audit()` function itself returns a report — it does not raise.

## TimefenceValidationError

Raised for general input validation failures.

```
TimefenceValidationError: Source requires either 'path' or 'df' parameter.
TimefenceValidationError: Source accepts either 'path' or 'df', not both.
TimefenceValidationError: Labels 'keys' cannot be empty. Provide at least one entity key column.
```

## Handling errors

```python
import timefence
from timefence.errors import (
    TimefenceError,
    TimefenceSchemaError,
    TimefenceLeakageError,
)

# Catch a specific error
try:
    result = timefence.build(labels=labels, features=[feature], output="train.parquet")
except TimefenceSchemaError as e:
    print(f"Schema issue: {e}")
    # Inspect the error message for suggested fixes

# Catch any Timefence error
try:
    result = timefence.build(labels=labels, features=[feature], output="train.parquet")
except TimefenceError as e:
    print(f"Timefence error: {e}")

# CI pattern: assert_clean raises TimefenceLeakageError
try:
    report = timefence.audit(data="train.parquet", features=[feature], keys=["user_id"], label_time="label_time")
    report.assert_clean()
except TimefenceLeakageError:
    print("Leakage found — failing build")
    sys.exit(1)
```
