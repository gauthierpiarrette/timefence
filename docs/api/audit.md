# audit()

Scan a dataset for temporal leakage. Two modes are available.

::: timefence.audit
    options:
      show_root_heading: true

## Mode 1: Rebuild-and-compare (full audit)

```python
report = timefence.audit(
    data="data/train.parquet",
    features=[rolling_spend, user_country],
    keys=["user_id"],
    label_time="label_time",
    max_lookback="365d",
    max_staleness=None,
    join="strict",
)
```

## Mode 2: Temporal check (lightweight)

```python
report = timefence.audit(
    data="data/train.parquet",
    feature_time_columns={
        "spend_30d": "spend_computed_at",
        "country": "country_updated_at",
    },
    label_time="label_time",
)
```

## Returns: AuditReport

| Attribute | Type | Description |
|-----------|------|-------------|
| `.has_leakage` | `bool` | `True` if any feature is leaky. |
| `.clean_features` | `list[str]` | Feature names with no leakage. |
| `.leaky_features` | `list[str]` | Feature names with leakage. |
| `.total_rows` | `int` | Total rows scanned. |
| `.mode` | `str` | `"rebuild"` or `"temporal"`. |
| `[name]` | `FeatureAuditDetail` | Per-feature detail via `report["feature_name"]`. |
| `.assert_clean()` | `None` | Raises `TimefenceLeakageError` if leakage found. |
| `.to_html(path)` | `None` | Export HTML report. |
| `.to_json(path)` | `None` | Export JSON report. |

## FeatureAuditDetail

| Attribute | Type | Description |
|-----------|------|-------------|
| `.name` | `str` | The feature name. |
| `.clean` | `bool` | No leakage detected. |
| `.leaky_row_count` | `int` | Number of leaky rows. |
| `.leaky_row_pct` | `float` | Fraction of rows with leakage (0.0–1.0). |
| `.severity` | `str` | `"HIGH"`, `"MEDIUM"`, `"LOW"`, or `"OK"`. |
| `.max_leakage` | `timedelta \| None` | Largest time violation. |
| `.median_leakage` | `timedelta \| None` | Median time violation. |
| `.total_rows` | `int` | Total rows examined. |
| `.null_rows` | `int` | Rows where feature was NULL. |
| `.leaky_rows` | `DataFrame \| None` | DataFrame of up to 1,000 violating rows (when leakage detected). |

!!! info "Severity levels"
    **HIGH** = >5% leaky rows or max leakage >7 days.
    **MEDIUM** = 1–5% or 1–7 days.
    **LOW** = <1% and <1 day.
    **OK** = no leakage.
