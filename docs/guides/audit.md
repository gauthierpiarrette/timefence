# Guide: Audit Existing Data

Already have a training dataset? Audit it for temporal leakage without rebuilding.

## Step 1: Inspect your data

```bash
timefence inspect data/train.parquet
```

This shows column types and suggests which columns are likely keys and timestamps.

## Step 2: Run the audit

=== "Python"

    ```python
    import timefence

    transactions = timefence.Source(
        path="data/transactions.parquet",
        keys=["user_id"],
        timestamp="created_at",
    )

    rolling_spend = timefence.Feature(
        source=transactions,
        columns=["amount"],
        embargo="1d",
    )

    report = timefence.audit(
        data="data/train.parquet",
        features=[rolling_spend],
        keys=["user_id"],
        label_time="label_time",
    )

    print(report)
    ```

=== "CLI"

    ```bash
    timefence audit data/train.parquet \
      --features features.py \
      --keys user_id \
      --label-time label_time
    ```

## Step 3: Inspect results

```python
import timefence

# (assuming `report` from step 2)

# Check overall status
report.has_leakage        # True/False
report.leaky_features     # ["rolling_spend_30d"]
report.clean_features     # ["user_country"]

# Per-feature details
detail = report["rolling_spend_30d"]
detail.leaky_row_count    # 1520
detail.leaky_row_pct      # 0.304
detail.severity           # "HIGH"
detail.max_leakage        # timedelta(days=15)

# Export
report.to_html("audit_report.html")
report.to_json("audit_report.json")
```

## Step 4: Assert in tests

```python
report.assert_clean()  # Raises TimefenceLeakageError if leakage found
```

See the [audit() API reference](../api/audit.md) for full parameter documentation.
