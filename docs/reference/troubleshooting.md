# Troubleshooting

Common issues and how to fix them.

## "Feature X is missing required key column"

**Error:** `TimefenceSchemaError`

Your feature source uses different column names than your labels. For example, labels have `user_id` but the source has `customer_id`.

**Fix:** Add `key_mapping` to the feature:

```python
feature = timefence.Feature(
    source=transactions,
    columns=["amount"],
    key_mapping={"user_id": "customer_id"},
)
```

## "Duplicate (key, feature_time) pairs"

**Error:** `TimefenceDuplicateError`

Your source has multiple rows with the same key and timestamp. The point-in-time join can't determine which row to select.

**Fix (pick one):**

1. Deduplicate your source data upstream
2. Accept non-determinism: `timefence.Feature(..., on_duplicate="keep_any")`

## "Mixed timezones between labels and feature"

**Error:** `TimefenceTimezoneError`

One timestamp is timezone-aware (e.g., `2024-01-01 10:00:00+00:00`) and the other is naive (e.g., `2024-01-01 10:00:00`). Comparing them directly could shift joins by hours.

**Fix:** Normalize all timestamps to the same type before passing to Timefence.

## "embargo must be less than max_lookback"

**Error:** `TimefenceConfigError`

Your embargo is larger than the lookback window, making it impossible for any feature to match.

**Fix:** Either increase `max_lookback` or decrease `embargo`:

```python
result = timefence.build(
    labels=labels,
    features=[feature],
    output="train.parquet",
    max_lookback="730d",  # Increase from default 365d
)
```

## Audit says "LEAKAGE DETECTED" — what now?

This means your existing training data has rows where `feature_time >= label_time` (in the default strict mode). In inclusive mode (`join="inclusive"`), leakage is `feature_time > label_time`. Either way, the feature was computed after the event you're predicting — meaning your model trained on the future.

**Options:**

1. **Rebuild** the dataset with `timefence build` to get temporally correct data
2. **Investigate** which features leak and by how much (check `report["feature_name"].severity`)
3. **Add to CI** with `--strict` to prevent it from happening again

## Build is slow

Timefence processes data through DuckDB's columnar engine. If builds are slow:

1. **Check data size:** How many label rows × how many features? See [benchmarks](../getting-started/installation.md#performance)
2. **Enable caching:** Pass a `Store` to avoid recomputing unchanged features
3. **Use Parquet over CSV:** Parquet is significantly faster due to columnar reads
4. **Check feature SQL complexity:** Complex window functions take longer — consider pre-computing

## "Cannot auto-detect format"

**Error:** `TimefenceValidationError`

The file extension isn't `.parquet`, `.pq`, or `.csv`.

**Fix:** Specify the format explicitly:

```python
source = timefence.Source(
    path="data/file.dat",
    keys=["user_id"],
    timestamp="ts",
    format="parquet",  # or "csv"
)
```

## "Feature requires exactly one of columns, sql, or transform"

**Error:** `TimefenceConfigError`

You passed zero or more than one of `columns`, `sql`, or `transform` to a Feature.

**Fix:** Use exactly one:

```python
# Column mode
timefence.Feature(source=src, columns=["country"])

# SQL mode
timefence.Feature(source=src, sql="SELECT ...", name="my_feature")

# Transform mode
timefence.Feature(source=src, transform=my_function)
```

## ASOF JOIN fallback warning

When the log shows "ASOF JOIN failed, falling back to ROW_NUMBER," this means DuckDB's ASOF JOIN couldn't handle the query (uncommon). Timefence automatically retried with the ROW_NUMBER strategy. The result is still correct — ROW_NUMBER is the universal fallback.

**No action needed.** To see why the fallback happened, run with `--debug`:

```bash
timefence --debug build --labels data/labels.parquet --features features.py -o train.parquet
```

## Still stuck?

1. Run `timefence doctor` to check your project setup
2. Run `timefence inspect data/your_file.parquet` to inspect columns and types
3. Open an issue at [github.com/gauthierpiarrette/timefence](https://github.com/gauthierpiarrette/timefence/issues)
