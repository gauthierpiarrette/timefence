# Join Logic

Timefence performs an **as-of join** (also called a point-in-time join) for each feature. For each label row, it finds the most recent feature value that satisfies the temporal constraint.

## Join strategies

Timefence automatically selects the best SQL strategy:

| Strategy | When Used | Performance |
|----------|-----------|-------------|
| **ASOF JOIN** | No embargo (fast path) | Fastest — native DuckDB operator |
| **ROW_NUMBER** | With embargo or complex constraints | Universal fallback, always correct |

If ASOF JOIN fails for any reason, Timefence automatically falls back to ROW_NUMBER with a warning.

## Additional constraints

### max_lookback

Maximum age of a feature value. Default: `"365d"`. Features older than this are treated as missing.

```python
result = timefence.build(
    labels=labels,
    features=features,
    max_lookback="90d",  # Only use features from last 90 days
)
```

### max_staleness

If set, features older than this threshold are treated as missing even if within the lookback window. Must satisfy: `max_staleness > embargo`.

```python
result = timefence.build(
    labels=labels,
    features=features,
    max_staleness="30d",  # Drop features older than 30 days
)
```

### on_missing

What to do when no valid feature value exists for a label row:

| Value | Behavior |
|-------|----------|
| `"null"` | Keep the row, fill feature columns with NULL (default) |
| `"skip"` | Drop the row entirely |

## Inspecting the join plan

Preview what SQL will be generated without executing:

```bash
timefence explain --labels data/labels.parquet --features features.py
```

```
JOIN PLAN for 5,000 label rows

For each label row (keys, label_time):

  1. rolling_spend_30d
     Source:  data/transactions.parquet
     Join:    feature_time < label_time - INTERVAL '1d'
     Window:  [label_time - 365d, label_time - 1d)
     Embargo: 1d
     Strategy: row_number

  2. country
     Source:  data/users.parquet
     Join:    feature_time < label_time
     Window:  [label_time - 365d, label_time)
     Embargo: none
     Strategy: asof
```

Or in Python:

```python
result = timefence.explain(labels=labels, features=[spend, country])
print(result)        # Same formatted output
print(result.plan)   # Raw list of dicts per feature
```

See the [explain() API reference](../api/explain.md) for full details on the `ExplainResult` object.
