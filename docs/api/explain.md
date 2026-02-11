# explain()

Preview the join logic that `build()` will use, without executing any queries.

::: timefence.explain
    options:
      show_root_heading: true

## Example

```python
import timefence

result = timefence.explain(
    labels=labels,
    features=[rolling_spend, user_country],
    max_lookback="365d",
    join="strict",
)

print(result)
```

### Sample output

```
JOIN PLAN for 5000 label rows

For each label row (keys, label_time):

  1. rolling_spend_30d
     Source:  data/transactions.parquet
     Join:    feature_time < label_time - INTERVAL '1d'
     Window:  [label_time - 365d, label_time - 1d)
     Embargo: 1d
     Strategy: row_number
     SQL:
       SELECT user_id, created_at AS feature_time, amount
       FROM 'data/transactions.parquet'
       WHERE user_id = {K}
         AND created_at < {T} - INTERVAL '1d'
         AND created_at >= {T} - INTERVAL '365d'
       ORDER BY created_at DESC
       LIMIT 1

  2. country
     Source:  data/users.parquet
     Join:    feature_time < label_time
     Window:  [label_time - 365d, label_time)
     Embargo: none
     Strategy: asof
     SQL:
       SELECT user_id, updated_at AS feature_time, country
       FROM 'data/users.parquet'
       WHERE user_id = {K}
         AND updated_at < {T}
         AND updated_at >= {T} - INTERVAL '365d'
       ORDER BY updated_at DESC
       LIMIT 1
```

`{K}` and `{T}` are placeholders for the entity key and label time of each row.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `Labels` | *required* | Label definition. |
| `features` | `Sequence[Feature \| FeatureSet]` | *required* | Features to explain. |
| `max_lookback` | `str \| timedelta` | `"365d"` | Maximum feature age. |
| `max_staleness` | `str \| timedelta \| None` | `None` | Max staleness threshold. |
| `join` | `str` | `"strict"` | `"strict"` (`<`) or `"inclusive"` (`<=`). |

## Returns: ExplainResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `.label_count` | `int` | Number of label rows. |
| `.plan` | `list[dict]` | Per-feature join plan (see below). |

Each item in `.plan` contains:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Feature name. |
| `source` | `str` | Source file path or name. |
| `join_condition` | `str` | The temporal join condition (e.g., `feature_time < label_time`). |
| `window` | `str` | The valid feature window (e.g., `[label_time - 365d, label_time)`). |
| `embargo_str` | `str` | Embargo duration or `"none"`. |
| `strategy` | `str` | `"asof"` or `"row_number"`. |
| `sql` | `str` | Example SQL for this feature. |
