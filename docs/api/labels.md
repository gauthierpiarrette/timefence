# Labels

Defines the prediction target: which entities, at what times, and what outcome.

::: timefence.Labels
    options:
      show_root_heading: true
      members: []

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path \| None` | Path to labels file (Parquet). Mutually exclusive with `df`. |
| `df` | `Any \| None` | DataFrame, DuckDB relation, or any object with a compatible interface. Mutually exclusive with `path`. |
| `keys` | `str \| list[str]` | Entity key column(s). Must match the keys used in features. |
| `label_time` | `str` | Column name for the label event timestamp. |
| `target` | `str \| list[str]` | Prediction target column(s) (e.g., `"churned"`). |

## Example

```python
labels = timefence.Labels(
    path="data/labels.parquet",
    keys=["user_id"],
    label_time="label_time",
    target="churned",
)
```
