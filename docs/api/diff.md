# diff()

Compare two training datasets for schema drift and value changes.

::: timefence.diff
    options:
      show_root_heading: true

## Example

```python
result = timefence.diff(
    old="train_v1.parquet",
    new="train_v2.parquet",
    keys=["user_id"],
    label_time="label_time",
    atol=1e-10,
    rtol=1e-7,
)
```

## Returns: DiffResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `.old_rows` | `int` | Row count in old dataset. |
| `.new_rows` | `int` | Row count in new dataset. |
| `.schema_changes` | `list[dict]` | Schema changes: `type` (`+` added, `-` removed, `~` changed, `=` unchanged, `?` comparison failed), `column`, `detail`. |
| `.value_changes` | `dict[str, dict]` | Per-column: `changed_count`, `changed_pct`, `mean_delta`, `max_delta`. `mean_delta` and `max_delta` are only present for numeric columns. |
