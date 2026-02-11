# build()

Constructs a point-in-time correct training dataset.

::: timefence.build
    options:
      show_root_heading: true

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `Labels` | *required* | Label definition. |
| `features` | `Sequence[Feature \| FeatureSet]` | *required* | Features to join. |
| `output` | `str \| Path \| None` | `None` | Output file path. If `None`, no file is written. |
| `max_lookback` | `str \| timedelta` | `"365d"` | Maximum feature age. |
| `max_staleness` | `str \| timedelta \| None` | `None` | Max feature age before treating as missing. |
| `join` | `str` | `"strict"` | `"strict"` (`<`) or `"inclusive"` (`<=`). |
| `on_missing` | `str` | `"null"` | `"null"` (keep row with NULLs) or `"skip"` (drop row). |
| `splits` | `dict \| None` | `None` | Time-based splits: `{"train": ("start", "end"), ...}`. |
| `store` | `Store \| None` | `None` | Build tracking and caching. |
| `flatten_columns` | `bool` | `False` | Strip feature name prefix from output columns. |
| `progress` | `Callable \| None` | `None` | Callback for progress reporting. |

## Returns: BuildResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `.output_path` | `str \| None` | Path to the output file. |
| `.stats` | `BuildStats` | Build statistics (see below). |
| `.sql` | `str` | The exact SQL queries executed. |
| `.splits` | `dict[str, Path] \| None` | Split output file paths. |
| `.manifest` | `dict` | Full build manifest (JSON-serializable). |
| `.validate()` | `bool` | Check if the post-build audit passed. |
| `.explain()` | `str` | Return the SQL used for joins. |

## BuildStats

| Attribute | Type | Description |
|-----------|------|-------------|
| `.row_count` | `int` | Total rows in the output dataset. |
| `.column_count` | `int` | Total columns in the output dataset. |
| `.feature_stats` | `dict[str, dict]` | Per-feature join statistics (see below). |
| `.duration_seconds` | `float` | Wall-clock time for the build. |

Each entry in `.feature_stats` is a dict:

| Key | Type | Description |
|-----|------|-------------|
| `"matched"` | `int` | Number of label rows that matched a feature value. |
| `"missing"` | `int` | Number of label rows with no valid feature (filled with NULL or skipped). |
| `"cached"` | `bool` | Whether this feature was loaded from cache. |

## Build manifest

Every build produces a JSON manifest (saved by `Store` or accessible via `.manifest`):

```json
{
  "timefence_version": "0.9.1",
  "build_id": "20240315T120000Z",
  "created_at": "2024-03-15T12:00:00+00:00",
  "duration_seconds": 1.8,
  "labels": {
    "path": "data/labels.parquet",
    "content_hash": "sha256:abc123...",
    "row_count": 5000,
    "time_range": ["2023-01-01", "2024-12-31"],
    "keys": ["user_id"],
    "label_time_column": "label_time",
    "target_columns": ["churned"]
  },
  "features": {
    "rolling_spend_30d": {
      "definition_hash": "sha256:def456...",
      "source_content_hash": "sha256:aaa789...",
      "embargo": "1d",
      "matched_rows": 4800,
      "missing_rows": 200,
      "output_columns": ["spend_30d"],
      "cached": false
    }
  },
  "parameters": {
    "max_lookback": "365d",
    "max_staleness": null,
    "join": "strict",
    "on_missing": "null"
  },
  "output": {
    "path": "train.parquet",
    "content_hash": "sha256:out321...",
    "file_size_bytes": 204800,
    "row_count": 5000,
    "column_count": 7
  },
  "audit": {
    "passed": true,
    "invariant": "feature_time < label_time - embargo",
    "rows_checked": 5000
  },
  "environment": {
    "python_version": "3.11.5",
    "duckdb_version": "0.10.1",
    "os": "Linux-6.1.0-x86_64"
  }
}
```

## Column naming

By default, feature columns in the output are namespaced as `{feature_name}__{column_name}` to avoid collisions:

```
user_id | label_time | churned | country__country | spend__spend_30d
```

Set `flatten_columns=True` to strip the prefix when there are no name collisions:

```
user_id | label_time | churned | country | spend_30d
```

## Progress reporting

Pass a callback to show build progress (used by the CLI for its Rich progress bar):

```python
def on_progress(message: str):
    print(message)

result = timefence.build(
    labels=labels,
    features=features,
    output="train.parquet",
    progress=on_progress,
)
# Prints: "Loading labels", "Computing spend (1/2)", "Joining spend (1/2)", ...
```
