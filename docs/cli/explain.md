# timefence explain

Preview join logic without executing any queries.

## Usage

```bash
# Full explain
timefence explain \
  --labels data/labels.parquet \
  --features features.py

# Single feature
timefence explain --features features.py:rolling_spend_30d

# JSON output
timefence explain --features features.py --json
```

## Options

| Option | Description |
|--------|-------------|
| `--labels` | Path to labels file. |
| `--features` | Path to features Python file. Append `:name` for single feature. |
| `--max-lookback` | Maximum lookback window. Default: `365d`. |
| `--join-mode` | `strict` or `inclusive`. Default: `strict`. |
| `--json` | Output as JSON. |
