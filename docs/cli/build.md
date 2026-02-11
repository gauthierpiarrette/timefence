# timefence build

Build a point-in-time correct training set. Uses `timefence.yaml` defaults if available.

## Usage

```bash
# Basic build
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet

# With all options
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet \
  --max-lookback 365d \
  --max-staleness 30d \
  --on-missing null \
  --join-mode strict

# Time-based splits
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet \
  --split train:2023-01-01:2024-01-01 \
  --split test:2024-01-01:2025-01-01

# Dry run (preview plan without executing)
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet \
  --dry-run
```

## Options

| Option | Description |
|--------|-------------|
| `--labels` | Path to labels file (required unless in config). |
| `--features` | Path to features Python file (required unless in config). |
| `-o, --output` | Output path for training set (required). |
| `--max-lookback` | Maximum lookback window (e.g., `"365d"`). Default: `365d`. |
| `--max-staleness` | Maximum feature staleness. |
| `--on-missing` | `null` or `skip`. Default: `null`. |
| `--join-mode` | `strict` or `inclusive`. Default: `strict`. |
| `--split` | Time split: `name:start:end` (repeatable). |
| `--dry-run` | Preview the join plan without executing queries. |
| `--flatten` | Strip the `{feature_name}__` prefix from output columns. For example, `spend__amount` becomes `amount`. |
| `--json` | Output the full build manifest as JSON instead of the Rich terminal display. |
