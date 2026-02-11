# timefence audit

Scan a dataset for temporal leakage.

## Usage

```bash
# Basic audit
timefence audit data/train.parquet

# With explicit options
timefence audit data/train.parquet \
  --features features.py \
  --keys user_id \
  --label-time label_time

# CI mode: exit code 1 if leakage found
timefence audit data/train.parquet --strict

# Export reports
timefence audit data/train.parquet --html report.html
timefence audit data/train.parquet --json
```

## Options

| Option | Description |
|--------|-------------|
| `data` | Positional. Path to the dataset file. |
| `--features` | Path to Python file with feature definitions. |
| `--keys` | Key column(s), comma-separated. |
| `--label-time` | Label time column name. Default: `"label_time"`. |
| `--strict` | Exit with code 1 if leakage found (for CI/CD). |
| `--html FILE` | Export interactive HTML report. |
| `--json` | Output as JSON. |
