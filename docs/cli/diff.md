# timefence diff

Compare two datasets for value changes or schema drift.

## Usage

```bash
timefence diff train_v1.parquet train_v2.parquet \
  --keys user_id \
  --label-time label_time

# Custom tolerance
timefence diff v1.parquet v2.parquet \
  --keys user_id \
  --label-time label_time \
  --atol 0.01 \
  --rtol 0.001

# JSON output
timefence diff v1.parquet v2.parquet \
  --keys user_id \
  --label-time label_time \
  --json
```

## Options

| Option | Description |
|--------|-------------|
| `old_path` | Positional. Path to first dataset. |
| `new_path` | Positional. Path to second dataset. |
| `--keys` | Key column(s), comma-separated (required). |
| `--label-time` | Label time column (required). |
| `--atol` | Absolute tolerance for numeric comparison (default: `1e-10`). |
| `--rtol` | Relative tolerance for numeric comparison (default: `1e-7`). |
| `--json` | Output as JSON. |
