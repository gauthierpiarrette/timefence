# timefence inspect

Analyze a data file and suggest which columns are keys and timestamps.

## Usage

```bash
timefence inspect data/transactions.parquet
timefence inspect data/events.csv --json
```

## Options

| Option | Description |
|--------|-------------|
| `path` | Positional. Path to data file (Parquet or CSV). |
| `--json` | Output as JSON. |

Output includes column names, types, uniqueness percentage, and auto-detected key/timestamp suggestions.
