# timefence doctor

Diagnose project setup and common issues.

## Usage

```bash
timefence doctor
timefence doctor --json
```

## Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON. |

## Checks performed

- Config file presence and validity
- DuckDB installation and version
- Feature file validity
- Source file accessibility
- Label schema compatibility
- Duplicate (key, timestamp) combinations
- Column name conflicts
