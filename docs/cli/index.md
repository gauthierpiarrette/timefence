# CLI Reference

Timefence provides a full command-line interface built on [Click](https://click.palletsprojects.com/) with [Rich](https://rich.readthedocs.io/) terminal output.

## Global Options

These flags apply to all commands when placed before the subcommand name.

```bash
timefence --version             # Show version and exit
timefence -v <command>          # Verbose: show generated SQL and details
timefence --debug <command>     # Debug: full output including DuckDB internals
```

| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `-v, --verbose` | Show generated SQL and details |
| `--debug` | Show full debug output including DuckDB internals |

## Commands

| Command | Description |
|---------|-------------|
| [`audit`](audit.md) | Scan a dataset for temporal leakage |
| [`build`](build.md) | Build a point-in-time correct training set |
| [`explain`](explain.md) | Preview join logic without executing |
| [`diff`](diff.md) | Compare two datasets |
| [`inspect`](inspect.md) | Analyze a data file |
| [`quickstart`](quickstart.md) | Generate a sample project |
| [`catalog`](catalog.md) | List all features |
| [`doctor`](doctor.md) | Diagnose project setup |
| [`init`](init.md) | Initialize a project |
