# Installation

## Requirements

- **Python 3.9+**
- pip or uv

## Install

```bash
pip install timefence
```

### Optional extras

```bash
# Jupyter notebook support
pip install timefence[notebook]

# Development dependencies
pip install timefence[dev]
```

!!! note "dbt integration"
    A `timefence[dbt]` extra is planned for a future release. The `from_dbt()` function exists but is not yet implemented.

## Dependencies

Timefence depends on:

| Package | Version | Purpose |
|---------|---------|---------|
| `duckdb` | >= 1.0.0 | Columnar SQL engine |
| `click` | >= 8.0.0 | CLI framework |
| `rich` | >= 13.0.0 | Terminal output |
| `pandas` | >= 1.5.0 | DataFrame interop |
| `pyyaml` | >= 6.0 | YAML config |

No Spark, no JVM, no cloud infrastructure.

## Performance

Built on DuckDB's columnar engine. Median of 3 runs after warmup (Intel i7, 16 GB):

| Scenario | Labels | Features | Build | Audit |
|----------|--------|----------|-------|-------|
| Small project | 100K | 1 | **0.5s** | 0.3s |
| Typical project | 100K | 10 | **1.9s** | 1.7s |
| Large project | 1M | 1 | **3.0s** | 2.0s |
| Large + many features | 1M | 10 | **12s** | 8.5s |

Adding embargo, staleness, and splits costs seconds, not minutes.

??? note "Run benchmarks yourself"

    ```bash
    uv run python benchmarks/bench.py --quick
    uv run python benchmarks/bench.py --quick --include-pandas
    ```

## Verify installation

```bash
timefence --version
timefence doctor
```
