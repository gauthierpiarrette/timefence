# Changelog

## 0.9.0 — 2026-02-06

Initial public release.

### Core

- **Point-in-time correct joins** with strict `feature_time < label_time - embargo` invariant
- **ASOF JOIN fast path** (embargo=0) and ROW_NUMBER fallback (embargo>0) strategy selection
- Data model: `Source`, `CSVSource`, `SQLSource`, `Feature`, `FeatureSet`, `Labels`
- Three feature modes: column selection, SQL, and Python transform
- Configurable `max_lookback`, `max_staleness`, `embargo`, `on_missing`, and `join` mode
- Time-based train/valid/test splits
- Key mapping for cross-source key name mismatches
- Duplicate row handling (`on_duplicate`: `error` or `keep_any`)

### Commands

- `timefence build` — Build point-in-time correct training sets
- `timefence audit` — Detect temporal leakage in any dataset (rebuild-and-compare or lightweight temporal check)
- `timefence explain` — Preview join logic and SQL without executing
- `timefence diff` — Compare two training datasets with configurable numeric tolerances
- `timefence inspect` — Suggest keys and timestamps for a data file
- `timefence catalog` — List all features defined in a project
- `timefence quickstart` — Generate a self-contained example project with planted leakage
- `timefence doctor` — Diagnose project setup and common issues
- `timefence init` — Scaffold a `timefence.yaml` config file

### Store

- Build tracking with content-hashed manifests
- Feature-level and build-level caching
- Reproducible builds via `Store` and `.timefence/` directory

### Configuration

- `timefence.yaml` for project-level defaults
- Precedence: CLI flags > Python API > `timefence.yaml` > built-in defaults

### Developer Experience

- Rich terminal output for all commands
- JSON export for audit reports (`--json`) and HTML reports (`--html`)
- `--strict` flag for CI integration (exit code 1 on leakage)
- Structured error messages with fix suggestions
- `--dry-run` for build previews
