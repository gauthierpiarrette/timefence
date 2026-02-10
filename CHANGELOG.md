# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.1] — 2026-02-10

### Security

- **SQL injection prevention.** All user-provided column names, table names, and file paths are now quoted through `_qi()`, `_ql()`, and `_safe_name()` helpers. Previously, these were interpolated directly into SQL via f-strings.
- Added collision detection for `_safe_name()` — if two different source names sanitize to the same table name, the build raises an error instead of silently overwriting data.

### Added

- **Type hints** on all public API classes and functions (`Source`, `Feature`, `Labels`, `FeatureSet`, `Store`, `build`, `audit`, `explain`, `diff`).
- **`py.typed` marker** (PEP 561) — enables IDE autocomplete and static analysis with mypy/pyright.
- **Progress reporting** — `build()` accepts a `progress` callback for long-running builds. CLI `build` command shows a Rich progress bar.
- **`--verbose` / `--debug` flags** — `-v` shows generated SQL and build details; `--debug` adds DuckDB internals.
- **Coverage reporting in CI** — tests run with `pytest-cov`, results uploaded to Codecov.
- **Windows CI** — test matrix expanded to include `windows-latest`.
- **Pre-commit hooks** — Ruff lint and format checks run automatically on commit.
- **Dependabot** — automated dependency updates for pip and GitHub Actions.
- **Comparison page** — `docs/compare.html` comparing Timefence to feature stores, data quality tools, and manual SQL.
- **PyPI metadata** — keywords, additional classifiers (`Typing :: Typed`, `Operating System :: OS Independent`, Python 3.13), and project URLs (Issues, Changelog).
- **`CONTRIBUTING.md`** — contributor guide with dev setup, testing, code style, and PR process.

### Added (Tests)

- DataFrame input mode (`df=` parameter) — build and audit with pandas DataFrames.
- Multi-key joins — composite keys like `(user_id, product_id)`.
- CSV source end-to-end — full build pipeline from `.csv` files.
- Transform mode through audit — audit a dataset built with Python transform features.
- YAML config loading — 20 tests covering valid/malformed/empty configs and resolution precedence.
- SQL safety helpers — 28 tests with adversarial inputs (quotes, semicolons, unicode, empty strings).
- Duplicate feature name detection.
- Verbose/debug CLI flag behavior.

### Changed

- **Replaced custom YAML parser with PyYAML.** The hand-rolled `_parse_simple_yaml()` function (78 lines) is replaced by `yaml.safe_load()`. PyYAML added as a runtime dependency.
- **Materialized final query in `build()`.** The result SQL is now executed once into a temp table. Previously, the same query was executed three times (COPY, DESCRIBE, COUNT). Significant performance improvement for large datasets.
- `Feature.source_keys` now returns raw source keys. Key mapping is applied by the engine during join SQL generation, not at the property level.

### Fixed

- Windows path handling in tests — paths written to Python source files use forward slashes to avoid `\U` unicode escape errors.
- `Feature.__init__` now initializes all internal attributes in every mode branch, preventing `AttributeError` on edge cases.
- `_check_duplicates` correctly formats example dict for multi-key features.
- Removed dead code in `_duration.py` (unreachable conditional block).
- `max_lookback` test assertion changed from tautological `assert missing >= 0` to meaningful `assert missing > 0`.

### Removed

- Custom `_parse_simple_yaml()` function from `cli.py` (replaced by PyYAML).

---

## [0.9.0] — 2026-02-06

Initial public release.

### Core

- **Point-in-time correct joins** with strict `feature_time < label_time - embargo` invariant.
- **ASOF JOIN fast path** (embargo=0) and ROW_NUMBER fallback (embargo>0) strategy selection.
- Data model: `Source`, `CSVSource`, `SQLSource`, `Feature`, `FeatureSet`, `Labels`.
- Three feature modes: column selection, SQL, and Python transform.
- Configurable `max_lookback`, `max_staleness`, `embargo`, `on_missing`, and `join` mode.
- Time-based train/valid/test splits.
- Key mapping for cross-source key name mismatches.
- Duplicate row handling (`on_duplicate`: `error` or `keep_any`).

### Commands

- `timefence build` — Build point-in-time correct training sets.
- `timefence audit` — Detect temporal leakage in any dataset.
- `timefence explain` — Preview join logic and SQL without executing.
- `timefence diff` — Compare two training datasets with configurable numeric tolerances.
- `timefence inspect` — Suggest keys and timestamps for a data file.
- `timefence catalog` — List all features defined in a project.
- `timefence quickstart` — Generate a self-contained example project with planted leakage.
- `timefence doctor` — Diagnose project setup and common issues.
- `timefence init` — Scaffold a `timefence.yaml` config file.

### Store

- Build tracking with content-hashed manifests.
- Feature-level and build-level caching.
- Reproducible builds via `Store` and `.timefence/` directory.

### Configuration

- `timefence.yaml` for project-level defaults.
- Precedence: CLI flags > Python API > `timefence.yaml` > built-in defaults.

### Developer Experience

- Rich terminal output for all commands.
- JSON export for audit reports (`--json`) and HTML reports (`--html`).
- `--strict` flag for CI integration (exit code 1 on leakage).
- Structured error messages with fix suggestions.
- `--dry-run` for build previews.

[Unreleased]: https://github.com/gauthierpiarrette/timefence/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/gauthierpiarrette/timefence/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/gauthierpiarrette/timefence/releases/tag/v0.9.0
