# Contributing to Timefence

Thanks for your interest in contributing to Timefence. This document covers everything you need to get started.

## Development Setup

**Prerequisites:** Python 3.9+ and [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Clone the repo
git clone https://github.com/gauthierpiarrette/timefence.git
cd timefence

# Create a virtual environment and install dev dependencies
uv sync --extra dev

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Pre-commit Hooks

Install pre-commit hooks to automatically lint and format on every commit:

```bash
uv run pre-commit install
```

This runs [Ruff](https://docs.astral.sh/ruff/) for linting and formatting before each commit.

## Running Tests

```bash
# Fast test suite (excludes slow property-based tests)
uv run task test

# Full suite including property-based tests
uv run task test-all

# Single test file
uv run pytest tests/test_engine.py

# Single test
uv run pytest tests/test_engine.py::TestBuildTemporalCorrectness::test_embargo

# With coverage
uv run pytest tests/ --cov=timefence --cov-report=term-missing
```

### Test Categories

| Directory / File | What it tests |
|---|---|
| `test_core.py` | Data model classes (Source, Feature, Labels, FeatureSet) |
| `test_engine.py` | Build, audit, explain, diff engine |
| `test_cli.py` | CLI commands and flag parsing |
| `test_integration.py` | End-to-end flows (build then audit, DataFrame input, CSV, multi-key) |
| `test_store.py` | Build tracking, caching, manifests |
| `test_config.py` | YAML config loading and resolution |
| `test_sql_safety.py` | SQL injection prevention helpers |
| `test_property.py` | Hypothesis property-based tests (marked `@pytest.mark.slow`) |
| `test_duration.py` | Duration string parsing |
| `test_errors.py` | Error class hierarchy and messages |

## Linting and Formatting

```bash
# Check for lint errors
uv run task lint

# Auto-fix lint errors
uv run ruff check src/ tests/ --fix

# Format code
uv run task format

# Check formatting without changing files
uv run task format-check

# Run all checks (lint + format-check + test)
uv run task check
```

Timefence uses Ruff with these rule sets: `E` (pycodestyle), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify), `RUF` (ruff-specific). Line length is 88.

All tasks are defined in `pyproject.toml` under `[tool.taskipy.tasks]`.

## Project Structure

```
src/timefence/
  __init__.py          # Public API surface
  core.py              # Data model: Source, Feature, Labels, FeatureSet
  engine.py            # Computational core: build, audit, explain, diff
  cli.py               # Click-based CLI
  store.py             # Build tracking and caching
  errors.py            # Error hierarchy with structured messages
  quickstart.py        # Example project generator
  _duration.py         # Duration string parsing (e.g., "30d" -> timedelta)
  _constants.py        # Default values
  _version.py          # Single source of version
  py.typed             # PEP 561 type marker

tests/                 # Mirrors src/ structure
docs/                  # Documentation site (timefence.dev)
blog/                  # Blog posts
```

## Making Changes

### Before You Start

1. Check [existing issues](https://github.com/gauthierpiarrette/timefence/issues) to avoid duplicate work.
2. For non-trivial changes, open an issue first to discuss the approach.
3. Fork the repo and create a branch from `main`.

### Code Style

- **Type hints on all public API functions and classes.** Timefence ships with `py.typed`.
- **No SQL injection.** Use `_qi()` for identifiers, `_ql()` for literals, `_safe_name()` for dynamic table names. Never use f-string interpolation with user input in SQL.
- **Structured error messages.** Errors should include: what happened, why it matters, where (specific data), and how to fix it.
- **No dead code.** Every branch, every import should be reachable.

### Writing Tests

- Every new feature or bug fix needs tests.
- Use the existing test structure: create a class per logical group, one method per case.
- For temporal correctness tests, verify the core invariant: `feature_time < label_time - embargo`.
- For integration tests that need optional dependencies (e.g., pandas), use `pytest.importorskip()`.
- Property-based tests go in `test_property.py` and should be marked with `@pytest.mark.slow`.

### Commit Messages

Use clear, concise commit messages that explain *why*, not just *what*:

```
Prevent SQL injection in dynamic table names

User-provided source names were interpolated directly into SQL.
Added _safe_name() helper with collision detection.
```

### Pull Request Process

1. Run the full check suite: `uv run task check` (lint + format + test).
2. Update `CHANGELOG.md` under an `## [Unreleased]` section.
3. Open a PR against `main` with a clear description of what changed and why.

## Reporting Bugs

Open an issue with:

1. What you expected to happen.
2. What actually happened (include the full error message).
3. Minimal reproduction steps (ideally a code snippet or CLI command).
4. Your environment: Python version, Timefence version (`timefence --version`), OS.

## Requesting Features

Open an issue with:

1. The problem you're trying to solve (not just the solution you want).
2. How you currently work around it (if applicable).
3. Why it belongs in Timefence specifically (vs. a separate tool).

Timefence is deliberately narrow in scope: temporal correctness for ML training data. Features that expand beyond this scope are unlikely to be accepted.

## Release Process

Releases are automated via GitHub Actions. When a tag matching `v*` is pushed:

1. The package is built with `uv build`.
2. Tests run against the built package.
3. On success, the package is published to PyPI via trusted publishing.

Only maintainers can create release tags.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
