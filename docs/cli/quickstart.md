# timefence quickstart

Generate a self-contained example project with synthetic data and leakage scenarios.

## Usage

```bash
# Default churn example
timefence quickstart churn-example

# Minimal version
timefence quickstart myproject --minimal
```

## Options

| Option | Description |
|--------|-------------|
| `project_name` | Positional. Directory name to create. Optional; defaults to `"churn-example"`. |
| `--template` | Example template. Default: `"churn"`. Accepted but currently has no effect. |
| `--minimal` | Generate a smaller example. |
