# Configuration

Timefence looks for a `timefence.yaml` (or `timefence.yml`) in the current directory. All fields are optional — CLI flags and Python API arguments take precedence.

## Full example

```yaml
name: churn-model
version: "1.0"

# Feature file(s)
features:
  - features.py

# Label configuration
labels:
  path: data/labels.parquet
  keys: [user_id]
  label_time: label_time
  target: [churned]

# Default parameters
defaults:
  max_lookback: 365d
  max_staleness: null    # or e.g. "30d"
  join: strict          # "strict" or "inclusive"
  on_missing: "null"    # "null" or "skip"

# Store directory for build tracking
store: .timefence/

# Output directory (relative paths in build resolve against this)
output:
  dir: artifacts/
```

!!! tip "Precedence"
    CLI flags > Python API arguments > `timefence.yaml` > built-in defaults.
