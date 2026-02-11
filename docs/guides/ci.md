# Guide: CI/CD Integration

Stop leakage before it reaches production.

## CLI

```bash
# Exits with code 1 if leakage is found
timefence audit data/train.parquet --strict
```

## Python

```python
report = timefence.audit(
    data="data/train.parquet",
    features=[rolling_spend, user_country],
    keys=["user_id"],
    label_time="label_time",
)

# Raises TimefenceLeakageError if any leakage detected
report.assert_clean()
```

## GitHub Actions

```yaml
- name: Audit training data
  run: |
    pip install timefence
    timefence audit data/train.parquet \
      --features features.py \
      --keys user_id \
      --label-time label_time \
      --strict
```

## GitLab CI

```yaml
audit:
  image: python:3.12
  script:
    - pip install timefence
    - timefence audit data/train.parquet --features features.py --strict
```

## What `--strict` does

- Exit code `0` = no leakage detected (pipeline continues)
- Exit code `1` = leakage detected (pipeline fails)

Combine with `--json` for machine-readable output:

```bash
timefence audit data/train.parquet --strict --json > audit-result.json
```

Or generate an HTML report for review:

```bash
timefence audit data/train.parquet --html audit-report.html
```
