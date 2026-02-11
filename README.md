<p align="center">
  <img src="docs/assets/logo.png" alt="Timefence" width="80">
</p>

<h1 align="center">Timefence</h1>

<p align="center">
  <strong>Your ML model may be trained on the future. Find out in one command.</strong>
</p>

<p align="center">
  <a href="https://github.com/gauthierpiarrette/timefence/actions/workflows/ci.yml"><img src="https://github.com/gauthierpiarrette/timefence/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/gauthierpiarrette/timefence"><img src="https://codecov.io/gh/gauthierpiarrette/timefence/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/timefence/"><img src="https://img.shields.io/pypi/v/timefence" alt="PyPI"></a>
  <a href="https://pypi.org/project/timefence/"><img src="https://img.shields.io/pypi/pyversions/timefence" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://timefence.dev">Docs</a> &middot;
  <a href="CHANGELOG.md">Changelog</a> &middot;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

Timefence finds and fixes temporal data leakage in ML training sets. No infrastructure required — runs locally, reads Parquet/CSV, and finishes in seconds.

If you build training data by joining features to labels, your model may be training on the future. A `LEFT JOIN` or `merge_asof` gives each label the latest feature row — including data from *after* the event you're predicting. The model trains on the future. Offline metrics look great. Production doesn't match. No error, no warning, no way to tell from the output alone.

## Try It in 60 Seconds

```bash
pip install timefence
timefence quickstart churn-example && cd churn-example
```

Audit the training set — Timefence finds 3 leaky features:

```bash
timefence audit data/train_LEAKY.parquet
```

<p align="center">
  <img src="docs/assets/terminal-audit.png" alt="timefence audit showing leakage in 3 of 4 features" width="800">
</p>

Rebuild with temporal correctness:

```bash
timefence build -o train_CLEAN.parquet
```

<p align="center">
  <img src="docs/assets/terminal-build.png" alt="timefence build producing a clean training set" width="800">
</p>

Verify the new dataset is clean:

```bash
timefence audit train_CLEAN.parquet
# ALL CLEAN — no temporal leakage detected
```

Already have a training set? Audit it directly — no config needed:

```bash
timefence audit your_data.parquet --features features.py --keys user_id --label-time label_time
```

See the [Getting Started guide](https://timefence.dev/getting-started/installation/) for more.

## Python API

```python
import timefence

users = timefence.Source(path="data/users.parquet", keys=["user_id"], timestamp="updated_at")
txns  = timefence.Source(path="data/txns.parquet", keys=["user_id"], timestamp="created_at")

country = timefence.Feature(source=users, columns=["country"])
spend   = timefence.Feature(source=txns, embargo="1d", name="spend_30d", sql="""
    SELECT user_id, created_at AS feature_time,
           SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at
               RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW) AS spend_30d
    FROM {source}
""")

labels = timefence.Labels(
    path="data/labels.parquet", keys=["user_id"],
    label_time="label_time", target=["churned"],
)

result = timefence.build(labels=labels, features=[country, spend], output="train.parquet")
result  # renders in Jupyter
```

<p align="center">
  <img src="docs/assets/jupyter-output.png" alt="BuildResult rendered in Jupyter" width="550">
</p>

Audit an existing dataset without rebuilding:

```python
report = timefence.audit(
    "train.parquet",
    features=[country, spend],
    keys=["user_id"],
    label_time="label_time",
)
report.assert_clean()  # raises if leakage found
```

## Add to CI

Stop leakage before it reaches production:

```yaml
- run: pip install timefence && timefence audit data/train.parquet --features features.py --strict
```

`--strict` exits with code 1 on leakage. Your pipeline fails before a leaky model ever trains.

## Performance

Built on DuckDB's columnar engine. Median of 3 runs after warmup (Intel i7, 16 GB):

| Scenario | Labels | Features | Build | Audit |
|----------|--------|----------|-------|-------|
| Small project | 100K | 1 | **0.5s** | 0.3s |
| Typical project | 100K | 10 | **1.9s** | 1.7s |
| Large project | 1M | 1 | **3.0s** | 2.0s |
| Large + many features | 1M | 10 | **12s** | 8.5s |

Adding embargo, staleness, and splits costs seconds, not minutes.

<details>
<summary>Run benchmarks yourself</summary>

```bash
uv run python benchmarks/bench.py --quick
uv run python benchmarks/bench.py --quick --include-pandas
```

</details>

## How It Works

1. **Define** — declare sources, features, and labels in Python or `timefence.yaml`
2. **Build** — Timefence generates SQL (ASOF JOIN or ROW_NUMBER) and runs it in an embedded DuckDB, enforcing `feature_time < label_time - embargo` for every row
3. **Audit** — point at any existing dataset to check for leakage, no rebuild needed

No server, no JVM, no Spark. Every query is inspectable via `timefence -v build` or `timefence explain`.

## All Features

| | |
|---|---|
| **Joins** | Point-in-time correct. ASOF JOIN fast path, ROW_NUMBER fallback |
| **Guardrails** | Embargo, max lookback, max staleness — all configurable |
| **Inputs** | Parquet, CSV, SQL query, DataFrame |
| **Feature modes** | Column selection, SQL, Python transform |
| **Splitting** | Time-based train / validation / test splits |
| **Caching** | Feature-level cache with content-hash keys |
| **Audit** | Full rebuild-and-compare or lightweight temporal check |
| **Reports** | Severity classification. JSON manifest, HTML report, Rich terminal |
| **CLI** | `quickstart` `build` `audit` `explain` `diff` `inspect` `catalog` `doctor` |
| **Flags** | `-v` verbose · `--debug` · `--strict` CI gate · `--json` · `--html` |

## What Timefence Is NOT

| Not This | Why | Use Instead |
|----------|-----|-------------|
| Feature store | No server, no online serving | Tecton, Feast |
| Data orchestrator | No scheduling, no DAGs | Airflow, Dagster |
| Data quality framework | Temporal correctness only | Great Expectations |
| ML pipeline framework | Produces training data only | MLflow, Metaflow |

One tool. One job. Temporal correctness for ML training data.

---

<p align="center">
  If Timefence helps you, consider giving it a <a href="https://github.com/gauthierpiarrette/timefence">⭐️ on GitHub</a> — it helps others find it.
</p>

<p align="center">
  <a href="https://timefence.dev">Documentation</a> &middot;
  <a href="CONTRIBUTING.md">Contributing</a> &middot;
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">MIT License</p>
