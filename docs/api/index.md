# API Reference

Timefence's Python API is organized around five core objects and four functions.

## Core Objects

| Class | Purpose |
|-------|---------|
| [`Source`](source.md) | A historical data source (Parquet, CSV, DataFrame) |
| [`Feature`](feature.md) | A named signal derived from a Source |
| [`Labels`](labels.md) | Prediction target definition |
| [`FeatureSet`](featureset.md) | Group features for reuse |
| [`Store`](store.md) | Build tracking and caching |

## Functions

| Function | Purpose |
|----------|---------|
| [`build()`](build.md) | Construct a point-in-time correct training dataset |
| [`audit()`](audit.md) | Scan a dataset for temporal leakage |
| [`explain()`](explain.md) | Preview join logic without executing |
| [`diff()`](diff.md) | Compare two datasets for changes |

## Quick example

```python
import timefence

source = timefence.Source(path="data/users.parquet", keys=["user_id"], timestamp="updated_at")
feature = timefence.Feature(source=source, columns=["country"])
labels = timefence.Labels(path="data/labels.parquet", keys=["user_id"], label_time="label_time", target="churned")

result = timefence.build(labels=labels, features=[feature], output="train.parquet")
```
