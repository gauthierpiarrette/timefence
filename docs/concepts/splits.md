# Time-Based Splits

Build separate train/validation/test files split by label time.

## Usage

```python
result = timefence.build(
    labels=labels,
    features=[rolling_spend, user_country],
    output="dataset.parquet",
    splits={
        "train": ("2023-01-01", "2024-01-01"),
        "valid": ("2024-01-01", "2024-07-01"),
        "test":  ("2024-07-01", "2025-01-01"),
    },
)

# result.splits = {"train": Path(...), "valid": Path(...), "test": Path(...)}
```

Each split file contains only labels whose `label_time` falls within the given range. All temporal correctness guarantees still apply per-row.

## CLI

```bash
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet \
  --split train:2023-01-01:2024-01-01 \
  --split test:2024-01-01:2025-01-01
```

## Why time-based splits?

Random splits in time-series data cause leakage: the model sees future patterns in the training set. Time-based splits ensure the model is always evaluated on data it has never seen from the future.
