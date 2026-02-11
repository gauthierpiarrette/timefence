# Temporal Correctness

The core invariant Timefence enforces:

!!! success "The Rule"

    ```
    feature_time < label_time - embargo
    ```

For every row in your training set, the feature value used must have been available *strictly before* the label event time (minus any embargo buffer). This prevents future data leakage — the most common and hardest-to-detect source of inflated ML metrics.

## Why it matters

When you join features to labels with a `LEFT JOIN` or `merge_asof`, each label gets the latest feature row — including data from *after* the event you're predicting. The model trains on the future. Offline metrics look great. Production doesn't match.

No error, no warning, no way to tell from the output alone.

## Strict vs Inclusive

In `inclusive` join mode, the condition relaxes to:

```
feature_time <= label_time - embargo
```

| Mode | Condition | Use When |
|------|-----------|----------|
| `"strict"` | `feature_time < label_time` | Default. No same-timestamp leakage. |
| `"inclusive"` | `feature_time <= label_time` | When same-timestamp is safe (e.g., static attributes). |

## How Timefence enforces it

Timefence generates SQL (ASOF JOIN or ROW_NUMBER) and runs it in an embedded DuckDB. For each label row, it finds the most recent feature value that satisfies the temporal constraint. Every query is inspectable via `timefence -v build` or `timefence explain`.
