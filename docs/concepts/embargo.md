# Embargo

Real-world ML pipelines have latency: data arrives late, ETL jobs run on schedules, and features take time to compute. The **embargo** parameter models this lag.

## Usage

```python
rolling_spend = timefence.Feature(
    source=transactions,
    sql="SELECT ...",
    embargo="1d"  # Feature available 1 day after event
)
```

## How it works

With `embargo="1d"`, a feature recorded at `2024-03-15 10:00` is only eligible for labels at `2024-03-16 10:00` or later. This prevents optimistic leakage from features that wouldn't actually be available in production.

The full temporal constraint becomes:

```
feature_time < label_time - embargo
```

## When to use embargo

| Scenario | Recommended Embargo |
|----------|-------------------|
| Real-time features | `"0d"` (no embargo) |
| Daily ETL pipeline | `"1d"` |
| Weekly batch features | `"7d"` |
| Monthly aggregates | `"30d"` |

!!! tip
    When in doubt, set embargo to match your production pipeline's worst-case latency. It's better to be conservative (larger embargo) than to train on features that wouldn't actually be available at prediction time.

## Duration format

Accepted formats: `"30d"`, `"1d12h"`, `"6h"`, `"30m"`, `"15s"`.

See [Duration Format](../reference/durations.md) for the full specification.
