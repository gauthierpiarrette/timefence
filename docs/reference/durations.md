# Duration Format

Timefence accepts human-readable duration strings wherever a time duration is expected (`embargo`, `max_lookback`, `max_staleness`).

## Formats

| Format | Example | Meaning |
|--------|---------|---------|
| `Nd` | `"30d"` | 30 days |
| `Nh` | `"6h"` | 6 hours |
| `Nm` | `"30m"` | 30 minutes |
| `Ns` | `"15s"` | 15 seconds |
| Combined | `"1d12h"` | 1 day and 12 hours |
| Zero | `"0d"` or `"0"` | No duration |

## Python API

You can also pass a Python `timedelta` object directly:

```python
from datetime import timedelta

feature = timefence.Feature(
    source=transactions,
    columns=["amount"],
    embargo=timedelta(days=1),
)

result = timefence.build(
    labels=labels,
    features=[feature],
    max_lookback=timedelta(days=365),
)
```
