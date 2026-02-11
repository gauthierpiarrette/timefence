# Source

Defines a historical data source (Parquet, CSV, or DataFrame).

::: timefence.Source
    options:
      show_root_heading: true
      members: []

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path \| None` | Path to the data file. Mutually exclusive with `df`. |
| `keys` | `str \| list[str]` | Column name(s) representing the entity (e.g., `"user_id"`). |
| `timestamp` | `str` | Column name containing the valid-at timestamp. |
| `name` | `str \| None` | Human-readable name. Defaults to filename stem. |
| `format` | `str \| None` | `"parquet"` or `"csv"`. Auto-detected from file extension. |
| `delimiter` | `str` | CSV delimiter. Default: `","`. |
| `timestamp_format` | `str \| None` | Optional strftime format for parsing timestamps (CSV only). |
| `df` | `Any \| None` | Pass a DataFrame, DuckDB relation, or any object with a compatible interface instead of a file path. |

!!! info "Mutual exclusivity"
    Provide exactly one of `path` or `df`. Passing both raises `TimefenceValidationError`.
    Passing neither also raises an error.

## Examples

```python
# Parquet source
transactions = timefence.Source(
    path="data/transactions.parquet",
    keys=["user_id"],
    timestamp="created_at",
)

# CSV source
events = timefence.Source(
    path="data/events.csv",
    keys=["user_id"],
    timestamp="event_time",
    format="csv",
    delimiter=",",
)

# DataFrame source
df_source = timefence.Source(
    df=my_dataframe,
    keys=["user_id"],
    timestamp="created_at",
)

# Multi-key source
orders = timefence.Source(
    path="data/orders.parquet",
    keys=["user_id", "product_id"],
    timestamp="order_time",
)
```

## Convenience aliases

```python
transactions = timefence.ParquetSource("data/tx.parquet", keys="user_id", timestamp="ts")
events = timefence.CSVSource("data/events.csv", keys="user_id", timestamp="ts")
```

These are thin wrappers that set `format` automatically.

## SQLSource

Define a source via a SQL query against DuckDB. Use this when your data requires pre-processing (joins, filters, aggregations) before it can serve as a feature source.

::: timefence.SQLSource
    options:
      show_root_heading: true
      members: []

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | SQL query string. Can use `read_parquet()` / `read_csv()` directly. |
| `keys` | `str \| list[str]` | Column name(s) used as entity keys. |
| `timestamp` | `str` | Column name containing the temporal key. |
| `name` | `str` | Human-readable name (required — cannot be auto-derived from a query). |
| `connection` | `str \| None` | Path to a DuckDB database file. Default: `None` (in-memory). |

### Examples

```python
import timefence

# Query across multiple parquet files
combined = timefence.SQLSource(
    query="""
        SELECT user_id, event_time, amount
        FROM read_parquet('data/transactions_*.parquet')
        WHERE amount > 0
    """,
    keys=["user_id"],
    timestamp="event_time",
    name="positive_transactions",
)

# Pre-aggregate before using as a feature source
daily_spend = timefence.SQLSource(
    query="""
        SELECT user_id, DATE_TRUNC('day', created_at) AS day,
               SUM(amount) AS daily_total
        FROM read_parquet('data/transactions.parquet')
        GROUP BY user_id, DATE_TRUNC('day', created_at)
    """,
    keys=["user_id"],
    timestamp="day",
    name="daily_spend",
)

# Join two files before use
enriched = timefence.SQLSource(
    query="""
        SELECT t.user_id, t.created_at, t.amount, u.country
        FROM read_parquet('data/transactions.parquet') t
        JOIN read_parquet('data/users.parquet') u
          ON t.user_id = u.user_id
    """,
    keys=["user_id"],
    timestamp="created_at",
    name="enriched_transactions",
)

# Use with an existing DuckDB database
warehouse = timefence.SQLSource(
    query="SELECT user_id, updated_at, score FROM user_scores",
    keys=["user_id"],
    timestamp="updated_at",
    name="user_scores",
    connection="analytics.duckdb",
)
```

### When to use SQLSource

| Scenario | Use |
|----------|-----|
| Single Parquet or CSV file | `Source` (simpler) |
| Glob patterns (`*.parquet`) | `SQLSource` |
| Pre-filtering rows | `SQLSource` |
| Joining multiple tables | `SQLSource` |
| Existing DuckDB database | `SQLSource` with `connection` |
| DataFrame in memory | `Source` with `df` |
