# Feature

A named signal derived from a Source. Exactly one of `columns`, `sql`, or `transform` must be provided.

::: timefence.Feature
    options:
      show_root_heading: true
      members: []

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Source \| SQLSource` | The data source object. |
| `columns` | `str \| list \| dict \| None` | **Mode 1:** Select columns directly. Pass a dict to rename: `{"source_col": "feature_col"}`. |
| `sql` | `str \| Path \| None` | **Mode 2:** SQL query or path to `.sql` file. Use `{source}` placeholder. |
| `transform` | `Callable \| None` | **Mode 3:** Python function `(conn, source_table) -> DuckDBPyRelation`. Use `conn.sql(...)` to return a relation. |
| `name` | `str \| None` | Feature name. Auto-derived when possible. Required for inline SQL strings. |
| `embargo` | `str \| timedelta \| None` | Computation lag buffer. See [Embargo](../concepts/embargo.md). |
| `key_mapping` | `dict[str, str] \| None` | Map label key names to source key names: `{"user_id": "customer_id"}`. |
| `on_duplicate` | `str` | `"error"` (default) or `"keep_any"` when duplicate `(key, feature_time)` pairs exist. |

## key_mapping

When label keys don't match source keys, use `key_mapping` to bridge them:

```python
# Labels use "user_id", but the source uses "customer_id"
spend = timefence.Feature(
    source=transactions,  # has column "customer_id"
    columns=["amount"],
    key_mapping={"user_id": "customer_id"},
)

# Multi-key mapping
orders_feature = timefence.Feature(
    source=orders,  # has "cust_id" and "prod_id"
    columns=["quantity"],
    key_mapping={"user_id": "cust_id", "product_id": "prod_id"},
)
```

The mapping format is `{label_key: source_key}`. During the join, Timefence rewrites `ON labels.user_id = source.customer_id` automatically.

## Feature modes

=== "Column Selection"

    ```python
    country = timefence.Feature(source=users, columns=["country"])
    ```

=== "SQL (inline)"

    ```python
    spend = timefence.Feature(
        source=transactions,
        sql="""
            SELECT user_id, created_at AS feature_time,
            SUM(amount) OVER (
                PARTITION BY user_id ORDER BY created_at
                RANGE INTERVAL 30 DAYS PRECEDING
            ) AS spend_30d
            FROM {source}
        """,
        name="rolling_spend_30d",
        embargo="1d",
    )
    ```

=== "SQL (file)"

    ```python
    spend = timefence.Feature(
        source=transactions,
        sql=Path("features/rolling_spend.sql"),
    )
    ```

=== "Python Transform"

    ```python
    def compute_score(conn, source_table):
        return conn.sql(f"""
            SELECT user_id, created_at AS feature_time,
                   raw_score * 2.5 AS adjusted_score
            FROM {source_table}
        """)

    score = timefence.Feature(source=transactions, transform=compute_score)
    ```
