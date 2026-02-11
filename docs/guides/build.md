# Guide: Build from Scratch

Build a point-in-time correct training dataset from raw data.

## Full example

```python
import timefence

# 1. Define sources
transactions = timefence.Source(
    path="data/transactions.parquet",
    keys=["user_id"],
    timestamp="created_at",
)

users = timefence.Source(
    path="data/users.parquet",
    keys=["user_id"],
    timestamp="updated_at",
)

# 2. Define features
rolling_spend = timefence.Feature(
    source=transactions,
    sql="""
        SELECT user_id, created_at AS feature_time,
        SUM(amount) OVER (
            PARTITION BY user_id
            ORDER BY created_at
            RANGE INTERVAL 30 DAYS PRECEDING
        ) AS spend_30d
        FROM {source}
    """,
    name="rolling_spend_30d",
    embargo="1d",
)

user_country = timefence.Feature(
    source=users,
    columns=["country"],
)

# 3. Define labels
labels = timefence.Labels(
    path="data/labels.parquet",
    keys=["user_id"],
    label_time="label_time",
    target="churned",
)

# 4. Build
result = timefence.build(
    labels=labels,
    features=[rolling_spend, user_country],
    output="train_CLEAN.parquet",
)

print(result)
# rows: 5000, columns: 4, duration: 0.8s
```

## Feature modes

Timefence supports three ways to define features:

=== "Column Selection"

    ```python
    country = timefence.Feature(source=users, columns=["country"])
    ```

=== "SQL"

    ```python
    spend = timefence.Feature(
        source=transactions,
        sql="""
            SELECT user_id, created_at AS feature_time,
            SUM(amount) OVER (...) AS spend_30d
            FROM {source}
        """,
        name="rolling_spend_30d",
        embargo="1d",
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

## CLI equivalent

```bash
timefence build \
  --labels data/labels.parquet \
  --features features.py \
  -o train.parquet
```

See the [build() API reference](../api/build.md) for full parameter documentation.
