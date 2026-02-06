"""Shared test fixtures for Timefence tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    conn = duckdb.connect()
    try:
        # Users: 100 users with known timestamps
        conn.execute(f"""
            COPY (
                SELECT
                    i AS user_id,
                    CASE (i % 3)
                        WHEN 0 THEN 'US'
                        WHEN 1 THEN 'UK'
                        WHEN 2 THEN 'DE'
                    END AS country,
                    DATE '2020-01-01' + INTERVAL (i * 10) DAY AS signup_date,
                    TIMESTAMP '2023-01-01' + INTERVAL (i * 3) DAY AS updated_at
                FROM generate_series(1, 100) t(i)
            ) TO '{data_dir}/users.parquet' (FORMAT PARQUET)
        """)

        # Transactions: multiple per user
        conn.execute(f"""
            COPY (
                SELECT
                    ((i - 1) % 100) + 1 AS user_id,
                    TIMESTAMP '2023-01-01' + INTERVAL (i * 7 % 365) DAY
                        + INTERVAL (i * 3 % 24) HOUR AS created_at,
                    ROUND((10 + (i * 17 % 200))::DOUBLE / 10.0, 2) AS amount
                FROM generate_series(1, 2000) t(i)
            ) TO '{data_dir}/transactions.parquet' (FORMAT PARQUET)
        """)

        # Labels: 50 labels for 50 users
        conn.execute(f"""
            COPY (
                SELECT
                    i AS user_id,
                    TIMESTAMP '2024-01-15' + INTERVAL (i * 5) DAY AS label_time,
                    CASE WHEN i % 4 = 0 THEN true ELSE false END AS churned
                FROM generate_series(1, 50) t(i)
            ) TO '{data_dir}/labels.parquet' (FORMAT PARQUET)
        """)
    finally:
        conn.close()

    return data_dir


@pytest.fixture
def sample_features(tmp_data: Path):
    """Create sample Feature objects for testing."""
    import timefence

    users = timefence.Source(
        path=str(tmp_data / "users.parquet"),
        keys=["user_id"],
        timestamp="updated_at",
        name="users",
    )

    transactions = timefence.Source(
        path=str(tmp_data / "transactions.parquet"),
        keys=["user_id"],
        timestamp="created_at",
        name="transactions",
    )

    user_country = timefence.Feature(
        source=users,
        columns=["country"],
        name="user_country",
    )

    spending = timefence.Feature(
        source=transactions,
        sql="""
            SELECT
                user_id,
                created_at AS feature_time,
                SUM(amount) OVER (
                    PARTITION BY user_id
                    ORDER BY created_at
                    RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
                ) AS spend_30d
            FROM {source}
        """,
        name="rolling_spend",
        embargo="1d",
    )

    labels = timefence.Labels(
        path=str(tmp_data / "labels.parquet"),
        keys=["user_id"],
        label_time="label_time",
        target=["churned"],
    )

    return {
        "users_source": users,
        "txn_source": transactions,
        "user_country": user_country,
        "spending": spending,
        "labels": labels,
    }
