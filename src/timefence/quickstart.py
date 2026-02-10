"""Quickstart project generation with synthetic data and planted leakage."""

from __future__ import annotations

from pathlib import Path

import duckdb


def generate_quickstart(project_name: str, *, minimal: bool = False) -> Path:
    """Generate a self-contained example project.

    Creates synthetic user/transaction data with planted temporal leakage
    for demonstration purposes.

    Returns the project directory path.
    """
    project_dir = Path(project_name)
    project_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    conn = duckdb.connect()
    try:
        _generate_users(conn, data_dir, n_users=10000 if not minimal else 1000)
        _generate_transactions(conn, data_dir, n_users=10000 if not minimal else 1000)
        _generate_labels(conn, data_dir, n_labels=5000 if not minimal else 500)
        _generate_leaky_dataset(conn, data_dir)
    finally:
        conn.close()

    _write_features_py(project_dir, minimal)
    _write_config(project_dir)
    _write_readme(project_dir)

    return project_dir


def _generate_users(
    conn: duckdb.DuckDBPyConnection, data_dir: Path, n_users: int
) -> None:
    """Generate synthetic user data.

    Timestamps are set so most users have updated_at within the 365-day
    lookback window of the label times (which start ~2023-06-01).
    """
    # Generate 3 snapshots per user to ensure good coverage across the
    # label time range. Each user appears at ~6-month intervals from
    # 2023-01-01 to 2024-07-01, guaranteeing that labels (starting
    # 2023-06-01) always find a recent user row within 365d lookback.
    conn.execute(f"""
        COPY (
            SELECT
                user_id, country, signup_date, updated_at, tier
            FROM (
                SELECT
                    i AS user_id,
                    CASE (i % 5)
                        WHEN 0 THEN 'US'
                        WHEN 1 THEN 'UK'
                        WHEN 2 THEN 'DE'
                        WHEN 3 THEN 'FR'
                        WHEN 4 THEN 'JP'
                    END AS country,
                    DATE '2020-01-01' + INTERVAL (i % 1000) DAY AS signup_date,
                    CASE (i % 3)
                        WHEN 0 THEN 'free'
                        WHEN 1 THEN 'pro'
                        WHEN 2 THEN 'enterprise'
                    END AS tier
                FROM generate_series(1, {n_users}) t(i)
            ) base
            CROSS JOIN (
                SELECT unnest([
                    TIMESTAMP '2023-01-15' ,
                    TIMESTAMP '2023-07-15',
                    TIMESTAMP '2024-01-15'
                ]) AS updated_at
            ) snapshots
            ORDER BY user_id, updated_at
        ) TO '{str(data_dir / "users.parquet").replace("'", "''")}' (FORMAT PARQUET)
    """)


def _generate_transactions(
    conn: duckdb.DuckDBPyConnection, data_dir: Path, n_users: int
) -> None:
    """Generate synthetic transaction data."""
    n_txns = n_users * 20
    conn.execute(f"""
        COPY (
            SELECT
                (i % {n_users}) + 1 AS user_id,
                TIMESTAMP '2022-01-01' + INTERVAL (i * 7 % 1095) DAY
                    + INTERVAL (i * 13 % 24) HOUR AS created_at,
                ROUND((50 + (i * 17 % 500))::DOUBLE / 10, 2) AS amount
            FROM generate_series(1, {n_txns}) t(i)
        ) TO '{str(data_dir / "transactions.parquet").replace("'", "''")}' (FORMAT PARQUET)
    """)


def _generate_labels(
    conn: duckdb.DuckDBPyConnection, data_dir: Path, n_labels: int
) -> None:
    """Generate synthetic churn labels."""
    conn.execute(f"""
        COPY (
            SELECT
                (i % {n_labels}) + 1 AS user_id,
                TIMESTAMP '2023-06-01' + INTERVAL (i * 11 % 548) DAY AS label_time,
                CASE WHEN i % 5 = 0 THEN true ELSE false END AS churned
            FROM generate_series(1, {n_labels}) t(i)
        ) TO '{str(data_dir / "labels.parquet").replace("'", "''")}' (FORMAT PARQUET)
    """)


def _generate_leaky_dataset(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Generate a pre-built dataset WITH planted temporal leakage.

    This dataset uses <= instead of < for joins (common source of leakage)
    and uses some features from the future for ~12% of rows.
    """
    conn.execute(f"""
        CREATE TEMP TABLE labels AS
        SELECT * FROM read_parquet('{str(data_dir / "labels.parquet").replace("'", "''")}')
    """)
    conn.execute(f"""
        CREATE TEMP TABLE users AS
        SELECT * FROM read_parquet('{str(data_dir / "users.parquet").replace("'", "''")}')
    """)
    conn.execute(f"""
        CREATE TEMP TABLE txns AS
        SELECT * FROM read_parquet('{str(data_dir / "transactions.parquet").replace("'", "''")}')
    """)

    # Build with leaky joins: use <= instead of < AND allow some future data
    conn.execute(f"""
        COPY (
            WITH latest_user AS (
                SELECT u.*, ROW_NUMBER() OVER (
                    PARTITION BY u.user_id ORDER BY u.updated_at DESC
                ) AS rn
                FROM users u
            ),
            user_feat AS (
                SELECT
                    l.user_id,
                    l.label_time,
                    l.churned,
                    u.country AS user_country__country,
                    DATE_DIFF('day', u.signup_date, l.label_time::DATE) AS account_age_days__acc_age_days,
                    u.updated_at AS user_country__feature_time,
                    u.signup_date AS account_age_days__feature_time
                FROM labels l
                LEFT JOIN latest_user u ON l.user_id = u.user_id AND u.rn = 1
            ),
            spend_feat AS (
                SELECT
                    l.user_id,
                    l.label_time,
                    SUM(t.amount) AS rolling_spend_30d__spend_30d,
                    MAX(t.created_at) AS rolling_spend_30d__feature_time
                FROM labels l
                LEFT JOIN txns t
                    ON l.user_id = t.user_id
                    AND t.created_at <= l.label_time + INTERVAL 14 DAY  -- LEAKY: allows future data
                    AND t.created_at >= l.label_time - INTERVAL 30 DAY
                GROUP BY l.user_id, l.label_time
            ),
            login_feat AS (
                SELECT
                    l.user_id,
                    l.label_time,
                    DATE_DIFF('day',
                        COALESCE(
                            (SELECT MAX(t2.created_at) FROM txns t2
                             WHERE t2.user_id = l.user_id
                             AND t2.created_at <= l.label_time + INTERVAL 2 DAY),  -- LEAKY
                            l.label_time - INTERVAL 30 DAY
                        ),
                        l.label_time
                    ) AS days_since_login__days_since_login,
                    COALESCE(
                        (SELECT MAX(t2.created_at) FROM txns t2
                         WHERE t2.user_id = l.user_id
                         AND t2.created_at <= l.label_time + INTERVAL 2 DAY),
                        l.label_time - INTERVAL 30 DAY
                    ) AS days_since_login__feature_time
                FROM labels l
            )
            SELECT
                uf.user_id,
                uf.label_time,
                uf.churned,
                uf.user_country__country,
                uf.account_age_days__acc_age_days,
                sf.rolling_spend_30d__spend_30d,
                lf.days_since_login__days_since_login
            FROM user_feat uf
            LEFT JOIN spend_feat sf ON uf.user_id = sf.user_id AND uf.label_time = sf.label_time
            LEFT JOIN login_feat lf ON uf.user_id = lf.user_id AND uf.label_time = lf.label_time
            ORDER BY uf.user_id, uf.label_time
        ) TO '{str(data_dir / "train_LEAKY.parquet").replace("'", "''")}' (FORMAT PARQUET)
    """)


def _write_features_py(project_dir: Path, minimal: bool) -> None:
    """Write feature definitions file."""
    content = '''"""Feature definitions for the churn example."""

import timefence

# Sources
users = timefence.Source(
    path="data/users.parquet",
    keys=["user_id"],
    timestamp="updated_at",
    name="users",
)

transactions = timefence.Source(
    path="data/transactions.parquet",
    keys=["user_id"],
    timestamp="created_at",
    name="transactions",
)

# Features
user_country = timefence.Feature(
    source=users,
    columns=["country"],
    name="user_country",
)

account_age_days = timefence.Feature(
    source=users,
    columns={"signup_date": "acc_age_days"},
    name="account_age_days",
)

rolling_spend_30d = timefence.Feature(
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
    name="rolling_spend_30d",
    embargo="1d",
)

days_since_login = timefence.Feature(
    source=transactions,
    sql="""
        SELECT
            user_id,
            created_at AS feature_time,
            1 AS days_since_login
        FROM {source}
    """,
    name="days_since_login",
)
'''
    (project_dir / "features.py").write_text(content)


def _write_config(project_dir: Path) -> None:
    """Write timefence.yaml config."""
    content = """# timefence.yaml
name: churn-example
version: "1.0"

data_dir: data/

features:
  - features.py

labels:
  path: data/labels.parquet
  keys: [user_id]
  label_time: label_time
  target: [churned]

defaults:
  max_lookback: 365d
  join: strict
  on_missing: "null"
"""
    (project_dir / "timefence.yaml").write_text(content)


def _write_readme(project_dir: Path) -> None:
    """Write project README."""
    content = """# Churn Prediction Example

Get started in 3 commands:

1. **Find leakage in the bad dataset:**

   ```bash
   timefence audit data/train_LEAKY.parquet
   ```

2. **Build a clean training set:**

   ```bash
   timefence build -o data/train_CLEAN.parquet
   ```

3. **Verify the new dataset is clean:**

   ```bash
   timefence audit data/train_CLEAN.parquet
   ```
"""
    (project_dir / "README.md").write_text(content)
