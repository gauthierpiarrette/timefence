"""Integration tests: full end-to-end flows."""

from __future__ import annotations

import duckdb
import pytest

import timefence
from timefence.engine import build


class TestBuildThenAuditRoundtrip:
    """A dataset built by Timefence always passes its own audit."""

    def test_roundtrip_columns_mode(self, sample_features, tmp_path):
        output = str(tmp_path / "out.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=output,
        )
        report = timefence.audit(
            output,
            features=[sample_features["user_country"]],
            keys=["user_id"],
            label_time="label_time",
        )
        assert not report.has_leakage

    def test_roundtrip_sql_mode(self, sample_features, tmp_path):
        output = str(tmp_path / "out.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],
            output=output,
        )
        report = timefence.audit(
            output,
            features=[sample_features["spending"]],
            keys=["user_id"],
            label_time="label_time",
        )
        assert not report.has_leakage

    def test_roundtrip_transform_mode(self, tmp_data, tmp_path):
        source = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
            name="users",
        )

        def compute_country(conn, source_table):
            return conn.sql(
                f"SELECT user_id, updated_at AS feature_time, country "
                f"FROM {source_table}"
            )

        feat = timefence.Feature(
            source=source, transform=compute_country, name="country_transform"
        )
        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        output = str(tmp_path / "out.parquet")
        build(labels=labels, features=[feat], output=output)

        report = timefence.audit(
            output,
            features=[feat],
            keys=["user_id"],
            label_time="label_time",
        )
        assert not report.has_leakage


class TestFeatureModesEquivalent:
    """columns, sql, and transform modes should produce identical output for equivalent logic."""

    def test_columns_vs_sql(self, tmp_data, tmp_path):
        users_source = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
            name="users",
        )

        # columns mode
        feat_col = timefence.Feature(
            source=users_source,
            columns=["country"],
            name="country_col",
        )

        # sql mode (equivalent)
        feat_sql = timefence.Feature(
            source=users_source,
            sql="SELECT user_id, updated_at AS feature_time, country FROM {source}",
            name="country_sql",
        )

        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        out_col = str(tmp_path / "col.parquet")
        out_sql = str(tmp_path / "sql.parquet")

        r1 = build(labels=labels, features=[feat_col], output=out_col)
        r2 = build(labels=labels, features=[feat_sql], output=out_sql)

        # Both should have the same row count
        assert r1.stats.row_count == r2.stats.row_count

        # Both should match on feature values
        conn = duckdb.connect()
        try:
            conn.execute(
                f"CREATE TEMP TABLE df1 AS SELECT * FROM read_parquet('{out_col}')"
            )
            conn.execute(
                f"CREATE TEMP TABLE df2 AS SELECT * FROM read_parquet('{out_sql}')"
            )
            df1 = conn.execute(
                "SELECT * FROM df1 ORDER BY user_id, label_time"
            ).fetchall()
            df2 = conn.execute(
                "SELECT * FROM df2 ORDER BY user_id, label_time"
            ).fetchall()

            cols1 = [c[0] for c in conn.execute("DESCRIBE df1").fetchall()]
            cols2 = [c[0] for c in conn.execute("DESCRIBE df2").fetchall()]

            # Find country columns
            country_idx1 = next(i for i, c in enumerate(cols1) if "country" in c)
            country_idx2 = next(i for i, c in enumerate(cols2) if "country" in c)

            for row1, row2 in zip(df1, df2):
                assert row1[country_idx1] == row2[country_idx2], (
                    f"Value mismatch: columns={row1[country_idx1]} vs sql={row2[country_idx2]}"
                )
        finally:
            conn.close()


class TestTransformMode:
    def test_transform_produces_output(self, tmp_data, tmp_path):
        users_source = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
            name="users",
        )

        def compute_country(conn, source_table):
            return conn.sql(
                f"SELECT user_id, updated_at AS feature_time, country FROM {source_table}"
            )

        feat = timefence.Feature(
            source=users_source,
            transform=compute_country,
            name="country_transform",
        )

        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        result = build(
            labels=labels,
            features=[feat],
            output=str(tmp_path / "out.parquet"),
        )
        assert result.stats.row_count == 50


class TestFeatureSetBuild:
    def test_feature_set_in_build(self, sample_features, tmp_path):
        fs = timefence.FeatureSet(
            name="user_features",
            features=[sample_features["user_country"]],
        )
        result = build(
            labels=sample_features["labels"],
            features=[fs, sample_features["spending"]],
            output=str(tmp_path / "out.parquet"),
        )
        assert "user_country" in result.stats.feature_stats
        assert "rolling_spend" in result.stats.feature_stats


class TestFlattenColumns:
    def test_flatten_when_no_conflicts(self, sample_features, tmp_path):
        output = str(tmp_path / "out.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=output,
            flatten_columns=True,
        )
        conn = duckdb.connect()
        try:
            conn.execute(
                f"CREATE TEMP TABLE __out AS SELECT * FROM read_parquet('{output}')"
            )
            cols = [c[0] for c in conn.execute("DESCRIBE __out").fetchall()]
            # Should have "country" not "user_country__country"
            assert "country" in cols
        finally:
            conn.close()


class TestDataFrameInput:
    """Source(df=...) and Labels(df=...) work through the full pipeline."""

    def test_build_with_dataframe_source_and_labels(self, tmp_path):
        pd = pytest.importorskip("pandas")

        users_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "country": ["US", "UK", "DE"],
                "updated_at": pd.to_datetime(
                    ["2023-01-01", "2023-06-01", "2023-09-01"]
                ),
            }
        )
        labels_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "label_time": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-03-01"]
                ),
                "churned": [True, False, True],
            }
        )

        source = timefence.Source(df=users_df, keys=["user_id"], timestamp="updated_at")
        feature = timefence.Feature(source=source, columns=["country"])
        labels = timefence.Labels(
            df=labels_df,
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        output = str(tmp_path / "out.parquet")
        result = build(labels=labels, features=[feature], output=output)

        assert result.stats.row_count == 3
        assert result.stats.feature_stats["country"]["matched"] == 3
        assert result.manifest["audit"]["passed"]

    def test_audit_with_dataframe_data(self, tmp_path):
        """Audit accepts a DataFrame as the data argument."""
        pd = pytest.importorskip("pandas")

        users_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "country": ["US", "UK", "DE"],
                "updated_at": pd.to_datetime(
                    ["2023-01-01", "2023-06-01", "2023-09-01"]
                ),
            }
        )
        labels_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "label_time": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-03-01"]
                ),
                "churned": [True, False, True],
            }
        )

        source = timefence.Source(df=users_df, keys=["user_id"], timestamp="updated_at")
        feature = timefence.Feature(source=source, columns=["country"])
        labels = timefence.Labels(
            df=labels_df,
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        output = str(tmp_path / "out.parquet")
        build(labels=labels, features=[feature], output=output)

        # Audit the output as a file path
        report = timefence.audit(
            output,
            features=[feature],
            keys=["user_id"],
            label_time="label_time",
        )
        assert not report.has_leakage


class TestMultiKeyJoins:
    """Composite keys like (user_id, product_id) work correctly."""

    def test_build_and_audit_with_composite_keys(self, tmp_path):
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT
                        (i % 5) + 1 AS user_id,
                        (i % 3) + 1 AS product_id,
                        ROUND((10 + (i * 7 % 100))::DOUBLE, 2) AS score,
                        TIMESTAMP '2023-01-01' + INTERVAL (i * 3) DAY AS computed_at
                    FROM generate_series(1, 60) t(i)
                ) TO '{tmp_path}/features.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT
                        (i % 5) + 1 AS user_id,
                        (i % 3) + 1 AS product_id,
                        TIMESTAMP '2024-06-01' + INTERVAL (i * 7) DAY AS label_time,
                        CASE WHEN i % 3 = 0 THEN 1 ELSE 0 END AS purchased
                    FROM generate_series(1, 15) t(i)
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        source = timefence.Source(
            path=str(tmp_path / "features.parquet"),
            keys=["user_id", "product_id"],
            timestamp="computed_at",
        )
        feature = timefence.Feature(source=source, columns=["score"])
        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys=["user_id", "product_id"],
            label_time="label_time",
            target=["purchased"],
        )

        output = str(tmp_path / "out.parquet")
        result = build(labels=labels, features=[feature], output=output)

        assert result.stats.row_count == 15
        assert result.manifest["audit"]["passed"]

        # Audit roundtrip: build output passes its own audit
        report = timefence.audit(
            output,
            features=[feature],
            keys=["user_id", "product_id"],
            label_time="label_time",
        )
        assert not report.has_leakage


class TestCSVSource:
    """CSV files work end-to-end through build and audit."""

    def test_build_from_csv_source(self, tmp_path):
        csv_path = tmp_path / "users.csv"
        csv_path.write_text(
            "user_id,country,updated_at\n"
            "1,US,2023-06-01 00:00:00\n"
            "2,UK,2023-08-01 00:00:00\n"
            "3,DE,2023-10-01 00:00:00\n"
        )

        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS user_id,
                        TIMESTAMP '2024-01-15' + INTERVAL (i * 30) DAY AS label_time,
                        CASE WHEN i = 1 THEN true ELSE false END AS churned
                    FROM generate_series(1, 3) t(i)
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        source = timefence.CSVSource(
            path=str(csv_path), keys=["user_id"], timestamp="updated_at"
        )
        feature = timefence.Feature(source=source, columns=["country"])
        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )

        output = str(tmp_path / "out.parquet")
        result = build(labels=labels, features=[feature], output=output)

        assert result.stats.row_count == 3
        assert result.stats.feature_stats["country"]["matched"] == 3
        assert result.manifest["audit"]["passed"]


class TestQuickstartFlow:
    """The exact flow from the 3-minute wow section works end-to-end."""

    def test_full_quickstart_flow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # 1. Generate quickstart
        from timefence.quickstart import generate_quickstart

        project_dir = generate_quickstart("test-churn", minimal=True)

        # Verify files exist
        assert (project_dir / "features.py").exists()
        assert (project_dir / "data" / "users.parquet").exists()
        assert (project_dir / "data" / "transactions.parquet").exists()
        assert (project_dir / "data" / "labels.parquet").exists()
        assert (project_dir / "data" / "train_LEAKY.parquet").exists()
        assert (project_dir / "timefence.yaml").exists()
        assert (project_dir / "README.md").exists()
