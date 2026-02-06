"""Integration tests: full end-to-end flows."""

from __future__ import annotations

import duckdb

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


class TestQuickstartFlow:
    """The exact flow from the 3-minute wow section works end-to-end."""

    def test_full_quickstart_flow(self, tmp_path):
        import os

        os.chdir(tmp_path)

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
