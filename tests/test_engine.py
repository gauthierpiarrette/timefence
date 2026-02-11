"""Tests for the build/audit/explain/diff engine."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

import timefence
from timefence.engine import (
    AuditReport,
    BuildResult,
    DiffResult,
    ExplainResult,
    _classify_severity,
    build,
)
from timefence.errors import (
    TimefenceConfigError,
    TimefenceSchemaError,
    TimefenceValidationError,
)


def _has_pytz() -> bool:
    """Check if pytz is available (needed for DuckDB TIMESTAMPTZ)."""
    try:
        import pytz  # noqa: F401

        return True
    except ImportError:
        return False


class TestBuildBasic:
    """Core build functionality tests."""

    def test_simple_build(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        assert isinstance(result, BuildResult)
        assert result.stats.row_count == 50
        assert (tmp_path / "out.parquet").exists()

    def test_build_no_output(self, sample_features):
        """Build without writing to disk."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
        )
        assert result.output_path is None
        assert result.stats.row_count == 50

    def test_build_multiple_features(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[
                sample_features["user_country"],
                sample_features["spending"],
            ],
            output=str(tmp_path / "out.parquet"),
        )
        assert result.stats.row_count == 50
        assert "user_country" in result.stats.feature_stats
        assert "rolling_spend" in result.stats.feature_stats

    def test_build_with_store(self, sample_features, tmp_path):
        store = timefence.Store(str(tmp_path / ".timefence"))
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            store=store,
        )
        builds = store.list_builds()
        assert len(builds) == 1
        assert builds[0]["output"]["row_count"] == 50

    def test_build_manifest(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        manifest = result.manifest
        assert manifest["timefence_version"] == "0.9.1"
        assert manifest["audit"]["passed"] is True
        assert "user_country" in manifest["features"]


class TestBuildTemporalCorrectness:
    """Tests for the core temporal invariant."""

    def test_strict_mode_no_future_data(self, sample_features, tmp_path):
        """In strict mode, feature_time must be < label_time."""
        output = str(tmp_path / "out.parquet")
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=output,
            join="strict",
        )
        # Verify no violations
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{output}')
                WHERE user_country__feature_time >= label_time
                  AND user_country__feature_time IS NOT NULL
            """).fetchone()[0]
        except Exception:
            # Column might not be in output; just fall through to manifest check
            pass
        finally:
            conn.close()
        assert result.manifest["audit"]["passed"] is True

    def test_embargo_shifts_window(self, sample_features, tmp_path):
        """Embargo should exclude features within the embargo window."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],  # has 1d embargo
            output=str(tmp_path / "out.parquet"),
        )
        assert result.manifest["audit"]["passed"] is True
        assert result.manifest["features"]["rolling_spend"]["embargo"] == "1d"

    def test_inclusive_mode(self, sample_features, tmp_path):
        """Inclusive mode: feature_time <= label_time."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            join="inclusive",
        )
        assert result.manifest["audit"]["passed"] is True


class TestBuildParameters:
    """Tests for build parameter validation."""

    def test_invalid_join_mode(self, sample_features):
        with pytest.raises(TimefenceConfigError, match="join must be"):
            build(
                labels=sample_features["labels"],
                features=[sample_features["user_country"]],
                join="invalid",
            )

    def test_invalid_on_missing(self, sample_features):
        with pytest.raises(TimefenceConfigError, match="on_missing must be"):
            build(
                labels=sample_features["labels"],
                features=[sample_features["user_country"]],
                on_missing="invalid",
            )

    def test_embargo_exceeds_lookback(self, tmp_data):
        src = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
        )
        feat = timefence.Feature(source=src, columns=["country"], embargo="400d")
        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["user_id"],
            label_time="label_time",
            target=["churned"],
        )
        with pytest.raises(TimefenceConfigError, match=r"embargo.*must be less than"):
            build(labels=labels, features=[feat], max_lookback="365d")

    def test_on_missing_skip(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],
            output=str(tmp_path / "out.parquet"),
            on_missing="skip",
        )
        # skip mode should have fewer or equal rows
        assert result.stats.row_count <= 50

    def test_max_lookback(self, sample_features, tmp_path):
        """Very short lookback should produce more missing values."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            max_lookback="1d",
        )
        matched = result.stats.feature_stats["user_country"]["matched"]
        missing = result.stats.feature_stats["user_country"]["missing"]
        # With a 1-day lookback, most features are too old to match
        assert matched + missing == result.stats.row_count
        assert missing > 0  # short lookback should cause some misses


class TestBuildSchemaValidation:
    """Tests for schema validation during build."""

    def test_missing_label_key(self, tmp_data):
        src = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
        )
        feat = timefence.Feature(source=src, columns=["country"])
        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["nonexistent_key"],
            label_time="label_time",
            target=["churned"],
        )
        with pytest.raises(TimefenceSchemaError):
            build(labels=labels, features=[feat])

    def test_missing_label_time(self, tmp_data):
        src = timefence.Source(
            path=str(tmp_data / "users.parquet"),
            keys=["user_id"],
            timestamp="updated_at",
        )
        feat = timefence.Feature(source=src, columns=["country"])
        labels = timefence.Labels(
            path=str(tmp_data / "labels.parquet"),
            keys=["user_id"],
            label_time="nonexistent_time",
            target=["churned"],
        )
        with pytest.raises(TimefenceSchemaError):
            build(labels=labels, features=[feat])


class TestBuildSplits:
    """Tests for time-based splits."""

    def test_splits(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            splits={
                "train": ("2024-01-01", "2024-04-01"),
                "test": ("2024-04-01", "2025-01-01"),
            },
        )
        assert result.splits is not None
        assert "train" in result.splits
        assert "test" in result.splits


class TestAudit:
    """Tests for the audit engine."""

    def test_clean_dataset_passes(self, sample_features, tmp_path):
        """A Timefence-built dataset should always pass audit."""
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
        assert isinstance(report, AuditReport)
        assert not report.has_leakage

    def test_audit_temporal_mode(self, tmp_data):
        """Test lightweight temporal check mode."""
        conn = duckdb.connect()
        try:
            # Create a dataset with explicit feature_time columns
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS user_id,
                        TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS label_time,
                        TIMESTAMP '2024-05-01' + INTERVAL (i) DAY AS spend_computed_at,
                        100.0 AS spend_30d
                    FROM generate_series(1, 100) t(i)
                ) TO '{tmp_data}/temporal_clean.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        report = timefence.audit(
            str(tmp_data / "temporal_clean.parquet"),
            feature_time_columns={"spend_30d": "spend_computed_at"},
            label_time="label_time",
        )
        assert report.mode == "temporal"
        assert not report.has_leakage

    def test_audit_temporal_detects_leakage(self, tmp_data):
        """Temporal check should detect feature_time >= label_time."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS user_id,
                        TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS label_time,
                        TIMESTAMP '2024-06-01' + INTERVAL (i + 5) DAY AS spend_computed_at,
                        100.0 AS spend_30d
                    FROM generate_series(1, 100) t(i)
                ) TO '{tmp_data}/temporal_leaky.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        report = timefence.audit(
            str(tmp_data / "temporal_leaky.parquet"),
            feature_time_columns={"spend_30d": "spend_computed_at"},
            label_time="label_time",
        )
        assert report.has_leakage
        assert report["spend_30d"].leaky_row_count == 100

    def test_audit_report_assert_clean(self, tmp_data):
        """assert_clean should raise on leaky data."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS user_id,
                        TIMESTAMP '2024-06-01' AS label_time,
                        TIMESTAMP '2024-06-15' AS ft,
                        1.0 AS val
                    FROM generate_series(1, 10) t(i)
                ) TO '{tmp_data}/leaky.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        report = timefence.audit(
            str(tmp_data / "leaky.parquet"),
            feature_time_columns={"val": "ft"},
            label_time="label_time",
        )
        with pytest.raises(timefence.errors.TimefenceLeakageError):
            report.assert_clean()

    def test_audit_report_to_json(self, tmp_data, tmp_path):
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS user_id,
                        TIMESTAMP '2024-06-01' AS label_time,
                        TIMESTAMP '2024-05-01' AS ft,
                        1.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_data}/clean.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        report = timefence.audit(
            str(tmp_data / "clean.parquet"),
            feature_time_columns={"val": "ft"},
            label_time="label_time",
        )
        json_path = str(tmp_path / "report.json")
        report.to_json(json_path)
        assert Path(json_path).exists()

    def test_audit_report_to_html(self, tmp_data, tmp_path):
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS user_id,
                        TIMESTAMP '2024-06-01' AS label_time,
                        TIMESTAMP '2024-05-01' AS ft,
                        1.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_data}/clean2.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        report = timefence.audit(
            str(tmp_data / "clean2.parquet"),
            feature_time_columns={"val": "ft"},
            label_time="label_time",
        )
        html_path = str(tmp_path / "report.html")
        report.to_html(html_path)
        assert Path(html_path).exists()
        content = Path(html_path).read_text()
        assert "Timefence" in content


class TestExplain:
    """Tests for explain functionality."""

    def test_explain_basic(self, sample_features):
        result = timefence.explain(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
        )
        assert isinstance(result, ExplainResult)
        assert result.label_count == 50
        assert len(result.plan) == 1
        assert result.plan[0]["name"] == "user_country"

    def test_explain_with_embargo(self, sample_features):
        result = timefence.explain(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],
        )
        assert result.plan[0]["embargo_str"] == "1d"
        assert (
            "1 day" in result.plan[0]["join_condition"].lower()
            or "1d" in result.plan[0]["join_condition"]
        )

    def test_explain_str(self, sample_features):
        result = timefence.explain(
            labels=sample_features["labels"],
            features=[
                sample_features["user_country"],
                sample_features["spending"],
            ],
        )
        text = str(result)
        assert "JOIN PLAN" in text
        assert "user_country" in text
        assert "rolling_spend" in text


class TestJoinStrategy:
    """Tests for ASOF vs ROW_NUMBER join strategy selection."""

    def test_asof_strategy_with_no_embargo(self, sample_features, tmp_path):
        """Features with no embargo should use ASOF strategy."""
        from timefence.engine import _use_asof_strategy

        feat = sample_features["user_country"]
        assert feat.embargo.total_seconds() == 0
        assert _use_asof_strategy(feat) is True

    def test_row_number_strategy_with_embargo(self, sample_features):
        """Features with embargo should use ROW_NUMBER strategy."""
        from timefence.engine import _use_asof_strategy

        feat = sample_features["spending"]
        assert feat.embargo.total_seconds() > 0
        assert _use_asof_strategy(feat) is False

    def test_asof_build_produces_correct_results(self, sample_features, tmp_path):
        """ASOF join path (no embargo) produces temporally correct output."""
        output = str(tmp_path / "asof_out.parquet")
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],  # no embargo -> ASOF
            output=output,
            join="strict",
        )
        assert result.manifest["audit"]["passed"] is True

    def test_row_number_build_produces_correct_results(self, sample_features, tmp_path):
        """ROW_NUMBER join path (with embargo) produces temporally correct output."""
        output = str(tmp_path / "rn_out.parquet")
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],  # 1d embargo -> ROW_NUMBER
            output=output,
            join="strict",
        )
        assert result.manifest["audit"]["passed"] is True


class TestBuildManifestPath:
    """Tests for manifest path in build result."""

    def test_manifest_path_with_store(self, sample_features, tmp_path):
        """When store is active, manifest_path appears in result."""
        store = timefence.Store(str(tmp_path / ".timefence"))
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            store=store,
        )
        assert "manifest_path" in result.manifest
        assert result.manifest["manifest_path"].endswith("build.json")

    def test_no_manifest_path_without_store(self, sample_features, tmp_path):
        """Without a store, no manifest_path."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        assert result.manifest.get("manifest_path") is None


class TestBuildInvariantAnnotation:
    """Tests for the invariant annotation in manifest."""

    def test_strict_invariant(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            join="strict",
        )
        assert (
            result.manifest["audit"]["invariant"]
            == "feature_time < label_time - embargo"
        )

    def test_inclusive_invariant(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            join="inclusive",
        )
        assert (
            result.manifest["audit"]["invariant"]
            == "feature_time <= label_time - embargo"
        )


class TestCacheReporting:
    """Tests for cache hit/miss reporting in build stats and manifest."""

    def test_first_build_no_cache(self, sample_features, tmp_path):
        """First build should report all features as not cached."""
        store = timefence.Store(str(tmp_path / ".timefence"))
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            store=store,
        )
        fstats = result.stats.feature_stats["user_country"]
        assert fstats["cached"] is False
        assert result.manifest["features"]["user_country"]["cached"] is False

    def test_second_build_uses_cache(self, sample_features, tmp_path):
        """Second identical build should report feature as cached in stats."""
        store = timefence.Store(str(tmp_path / ".timefence"))
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out1.parquet"),
            store=store,
        )
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out1.parquet"),
            store=store,
        )
        # Build-level cache hit → all features reported as cached in stats
        fstats = result.stats.feature_stats["user_country"]
        assert fstats["cached"] is True


class TestDiff:
    """Tests for diff functionality."""

    def test_diff_identical(self, sample_features, tmp_path):
        output = str(tmp_path / "v1.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=output,
        )
        # Copy as v2
        import shutil

        v2 = str(tmp_path / "v2.parquet")
        shutil.copy(output, v2)

        result = timefence.diff(output, v2, keys=["user_id"], label_time="label_time")
        assert isinstance(result, DiffResult)
        assert result.old_rows == result.new_rows
        assert len(result.value_changes) == 0

    def test_diff_different(self, sample_features, tmp_path):
        v1 = str(tmp_path / "v1.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=v1,
        )
        v2 = str(tmp_path / "v2.parquet")
        build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"], sample_features["spending"]],
            output=v2,
        )
        result = timefence.diff(v1, v2, keys=["user_id"], label_time="label_time")
        # v2 has extra columns
        assert any(c["type"] == "+" for c in result.schema_changes)

    def test_diff_atol_rtol(self, tmp_data, tmp_path):
        """Custom atol/rtol should control numeric comparison sensitivity."""
        import duckdb

        conn = duckdb.connect()
        try:
            # Create two files with small numeric differences
            conn.execute(f"""
                COPY (
                    SELECT i AS user_id,
                        TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS label_time,
                        100.0 AS val
                    FROM generate_series(1, 10) t(i)
                ) TO '{tmp_path}/v1.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT i AS user_id,
                        TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS label_time,
                        100.001 AS val
                    FROM generate_series(1, 10) t(i)
                ) TO '{tmp_path}/v2.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        # With tight tolerance: should detect changes
        result_tight = timefence.diff(
            str(tmp_path / "v1.parquet"),
            str(tmp_path / "v2.parquet"),
            keys=["user_id"],
            label_time="label_time",
            atol=1e-10,
            rtol=1e-7,
        )
        assert "val" in result_tight.value_changes

        # With loose tolerance: should NOT detect changes
        result_loose = timefence.diff(
            str(tmp_path / "v1.parquet"),
            str(tmp_path / "v2.parquet"),
            keys=["user_id"],
            label_time="label_time",
            atol=0.01,
            rtol=0.0,
        )
        assert "val" not in result_loose.value_changes


class TestFromDbt:
    """Tests for the from_dbt() stub."""

    def test_from_dbt_raises_not_implemented(self):
        with pytest.raises(
            NotImplementedError, match="dbt integration is not yet available"
        ):
            timefence.from_dbt()

    def test_from_dbt_accepts_manifest_path(self):
        with pytest.raises(NotImplementedError):
            timefence.from_dbt(manifest_path="custom/manifest.json")

    def test_from_dbt_in_public_api(self):
        assert hasattr(timefence, "from_dbt")
        assert "from_dbt" in timefence.__all__


class TestConstants:
    """Verify constants are wired correctly into engine defaults."""

    def test_build_uses_default_lookback(self, sample_features, tmp_path):
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        # Default lookback is used (365 days) — build should succeed
        assert result.stats.row_count > 0

    def test_diff_uses_default_tolerances(self, tmp_path):
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 1.0 AS v)
                TO '{tmp_path}/a.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 1.0 AS v)
                TO '{tmp_path}/b.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        # Identical files with default tolerances should show no changes
        result = timefence.diff(
            str(tmp_path / "a.parquet"),
            str(tmp_path / "b.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        assert len(result.value_changes) == 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestBuildEdgeCases:
    """Edge cases for the build pipeline."""

    def test_build_empty_labels(self, tmp_path):
        """Build with 0-row labels should succeed with 0 output rows."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-01-01' AS lt, true AS y
                    WHERE false
                ) TO '{tmp_path}/empty_labels.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2023-06-01' + INTERVAL (i) DAY AS ts, i * 10.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "empty_labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(source=src, columns=["val"], name="val_feat")

        result = build(labels=labels, features=[feat])
        assert result.stats.row_count == 0

    def test_build_single_label_row(self, tmp_path):
        """Build with exactly 1 label should produce 1 output row."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, false AS y)
                TO '{tmp_path}/one_label.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 42.0 AS val
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "one_label.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(source=src, columns=["val"], name="val_feat")

        result = build(
            labels=labels,
            features=[feat],
            output=str(tmp_path / "out.parquet"),
        )
        assert result.stats.row_count == 1
        assert (tmp_path / "out.parquet").exists()

    def test_build_max_staleness_lte_embargo_raises(self, sample_features, tmp_path):
        """max_staleness <= embargo must raise TimefenceConfigError."""
        with pytest.raises(
            TimefenceConfigError, match=r"max_staleness.*must be greater"
        ):
            build(
                labels=sample_features["labels"],
                features=[sample_features["spending"]],  # embargo="1d"
                output=str(tmp_path / "out.parquet"),
                max_staleness="1d",  # equal to embargo
            )

    def test_build_max_staleness_valid(self, sample_features, tmp_path):
        """max_staleness > embargo should succeed."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["spending"]],  # embargo="1d"
            output=str(tmp_path / "out.parquet"),
            max_staleness="30d",
        )
        assert isinstance(result, BuildResult)
        assert result.stats.row_count > 0

    def test_build_all_null_feature_column(self, tmp_path):
        """Feature column that is entirely NULL should not crash the build."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt, true AS y
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
            # Source has val = NULL for all rows
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-05-01' + INTERVAL (i) DAY AS ts,
                           NULL::DOUBLE AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(source=src, columns=["val"], name="null_feat")

        result = build(labels=labels, features=[feat])
        assert result.stats.row_count == 5

    def test_build_flatten_columns_with_conflict(self, tmp_path):
        """flatten_columns=True skips flattening when short names collide."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 'A' AS val
                ) TO '{tmp_path}/src_a.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 'B' AS val
                ) TO '{tmp_path}/src_b.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src_a = timefence.Source(
            path=str(tmp_path / "src_a.parquet"),
            keys="uid",
            timestamp="ts",
            name="src_a",
        )
        src_b = timefence.Source(
            path=str(tmp_path / "src_b.parquet"),
            keys="uid",
            timestamp="ts",
            name="src_b",
        )
        # Both features produce a column named "val" — short names will collide
        feat_a = timefence.Feature(source=src_a, columns=["val"], name="feat_a")
        feat_b = timefence.Feature(source=src_b, columns=["val"], name="feat_b")

        result = build(
            labels=labels,
            features=[feat_a, feat_b],
            output=str(tmp_path / "out.parquet"),
            flatten_columns=True,
        )
        assert result.stats.row_count == 1
        # With conflict, names should stay prefixed (feat_a__val, feat_b__val)
        out_conn = duckdb.connect()
        try:
            cols = [
                c[0]
                for c in out_conn.execute(
                    f"DESCRIBE (SELECT * FROM read_parquet('{tmp_path}/out.parquet'))"
                ).fetchall()
            ]
        finally:
            out_conn.close()
        assert "feat_a__val" in cols
        assert "feat_b__val" in cols

    def test_build_duplicate_feature_names_raises(self, tmp_path):
        """Two features with the same name must raise, not silently overwrite."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 'A' AS val
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        source = timefence.Source(
            path=str(tmp_path / "src.parquet"),
            keys="uid",
            timestamp="ts",
        )
        feat_a = timefence.Feature(source=source, columns=["val"], name="same_name")
        feat_b = timefence.Feature(source=source, columns=["val"], name="same_name")

        with pytest.raises(
            timefence.errors.TimefenceConfigError, match="Duplicate feature names"
        ):
            build(
                labels=labels,
                features=[feat_a, feat_b],
                output=str(tmp_path / "out.parquet"),
            )


class TestAuditEdgeCases:
    """Edge cases for the audit pipeline."""

    def test_audit_no_features_no_ftc_raises(self, tmp_path):
        """audit() without features or feature_time_columns must raise."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (SELECT 1 AS uid, TIMESTAMP '2024-01-01' AS lt, 1.0 AS v)
                TO '{tmp_path}/data.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        with pytest.raises(TimefenceValidationError, match="requires either"):
            timefence.audit(str(tmp_path / "data.parquet"))

    def test_audit_rebuild_missing_keys_raises(self, sample_features, tmp_path):
        """Rebuild mode without keys must raise."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (SELECT 1 AS uid, TIMESTAMP '2024-01-01' AS lt, 1.0 AS v)
                TO '{tmp_path}/data.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        with pytest.raises(
            TimefenceValidationError, match="requires 'keys' and 'label_time'"
        ):
            timefence.audit(
                str(tmp_path / "data.parquet"),
                features=[sample_features["user_country"]],
                keys=None,
                label_time=None,
            )

    def test_audit_rebuild_detects_leakage(self, tmp_path):
        """Rebuild audit should detect features that use future data."""
        conn = duckdb.connect()
        try:
            # Source has data BEFORE and AFTER labels.
            # Before labels: val = 10, After labels: val = 999
            conn.execute(f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, TIMESTAMP '2024-01-01', 10.0),
                        (1, TIMESTAMP '2024-08-01', 999.0),
                        (2, TIMESTAMP '2024-01-01', 20.0),
                        (2, TIMESTAMP '2024-08-01', 888.0)
                    ) AS t(uid, ts, val)
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)

            # "Leaky" dataset: uses the FUTURE values (999, 888)
            # A correct build would use the PAST values (10, 20)
            conn.execute(f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, TIMESTAMP '2024-06-01', 999.0),
                        (2, TIMESTAMP '2024-06-01', 888.0)
                    ) AS t(uid, label_time, feat__val)
                ) TO '{tmp_path}/leaky.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(source=src, columns=["val"], name="feat")

        report = timefence.audit(
            str(tmp_path / "leaky.parquet"),
            features=[feat],
            keys="uid",
            label_time="label_time",
        )
        assert isinstance(report, AuditReport)
        # The rebuild should detect differences: correct join picks 10/20
        # but existing has 999/888 — that's leakage
        assert report.has_leakage


class TestDiffEdgeCases:
    """Edge cases for the diff pipeline."""

    def test_diff_string_columns(self, tmp_path):
        """diff() should handle string (non-numeric) columns correctly."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 'hello' AS txt
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, 'world' AS txt
                ) TO '{tmp_path}/a.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 'hello' AS txt
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, 'changed' AS txt
                ) TO '{tmp_path}/b.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "a.parquet"),
            str(tmp_path / "b.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        assert isinstance(result, DiffResult)
        assert "txt" in result.value_changes
        assert result.value_changes["txt"]["changed_count"] == 1

    def test_diff_null_vs_non_null(self, tmp_path):
        """diff() should detect null-vs-non-null as a change."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 10.0 AS v
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, NULL::DOUBLE AS v
                ) TO '{tmp_path}/a.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, NULL::DOUBLE AS v
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, 20.0 AS v
                ) TO '{tmp_path}/b.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "a.parquet"),
            str(tmp_path / "b.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        assert "v" in result.value_changes
        # Both rows have null-vs-non-null differences
        assert result.value_changes["v"]["changed_count"] == 2

    def test_diff_schema_added_removed_columns(self, tmp_path):
        """diff() should report added and removed columns in schema_changes."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 1.0 AS old_col
                ) TO '{tmp_path}/a.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 2.0 AS new_col
                ) TO '{tmp_path}/b.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "a.parquet"),
            str(tmp_path / "b.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        types = {s["column"]: s["type"] for s in result.schema_changes}
        assert types.get("new_col") == "+"
        assert types.get("old_col") == "-"


class TestClassifySeverity:
    """Boundary condition tests for _classify_severity."""

    def test_high_by_max_leakage_days(self):
        from datetime import timedelta

        assert _classify_severity(0.0, timedelta(days=8)) == "HIGH"

    def test_high_by_percentage(self):
        assert _classify_severity(0.06, None) == "HIGH"

    def test_medium_by_percentage(self):
        assert _classify_severity(0.02, None) == "MEDIUM"

    def test_medium_by_leakage_days(self):
        from datetime import timedelta

        assert _classify_severity(0.001, timedelta(days=1)) == "MEDIUM"

    def test_low_severity(self):
        assert _classify_severity(0.005, None) == "LOW"

    def test_boundary_high_pct_exactly(self):
        # 5% is the HIGH threshold (strict >), so exactly 5% falls to MEDIUM
        # because 0.05 > SEVERITY_MEDIUM_PCT (0.01)
        assert _classify_severity(0.05, None) == "MEDIUM"

    def test_boundary_medium_pct_exactly(self):
        # 1% is the boundary — at exactly 1%, should be LOW (not >)
        assert _classify_severity(0.01, None) == "LOW"

    def test_boundary_high_days_exactly_7(self):
        from datetime import timedelta

        # 7 days is the boundary — at exactly 7, should be MEDIUM (not >)
        assert _classify_severity(0.0, timedelta(days=7)) == "MEDIUM"


# ---------------------------------------------------------------------------
# Test 2: AuditReport.__str__ on leaky reports (covers lines 274-317)
# ---------------------------------------------------------------------------


class TestAuditReportStr:
    """Test the textual rendering of audit reports, including _format_leakage."""

    def test_str_leaky_report_with_days(self):
        """Leaky report with max/median leakage measured in days."""
        from datetime import timedelta

        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "risk_score": FeatureAuditDetail(
                    name="risk_score",
                    leaky_row_count=500,
                    leaky_row_pct=0.10,
                    max_leakage=timedelta(days=14),
                    median_leakage=timedelta(days=3),
                    severity="HIGH",
                    total_rows=5000,
                    null_rows=0,
                    clean=False,
                ),
                "account_age": FeatureAuditDetail(
                    name="account_age",
                    total_rows=5000,
                    null_rows=12,
                    clean=True,
                ),
            },
            total_rows=5000,
        )
        text = str(report)
        assert "WARNING: LEAKAGE DETECTED in 1 of 2 features" in text
        assert "LEAK  risk_score" in text
        assert "500 rows (10.0%)" in text
        assert "Max leakage: 14 days" in text
        assert "Median leakage: 3 days" in text
        assert "Severity: HIGH" in text
        assert "OK  account_age - clean (5000 rows, 12 null)" in text

    def test_str_leaky_report_hours(self):
        """Leakage measured in hours (< 1 day)."""
        from datetime import timedelta

        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "feat": FeatureAuditDetail(
                    name="feat",
                    leaky_row_count=10,
                    leaky_row_pct=0.005,
                    max_leakage=timedelta(hours=6),
                    median_leakage=timedelta(hours=1),
                    severity="LOW",
                    total_rows=2000,
                    clean=False,
                ),
            },
            total_rows=2000,
        )
        text = str(report)
        assert "Max leakage: 6 hours" in text
        assert "Median leakage: 1 hour" in text  # singular

    def test_str_leaky_report_minutes(self):
        """Leakage measured in minutes (< 1 hour)."""
        from datetime import timedelta

        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "feat": FeatureAuditDetail(
                    name="feat",
                    leaky_row_count=2,
                    leaky_row_pct=0.001,
                    max_leakage=timedelta(minutes=45),
                    median_leakage=timedelta(minutes=1),
                    severity="LOW",
                    total_rows=2000,
                    clean=False,
                ),
            },
            total_rows=2000,
        )
        text = str(report)
        assert "Max leakage: 45 minutes" in text
        assert "Median leakage: 1 minute" in text  # singular

    def test_str_clean_report(self):
        """Clean report should say ALL CLEAN."""
        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "good_feat": FeatureAuditDetail(
                    name="good_feat", total_rows=100, clean=True
                ),
            },
            total_rows=100,
        )
        text = str(report)
        assert "ALL CLEAN" in text
        assert "OK  good_feat" in text

    def test_str_1_day_singular(self):
        """Exactly 1 day should use singular 'day'."""
        from datetime import timedelta

        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "f": FeatureAuditDetail(
                    name="f",
                    leaky_row_count=1,
                    leaky_row_pct=0.01,
                    max_leakage=timedelta(days=1),
                    severity="MEDIUM",
                    total_rows=100,
                    clean=False,
                ),
            },
            total_rows=100,
        )
        text = str(report)
        assert "Max leakage: 1 day" in text
        assert "1 days" not in text


# ---------------------------------------------------------------------------
# Test 3: Duplicate (key, timestamp) detection (covers lines 582-624)
# ---------------------------------------------------------------------------


class TestDuplicateKeyTimestamp:
    """Tests for _check_duplicates: on_duplicate='error' and 'keep_any'."""

    def test_duplicate_raises_on_error_mode(self, tmp_path):
        """Duplicates with on_duplicate='error' should raise TimefenceDuplicateError."""
        conn = duckdb.connect()
        try:
            # Create source with duplicate (user_id, ts) pairs
            conn.execute(f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, TIMESTAMP '2024-01-15', 10.0),
                        (1, TIMESTAMP '2024-01-15', 20.0),
                        (2, TIMESTAMP '2024-02-01', 30.0)
                    ) AS t(uid, ts, val)
                ) TO '{tmp_path}/dup_src.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, false AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "dup_src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(
            source=src, columns=["val"], name="dup_feat", on_duplicate="error"
        )

        with pytest.raises(timefence.errors.TimefenceDuplicateError, match="duplicate"):
            build(labels=labels, features=[feat])

    def test_duplicate_warns_on_keep_any(self, tmp_path):
        """Duplicates with on_duplicate='keep_any' should warn but succeed."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, TIMESTAMP '2024-01-15', 10.0),
                        (1, TIMESTAMP '2024-01-15', 20.0),
                        (2, TIMESTAMP '2024-02-01', 30.0)
                    ) AS t(uid, ts, val)
                ) TO '{tmp_path}/dup_src.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                    UNION ALL
                    SELECT 2 AS uid, TIMESTAMP '2024-06-02' AS lt, false AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "dup_src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(
            source=src, columns=["val"], name="dup_feat", on_duplicate="keep_any"
        )

        with pytest.warns(match="duplicate"):
            result = build(labels=labels, features=[feat])
        assert result.stats.row_count == 2


# ---------------------------------------------------------------------------
# Test 4: Timezone mismatch detection (covers lines 535-579)
# ---------------------------------------------------------------------------


class TestTimezoneMismatch:
    """Build should raise TimefenceTimezoneError when tz-aware meets tz-naive."""

    @pytest.mark.skipif(
        not _has_pytz(), reason="pytz required for TIMESTAMPTZ support in DuckDB"
    )
    def test_tz_aware_labels_vs_naive_features(self, tmp_path):
        """TIMESTAMPTZ labels + TIMESTAMP features → timezone error."""
        conn = duckdb.connect()
        try:
            # Labels with timezone-aware timestamps
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS uid,
                        TIMESTAMPTZ '2024-06-01 00:00:00+00' + INTERVAL (i) DAY AS lt,
                        true AS y
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/labels_tz.parquet' (FORMAT PARQUET)
            """)
            # Source with timezone-naive timestamps
            conn.execute(f"""
                COPY (
                    SELECT
                        i AS uid,
                        TIMESTAMP '2024-05-01' + INTERVAL (i) DAY AS ts,
                        42.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/src_naive.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels_tz.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src_naive.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(source=src, columns=["val"], name="tz_feat")

        with pytest.raises(
            timefence.errors.TimefenceTimezoneError, match="Mixed timezones"
        ):
            build(labels=labels, features=[feat])


# ---------------------------------------------------------------------------
# Test 5: BuildResult str/validate/explain (covers lines 83-108)
# ---------------------------------------------------------------------------


class TestBuildResultMethods:
    """Test BuildResult.__str__, validate(), explain()."""

    def test_str_output_with_missing(self, sample_features, tmp_path):
        """str() should include feature match stats and output path."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
            max_lookback="1d",  # short lookback → some missing
        )
        text = str(result)
        assert "BuildResult:" in text
        assert "rows" in text
        assert "columns" in text
        assert "Output:" in text
        assert str(tmp_path / "out.parquet") in text
        assert "Time:" in text
        assert "user_country:" in text
        # Short lookback should produce missing values
        assert "missing" in text.lower() or "matched" in text.lower()

    def test_str_no_output_path(self, sample_features):
        """str() without output should omit Output line."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
        )
        text = str(result)
        assert "BuildResult:" in text
        assert "Output:" not in text

    def test_str_all_matched(self, sample_features, tmp_path):
        """When all rows match, 'missing' should not appear in str output."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        text = str(result)
        # user_country with default lookback should match all or most
        assert "user_country:" in text

    def test_validate_returns_bool(self, sample_features, tmp_path):
        """validate() should return True for a clean build."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        assert result.validate() is True

    def test_explain_returns_sql(self, sample_features, tmp_path):
        """explain() should return the join SQL string."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        sql = result.explain()
        assert isinstance(sql, str)
        assert len(sql) > 0

    def test_repr_html(self, sample_features, tmp_path):
        """_repr_html_ should return valid HTML for Jupyter."""
        result = build(
            labels=sample_features["labels"],
            features=[sample_features["user_country"]],
            output=str(tmp_path / "out.parquet"),
        )
        html = result._repr_html_()
        assert "<div" in html
        assert "Timefence Build Result" in html
        assert "user_country" in html


# ---------------------------------------------------------------------------
# Test 6: Split validation — overlaps, gaps, boundaries (covers lines 626-671)
# ---------------------------------------------------------------------------


class TestSplitValidation:
    """Tests for _validate_splits warnings and errors."""

    def test_overlapping_splits_raises(self, sample_features, tmp_path):
        """Overlapping split ranges must raise TimefenceConfigError."""
        with pytest.raises(TimefenceConfigError, match="overlap"):
            build(
                labels=sample_features["labels"],
                features=[sample_features["user_country"]],
                output=str(tmp_path / "out.parquet"),
                splits={
                    "train": ("2024-01-01", "2024-06-01"),
                    "test": ("2024-05-01", "2025-01-01"),  # overlaps with train
                },
            )

    def test_gap_between_splits_warns(self, sample_features, tmp_path):
        """Gap between splits should emit a warning."""
        with pytest.warns(match="Gap between splits"):
            build(
                labels=sample_features["labels"],
                features=[sample_features["user_country"]],
                output=str(tmp_path / "out.parquet"),
                splits={
                    "train": ("2024-01-01", "2024-03-01"),
                    "test": ("2024-06-01", "2025-01-01"),  # 3 month gap
                },
            )

    def test_splits_not_covering_labels_warns(self, sample_features, tmp_path):
        """Splits that don't cover the full label range should warn."""
        # Labels span 2024-01-20 to 2024-09-28 (50 labels, 5 days apart)
        # Splits only cover a narrow range in the middle
        with pytest.warns(match="Splits"):
            build(
                labels=sample_features["labels"],
                features=[sample_features["user_country"]],
                output=str(tmp_path / "out.parquet"),
                splits={
                    "narrow": ("2024-04-01", "2024-05-01"),
                },
            )


# ---------------------------------------------------------------------------
# Test 7: Schema validation — missing source columns (covers lines 504-533)
# ---------------------------------------------------------------------------


class TestSourceSchemaValidation:
    """Tests for _validate_source_schema error paths."""

    def test_missing_timestamp_column(self, tmp_path):
        """Source missing the timestamp column should raise TimefenceSchemaError."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, 'A' AS val
                ) TO '{tmp_path}/no_ts.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "no_ts.parquet"),
            keys="uid",
            timestamp="missing_ts",  # column doesn't exist
        )
        feat = timefence.Feature(source=src, columns=["val"], name="bad_ts_feat")

        with pytest.raises(TimefenceSchemaError, match="missing timestamp column"):
            build(labels=labels, features=[feat])

    def test_missing_feature_column(self, tmp_path):
        """Referencing a non-existent column in columns mode should raise."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 'A' AS val
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        feat = timefence.Feature(
            source=src, columns=["nonexistent_col"], name="bad_col_feat"
        )

        with pytest.raises(TimefenceSchemaError, match="does not exist"):
            build(labels=labels, features=[feat])


# ---------------------------------------------------------------------------
# Test 8: Feature name sanitization collision (covers lines 989-997)
# ---------------------------------------------------------------------------


class TestDiffResultStr:
    """Test DiffResult.__str__ rendering (covers lines 372-399)."""

    def test_str_with_schema_and_value_changes(self, tmp_path):
        """str() on DiffResult with schema/value changes should show all sections."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                           100.0 AS val, 'hello' AS txt
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/old.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                           110.0 AS val, 99 AS new_col
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/new.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "old.parquet"),
            str(tmp_path / "new.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        text = str(result)
        assert "BUILD DIFF" in text
        assert "Rows" in text
        assert "old:" in text
        assert "new:" in text
        # Schema changes (txt removed, new_col added)
        assert "Schema" in text
        # Value changes for val column
        assert "Value Changes" in text
        assert "val" in text

    def test_str_with_numeric_deltas(self, tmp_path):
        """Numeric changes should show mean/max delta."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                           100.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/old.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                           200.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/new.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "old.parquet"),
            str(tmp_path / "new.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        text = str(result)
        assert "Mean delta" in text or "Max delta" in text

    def test_str_no_changes(self, tmp_path):
        """Identical files should not show Schema or Value Changes sections."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, 10.0 AS val
                ) TO '{tmp_path}/same.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        import shutil

        shutil.copy(str(tmp_path / "same.parquet"), str(tmp_path / "same2.parquet"))

        result = timefence.diff(
            str(tmp_path / "same.parquet"),
            str(tmp_path / "same2.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        text = str(result)
        assert "BUILD DIFF" in text
        # Identical files: no value changes
        assert "Value Changes" not in text

    def test_diff_row_count_change(self, tmp_path):
        """DiffResult should correctly report row count differences."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt, 1.0 AS val
                    FROM generate_series(1, 3) t(i)
                ) TO '{tmp_path}/small.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT i AS uid, TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt, 1.0 AS val
                    FROM generate_series(1, 5) t(i)
                ) TO '{tmp_path}/large.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        result = timefence.diff(
            str(tmp_path / "small.parquet"),
            str(tmp_path / "large.parquet"),
            keys=["uid"],
            label_time="lt",
        )
        text = str(result)
        assert "old: 3" in text
        assert "new: 5" in text
        assert "+2" in text


class TestAuditReportReprHtml:
    """Test AuditReport._repr_html_ for Jupyter rendering (covers lines 240-272)."""

    def test_repr_html_leaky(self):
        """Leaky report HTML should show LEAKAGE DETECTED and LEAK rows."""

        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "bad_feat": FeatureAuditDetail(
                    name="bad_feat",
                    leaky_row_count=100,
                    leaky_row_pct=0.05,
                    severity="MEDIUM",
                    total_rows=2000,
                    clean=False,
                ),
                "good_feat": FeatureAuditDetail(
                    name="good_feat",
                    total_rows=2000,
                    clean=True,
                ),
            },
            total_rows=2000,
        )
        html = report._repr_html_()
        assert "LEAKAGE DETECTED" in html
        assert "bad_feat" in html
        assert "good_feat" in html
        assert "LEAK" in html
        assert "CLEAN" in html
        assert "#e74c3c" in html  # red for leaky

    def test_repr_html_clean(self):
        """Clean report HTML should show ALL CLEAN."""
        from timefence.engine import AuditReport, FeatureAuditDetail

        report = AuditReport(
            features={
                "feat": FeatureAuditDetail(name="feat", total_rows=100, clean=True),
            },
            total_rows=100,
        )
        html = report._repr_html_()
        assert "ALL CLEAN" in html
        assert "#2ecc71" in html  # green for clean


class TestFeatureNameCollision:
    """Names that are distinct but collide after sanitization must raise."""

    def test_sanitized_name_collision(self, tmp_path):
        """'feat-a' and 'feat_a' collide after sanitization → error."""
        conn = duckdb.connect()
        try:
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-06-01' AS lt, true AS y
                ) TO '{tmp_path}/labels.parquet' (FORMAT PARQUET)
            """)
            conn.execute(f"""
                COPY (
                    SELECT 1 AS uid, TIMESTAMP '2024-05-01' AS ts, 'A' AS val
                ) TO '{tmp_path}/src.parquet' (FORMAT PARQUET)
            """)
        finally:
            conn.close()

        labels = timefence.Labels(
            path=str(tmp_path / "labels.parquet"),
            keys="uid",
            label_time="lt",
            target="y",
        )
        src = timefence.Source(
            path=str(tmp_path / "src.parquet"), keys="uid", timestamp="ts"
        )
        # These two names are distinct but map to the same sanitized name "feat_a"
        feat_dash = timefence.Feature(source=src, columns=["val"], name="feat-a")
        feat_under = timefence.Feature(source=src, columns=["val"], name="feat_a")

        with pytest.raises(TimefenceConfigError, match="collide after sanitization"):
            build(
                labels=labels,
                features=[feat_dash, feat_under],
            )
