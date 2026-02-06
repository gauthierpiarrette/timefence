"""Tests for core data model: Source, Feature, Labels, FeatureSet."""

from datetime import timedelta

import pytest

import timefence
from timefence.core import FeatureSet, flatten_features
from timefence.errors import TimefenceConfigError, TimefenceValidationError


class TestSource:
    def test_parquet_auto_detect(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        assert src.format == "parquet"
        assert src.keys == ["id"]
        assert src.timestamp == "ts"
        assert src.name == "data"

    def test_csv_auto_detect(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        src = timefence.Source(path=str(p), keys="id", timestamp="ts")
        assert src.format == "csv"
        assert src.keys == ["id"]  # string -> list

    def test_unknown_extension_errors(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.touch()
        with pytest.raises(TimefenceValidationError, match="Cannot auto-detect"):
            timefence.Source(path=str(p), keys=["id"], timestamp="ts")

    def test_explicit_format(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.touch()
        src = timefence.Source(
            path=str(p), keys=["id"], timestamp="ts", format="parquet"
        )
        assert src.format == "parquet"

    def test_no_path_or_df_errors(self):
        with pytest.raises(TimefenceValidationError, match="either 'path' or 'df'"):
            timefence.Source(keys=["id"], timestamp="ts")

    def test_both_path_and_df_errors(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        with pytest.raises(TimefenceValidationError, match="either 'path' or 'df'"):
            timefence.Source(path=str(p), df=[], keys=["id"], timestamp="ts")

    def test_parquet_source_alias(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.ParquetSource(path=str(p), keys=["id"], timestamp="ts")
        assert src.format == "parquet"

    def test_csv_source_alias(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        src = timefence.CSVSource(path=str(p), keys=["id"], timestamp="ts")
        assert src.format == "csv"


class TestSQLSource:
    def test_basic(self):
        src = timefence.SQLSource(
            query="SELECT * FROM t",
            keys=["id"],
            timestamp="ts",
            name="my_source",
        )
        assert src.format == "sql"
        assert src.name == "my_source"
        assert src.keys == ["id"]

    def test_string_key(self):
        src = timefence.SQLSource(
            query="SELECT * FROM t",
            keys="id",
            timestamp="ts",
            name="src",
        )
        assert src.keys == ["id"]


class TestFeature:
    def test_columns_mode_string(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, columns="name")
        assert feat.mode == "columns"
        assert feat.name == "name"
        assert feat._columns == {"name": "name"}

    def test_columns_mode_list(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, columns=["a", "b"])
        assert feat.name == "a_b"
        assert feat.output_columns == ["a", "b"]

    def test_columns_mode_dict(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, columns={"src_col": "out_col"})
        assert feat._columns == {"src_col": "out_col"}
        assert feat.output_columns == ["out_col"]
        assert feat.name == "out_col"

    def test_sql_mode_inline(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(
            source=src,
            sql="SELECT id, ts AS feature_time, val FROM {source}",
            name="my_feat",
        )
        assert feat.mode == "sql"
        assert feat.name == "my_feat"

    def test_sql_mode_inline_requires_name(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        with pytest.raises(TimefenceConfigError, match=r"name.*required.*inline SQL"):
            timefence.Feature(source=src, sql="SELECT * FROM {source}")

    def test_sql_mode_path(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        sql_file = tmp_path / "my_feature.sql"
        sql_file.write_text("SELECT id, ts AS feature_time, val FROM {source}")
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, sql=sql_file)
        assert feat.mode == "sql"
        assert feat.name == "my_feature"

    def test_transform_mode(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")

        def compute_thing(conn, table):
            return conn.sql(f"SELECT id, ts AS feature_time, 1 AS val FROM {table}")

        feat = timefence.Feature(source=src, transform=compute_thing)
        assert feat.mode == "transform"
        assert feat.name == "compute_thing"

    def test_no_mode_errors(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        with pytest.raises(TimefenceConfigError, match="exactly one"):
            timefence.Feature(source=src)

    def test_multiple_modes_errors(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        with pytest.raises(TimefenceConfigError, match="exactly one"):
            timefence.Feature(source=src, columns=["a"], sql="SELECT 1")

    def test_embargo(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, columns=["a"], embargo="1d")
        assert feat.embargo == timedelta(days=1)

    def test_default_embargo(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        feat = timefence.Feature(source=src, columns=["a"])
        assert feat.embargo == timedelta(0)

    def test_invalid_on_duplicate(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        with pytest.raises(TimefenceConfigError, match="on_duplicate"):
            timefence.Feature(source=src, columns=["a"], on_duplicate="invalid")

    def test_key_mapping(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["user_id"], timestamp="ts")
        feat = timefence.Feature(
            source=src, columns=["a"], key_mapping={"user_id": "customer_id"}
        )
        assert feat.source_keys == ["customer_id"]


class TestLabels:
    def test_basic(self, tmp_path):
        p = tmp_path / "labels.parquet"
        p.touch()
        labels = timefence.Labels(
            path=str(p), keys=["user_id"], label_time="lt", target="churned"
        )
        assert labels.keys == ["user_id"]
        assert labels.target == ["churned"]

    def test_no_path_or_df(self):
        with pytest.raises(TimefenceValidationError):
            timefence.Labels(keys=["id"], label_time="lt", target="t")

    def test_both_path_and_df(self, tmp_path):
        p = tmp_path / "labels.parquet"
        p.touch()
        with pytest.raises(TimefenceValidationError):
            timefence.Labels(
                path=str(p), df=[], keys=["id"], label_time="lt", target="t"
            )


class TestFeatureSet:
    def test_basic(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        f1 = timefence.Feature(source=src, columns=["a"])
        f2 = timefence.Feature(source=src, columns=["b"])
        fs = FeatureSet(name="test", features=[f1, f2])
        assert len(fs) == 2
        assert list(fs) == [f1, f2]

    def test_flatten(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.touch()
        src = timefence.Source(path=str(p), keys=["id"], timestamp="ts")
        f1 = timefence.Feature(source=src, columns=["a"])
        f2 = timefence.Feature(source=src, columns=["b"])
        f3 = timefence.Feature(source=src, columns=["c"])
        fs = FeatureSet(name="grp", features=[f1, f2])
        flat = flatten_features([fs, f3])
        assert flat == [f1, f2, f3]
