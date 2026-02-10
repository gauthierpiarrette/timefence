"""Tests for the CLI interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from timefence.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIVersion:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.9.1" in result.output


class TestCLIVerboseDebug:
    def test_verbose_flag_accepted(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["-v", "--version"])
            assert result.exit_code == 0

    def test_debug_flag_accepted(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--debug", "--version"])
            assert result.exit_code == 0

    def test_verbose_build_emits_sql_logs(self, runner, tmp_path, caplog):
        import logging

        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")
            with caplog.at_level(logging.INFO, logger="timefence.engine"):
                result = runner.invoke(cli, ["-v", "build", "-o", "out.parquet"])
            assert result.exit_code == 0
            log_text = caplog.text
            assert (
                "Feature SQL" in log_text
                or "Join SQL" in log_text
                or "Labels:" in log_text
            )


class TestCLIInit:
    def test_init(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path("timefence.yaml").exists()

    def test_init_idempotent(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "already exists" in result.output


class TestCLIDoctor:
    def test_doctor_no_config(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["doctor"])
            assert result.exit_code == 0
            assert "DuckDB" in result.output


class TestCLIInspect:
    def test_inspect_parquet(self, runner, tmp_data):
        result = runner.invoke(cli, ["inspect", str(tmp_data / "users.parquet")])
        assert result.exit_code == 0
        assert "user_id" in result.output
        assert "country" in result.output

    def test_inspect_nonexistent(self, runner):
        result = runner.invoke(cli, ["inspect", "nonexistent.parquet"])
        assert result.exit_code != 0


class TestCLICatalog:
    def test_catalog(self, runner, tmp_data):
        features_file = tmp_data.parent / "features.py"
        data_path = (tmp_data / "users.parquet").as_posix()
        features_file.write_text(f"""
import timefence

users = timefence.Source(
    path="{data_path}",
    keys=["user_id"],
    timestamp="updated_at",
    name="users",
)

user_country = timefence.Feature(
    source=users,
    columns=["country"],
    name="user_country",
)
""")
        result = runner.invoke(cli, ["catalog", "--features", str(features_file)])
        assert result.exit_code == 0
        assert "user_country" in result.output


class TestCLIQuickstart:
    def test_quickstart(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["quickstart", "test-project", "--minimal"])
            assert result.exit_code == 0
            assert Path("test-project").exists()
            assert (Path("test-project") / "features.py").exists()
            assert (Path("test-project") / "data" / "users.parquet").exists()
            assert (Path("test-project") / "data" / "train_LEAKY.parquet").exists()


class TestCLIBuild:
    def test_build_with_config(self, runner, tmp_path):
        """Build command resolves labels and features from timefence.yaml."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Generate a quickstart project first
            result = runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            assert result.exit_code == 0

            import os

            os.chdir("proj")

            result = runner.invoke(cli, ["build", "-o", "data/train_CLEAN.parquet"])
            assert result.exit_code == 0, result.output
            assert Path("data/train_CLEAN.parquet").exists()

    def test_build_json_output(self, runner, tmp_path):
        import json as json_mod

        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(cli, ["build", "-o", "out.parquet", "--json"])
            assert result.exit_code == 0, result.output
            data = json_mod.loads(result.output)
            assert "timefence_version" in data
            assert "audit" in data

    def test_build_dry_run(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(cli, ["build", "-o", "out.parquet", "--dry-run"])
            assert result.exit_code == 0
            assert "JOIN PLAN" in result.output
            assert not Path("out.parquet").exists()


class TestCLIAudit:
    def test_audit_with_config(self, runner, tmp_path):
        """Audit resolves features and keys from timefence.yaml."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(cli, ["audit", "data/train_LEAKY.parquet"])
            assert result.exit_code == 0, result.output
            assert "TEMPORAL AUDIT REPORT" in result.output

    def test_audit_json_output(self, runner, tmp_path):
        import json as json_mod

        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(cli, ["audit", "data/train_LEAKY.parquet", "--json"])
            assert result.exit_code == 0, result.output
            data = json_mod.loads(result.output)
            assert "has_leakage" in data

    def test_audit_strict_leaky(self, runner, tmp_path):
        """--strict should exit 1 if leakage found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(
                cli, ["audit", "data/train_LEAKY.parquet", "--strict"]
            )
            # Leaky dataset + --strict = non-zero exit
            assert result.exit_code != 0


class TestCLIFeatureFiltering:
    def test_explain_single_feature(self, runner, tmp_path):
        """--features features.py:feature_name filters to one feature."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            # Get the feature names from the project
            result = runner.invoke(cli, ["catalog", "--json"])
            assert result.exit_code == 0, result.output
            import json as json_mod

            data = json_mod.loads(result.output)
            first_feat = data["features"][0]["name"]

            # Explain with single feature filter
            result = runner.invoke(
                cli, ["explain", "--features", f"features.py:{first_feat}", "--json"]
            )
            assert result.exit_code == 0, result.output
            plan = json_mod.loads(result.output)
            assert len(plan["plan"]) == 1
            assert plan["plan"][0]["name"] == first_feat

    def test_explain_nonexistent_feature(self, runner, tmp_path):
        """Filtering to nonexistent feature should fail."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            result = runner.invoke(
                cli, ["explain", "--features", "features.py:nonexistent"]
            )
            assert result.exit_code != 0


class TestCLIDiffTolerance:
    def test_diff_atol_rtol_flags(self, runner, tmp_path):
        """diff command accepts --atol and --rtol flags."""
        import duckdb

        with runner.isolated_filesystem(temp_dir=tmp_path):
            conn = duckdb.connect()
            try:
                conn.execute("""
                    COPY (
                        SELECT i AS uid,
                            TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                            100.0 AS val
                        FROM generate_series(1, 5) t(i)
                    ) TO 'v1.parquet' (FORMAT PARQUET)
                """)
                conn.execute("""
                    COPY (
                        SELECT i AS uid,
                            TIMESTAMP '2024-06-01' + INTERVAL (i) DAY AS lt,
                            100.0001 AS val
                        FROM generate_series(1, 5) t(i)
                    ) TO 'v2.parquet' (FORMAT PARQUET)
                """)
            finally:
                conn.close()

            # Loose tolerance: no changes
            result = runner.invoke(
                cli,
                [
                    "diff",
                    "v1.parquet",
                    "v2.parquet",
                    "--keys",
                    "uid",
                    "--label-time",
                    "lt",
                    "--atol",
                    "0.01",
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output


class TestCLIOutputDir:
    def test_output_dir_from_config(self, runner, tmp_path):
        """output.dir in timefence.yaml should prefix the output path."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["quickstart", "proj", "--minimal"])
            import os

            os.chdir("proj")

            # Read and modify timefence.yaml to add output.dir
            config = Path("timefence.yaml").read_text()
            config += "\noutput:\n  dir: artifacts/\n"
            Path("timefence.yaml").write_text(config)
            Path("artifacts").mkdir()

            result = runner.invoke(cli, ["build", "-o", "train.parquet"])
            assert result.exit_code == 0, result.output
            assert Path("artifacts/train.parquet").exists()


class TestCLIDoctorJson:
    def test_doctor_json(self, runner, tmp_path):
        import json as json_mod

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["doctor", "--json"])
            assert result.exit_code == 0
            data = json_mod.loads(result.output)
            assert "checks" in data


class TestCLIInspectJson:
    def test_inspect_json(self, runner, tmp_data):
        import json as json_mod

        result = runner.invoke(
            cli, ["inspect", str(tmp_data / "users.parquet"), "--json"]
        )
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        assert "columns" in data
        assert data["row_count"] > 0
