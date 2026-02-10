"""Tests for YAML config loading, resolution, and override behavior."""

from __future__ import annotations

from timefence.cli import (
    _load_config,
    _resolve_defaults,
    _resolve_features_file,
    _resolve_labels_info,
    _try_load_config,
)


class TestLoadConfig:
    """Config loading from timefence.yaml."""

    def test_loads_valid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text(
            "name: my-project\n"
            "version: '1.0'\n"
            "store: .timefence/\n"
        )
        config = _load_config()
        assert config["name"] == "my-project"
        assert config["version"] == "1.0"
        assert config["store"] == ".timefence/"

    def test_loads_yml_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yml").write_text("name: alt-ext\n")
        config = _load_config()
        assert config["name"] == "alt-ext"

    def test_returns_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _load_config() == {}

    def test_nested_labels_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text(
            "labels:\n"
            "  path: data/labels.parquet\n"
            "  keys: [user_id]\n"
            "  label_time: label_time\n"
            "  target: [churned]\n"
        )
        config = _load_config()
        assert config["labels"]["path"] == "data/labels.parquet"
        assert config["labels"]["keys"] == ["user_id"]
        assert config["labels"]["target"] == ["churned"]

    def test_features_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text(
            "features:\n" "  - features.py\n" "  - extra_features.py\n"
        )
        config = _load_config()
        assert config["features"] == ["features.py", "extra_features.py"]

    def test_defaults_section(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text(
            "defaults:\n"
            "  max_lookback: 180d\n"
            "  join: inclusive\n"
            "  on_missing: skip\n"
        )
        config = _load_config()
        assert config["defaults"]["max_lookback"] == "180d"
        assert config["defaults"]["join"] == "inclusive"
        assert config["defaults"]["on_missing"] == "skip"


class TestTryLoadConfig:
    """Graceful fallback for malformed configs."""

    def test_malformed_yaml_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text(":\n  - [\n  bad: yaml: here\n")
        config = _try_load_config()
        assert config == {}

    def test_non_dict_yaml_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text("42\n")
        config = _try_load_config()
        assert config == {}

    def test_list_yaml_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text("- one\n- two\n")
        config = _try_load_config()
        assert config == {}

    def test_empty_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "timefence.yaml").write_text("")
        config = _try_load_config()
        assert config == {}


class TestResolveFeatures:
    """Feature file resolution: CLI > config > convention."""

    def test_cli_arg_wins(self):
        assert _resolve_features_file("my.py", {"features": ["other.py"]}) == "my.py"

    def test_config_list(self):
        assert _resolve_features_file(None, {"features": ["feat.py"]}) == "feat.py"

    def test_config_string(self):
        assert _resolve_features_file(None, {"features": "feat.py"}) == "feat.py"

    def test_convention_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "features.py").write_text("# empty")
        assert _resolve_features_file(None, {}) == "features.py"

    def test_returns_none_when_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _resolve_features_file(None, {}) is None


class TestResolveLabels:
    """Labels info extraction from config."""

    def test_full_labels_config(self):
        config = {
            "labels": {
                "path": "data/labels.parquet",
                "keys": ["user_id"],
                "label_time": "lt",
                "target": ["churned"],
            }
        }
        path, keys, label_time, target = _resolve_labels_info(config)
        assert path == "data/labels.parquet"
        assert keys == ["user_id"]
        assert label_time == "lt"
        assert target == ["churned"]

    def test_string_keys_coerced_to_list(self):
        config = {"labels": {"keys": "user_id"}}
        _, keys, _, _ = _resolve_labels_info(config)
        assert keys == ["user_id"]

    def test_missing_labels_returns_defaults(self):
        path, keys, label_time, target = _resolve_labels_info({})
        assert path is None
        assert keys is None
        assert label_time == "label_time"
        assert target == []


class TestResolveDefaults:
    """Defaults extraction from config."""

    def test_extracts_defaults(self):
        config = {"defaults": {"max_lookback": "90d", "join": "inclusive"}}
        defaults = _resolve_defaults(config)
        assert defaults["max_lookback"] == "90d"
        assert defaults["join"] == "inclusive"

    def test_missing_defaults_returns_empty(self):
        assert _resolve_defaults({}) == {}
