"""Tests for the Store module."""

from __future__ import annotations

from timefence.store import Store


class TestStore:
    def test_init_creates_dirs(self, tmp_path):
        Store(tmp_path / ".timefence")
        assert (tmp_path / ".timefence" / "builds").exists()
        assert (tmp_path / ".timefence" / "cache").exists()

    def test_save_and_list_builds(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        manifest = {"timefence_version": "2.0.0", "output": {"row_count": 100}}
        path = store.save_build(manifest)
        assert path.exists()

        builds = store.list_builds()
        assert len(builds) == 1
        assert builds[0]["output"]["row_count"] == 100

    def test_get_build(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        manifest = {"timefence_version": "2.0.0", "test_key": "test_value"}
        store.save_build(manifest)

        builds = store.list_builds()
        build_id = builds[0]["build_id"]

        result = store.get_build(build_id)
        assert result is not None
        assert result["test_key"] == "test_value"

    def test_get_nonexistent_build(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        assert store.get_build("nonexistent") is None

    def test_content_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = Store.content_hash(f)
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_content_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic content")
        h1 = Store.content_hash(f)
        h2 = Store.content_hash(f)
        assert h1 == h2

    def test_cached_content_hash(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        f = tmp_path / "data.txt"
        f.write_text("some data")

        h1 = store.cached_content_hash(f)
        h2 = store.cached_content_hash(f)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_cached_hash_invalidates_on_change(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        f = tmp_path / "data.txt"
        f.write_text("original")

        h1 = store.cached_content_hash(f)
        f.write_text("modified content that is longer")
        h2 = store.cached_content_hash(f)
        assert h1 != h2


class TestFeatureCache:
    def test_feature_cache_key_deterministic(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        k1 = store.feature_cache_key("def_hash", "src_hash", "1d")
        k2 = store.feature_cache_key("def_hash", "src_hash", "1d")
        assert k1 == k2

    def test_feature_cache_key_varies_with_inputs(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        k1 = store.feature_cache_key("def_hash_a", "src_hash", "1d")
        k2 = store.feature_cache_key("def_hash_b", "src_hash", "1d")
        assert k1 != k2

    def test_feature_cache_path(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        path = store.feature_cache_path("my_feat", "abc123")
        assert "my_feat__abc123.parquet" in str(path)

    def test_has_feature_cache(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        assert not store.has_feature_cache("feat", "key123")
        # Write a dummy file
        cache_path = store.feature_cache_path("feat", "key123")
        cache_path.write_text("dummy")
        assert store.has_feature_cache("feat", "key123")

    def test_init_creates_feature_cache_dir(self, tmp_path):
        Store(tmp_path / ".timefence")
        assert (tmp_path / ".timefence" / "cache" / "features").exists()


class TestBuildCache:
    def test_build_cache_key_deterministic(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        k1 = store.build_cache_key(
            "lbl_hash", ["fk1", "fk2"], "365d", None, "strict", "null"
        )
        k2 = store.build_cache_key(
            "lbl_hash", ["fk1", "fk2"], "365d", None, "strict", "null"
        )
        assert k1 == k2

    def test_build_cache_key_varies(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        k1 = store.build_cache_key("lbl_hash", ["fk1"], "365d", None, "strict", "null")
        k2 = store.build_cache_key(
            "lbl_hash", ["fk1"], "365d", None, "inclusive", "null"
        )
        assert k1 != k2

    def test_find_cached_build_no_match(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        assert store.find_cached_build("nonexistent_key") is None

    def test_find_cached_build_match(self, tmp_path):
        store = Store(tmp_path / ".timefence")
        # Create a fake output file
        out_file = tmp_path / "output.parquet"
        out_file.write_text("dummy")
        manifest = {
            "timefence_version": "2.0.0",
            "build_cache_key": "test_cache_key",
            "output": {"path": str(out_file), "row_count": 100},
        }
        store.save_build(manifest)
        result = store.find_cached_build("test_cache_key")
        assert result is not None
        assert result["output"]["row_count"] == 100
