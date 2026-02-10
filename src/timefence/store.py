"""Timefence Store: build tracking, content hashing, and caching."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from timefence._constants import CACHE_KEY_LENGTH, DEFAULT_STORE_PATH
from timefence._version import __version__


class Store:
    """Local directory that tracks builds and manifests.

    Args:
        path: Directory path for the store (default: ".timefence").
    """

    path: Path

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH):
        self.path = Path(path)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        (self.path / "builds").mkdir(parents=True, exist_ok=True)
        (self.path / "cache").mkdir(parents=True, exist_ok=True)
        (self.path / "cache" / "features").mkdir(parents=True, exist_ok=True)

    def save_build(self, manifest: dict[str, Any]) -> Path:
        """Save a build manifest and create a symlink to the output."""
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        build_dir = self.path / "builds" / build_id
        build_dir.mkdir(parents=True, exist_ok=True)

        manifest["build_id"] = build_id
        manifest_path = build_dir / "build.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        # Create symlink to output file
        output_path = manifest.get("output", {}).get("path")
        if output_path:
            output_abs = Path(output_path).resolve()
            if output_abs.exists():
                import contextlib

                link_path = build_dir / output_abs.name
                with contextlib.suppress(OSError):
                    link_path.symlink_to(output_abs)

        return manifest_path

    def list_builds(self) -> list[dict[str, Any]]:
        """List all builds in the store, newest first."""
        builds_dir = self.path / "builds"
        if not builds_dir.exists():
            return []

        builds = []
        for build_dir in sorted(builds_dir.iterdir(), reverse=True):
            manifest_path = build_dir / "build.json"
            if manifest_path.exists():
                builds.append(json.loads(manifest_path.read_text()))
        return builds

    def get_build(self, build_id: str) -> dict[str, Any] | None:
        """Get a specific build manifest by ID."""
        manifest_path = self.path / "builds" / build_id / "build.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return None

    # ------------------------------------------------------------------
    # Content hashing
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(path: str | Path) -> str:
        """Compute full SHA-256 content hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    def cached_content_hash(self, path: str | Path) -> str:
        """Content hash with (path, size, mtime_ns) caching for speed."""
        path = Path(path).resolve()
        cache_file = self.path / "cache" / "hashes.json"

        cache: dict[str, str] = {}
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())

        stat = path.stat()
        cache_key = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"

        if cache_key in cache:
            return cache[cache_key]

        content_hash = self.content_hash(path)
        cache[cache_key] = content_hash
        cache_file.write_text(json.dumps(cache, indent=2))
        return content_hash

    # ------------------------------------------------------------------
    # Feature-level cache
    # ------------------------------------------------------------------

    def feature_cache_key(
        self,
        definition_hash: str,
        source_content_hash: str | None,
        embargo: str | None,
    ) -> str:
        """Compute a cache key for a single feature computation."""
        key_input = (
            f"{definition_hash}:{source_content_hash or ''}:"
            f"{embargo or '0d'}:{__version__}"
        )
        return hashlib.sha256(key_input.encode()).hexdigest()[:CACHE_KEY_LENGTH]

    def feature_cache_path(self, feature_name: str, cache_key: str) -> Path:
        """Path where a cached feature table would be stored."""
        return self.path / "cache" / "features" / f"{feature_name}__{cache_key}.parquet"

    def has_feature_cache(self, feature_name: str, cache_key: str) -> bool:
        return self.feature_cache_path(feature_name, cache_key).exists()

    # ------------------------------------------------------------------
    # Build-level cache
    # ------------------------------------------------------------------

    def build_cache_key(
        self,
        label_content_hash: str | None,
        feature_cache_keys: list[str],
        max_lookback: str | None,
        max_staleness: str | None,
        join_mode: str,
        on_missing: str,
    ) -> str:
        """Compute a cache key for an entire build."""
        key_input = (
            f"{label_content_hash or ''}:"
            f"{sorted(feature_cache_keys)}:"
            f"{max_lookback}:{max_staleness}:{join_mode}:{on_missing}"
        )
        return hashlib.sha256(key_input.encode()).hexdigest()[:CACHE_KEY_LENGTH]

    def find_cached_build(self, build_cache_key: str) -> dict[str, Any] | None:
        """Find a previous build matching this cache key."""
        for build in self.list_builds():
            if build.get("build_cache_key") == build_cache_key:
                output_path = build.get("output", {}).get("path")
                if output_path and Path(output_path).exists():
                    return build
        return None
