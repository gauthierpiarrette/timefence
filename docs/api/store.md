# Store

Local directory for build tracking and feature-level caching.

::: timefence.Store
    options:
      show_root_heading: true

## How it works

When you pass a `Store` to `build()`, Timefence:

1. **Hashes inputs** — content hash (SHA-256) of source files, feature definitions, embargo values, and build parameters.
2. **Checks feature cache** — if a feature's inputs haven't changed, the cached intermediate table is loaded instead of recomputed.
3. **Checks build cache** — if all features + labels + parameters match a previous build, the entire result is returned immediately.
4. **Saves the manifest** — every build writes a JSON manifest with full provenance to `.timefence/builds/<build_id>/build.json`.

## Directory structure

```
.timefence/
├── builds/
│   ├── 20240315T120000Z/
│   │   ├── build.json        # Full build manifest
│   │   └── train.parquet     # Symlink to output file
│   └── 20240316T090000Z/
│       └── build.json
└── cache/
    ├── hashes.json            # Content hash cache (path:size:mtime → hash)
    └── features/
        ├── rolling_spend__a1b2c3d4.parquet
        └── user_country__e5f6g7h8.parquet
```

## Example

```python
import timefence

store = timefence.Store(".timefence")

# First build: computes everything, caches results
result = timefence.build(
    labels=labels,
    features=[rolling_spend, user_country],
    output="train.parquet",
    store=store,
)

# Second build (same inputs): returns cached result in milliseconds
result = timefence.build(
    labels=labels,
    features=[rolling_spend, user_country],
    output="train.parquet",
    store=store,
)

# Add a new feature: only the new one is computed, others loaded from cache
result = timefence.build(
    labels=labels,
    features=[rolling_spend, user_country, login_count],
    output="train.parquet",
    store=store,
)
```

## Build history

```python
store = timefence.Store(".timefence")

# List all past builds (newest first)
builds = store.list_builds()
for b in builds:
    print(f"{b['build_id']}  {b['output']['row_count']} rows  {b['duration_seconds']:.1f}s")

# Get a specific build by ID
manifest = store.get_build("20240315T120000Z")
if manifest:
    print(manifest["features"])    # Feature-level stats
    print(manifest["parameters"])  # max_lookback, join, etc.
    print(manifest["audit"])       # Post-build audit result
```

## Cache invalidation

Cache keys are recomputed from content hashes on every build. The cache is automatically invalidated when:

| Change | What happens |
|--------|-------------|
| Source data changes (any byte) | Feature recomputed from scratch |
| Feature definition changes (SQL, columns, transform) | Feature recomputed |
| Embargo value changes | Feature recomputed |
| Timefence version changes | Feature recomputed |
| Labels change | Build recomputed (features may still be cached) |
| `max_lookback` / `max_staleness` / `join` / `on_missing` change | Build recomputed |

To manually clear the cache, delete the `.timefence/cache/` directory:

```bash
rm -rf .timefence/cache/
```

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.save_build(manifest)` | `Path` | Save build manifest, return manifest path. |
| `.list_builds()` | `list[dict]` | List all builds (newest first). |
| `.get_build(build_id)` | `dict \| None` | Get a specific build manifest by ID. |
| `.content_hash(path)` | `str` | Compute SHA-256 hash of a file (e.g., `"sha256:abc123..."`). |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | `".timefence"` | Directory path for the store. |

!!! tip
    Add `.timefence/` to your `.gitignore`. The store is local-only and should not be committed.
