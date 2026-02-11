# Caching & Store

Timefence can track builds and cache intermediate results using a local `Store`.

## Usage

```python
store = timefence.Store(".timefence")

result = timefence.build(
    labels=labels,
    features=[rolling_spend],
    output="train.parquet",
    store=store,
)
```

## How it works

Cache keys are computed from content hashes of:

- Source files (SHA-256 of file contents)
- Feature definitions (SQL, columns, embargo, etc.)
- Build parameters (max_lookback, max_staleness, join mode)

If nothing has changed, subsequent builds return cached results in milliseconds.

## Feature-level caching

Individual features are cached independently. If you add a new feature to an existing build, only the new feature is computed — the others are loaded from cache.

## Build history

```python
store = timefence.Store(".timefence")

# List past builds
builds = store.list_builds()  # Newest first

# Get a specific build
manifest = store.get_build("abc123")
```

See the [Store API reference](../api/store.md) for the full interface.
