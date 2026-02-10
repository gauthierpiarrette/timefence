"""Core data model: Source, Feature, Labels, FeatureSet."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Union

from timefence._duration import parse_duration
from timefence.errors import TimefenceConfigError, TimefenceValidationError


def _as_list(value: str | list[str]) -> list[str]:
    """Normalize a string-or-list argument to a list."""
    return [value] if isinstance(value, str) else list(value)


class Source:
    """A table of historical data with timestamps.

    Args:
        path: Path to the data file (Parquet or CSV).
        keys: Column name(s) used as entity keys.
        timestamp: Column name containing the temporal key.
        name: Human-readable name (defaults to filename stem).
        format: File format ("parquet" or "csv"). Auto-detected from extension.
        delimiter: CSV delimiter (only for CSV files).
        timestamp_format: strftime format for parsing timestamp strings.
    """

    path: Path | None
    df: Any
    keys: list[str]
    timestamp: str
    name: str
    delimiter: str
    timestamp_format: str | None
    format: str

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        keys: str | list[str],
        timestamp: str,
        name: str | None = None,
        format: str | None = None,
        delimiter: str = ",",
        timestamp_format: str | None = None,
        df: Any = None,
    ):
        if path is None and df is None:
            raise TimefenceValidationError(
                "Source requires either 'path' or 'df' parameter."
            )
        if path is not None and df is not None:
            raise TimefenceValidationError(
                "Source accepts either 'path' or 'df', not both."
            )

        self.path = Path(path) if path is not None else None
        self.df = df
        self.keys = _as_list(keys)
        if not self.keys:
            raise TimefenceValidationError(
                "Source 'keys' cannot be empty. Provide at least one entity key column."
            )
        self.timestamp = timestamp
        self.name = name or (self.path.stem if self.path else "dataframe")
        self.delimiter = delimiter
        self.timestamp_format = timestamp_format

        if format is not None:
            self.format = format
        elif self.path is not None:
            ext = self.path.suffix.lower()
            if ext in (".parquet", ".pq"):
                self.format = "parquet"
            elif ext == ".csv":
                self.format = "csv"
            else:
                raise TimefenceValidationError(
                    f"Cannot auto-detect format for '{self.path}'. "
                    "Specify format='parquet' or format='csv'."
                )
        else:
            self.format = "arrow"

    def __repr__(self) -> str:
        src = str(self.path) if self.path else "DataFrame"
        return f"Source(name='{self.name}', path='{src}', keys={self.keys}, timestamp='{self.timestamp}')"


class ParquetSource(Source):
    """Convenience alias for Source with format='parquet'."""

    def __init__(self, path: str | Path, **kwargs: Any):
        super().__init__(path=path, format="parquet", **kwargs)


class CSVSource(Source):
    """Convenience alias for Source with format='csv'."""

    def __init__(self, path: str | Path, **kwargs: Any):
        super().__init__(path=path, format="csv", **kwargs)


class SQLSource:
    """A source defined by a SQL query against DuckDB.

    Args:
        query: SQL query string. Can use read_parquet/read_csv directly.
        keys: Column name(s) used as entity keys.
        timestamp: Column name containing the temporal key.
        name: Human-readable name.
        connection: Path to a DuckDB database file (optional, uses in-memory by default).
    """

    query: str
    keys: list[str]
    timestamp: str
    name: str
    connection: str | None
    path: Path | None  # Always None for SQLSource
    df: Any  # Always None for SQLSource
    format: Literal["sql"]

    def __init__(
        self,
        query: str,
        *,
        keys: str | list[str],
        timestamp: str,
        name: str,
        connection: str | None = None,
    ):
        self.query = query
        self.keys = _as_list(keys)
        self.timestamp = timestamp
        self.name = name
        self.connection = connection
        self.path = None
        self.df = None
        self.format = "sql"

    def __repr__(self) -> str:
        return f"SQLSource(name='{self.name}', keys={self.keys}, timestamp='{self.timestamp}')"


SourceType = Union[Source, SQLSource]


class Feature:
    """A named, versioned column derived from a source.

    Exactly one of columns, sql, or transform must be provided.

    Args:
        source: The data source for this feature.
        columns: Column name(s) to select (Mode 1).
        sql: SQL query string or path to .sql file (Mode 2).
        transform: Python callable (Mode 3).
        name: Feature name (auto-derived if possible).
        embargo: Computation lag buffer (e.g., "1d").
        key_mapping: Map label key names to source key names.
        on_duplicate: "error" (default) or "keep_any".
    """

    source: SourceType
    mode: Literal["columns", "sql", "transform"]
    name: str
    embargo: timedelta
    key_mapping: dict[str, str]
    on_duplicate: str
    _columns: dict[str, str]
    _sql_text: str | None
    _sql_path: Path | None
    _transform: Callable[..., Any] | None

    def __init__(
        self,
        source: SourceType,
        *,
        columns: str | list[str] | dict[str, str] | None = None,
        sql: str | Path | None = None,
        transform: Callable | None = None,
        name: str | None = None,
        embargo: str | timedelta | None = None,
        key_mapping: dict[str, str] | None = None,
        on_duplicate: str = "error",
    ):
        self.source = source

        # Validate exactly one mode
        modes = sum(x is not None for x in [columns, sql, transform])
        if modes != 1:
            raise TimefenceConfigError(
                "Feature requires exactly one of 'columns', 'sql', or 'transform'. "
                f"Got {modes} of them."
            )

        # Determine mode and normalize
        if columns is not None:
            self.mode = "columns"
            if isinstance(columns, str):
                self._columns = {columns: columns}
            elif isinstance(columns, list):
                self._columns = {c: c for c in columns}
            else:
                self._columns = dict(columns)
            if not self._columns:
                raise TimefenceConfigError(
                    "Feature 'columns' cannot be empty. "
                    "Provide at least one column name."
                )
            self._sql_text = None
            self._sql_path = None
            self._transform = None
        elif sql is not None:
            self.mode = "sql"
            if isinstance(sql, Path):
                self._sql_path = sql
                self._sql_text = sql.read_text()
            else:
                self._sql_path = None
                self._sql_text = sql
            self._columns = {}
            self._transform = None
        else:
            self.mode = "transform"
            self._transform = transform
            self._columns = {}
            self._sql_text = None
            self._sql_path = None

        # Derive name
        if name is not None:
            self.name = name
        elif self.mode == "columns":
            self.name = "_".join(self._columns.values())
        elif (
            self.mode == "sql"
            and hasattr(self, "_sql_path")
            and self._sql_path is not None
        ):
            self.name = self._sql_path.stem
        elif self.mode == "transform":
            self.name = transform.__name__  # type: ignore[union-attr]
        else:
            raise TimefenceConfigError(
                "Feature 'name' is required when using inline SQL. "
                "Timefence cannot auto-derive a name from a SQL string."
            )

        self.embargo = parse_duration(embargo) or timedelta(0)
        self.key_mapping = key_mapping or {}
        self.on_duplicate = on_duplicate

        if on_duplicate not in ("error", "keep_any"):
            raise TimefenceConfigError(
                f"on_duplicate must be 'error' or 'keep_any', got '{on_duplicate}'."
            )

    @property
    def output_columns(self) -> list[str]:
        """Column names this feature produces in the output."""
        if self.mode == "columns":
            return list(self._columns.values())
        return []

    @property
    def source_keys(self) -> list[str]:
        """Key column names as they appear in the source.

        Note: key_mapping is applied by the engine during joins (label_key -> source_key),
        not here. This returns the raw source key columns.
        """
        return list(self.source.keys)

    @property
    def definition_hash_input(self) -> str:
        """String used to compute the feature definition hash."""
        if self.mode == "columns":
            return f"columns:{sorted(self._columns.items())}:{self.source.name}:{self.key_mapping}"
        elif self.mode == "sql":
            return f"sql:{self._sql_text}:{self.source.name}"
        else:
            try:
                src = inspect.getsource(self._transform)
            except (OSError, TypeError):
                src = "<dynamic>"
            return f"transform:{src}:{self.source.name}"

    def __repr__(self) -> str:
        return f"Feature(name='{self.name}', source='{self.source.name}', mode='{self.mode}')"


class Labels:
    """Prediction targets with entity keys and event times.

    Args:
        path: Path to the labels file.
        df: DataFrame with labels (mutually exclusive with path).
        keys: Column name(s) used as entity keys.
        label_time: Column name containing the label event time.
        target: Column name(s) for the prediction target.
    """

    path: Path | None
    df: Any
    keys: list[str]
    label_time: str
    target: list[str]

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        df: Any = None,
        keys: str | list[str],
        label_time: str,
        target: str | list[str],
    ):
        if path is None and df is None:
            raise TimefenceValidationError(
                "Labels requires either 'path' or 'df' parameter."
            )
        if path is not None and df is not None:
            raise TimefenceValidationError(
                "Labels accepts either 'path' or 'df', not both."
            )

        self.path = Path(path) if path is not None else None
        self.df = df
        self.keys = _as_list(keys)
        if not self.keys:
            raise TimefenceValidationError(
                "Labels 'keys' cannot be empty. Provide at least one entity key column."
            )
        self.label_time = label_time
        self.target = _as_list(target)
        if not self.target:
            raise TimefenceValidationError(
                "Labels 'target' cannot be empty. Provide at least one target column."
            )

    def __repr__(self) -> str:
        src = str(self.path) if self.path else "DataFrame"
        return f"Labels(path='{src}', keys={self.keys}, label_time='{self.label_time}')"


class FeatureSet:
    """A named group of features for reuse across models.

    A FeatureSet is a flat list with a name. No nesting, no inheritance.
    """

    name: str
    features: list[Feature]

    def __init__(
        self,
        name: str,
        features: Sequence[Feature],
    ):
        self.name = name
        self.features = list(features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        names = [f.name for f in self.features]
        return f"FeatureSet(name='{self.name}', features={names})"


def flatten_features(
    features: Sequence[Feature | FeatureSet],
) -> list[Feature]:
    """Flatten a mix of Features and FeatureSets into a list of Features."""
    result = []
    for f in features:
        if isinstance(f, FeatureSet):
            result.extend(f.features)
        else:
            result.append(f)
    return result
