"""Timefence engine: build, audit, explain, diff.

Single DuckDB engine for all operations.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import duckdb

if TYPE_CHECKING:
    from timefence.store import Store

from timefence._constants import (
    CACHE_KEY_LENGTH,
    DEFAULT_ATOL,
    DEFAULT_MAX_LOOKBACK,
    DEFAULT_MAX_LOOKBACK_DAYS,
    DEFAULT_ON_MISSING,
    DEFAULT_RTOL,
    SEVERITY_HIGH_DAYS,
    SEVERITY_HIGH_PCT,
    SEVERITY_MEDIUM_DAYS,
    SEVERITY_MEDIUM_PCT,
)
from timefence._duration import (
    duration_to_sql_interval,
    format_duration,
    parse_duration,
)
from timefence._version import __version__
from timefence.core import (
    Feature,
    FeatureSet,
    Labels,
    Source,
    SQLSource,
    flatten_features,
)
from timefence.errors import (
    TimefenceConfigError,
    TimefenceLeakageError,
    TimefenceSchemaError,
    TimefenceTimezoneError,
    TimefenceValidationError,
    duplicate_error,
    schema_error_missing_key,
    timezone_error,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build Result
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    row_count: int = 0
    column_count: int = 0
    feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class BuildResult:
    output_path: str | None
    manifest: dict[str, Any]
    stats: BuildStats
    splits: dict[str, Path] | None = None
    sql: str = ""

    def __str__(self) -> str:
        lines = [
            f"BuildResult: {self.stats.row_count} rows, {self.stats.column_count} columns"
        ]
        if self.output_path:
            lines.append(f"  Output: {self.output_path}")
        lines.append(f"  Time: {self.stats.duration_seconds:.1f}s")
        for fname, fstats in self.stats.feature_stats.items():
            matched = fstats.get("matched", 0)
            missing = fstats.get("missing", 0)
            total = matched + missing
            if missing:
                lines.append(
                    f"  {fname}: {matched}/{total} matched ({missing} missing -> null)"
                )
            else:
                lines.append(f"  {fname}: {matched}/{total} matched")
        return "\n".join(lines)

    def validate(self) -> bool:
        """Re-run audit on the output (should always pass)."""
        return self.manifest.get("audit", {}).get("passed", False)

    def explain(self) -> str:
        """Return the join logic explanation."""
        return self.sql

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows = []
        for fname, fstats in self.stats.feature_stats.items():
            matched = fstats.get("matched", 0)
            missing = fstats.get("missing", 0)
            total = matched + missing
            status = "OK" if missing == 0 else "OK (nulls)"
            color = "#2ecc71"
            rows.append(
                f"<tr><td style='color:{color};font-weight:bold'>{status}</td>"
                f"<td>{fname}</td><td>{matched:,}/{total:,}</td>"
                f"<td>{missing:,}</td></tr>"
            )
        audit_status = (
            "PASSED" if self.manifest.get("audit", {}).get("passed") else "FAILED"
        )
        audit_color = "#2ecc71" if audit_status == "PASSED" else "#e74c3c"
        return (
            f"<div style='font-family:monospace;max-width:700px'>"
            f"<h3>Timefence Build Result</h3>"
            f"<p>{self.stats.row_count:,} rows, {self.stats.column_count} columns "
            f"in {self.stats.duration_seconds:.1f}s</p>"
            f"<p>Audit: <span style='color:{audit_color};font-weight:bold'>{audit_status}</span></p>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"<tr style='background:#f5f5f5'><th style='border:1px solid #ddd;padding:6px'>Status</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Feature</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Matched</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Missing</th></tr>"
            f"{''.join(rows)}</table></div>"
        )


# ---------------------------------------------------------------------------
# Audit Report
# ---------------------------------------------------------------------------


@dataclass
class FeatureAuditDetail:
    name: str
    leaky_row_count: int = 0
    leaky_row_pct: float = 0.0
    max_leakage: timedelta | None = None
    median_leakage: timedelta | None = None
    severity: str = "OK"
    total_rows: int = 0
    null_rows: int = 0
    clean: bool = True
    leaky_rows: Any = None  # DataFrame of violating rows when available


@dataclass
class AuditReport:
    features: dict[str, FeatureAuditDetail] = field(default_factory=dict)
    total_rows: int = 0
    mode: str = "rebuild"

    @property
    def has_leakage(self) -> bool:
        return any(not d.clean for d in self.features.values())

    @property
    def clean_features(self) -> list[str]:
        return [n for n, d in self.features.items() if d.clean]

    @property
    def leaky_features(self) -> list[str]:
        return [n for n, d in self.features.items() if not d.clean]

    def __getitem__(self, key: str) -> FeatureAuditDetail:
        return self.features[key]

    def assert_clean(self) -> None:
        if self.has_leakage:
            leaky = ", ".join(self.leaky_features)
            raise TimefenceLeakageError(
                f"Temporal leakage detected in features: {leaky}"
            )

    def to_json(self, path: str) -> None:
        data = {
            "has_leakage": self.has_leakage,
            "total_rows": self.total_rows,
            "mode": self.mode,
            "features": {},
        }
        for name, detail in self.features.items():
            data["features"][name] = {
                "clean": detail.clean,
                "leaky_row_count": detail.leaky_row_count,
                "leaky_row_pct": detail.leaky_row_pct,
                "max_leakage_seconds": (
                    detail.max_leakage.total_seconds() if detail.max_leakage else None
                ),
                "median_leakage_seconds": (
                    detail.median_leakage.total_seconds()
                    if detail.median_leakage
                    else None
                ),
                "severity": detail.severity,
                "total_rows": detail.total_rows,
                "null_rows": detail.null_rows,
            }
        Path(path).write_text(json.dumps(data, indent=2))

    def to_html(self, path: str) -> None:
        rows_html = []
        for name, detail in self.features.items():
            status = "CLEAN" if detail.clean else "LEAK"
            color = "#2ecc71" if detail.clean else "#e74c3c"
            rows_html.append(
                f"<tr><td style='color:{color};font-weight:bold'>{status}</td>"
                f"<td>{name}</td>"
                f"<td>{detail.leaky_row_count}</td>"
                f"<td>{detail.leaky_row_pct:.1%}</td>"
                f"<td>{detail.severity}</td></tr>"
            )
        html = f"""<!DOCTYPE html>
<html><head><title>Timefence Audit Report</title>
<style>body{{font-family:monospace;max-width:800px;margin:40px auto;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}
th{{background:#f5f5f5;}}</style></head>
<body><h1>Timefence Temporal Audit Report</h1>
<p>Scanned {self.total_rows} rows</p>
<table><tr><th>Status</th><th>Feature</th><th>Leaky Rows</th><th>%</th><th>Severity</th></tr>
{"".join(rows_html)}</table></body></html>"""
        Path(path).write_text(html)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows_html = []
        for name, detail in self.features.items():
            if detail.clean:
                status = "CLEAN"
                color = "#2ecc71"
            else:
                status = "LEAK"
                color = "#e74c3c"
            rows_html.append(
                f"<tr><td style='color:{color};font-weight:bold'>{status}</td>"
                f"<td>{name}</td>"
                f"<td>{detail.leaky_row_count:,}</td>"
                f"<td>{detail.leaky_row_pct:.1%}</td>"
                f"<td>{detail.severity}</td></tr>"
            )
        header_color = "#e74c3c" if self.has_leakage else "#2ecc71"
        header_text = "LEAKAGE DETECTED" if self.has_leakage else "ALL CLEAN"
        return (
            f"<div style='font-family:monospace;max-width:700px'>"
            f"<h3>Timefence Temporal Audit Report</h3>"
            f"<p>Scanned {self.total_rows:,} rows — "
            f"<span style='color:{header_color};font-weight:bold'>{header_text}</span></p>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"<tr style='background:#f5f5f5'>"
            f"<th style='border:1px solid #ddd;padding:6px'>Status</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Feature</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Leaky Rows</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>%</th>"
            f"<th style='border:1px solid #ddd;padding:6px'>Severity</th></tr>"
            f"{''.join(rows_html)}</table></div>"
        )

    def __str__(self) -> str:
        lines = ["TEMPORAL AUDIT REPORT", f"Scanned {self.total_rows} rows"]
        if self.has_leakage:
            leaky_count = len(self.leaky_features)
            total_count = len(self.features)
            lines.append(
                f"WARNING: LEAKAGE DETECTED in {leaky_count} of {total_count} features"
            )
        else:
            lines.append("ALL CLEAN - no temporal leakage detected")
        lines.append("")
        for name, detail in self.features.items():
            if detail.clean:
                null_info = f", {detail.null_rows} null" if detail.null_rows else ""
                lines.append(
                    f"  OK  {name} - clean ({detail.total_rows} rows{null_info})"
                )
            else:
                lines.append(f"  LEAK  {name}")
                lines.append(
                    f"        {detail.leaky_row_count} rows ({detail.leaky_row_pct:.1%}) use feature data from the future"
                )
                if detail.max_leakage:
                    lines.append(
                        f"        Max leakage: {_format_leakage(detail.max_leakage)}"
                    )
                if detail.median_leakage:
                    lines.append(
                        f"        Median leakage: {_format_leakage(detail.median_leakage)}"
                    )
                lines.append(f"        Severity: {detail.severity}")
        return "\n".join(lines)


def _format_leakage(td: timedelta) -> str:
    days = td.days
    if days > 0:
        return f"{days} day{'s' if days != 1 else ''}"
    hours = int(td.total_seconds() // 3600)
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    minutes = int(td.total_seconds() // 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def _classify_severity(pct: float, max_leakage: timedelta | None) -> str:
    if max_leakage and max_leakage.days > SEVERITY_HIGH_DAYS:
        return "HIGH"
    if pct > SEVERITY_HIGH_PCT:
        return "HIGH"
    if pct > SEVERITY_MEDIUM_PCT or (
        max_leakage and max_leakage.days >= SEVERITY_MEDIUM_DAYS
    ):
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Explain Result
# ---------------------------------------------------------------------------


@dataclass
class ExplainResult:
    plan: list[dict[str, Any]] = field(default_factory=list)
    label_count: int = 0

    def __str__(self) -> str:
        lines = [f"JOIN PLAN for {self.label_count} label rows", ""]
        lines.append("For each label row (keys, label_time):")
        lines.append("")
        for i, item in enumerate(self.plan, 1):
            lines.append(f"  {i}. {item['name']}")
            lines.append(f"     Source:  {item['source']}")
            lines.append(f"     Join:    {item['join_condition']}")
            lines.append(f"     Window:  {item['window']}")
            embargo_str = item.get("embargo_str", "none")
            lines.append(f"     Embargo: {embargo_str}")
            lines.append(f"     Strategy: {item.get('strategy', 'row_number')}")
            lines.append("     SQL:")
            for sql_line in item["sql"].split("\n"):
                lines.append(f"       {sql_line}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diff Result
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    old_rows: int = 0
    new_rows: int = 0
    schema_changes: list[dict[str, str]] = field(default_factory=list)
    value_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["BUILD DIFF", ""]
        lines.append("Rows")
        row_diff = self.new_rows - self.old_rows
        sign = "+" if row_diff >= 0 else ""
        lines.append(
            f"  old: {self.old_rows}    new: {self.new_rows}    ({sign}{row_diff})"
        )
        lines.append("")
        if self.schema_changes:
            lines.append("Schema")
            for change in self.schema_changes:
                lines.append(
                    f"  {change['type']} {change['column']}    {change.get('detail', '')}"
                )
            lines.append("")
        if self.value_changes:
            lines.append("Value Changes")
            for col, stats in self.value_changes.items():
                changed = stats.get("changed_count", 0)
                pct = stats.get("changed_pct", 0)
                lines.append(f"  {col}: {changed} values changed ({pct:.1%})")
                if "mean_delta" in stats:
                    lines.append(f"    Mean delta: {stats['mean_delta']:.3f}")
                if "max_delta" in stats:
                    lines.append(f"    Max delta: {stats['max_delta']:.3f}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL safety helpers
# ---------------------------------------------------------------------------


def _qi(name: str) -> str:
    """Quote a SQL identifier (column name, table name) for DuckDB.

    Wraps the name in double quotes and escapes any internal double quotes.
    Example: my col -> "my col", it"s -> "it""s"
    """
    return '"' + name.replace('"', '""') + '"'


def _ql(value: str | Path) -> str:
    """Quote a value as a SQL single-quoted string literal.

    Wraps in single quotes and escapes any internal single quotes.
    Example: it's -> 'it''s'
    """
    return "'" + str(value).replace("'", "''") + "'"


def _safe_name(name: str) -> str:
    """Sanitize a string for use in SQL table/alias names.

    Replaces non-alphanumeric characters (except underscores) with underscores.
    """
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "_unnamed"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_hash_safe(path: Path | None, store: Any) -> str | None:
    """Compute content hash if path and store are available."""
    if path is None:
        return None
    import hashlib

    try:
        if store is not None and hasattr(store, "cached_content_hash"):
            return store.cached_content_hash(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except (OSError, TypeError) as exc:
        logger.debug("Content hash failed for %s: %s", path, exc)
        return None


def _definition_hash(feat: Feature) -> str:
    """Compute a hash of the feature definition for cache keying."""
    import hashlib

    input_str = feat.definition_hash_input
    return f"sha256:{hashlib.sha256(input_str.encode()).hexdigest()[:CACHE_KEY_LENGTH]}"


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _os_identifier() -> str:
    import platform

    return f"{platform.system().lower()}-{platform.machine()}"


def _load_data_as_table(
    conn: duckdb.DuckDBPyConnection,
    data: str | Path | Any,
    table_name: str,
) -> None:
    """Load a file path or DataFrame into a DuckDB temp table."""
    if isinstance(data, (str, Path)):
        conn.execute(
            f"CREATE TEMP TABLE {table_name} AS SELECT * FROM read_parquet({_ql(data)})"
        )
    else:
        _register_df(conn, data, f"{table_name}_df")
        conn.execute(f"CREATE TEMP TABLE {table_name} AS SELECT * FROM {table_name}_df")


def _register_source(
    conn: duckdb.DuckDBPyConnection, source: Source | SQLSource, table_name: str
) -> None:
    """Register a source as a DuckDB table."""
    if isinstance(source, SQLSource):
        conn.execute(f"CREATE OR REPLACE TEMP TABLE {table_name} AS ({source.query})")
    elif source.df is not None:
        _register_df(conn, source.df, table_name)
    elif source.format == "parquet":
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
            f"SELECT * FROM read_parquet({_ql(source.path)})"
        )
    elif source.format == "csv":
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
            f"SELECT * FROM read_csv({_ql(source.path)}, delim={_ql(source.delimiter)})"
        )
    else:
        raise TimefenceValidationError(f"Unsupported source format: {source.format}")


def _register_df(conn: duckdb.DuckDBPyConnection, df: Any, table_name: str) -> None:
    """Register a DataFrame via Arrow PyCapsule protocol or direct registration."""
    if hasattr(df, "__arrow_c_stream__"):
        conn.register(table_name, df)
    else:
        try:
            conn.register(table_name, df)
        except (TypeError, duckdb.Error) as exc:
            raise TimefenceValidationError(
                f"Cannot register DataFrame of type {type(df).__name__}. "
                "Try converting to Arrow with .to_arrow() or saving to .parquet first."
            ) from exc


def _validate_source_schema(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    feature: Feature,
    label_keys: list[str],
) -> None:
    """Validate that source has required columns."""
    columns = [col[0] for col in conn.execute(f"DESCRIBE {_qi(table_name)}").fetchall()]

    source_keys = feature.source_keys
    for key in source_keys:
        if key not in columns:
            raise schema_error_missing_key(feature.name, source_keys, columns)

    ts = feature.source.timestamp
    if ts not in columns:
        raise TimefenceSchemaError(
            f"Feature '{feature.name}' source is missing timestamp column '{ts}'.\n\n"
            f"  Available columns: {columns}\n"
        )

    if feature.mode == "columns":
        for src_col in feature._columns:
            if src_col not in columns:
                raise TimefenceSchemaError(
                    f"Feature '{feature.name}' references column '{src_col}' "
                    f"which does not exist in source '{feature.source.name}'.\n\n"
                    f"  Available columns: {columns}\n"
                )


def _validate_timezones(
    conn: duckdb.DuckDBPyConnection,
    label_time_col: str,
    feature: Feature,
    feat_table: str,
) -> None:
    """Check for timezone mismatches between labels and features."""
    try:
        label_type = conn.execute(
            f"SELECT typeof({_qi(label_time_col)}) FROM __labels LIMIT 1"
        ).fetchone()
        feat_type = conn.execute(
            f"SELECT typeof(feature_time) FROM {feat_table} LIMIT 1"
        ).fetchone()
        if label_type is None or feat_type is None:
            return

        label_type_str = label_type[0].upper()
        feat_type_str = feat_type[0].upper()

        label_tz_aware = (
            "WITH TIME ZONE" in label_type_str or "TIMESTAMPTZ" in label_type_str
        )
        feat_tz_aware = (
            "WITH TIME ZONE" in feat_type_str or "TIMESTAMPTZ" in feat_type_str
        )

        if label_tz_aware != feat_tz_aware:
            label_sample = conn.execute(
                f"SELECT {_qi(label_time_col)} FROM __labels WHERE {_qi(label_time_col)} IS NOT NULL LIMIT 1"
            ).fetchone()
            feat_sample = conn.execute(
                f"SELECT feature_time FROM {feat_table} WHERE feature_time IS NOT NULL LIMIT 1"
            ).fetchone()
            raise timezone_error(
                feature.name,
                "UTC" if label_tz_aware else None,
                "UTC" if feat_tz_aware else None,
                str(label_sample[0]) if label_sample else "N/A",
                str(feat_sample[0]) if feat_sample else "N/A",
            )
    except TimefenceTimezoneError:
        raise
    except (duckdb.Error, TypeError, IndexError) as exc:
        logger.debug("Skipping timezone check for %s: %s", feature.name, exc)


def _check_duplicates(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    feature: Feature,
) -> None:
    """Check for duplicate (key, timestamp) pairs in feature table."""
    key_cols = ", ".join(_qi(k) for k in feature.source_keys)
    ts = _qi(feature.source.timestamp)

    count_sql = f"""
        SELECT COUNT(*) FROM (
            SELECT {key_cols}, {ts}, COUNT(*) as cnt
            FROM {_qi(table_name)}
            GROUP BY {key_cols}, {ts}
            HAVING cnt > 1
        ) t
    """
    dup_count = conn.execute(count_sql).fetchone()[0]

    if dup_count > 0:
        if feature.on_duplicate == "error":
            examples_sql = f"""
                SELECT {key_cols}, {ts}, COUNT(*) as cnt
                FROM {_qi(table_name)}
                GROUP BY {key_cols}, {ts}
                HAVING cnt > 1
                ORDER BY cnt DESC
                LIMIT 3
            """
            col_names = [_qi(k) for k in feature.source_keys] + [ts, "cnt"]
            examples = [
                dict(zip(col_names, row))
                for row in conn.execute(examples_sql).fetchall()
            ]
            raise duplicate_error(feature.name, dup_count, examples)
        else:
            warnings.warn(
                f"Feature '{feature.name}' has {dup_count} duplicate "
                f"(key, feature_time) pairs. Using on_duplicate='keep_any' — "
                f"one row will be selected arbitrarily.",
                stacklevel=3,
            )


def _validate_splits(
    splits: dict[str, tuple[str, str]],
    conn: duckdb.DuckDBPyConnection,
    label_time_col: str,
) -> None:
    """Validate that split ranges are non-overlapping and warn about gaps."""
    sorted_splits = sorted(splits.items(), key=lambda x: x[1][0])

    for i in range(len(sorted_splits) - 1):
        name_a, (_, end_a) = sorted_splits[i]
        name_b, (start_b, _) = sorted_splits[i + 1]
        if end_a > start_b:
            raise TimefenceConfigError(
                f"Split ranges overlap: '{name_a}' ends at {end_a} "
                f"but '{name_b}' starts at {start_b}."
            )
        if end_a < start_b:
            warnings.warn(
                f"Gap between splits '{name_a}' (ends {end_a}) "
                f"and '{name_b}' (starts {start_b}). "
                f"Labels in this range will not appear in any split.",
                stacklevel=3,
            )

    # Warn if splits don't cover full label range
    try:
        time_range = conn.execute(
            f"SELECT MIN({_qi(label_time_col)}), MAX({_qi(label_time_col)}) FROM __labels"
        ).fetchone()
        if time_range and sorted_splits:
            first_start = sorted_splits[0][1][0]
            last_end = sorted_splits[-1][1][1]
            min_label = str(time_range[0])[:19]
            max_label = str(time_range[1])[:19]
            if first_start > min_label:
                warnings.warn(
                    f"Splits start at {first_start} but labels start at {min_label}.",
                    stacklevel=3,
                )
            if last_end < max_label:
                warnings.warn(
                    f"Splits end at {last_end} but labels extend to {max_label}.",
                    stacklevel=3,
                )
    except (duckdb.Error, ValueError, KeyError) as exc:
        logger.debug("Could not validate split boundaries: %s", exc)


def _compute_feature_table(
    conn: duckdb.DuckDBPyConnection,
    feat: Feature,
    src_table: str,
    feat_table: str,
) -> tuple[str, list[str]]:
    """Compute a feature table from source. Returns (feat_table, output_cols).

    Handles all three modes (columns, sql, transform) and introspects the
    resulting table to determine output columns. Shared between build() and audit().
    """
    all_sql: list[str] = []

    if feat.mode == "transform":
        result_rel = feat._transform(conn, src_table)  # noqa: F841
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE {feat_table} AS SELECT * FROM result_rel"
        )
    elif feat.mode == "sql":
        feat_sql, _ = _build_feature_sql(feat, src_table)
        create_sql = (
            f"CREATE OR REPLACE TEMP TABLE {feat_table} AS SELECT * FROM {feat_sql}"
        )
        conn.execute(create_sql)
        all_sql.append(create_sql)
    else:
        feat_sql, cols = _build_feature_sql(feat, src_table)
        create_sql = f"CREATE OR REPLACE TEMP TABLE {feat_table} AS {feat_sql}"
        conn.execute(create_sql)
        all_sql.append(create_sql)
        return all_sql, cols

    # For sql/transform modes, introspect the table to get output columns
    feat_cols = [c[0] for c in conn.execute(f"DESCRIBE {feat_table}").fetchall()]
    output_cols = [
        c for c in feat_cols if c != "feature_time" and c not in feat.source_keys
    ]
    return all_sql, output_cols


def _build_feature_sql(
    feature: Feature,
    source_table: str,
) -> tuple[str, list[str]]:
    """Generate SQL for a feature's computation. Returns (sql, output_column_names)."""
    key_cols = ", ".join(_qi(k) for k in feature.source_keys)
    ts = _qi(feature.source.timestamp)

    if feature.mode == "columns":
        select_cols = []
        output_cols = []
        for src_col, out_col in feature._columns.items():
            select_cols.append(
                f"{_qi(src_col)} AS {_qi(out_col)}"
                if src_col != out_col
                else _qi(src_col)
            )
            output_cols.append(out_col)

        sql = (
            f"SELECT {key_cols}, {ts} AS feature_time, {', '.join(select_cols)} "
            f"FROM {source_table}"
        )
        return sql, output_cols

    elif feature.mode == "sql":
        user_sql = feature._sql_text.replace("{source}", source_table)
        sql = f"({user_sql})"
        return sql, []

    else:
        return "", []


# ---------------------------------------------------------------------------
# Join strategy selection
# ---------------------------------------------------------------------------


def _use_asof_strategy(feature: Feature) -> bool:
    """ASOF JOIN is used when no embargo shifts the upper bound."""
    return feature.embargo.total_seconds() == 0


def _build_row_number_join_sql(
    feature: Feature,
    feat_table: str,
    label_keys: list[str],
    label_time_col: str,
    join_mode: str,
    max_lookback: timedelta,
    max_staleness: timedelta | None,
    feat_output_cols: list[str],
) -> str:
    """Generate ROW_NUMBER-based join SQL (handles all cases)."""
    prefix = feature.name
    safe_prefix = _safe_name(prefix)
    lt = _qi(label_time_col)

    key_conditions = []
    for lk in label_keys:
        sk = feature.key_mapping.get(lk, lk)
        key_conditions.append(f"f.{_qi(sk)} = l.{_qi(lk)}")
    key_join = " AND ".join(key_conditions)

    embargo_interval = duration_to_sql_interval(feature.embargo)
    lookback_interval = duration_to_sql_interval(max_lookback)
    temporal_op = "<" if join_mode == "strict" else "<="

    if feature.embargo.total_seconds() > 0:
        upper_bound = f"f.feature_time {temporal_op} l.{lt} - {embargo_interval}"
    else:
        upper_bound = f"f.feature_time {temporal_op} l.{lt}"

    lower_bound = f"f.feature_time >= l.{lt} - {lookback_interval}"

    col_selects = []
    col_names = []
    for col in feat_output_cols:
        namespaced = f"{prefix}__{col}"
        col_selects.append(f"f.{_qi(col)} AS {_qi(namespaced)}")
        col_names.append(_qi(namespaced))
    ft_col_name = f"{prefix}__feature_time"
    col_selects.append(f"f.feature_time AS {_qi(ft_col_name)}")
    select_clause = ", ".join(col_selects)

    staleness_filter = ""
    if max_staleness is not None:
        staleness_interval = duration_to_sql_interval(max_staleness)
        staleness_filter = (
            f"\n      AND f.feature_time >= l.{lt} - {staleness_interval}"
        )

    return f"""CREATE OR REPLACE TEMP TABLE __joined_{safe_prefix} AS
    WITH ranked AS (
        SELECT
            l.__label_rowid,
            {select_clause},
            ROW_NUMBER() OVER (
                PARTITION BY l.__label_rowid
                ORDER BY f.feature_time DESC
            ) AS __rn
        FROM __labels l
        LEFT JOIN {feat_table} f
            ON {key_join}
           AND {upper_bound}
           AND {lower_bound}{staleness_filter}
    )
    SELECT __label_rowid, {", ".join(col_names)}, {_qi(ft_col_name)}
    FROM ranked
    WHERE __rn = 1 OR __rn IS NULL"""


def _build_join_sql(
    feature: Feature,
    feat_table: str,
    label_keys: list[str],
    label_time_col: str,
    join_mode: str,
    max_lookback: timedelta,
    max_staleness: timedelta | None,
    feat_output_cols: list[str],
) -> tuple[str, str]:
    """Generate point-in-time join SQL. Returns (sql, strategy_name).

    Uses ASOF JOIN when possible (no embargo), falls back to ROW_NUMBER.
    """
    if _use_asof_strategy(feature):
        sql = _build_asof_join_sql_impl(
            feature,
            feat_table,
            label_keys,
            label_time_col,
            join_mode,
            max_lookback,
            max_staleness,
            feat_output_cols,
        )
        return sql, "asof"
    else:
        sql = _build_row_number_join_sql(
            feature,
            feat_table,
            label_keys,
            label_time_col,
            join_mode,
            max_lookback,
            max_staleness,
            feat_output_cols,
        )
        return sql, "row_number"


def _build_asof_join_sql_impl(
    feature: Feature,
    feat_table: str,
    label_keys: list[str],
    label_time_col: str,
    join_mode: str,
    max_lookback: timedelta,
    max_staleness: timedelta | None,
    feat_output_cols: list[str],
) -> str:
    """Generate ASOF JOIN SQL (faster, used when embargo == 0)."""
    prefix = feature.name
    safe_prefix = _safe_name(prefix)
    lt = _qi(label_time_col)
    lookback_interval = duration_to_sql_interval(max_lookback)

    # Equality conditions
    on_parts = []
    for lk in label_keys:
        sk = feature.key_mapping.get(lk, lk)
        on_parts.append(f"l.{_qi(lk)} = f.{_qi(sk)}")

    # ASOF condition
    asof_op = ">" if join_mode == "strict" else ">="
    on_parts.append(f"l.{lt} {asof_op} f.feature_time")
    on_clause = " AND ".join(on_parts)

    # Validity: must be within lookback (and staleness if set)
    valid_parts = [f"f.feature_time >= l.{lt} - {lookback_interval}"]
    if max_staleness is not None:
        staleness_interval = duration_to_sql_interval(max_staleness)
        valid_parts.append(f"f.feature_time >= l.{lt} - {staleness_interval}")
    valid_check = " AND ".join(valid_parts)

    select_parts = ["l.__label_rowid"]
    col_names = []
    for col in feat_output_cols:
        namespaced = f"{prefix}__{col}"
        col_names.append(namespaced)
        select_parts.append(
            f"CASE WHEN {valid_check} THEN f.{_qi(col)} ELSE NULL END AS {_qi(namespaced)}"
        )
    ft_col_name = f"{prefix}__feature_time"
    select_parts.append(
        f"CASE WHEN {valid_check} THEN f.feature_time ELSE NULL END "
        f"AS {_qi(ft_col_name)}"
    )

    return (
        f"CREATE OR REPLACE TEMP TABLE __joined_{safe_prefix} AS\n"
        f"SELECT {', '.join(select_parts)}\n"
        f"FROM __labels l\n"
        f"ASOF LEFT JOIN {feat_table} f\n"
        f"    ON {on_clause}"
    )


# ---------------------------------------------------------------------------
# Public API: build
# ---------------------------------------------------------------------------


def build(
    labels: Labels,
    features: Sequence[Feature | FeatureSet],
    output: str | Path | None = None,
    *,
    max_lookback: str | timedelta = DEFAULT_MAX_LOOKBACK,
    max_staleness: str | timedelta | None = None,
    join: str = "strict",
    on_missing: str = DEFAULT_ON_MISSING,
    splits: dict[str, tuple[str, str]] | None = None,
    store: Store | None = None,
    flatten_columns: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BuildResult:
    """Build a point-in-time correct training set.

    Args:
        progress: Optional callback invoked with a status message at each step.
            Useful for progress bars. Called with messages like "Loading labels",
            "Computing feature_name", "Joining feature_name", "Writing output".
    """
    start_time = time.time()

    def _emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    max_lookback_td = parse_duration(max_lookback) or timedelta(
        days=DEFAULT_MAX_LOOKBACK_DAYS
    )
    max_staleness_td = parse_duration(max_staleness)

    if join not in ("strict", "inclusive"):
        raise TimefenceConfigError(
            f"join must be 'strict' or 'inclusive', got '{join}'."
        )
    if on_missing not in ("null", "skip"):
        raise TimefenceConfigError(
            f"on_missing must be 'null' or 'skip', got '{on_missing}'."
        )

    flat_features = flatten_features(features)

    # Validate feature names are unique (both exact and after sanitization)
    seen_names: dict[str, int] = {}
    seen_safe: dict[str, list[str]] = {}  # safe_name -> [original names]
    for feat in flat_features:
        seen_names[feat.name] = seen_names.get(feat.name, 0) + 1
        safe = _safe_name(feat.name)
        seen_safe.setdefault(safe, []).append(feat.name)
    duplicates = {n: c for n, c in seen_names.items() if c > 1}
    if duplicates:
        dup_str = ", ".join(f"'{n}' (x{c})" for n, c in duplicates.items())
        raise TimefenceConfigError(
            f"Duplicate feature names: {dup_str}.\n\n"
            "  Each feature must have a unique name. Duplicate names would cause\n"
            "  one feature to silently overwrite another.\n\n"
            "  Fix: Set an explicit name on each feature:\n"
            '    timefence.Feature(..., name="unique_name")\n'
        )
    collisions = {s: names for s, names in seen_safe.items() if len(set(names)) > 1}
    if collisions:
        pairs = [f"{sorted(set(names))}" for names in collisions.values()]
        raise TimefenceConfigError(
            f"Feature names collide after sanitization: {', '.join(pairs)}.\n\n"
            "  These names are distinct but map to the same internal identifier,\n"
            "  which would cause one feature to silently overwrite another.\n\n"
            "  Fix: Rename features to avoid ambiguity (e.g., use underscores consistently).\n"
        )

    for feat in flat_features:
        if feat.embargo >= max_lookback_td:
            from timefence.errors import config_error_embargo_lookback

            raise config_error_embargo_lookback(
                format_duration(feat.embargo) or "0d",
                format_duration(max_lookback_td) or DEFAULT_MAX_LOOKBACK,
            )
        if max_staleness_td is not None and max_staleness_td <= feat.embargo:
            raise TimefenceConfigError(
                f"max_staleness ({format_duration(max_staleness_td)}) must be greater than "
                f"embargo ({format_duration(feat.embargo)}) for feature '{feat.name}'."
            )

    # Check build-level cache
    if store is not None and output is not None:
        label_hash = _content_hash_safe(labels.path, store)
        feat_cache_keys = []
        for feat in flat_features:
            src_hash = _content_hash_safe(feat.source.path, store)
            fck = store.feature_cache_key(
                _definition_hash(feat), src_hash, format_duration(feat.embargo)
            )
            feat_cache_keys.append(fck)

        bck = store.build_cache_key(
            label_hash,
            feat_cache_keys,
            format_duration(max_lookback_td),
            format_duration(max_staleness_td),
            join,
            on_missing,
        )
        cached_build = store.find_cached_build(bck)
        if cached_build is not None:
            elapsed = time.time() - start_time
            cached_build["duration_seconds"] = elapsed
            return BuildResult(
                output_path=cached_build.get("output", {}).get("path"),
                manifest=cached_build,
                stats=BuildStats(
                    row_count=cached_build.get("output", {}).get("row_count", 0),
                    column_count=cached_build.get("output", {}).get("column_count", 0),
                    feature_stats={
                        k: {
                            "matched": v.get("matched_rows", 0),
                            "missing": v.get("missing_rows", 0),
                            "cached": True,
                        }
                        for k, v in cached_build.get("features", {}).items()
                    },
                    duration_seconds=elapsed,
                ),
                sql="-- cached build",
            )

    conn = duckdb.connect()
    all_sql = []

    try:
        # Step 1: Register labels
        _emit("Loading labels")
        if labels.path is not None:
            _load_data_as_table(conn, labels.path, "__labels_raw")
        elif labels.df is not None:
            _load_data_as_table(conn, labels.df, "__labels_raw")
        else:
            raise TimefenceValidationError("Labels must have either path or df.")

        # Validate label schema
        label_cols = [
            col[0] for col in conn.execute("DESCRIBE __labels_raw").fetchall()
        ]
        for key in labels.keys:
            if key not in label_cols:
                raise TimefenceSchemaError(
                    f"Labels missing key column '{key}'.\n  Available: {label_cols}"
                )
        if labels.label_time not in label_cols:
            raise TimefenceSchemaError(
                f"Labels missing label_time column '{labels.label_time}'.\n  Available: {label_cols}"
            )

        # Add rowid for join tracking
        conn.execute(
            "CREATE TEMP TABLE __labels AS "
            "SELECT ROW_NUMBER() OVER () AS __label_rowid, * FROM __labels_raw"
        )
        label_count = conn.execute("SELECT COUNT(*) FROM __labels").fetchone()[0]
        logger.info(
            "Labels: %d rows, keys=%s, label_time=%s",
            label_count,
            labels.keys,
            labels.label_time,
        )

        # Get label time range for manifest
        time_range_row = conn.execute(
            f"SELECT MIN({_qi(labels.label_time)}), MAX({_qi(labels.label_time)}) FROM __labels"
        ).fetchone()
        label_time_range = (
            [str(time_range_row[0]), str(time_range_row[1])]
            if time_range_row and time_range_row[0] is not None
            else None
        )

        # Validate splits if provided
        if splits:
            _validate_splits(splits, conn, labels.label_time)

        # Step 2: Register sources and compute features
        registered_sources: dict[str, str] = {}
        feature_tables: dict[str, tuple[str, list[str]]] = {}
        feature_cache_keys: list[str] = []
        feature_cache_status: dict[str, bool] = {}  # name -> was_cached

        for i, feat in enumerate(flat_features, 1):
            _emit(f"Computing {feat.name} ({i}/{len(flat_features)})")
            src_name = feat.source.name
            if src_name not in registered_sources:
                table_name = f"__src_{_safe_name(src_name)}"
                _register_source(conn, feat.source, table_name)
                registered_sources[src_name] = table_name

            src_table = registered_sources[src_name]
            _validate_source_schema(conn, src_table, feat, labels.keys)
            _check_duplicates(conn, src_table, feat)

            feat_table = f"__feat_{_safe_name(feat.name)}"

            # Check feature-level cache
            cached = False
            fck = None
            if store is not None:
                src_hash = _content_hash_safe(feat.source.path, store)
                fck = store.feature_cache_key(
                    _definition_hash(feat), src_hash, format_duration(feat.embargo)
                )
                feature_cache_keys.append(fck)
                if store.has_feature_cache(feat.name, fck):
                    cache_path = store.feature_cache_path(feat.name, fck)
                    conn.execute(
                        f"CREATE OR REPLACE TEMP TABLE {feat_table} AS "
                        f"SELECT * FROM read_parquet({_ql(cache_path)})"
                    )
                    feat_cols = [
                        c[0] for c in conn.execute(f"DESCRIBE {feat_table}").fetchall()
                    ]
                    output_cols = [
                        c
                        for c in feat_cols
                        if c != "feature_time" and c not in feat.source_keys
                    ]
                    cached = True
                    feature_cache_status[feat.name] = True

            if not cached:
                feature_cache_status[feat.name] = False
                feat_sqls, output_cols = _compute_feature_table(
                    conn, feat, src_table, feat_table
                )
                all_sql.extend(feat_sqls)
                for s in feat_sqls:
                    logger.info("Feature SQL [%s]:\n  %s", feat.name, s)
            else:
                logger.info("Feature [%s]: loaded from cache", feat.name)

                # Save to feature cache
                if store is not None and fck is not None:
                    cache_path = store.feature_cache_path(feat.name, fck)
                    try:
                        conn.execute(
                            f"COPY (SELECT * FROM {feat_table}) TO {_ql(cache_path)} (FORMAT PARQUET)"
                        )
                    except (duckdb.Error, OSError) as exc:
                        logger.warning(
                            "Feature cache write failed for %s: %s", feat.name, exc
                        )

            feature_tables[feat.name] = (feat_table, output_cols)

            # Timezone validation
            if output_cols:
                _validate_timezones(conn, labels.label_time, feat, feat_table)

        # Step 3: Point-in-time joins
        for i, feat in enumerate(flat_features, 1):
            _emit(f"Joining {feat.name} ({i}/{len(flat_features)})")
            feat_table, output_cols = feature_tables[feat.name]
            join_sql, strategy = _build_join_sql(
                feat,
                feat_table,
                labels.keys,
                labels.label_time,
                join,
                max_lookback_td,
                max_staleness_td,
                output_cols,
            )
            logger.info(
                "Join SQL [%s] (strategy=%s):\n  %s", feat.name, strategy, join_sql
            )
            try:
                conn.execute(join_sql)
            except duckdb.Error as exc:
                # ASOF fallback: if ASOF fails, retry with ROW_NUMBER
                if strategy == "asof":
                    logger.debug(
                        "ASOF JOIN failed for %s, falling back to ROW_NUMBER: %s",
                        feat.name,
                        exc,
                    )
                    join_sql = _build_row_number_join_sql(
                        feat,
                        feat_table,
                        labels.keys,
                        labels.label_time,
                        join,
                        max_lookback_td,
                        max_staleness_td,
                        output_cols,
                    )
                    conn.execute(join_sql)
                    strategy = "row_number"
                else:
                    raise
            all_sql.append(join_sql)

        # Step 4: Combine all joins
        key_cols = ", ".join(f"l.{_qi(k)}" for k in labels.keys)
        target_cols = ", ".join(f"l.{_qi(t)}" for t in labels.target)
        join_clauses = []
        select_cols = [key_cols, f"l.{_qi(labels.label_time)}", target_cols]

        for feat in flat_features:
            prefix = feat.name
            safe_prefix = _safe_name(prefix)
            _, output_cols = feature_tables[feat.name]
            for col in output_cols:
                select_cols.append(f"j_{safe_prefix}.{_qi(f'{prefix}__{col}')}")
            join_clauses.append(
                f"LEFT JOIN __joined_{safe_prefix} j_{safe_prefix} "
                f"ON l.__label_rowid = j_{safe_prefix}.__label_rowid"
            )

        order_cols = (
            ", ".join(f"l.{_qi(k)}" for k in labels.keys)
            + f", l.{_qi(labels.label_time)}"
        )
        final_sql = (
            f"SELECT {', '.join(select_cols)} "
            f"FROM __labels l "
            f"{' '.join(join_clauses)} "
            f"ORDER BY {order_cols}"
        )

        # Handle on_missing="skip"
        if on_missing == "skip":
            not_null_conditions = []
            for feat in flat_features:
                safe_pref = _safe_name(feat.name)
                _, output_cols = feature_tables[feat.name]
                for col in output_cols:
                    not_null_conditions.append(
                        f"j_{safe_pref}.{_qi(f'{feat.name}__{col}')} IS NOT NULL"
                    )
            if not_null_conditions:
                final_sql = (
                    f"SELECT {', '.join(select_cols)} "
                    f"FROM __labels l "
                    f"{' '.join(join_clauses)} "
                    f"WHERE {' AND '.join(not_null_conditions)} "
                    f"ORDER BY {order_cols}"
                )

        all_sql.append(final_sql)
        logger.info("Final SQL:\n  %s", final_sql)

        # Flatten column names if requested
        if flatten_columns:
            result_rel = conn.execute(final_sql)
            col_descriptions = result_rel.description
            seen: set[str] = set()
            can_flatten = True
            for desc in col_descriptions:
                name = desc[0]
                short = name.split("__", 1)[1] if "__" in name else name
                if short in seen:
                    can_flatten = False
                    break
                seen.add(short)

            if can_flatten:
                renames = []
                for desc in col_descriptions:
                    name = desc[0]
                    if "__" in name:
                        short = name.split("__", 1)[1]
                        renames.append(f"{_qi(name)} AS {_qi(short)}")
                    else:
                        renames.append(_qi(name))
                final_sql = f"SELECT {', '.join(renames)} FROM ({final_sql})"

        # Step 5: Write output
        _emit("Writing output")
        if output is not None:
            output = str(output)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            conn.execute(f"COPY ({final_sql}) TO {_ql(output)} (FORMAT PARQUET)")

        # Collect stats
        result_df = conn.execute(final_sql)
        result_cols = [desc[0] for desc in result_df.description]
        result_count = conn.execute(f"SELECT COUNT(*) FROM ({final_sql})").fetchone()[0]

        feature_stats = {}
        for feat in flat_features:
            prefix = feat.name
            _, output_cols = feature_tables[feat.name]
            if output_cols:
                first_col = f"{prefix}__{output_cols[0]}"
                if flatten_columns and output_cols[0] in result_cols:
                    first_col = output_cols[0]
                try:
                    null_count = conn.execute(
                        f"SELECT COUNT(*) FROM ({final_sql}) WHERE {_qi(first_col)} IS NULL"
                    ).fetchone()[0]
                except duckdb.Error as exc:
                    logger.debug(
                        "Could not compute null count for %s: %s", feat.name, exc
                    )
                    null_count = 0
                feature_stats[feat.name] = {
                    "matched": result_count - null_count,
                    "missing": null_count,
                    "cached": feature_cache_status.get(feat.name, False),
                }

        # Step 6: Post-build verification
        _emit("Verifying temporal correctness")
        audit_passed = True
        for feat in flat_features:
            prefix = feat.name
            safe_prefix = _safe_name(prefix)
            ft_col = _qi(f"{prefix}__feature_time")
            lt = _qi(labels.label_time)
            embargo_interval = duration_to_sql_interval(feat.embargo)

            if join == "strict":
                if feat.embargo.total_seconds() > 0:
                    check_sql = (
                        f"SELECT COUNT(*) FROM __joined_{safe_prefix} j "
                        f"JOIN __labels l ON j.__label_rowid = l.__label_rowid "
                        f"WHERE j.{ft_col} IS NOT NULL "
                        f"AND j.{ft_col} >= l.{lt} - {embargo_interval}"
                    )
                else:
                    check_sql = (
                        f"SELECT COUNT(*) FROM __joined_{safe_prefix} j "
                        f"JOIN __labels l ON j.__label_rowid = l.__label_rowid "
                        f"WHERE j.{ft_col} IS NOT NULL "
                        f"AND j.{ft_col} >= l.{lt}"
                    )
            else:
                if feat.embargo.total_seconds() > 0:
                    check_sql = (
                        f"SELECT COUNT(*) FROM __joined_{safe_prefix} j "
                        f"JOIN __labels l ON j.__label_rowid = l.__label_rowid "
                        f"WHERE j.{ft_col} IS NOT NULL "
                        f"AND j.{ft_col} > l.{lt} - {embargo_interval}"
                    )
                else:
                    check_sql = (
                        f"SELECT COUNT(*) FROM __joined_{safe_prefix} j "
                        f"JOIN __labels l ON j.__label_rowid = l.__label_rowid "
                        f"WHERE j.{ft_col} IS NOT NULL "
                        f"AND j.{ft_col} > l.{lt}"
                    )
            violations = conn.execute(check_sql).fetchone()[0]
            if violations > 0:
                audit_passed = False

        # Handle splits
        split_paths = None
        if splits and output:
            split_paths = {}
            output_path = Path(output)
            for split_name, (start, end) in splits.items():
                split_file = (
                    output_path.parent
                    / f"{output_path.stem}_{split_name}{output_path.suffix}"
                )
                split_sql = (
                    f"COPY (SELECT * FROM ({final_sql}) "
                    f"WHERE {_qi(labels.label_time)} >= {_ql(start)} "
                    f"AND {_qi(labels.label_time)} < {_ql(end)}) "
                    f"TO {_ql(split_file)} (FORMAT PARQUET)"
                )
                conn.execute(split_sql)
                split_paths[split_name] = split_file

        elapsed = time.time() - start_time

        stats = BuildStats(
            row_count=result_count,
            column_count=len(result_cols),
            feature_stats=feature_stats,
            duration_seconds=elapsed,
        )

        # Compute output file size
        output_file_size = None
        if output is not None:
            import contextlib

            with contextlib.suppress(OSError):
                output_file_size = Path(output).stat().st_size

        # Build manifest
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = {
            "timefence_version": __version__,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": elapsed,
            "labels": {
                "path": str(labels.path) if labels.path else None,
                "content_hash": _content_hash_safe(labels.path, store),
                "row_count": label_count,
                "time_range": label_time_range,
                "keys": labels.keys,
                "label_time_column": labels.label_time,
                "target_columns": labels.target,
            },
            "features": {},
            "parameters": {
                "max_lookback": format_duration(max_lookback_td),
                "max_staleness": format_duration(max_staleness_td),
                "join": join,
                "on_missing": on_missing,
            },
            "output": {
                "path": str(output) if output else None,
                "content_hash": _content_hash_safe(
                    Path(output) if output else None, store
                ),
                "row_count": result_count,
                "column_count": len(result_cols),
                "file_size_bytes": output_file_size,
            },
            "audit": {
                "passed": audit_passed,
                "invariant": f"feature_time {'<' if join == 'strict' else '<='} label_time - embargo",
                "rows_checked": result_count,
            },
            "environment": {
                "python_version": _python_version(),
                "duckdb_version": duckdb.__version__,
                "os": _os_identifier(),
            },
        }
        for feat in flat_features:
            fstats = feature_stats.get(feat.name, {})
            manifest["features"][feat.name] = {
                "definition_hash": _definition_hash(feat),
                "source_content_hash": _content_hash_safe(feat.source.path, store),
                "embargo": format_duration(feat.embargo),
                "matched_rows": fstats.get("matched", 0),
                "missing_rows": fstats.get("missing", 0),
                "output_columns": feature_tables[feat.name][1],
                "cached": feature_cache_status.get(feat.name, False),
            }

        # Store build cache key in manifest for future lookups
        if store is not None and feature_cache_keys:
            bck = store.build_cache_key(
                _content_hash_safe(labels.path, store),
                feature_cache_keys,
                format_duration(max_lookback_td),
                format_duration(max_staleness_td),
                join,
                on_missing,
            )
            manifest["build_cache_key"] = bck
            manifest_path = store.save_build(manifest)
            manifest["manifest_path"] = str(manifest_path)

        return BuildResult(
            output_path=str(output) if output else None,
            manifest=manifest,
            stats=stats,
            splits=split_paths,
            sql="\n\n".join(all_sql),
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API: audit
# ---------------------------------------------------------------------------


def audit(
    data: str | Path | Any,
    features: Sequence[Feature | FeatureSet] | None = None,
    *,
    keys: str | list[str] | None = None,
    label_time: str | None = None,
    feature_time_columns: dict[str, str] | None = None,
    max_lookback: str | timedelta = DEFAULT_MAX_LOOKBACK,
    max_staleness: str | timedelta | None = None,
    join: str = "strict",
) -> AuditReport:
    """Audit a dataset for temporal leakage.

    Two modes:
    1. Rebuild-and-compare: provide features, keys, label_time.
    2. Temporal check: provide feature_time_columns.
    """
    if feature_time_columns is not None:
        return _audit_temporal(data, feature_time_columns, label_time or "label_time")

    if features is None:
        raise TimefenceValidationError(
            "audit() requires either 'features' (for rebuild-and-compare) "
            "or 'feature_time_columns' (for temporal check)."
        )
    if keys is None or label_time is None:
        raise TimefenceValidationError(
            "audit() in rebuild-and-compare mode requires 'keys' and 'label_time'."
        )

    return _audit_rebuild(
        data,
        features,
        keys,
        label_time,
        max_lookback=max_lookback,
        max_staleness=max_staleness,
        join=join,
    )


def _audit_temporal_api(
    data: str | Path | Any,
    feature_time_columns: dict[str, str],
    label_time: str = "label_time",
) -> AuditReport:
    """Lightweight temporal check mode (public API: audit.temporal)."""
    return _audit_temporal(data, feature_time_columns, label_time)


audit.temporal = _audit_temporal_api  # type: ignore[attr-defined]


def _audit_temporal(
    data: str | Path | Any,
    feature_time_columns: dict[str, str],
    label_time: str,
) -> AuditReport:
    """Lightweight temporal check: feature_time < label_time for each row."""
    conn = duckdb.connect()
    try:
        _load_data_as_table(conn, data, "__audit_data")

        total = conn.execute("SELECT COUNT(*) FROM __audit_data").fetchone()[0]
        report = AuditReport(total_rows=total, mode="temporal")

        for feat_col, ft_col in feature_time_columns.items():
            qft = _qi(ft_col)
            qlt = _qi(label_time)
            leak_sql = (
                f"SELECT COUNT(*) FROM __audit_data "
                f"WHERE {qft} IS NOT NULL AND {qft} >= {qlt}"
            )
            leaky_count = conn.execute(leak_sql).fetchone()[0]

            if leaky_count > 0:
                stats_sql = (
                    f"SELECT MAX({qft} - {qlt}) as max_leak, "
                    f"MEDIAN({qft} - {qlt}) as med_leak "
                    f"FROM __audit_data "
                    f"WHERE {qft} >= {qlt}"
                )
                stats = conn.execute(stats_sql).fetchone()
                max_leak = stats[0] if stats[0] else None
                med_leak = stats[1] if stats[1] else None

                pct = leaky_count / total if total > 0 else 0

                # Capture leaky rows
                leaky_rows_df = None
                try:
                    leaky_rows_df = conn.execute(
                        f"SELECT * FROM __audit_data WHERE {qft} >= {qlt} LIMIT 1000"
                    ).fetchdf()
                except duckdb.Error as exc:
                    logger.debug(
                        "Could not capture leaky rows for %s: %s", feat_col, exc
                    )

                detail = FeatureAuditDetail(
                    name=feat_col,
                    leaky_row_count=leaky_count,
                    leaky_row_pct=pct,
                    max_leakage=max_leak,
                    median_leakage=med_leak,
                    severity=_classify_severity(pct, max_leak),
                    total_rows=total,
                    clean=False,
                    leaky_rows=leaky_rows_df,
                )
            else:
                null_count = conn.execute(
                    f"SELECT COUNT(*) FROM __audit_data WHERE {qft} IS NULL"
                ).fetchone()[0]
                detail = FeatureAuditDetail(
                    name=feat_col,
                    total_rows=total,
                    null_rows=null_count,
                    clean=True,
                )
            report.features[feat_col] = detail

        return report
    finally:
        conn.close()


def _audit_rebuild(
    data: str | Path | Any,
    features: Sequence[Feature | FeatureSet],
    keys: str | list[str],
    label_time: str,
    *,
    max_lookback: str | timedelta = DEFAULT_MAX_LOOKBACK,
    max_staleness: str | timedelta | None = None,
    join: str = "strict",
) -> AuditReport:
    """Rebuild-and-compare audit mode."""
    import tempfile

    keys_list = [keys] if isinstance(keys, str) else list(keys)
    flat_features = flatten_features(features)
    max_lookback_td = parse_duration(max_lookback) or timedelta(
        days=DEFAULT_MAX_LOOKBACK_DAYS
    )
    max_staleness_td = parse_duration(max_staleness)

    conn = duckdb.connect()
    try:
        # Load existing dataset
        _load_data_as_table(conn, data, "__existing")

        total = conn.execute("SELECT COUNT(*) FROM __existing").fetchone()[0]
        existing_cols = [c[0] for c in conn.execute("DESCRIBE __existing").fetchall()]

        # Extract label spine
        key_cols = ", ".join(_qi(k) for k in keys_list)
        possible_targets = [
            c
            for c in existing_cols
            if c not in keys_list and c != label_time and "__" not in c
        ]
        target = possible_targets[:1] if possible_targets else ["__dummy"]

        with tempfile.TemporaryDirectory() as tmpdir:
            synth_labels_path = Path(tmpdir) / "synth_labels.parquet"

            if target[0] == "__dummy":
                conn.execute(
                    f"COPY (SELECT {key_cols}, {_qi(label_time)}, 1 as __dummy FROM __existing) "
                    f"TO {_ql(synth_labels_path)} (FORMAT PARQUET)"
                )
            else:
                target_select = ", ".join(_qi(t) for t in target)
                conn.execute(
                    f"COPY (SELECT {key_cols}, {_qi(label_time)}, {target_select} FROM __existing) "
                    f"TO {_ql(synth_labels_path)} (FORMAT PARQUET)"
                )

            conn.execute(
                f"CREATE OR REPLACE TEMP TABLE __labels_raw AS "
                f"SELECT * FROM read_parquet({_ql(synth_labels_path)})"
            )
            conn.execute(
                "CREATE OR REPLACE TEMP TABLE __labels AS "
                "SELECT ROW_NUMBER() OVER () AS __label_rowid, * FROM __labels_raw"
            )

            registered: dict[str, str] = {}
            feat_tables: dict[str, tuple[str, list[str]]] = {}

            for feat in flat_features:
                src_name = feat.source.name
                if src_name not in registered:
                    tbl = f"__src_{_safe_name(src_name)}"
                    _register_source(conn, feat.source, tbl)
                    registered[src_name] = tbl

                src_tbl = registered[src_name]
                feat_tbl = f"__feat_{_safe_name(feat.name)}"

                _, out_cols = _compute_feature_table(conn, feat, src_tbl, feat_tbl)

                feat_tables[feat.name] = (feat_tbl, out_cols)

                join_sql, _ = _build_join_sql(
                    feat,
                    feat_tbl,
                    keys_list,
                    label_time,
                    join,
                    max_lookback_td,
                    max_staleness_td,
                    out_cols,
                )
                try:
                    conn.execute(join_sql)
                except duckdb.Error as exc:
                    # Fallback to ROW_NUMBER
                    logger.debug(
                        "ASOF JOIN failed for %s in audit, falling back to ROW_NUMBER: %s",
                        feat.name,
                        exc,
                    )
                    join_sql = _build_row_number_join_sql(
                        feat,
                        feat_tbl,
                        keys_list,
                        label_time,
                        join,
                        max_lookback_td,
                        max_staleness_td,
                        out_cols,
                    )
                    conn.execute(join_sql)

            # Compare existing vs correct
            report = AuditReport(total_rows=total, mode="rebuild")

            conn.execute(
                "CREATE OR REPLACE TEMP TABLE __existing_numbered AS "
                "SELECT ROW_NUMBER() OVER () AS __rowid, * FROM __existing"
            )

            for feat in flat_features:
                prefix = feat.name
                safe_prefix = _safe_name(prefix)
                _, out_cols = feat_tables[feat.name]

                matching_cols = []
                for col in out_cols:
                    namespaced = f"{prefix}__{col}"
                    if namespaced in existing_cols:
                        matching_cols.append((namespaced, namespaced))
                    elif col in existing_cols:
                        matching_cols.append((col, f"{prefix}__{col}"))

                if not matching_cols:
                    report.features[feat.name] = FeatureAuditDetail(
                        name=feat.name,
                        total_rows=total,
                        clean=True,
                    )
                    continue

                leaky_count = 0
                leaky_rows_df = None
                for exist_col, correct_col in matching_cols:
                    # numpy.allclose-style: |a - b| > atol + rtol * |b|
                    try:
                        compare_sql = (
                            f"SELECT COUNT(*) FROM __existing_numbered e "
                            f"JOIN __joined_{safe_prefix} c ON e.__rowid = c.__label_rowid "
                            f"WHERE e.{_qi(exist_col)} IS NOT NULL "
                            f"AND c.{_qi(correct_col)} IS NOT NULL "
                            f"AND ABS(CAST(e.{_qi(exist_col)} AS DOUBLE) - CAST(c.{_qi(correct_col)} AS DOUBLE)) "
                            f"> {DEFAULT_ATOL} + {DEFAULT_RTOL} * ABS(CAST(c.{_qi(correct_col)} AS DOUBLE))"
                        )
                        diff_count = conn.execute(compare_sql).fetchone()[0]
                    except (duckdb.Error, duckdb.ConversionException):
                        # Non-numeric: exact string comparison
                        compare_sql = (
                            f"SELECT COUNT(*) FROM __existing_numbered e "
                            f"JOIN __joined_{safe_prefix} c ON e.__rowid = c.__label_rowid "
                            f"WHERE e.{_qi(exist_col)} IS NOT NULL "
                            f"AND c.{_qi(correct_col)} IS NOT NULL "
                            f"AND CAST(e.{_qi(exist_col)} AS VARCHAR) != CAST(c.{_qi(correct_col)} AS VARCHAR)"
                        )
                        diff_count = conn.execute(compare_sql).fetchone()[0]

                    if diff_count > leaky_count:
                        leaky_count = diff_count
                        # Capture leaky rows
                        try:
                            leaky_rows_df = conn.execute(
                                f"SELECT e.* FROM __existing_numbered e "
                                f"JOIN __joined_{safe_prefix} c ON e.__rowid = c.__label_rowid "
                                f"WHERE e.{_qi(exist_col)} IS NOT NULL "
                                f"AND c.{_qi(correct_col)} IS NOT NULL "
                                f"AND CAST(e.{_qi(exist_col)} AS VARCHAR) != CAST(c.{_qi(correct_col)} AS VARCHAR) "
                                f"LIMIT 1000"
                            ).fetchdf()
                        except duckdb.Error as exc:
                            logger.debug(
                                "Could not capture leaky rows for %s: %s",
                                feat.name,
                                exc,
                            )

                if leaky_count > 0:
                    pct = leaky_count / total if total > 0 else 0
                    ft_col = _qi(f"{prefix}__feature_time")
                    try:
                        leak_stats = conn.execute(
                            f"SELECT MAX(l.{_qi(label_time)} - j.{ft_col}), "
                            f"MEDIAN(l.{_qi(label_time)} - j.{ft_col}) "
                            f"FROM __joined_{safe_prefix} j "
                            f"JOIN __labels l ON j.__label_rowid = l.__label_rowid "
                            f"WHERE j.{ft_col} IS NOT NULL"
                        ).fetchone()
                        max_leak = leak_stats[0] if leak_stats else None
                        med_leak = leak_stats[1] if leak_stats else None
                    except duckdb.Error as exc:
                        logger.debug(
                            "Could not compute leak stats for %s: %s", feat.name, exc
                        )
                        max_leak = None
                        med_leak = None

                    detail = FeatureAuditDetail(
                        name=feat.name,
                        leaky_row_count=leaky_count,
                        leaky_row_pct=pct,
                        max_leakage=max_leak,
                        median_leakage=med_leak,
                        severity=_classify_severity(pct, max_leak),
                        total_rows=total,
                        clean=False,
                        leaky_rows=leaky_rows_df,
                    )
                else:
                    null_count = 0
                    if out_cols:
                        try:
                            null_count = conn.execute(
                                f"SELECT COUNT(*) FROM __joined_{safe_prefix} "
                                f"WHERE {_qi(f'{prefix}__{out_cols[0]}')} IS NULL"
                            ).fetchone()[0]
                        except duckdb.Error as exc:
                            logger.debug(
                                "Could not compute null count for %s: %s",
                                feat.name,
                                exc,
                            )
                    detail = FeatureAuditDetail(
                        name=feat.name,
                        total_rows=total,
                        null_rows=null_count,
                        clean=True,
                    )
                report.features[feat.name] = detail

        return report
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API: explain
# ---------------------------------------------------------------------------


def explain(
    labels: Labels,
    features: Sequence[Feature | FeatureSet],
    *,
    max_lookback: str | timedelta = DEFAULT_MAX_LOOKBACK,
    max_staleness: str | timedelta | None = None,
    join: str = "strict",
) -> ExplainResult:
    """Preview join logic without executing."""
    max_lookback_td = parse_duration(max_lookback) or timedelta(
        days=DEFAULT_MAX_LOOKBACK_DAYS
    )
    flat_features = flatten_features(features)

    conn = duckdb.connect()
    try:
        if labels.path is not None:
            label_count = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet({_ql(labels.path)})"
            ).fetchone()[0]
        elif labels.df is not None:
            conn.register("__lbl", labels.df)
            label_count = conn.execute("SELECT COUNT(*) FROM __lbl").fetchone()[0]
        else:
            label_count = 0
    finally:
        conn.close()

    result = ExplainResult(label_count=label_count)

    for feat in flat_features:
        embargo_str = format_duration(feat.embargo) or "none"
        lookback_str = format_duration(max_lookback_td)
        strategy = "asof" if _use_asof_strategy(feat) else "row_number"

        op = "<" if join == "strict" else "<="

        if feat.embargo.total_seconds() > 0:
            join_cond = f"feature_time {op} label_time - INTERVAL '{embargo_str}'"
            window = f"[label_time - {lookback_str}, label_time - {embargo_str})"
        else:
            join_cond = f"feature_time {op} label_time"
            window = f"[label_time - {lookback_str}, label_time)"

        source_ref = str(feat.source.path) if feat.source.path else feat.source.name
        key_placeholder = "{K}"
        time_placeholder = "{T}"

        if feat.mode == "columns":
            cols = ", ".join(feat._columns.values())
            ts = feat.source.timestamp
            key_col = feat.source_keys[0]
            embargo_clause = (
                f" - INTERVAL '{embargo_str}'"
                if feat.embargo.total_seconds() > 0
                else ""
            )
            example_sql = (
                f"SELECT {key_col}, {ts} AS feature_time, {cols}\n"
                f"FROM '{source_ref}'\n"
                f"WHERE {key_col} = {key_placeholder}\n"
                f"  AND {ts} {op} {time_placeholder}{embargo_clause}\n"
                f"  AND {ts} >= {time_placeholder} - INTERVAL '{lookback_str}'\n"
                f"ORDER BY {ts} DESC\nLIMIT 1"
            )
        elif feat.mode == "sql":
            example_sql = f"WITH feature AS (\n  {feat._sql_text.strip()}\n)\nSELECT * FROM feature\n..."
        else:
            example_sql = f"-- Python transform: {feat._transform.__name__}"

        result.plan.append(
            {
                "name": feat.name,
                "source": source_ref,
                "join_condition": join_cond,
                "window": window,
                "embargo_str": (
                    embargo_str if feat.embargo.total_seconds() > 0 else "none"
                ),
                "strategy": strategy,
                "sql": example_sql,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Public API: diff
# ---------------------------------------------------------------------------


def diff(
    old: str | Path,
    new: str | Path,
    *,
    keys: str | list[str],
    label_time: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> DiffResult:
    """Compare two training datasets."""
    keys_list = [keys] if isinstance(keys, str) else list(keys)

    conn = duckdb.connect()
    try:
        conn.execute(
            f"CREATE TEMP TABLE __old AS SELECT * FROM read_parquet({_ql(old)})"
        )
        conn.execute(
            f"CREATE TEMP TABLE __new AS SELECT * FROM read_parquet({_ql(new)})"
        )

        old_count = conn.execute("SELECT COUNT(*) FROM __old").fetchone()[0]
        new_count = conn.execute("SELECT COUNT(*) FROM __new").fetchone()[0]

        old_cols = {c[0] for c in conn.execute("DESCRIBE __old").fetchall()}
        new_cols = {c[0] for c in conn.execute("DESCRIBE __new").fetchall()}

        result = DiffResult(old_rows=old_count, new_rows=new_count)

        meta_cols = set(keys_list) | {label_time}
        added = new_cols - old_cols
        removed = old_cols - new_cols
        common = (old_cols & new_cols) - meta_cols

        for col in sorted(added):
            result.schema_changes.append(
                {"type": "+", "column": col, "detail": "(new column)"}
            )
        for col in sorted(removed):
            result.schema_changes.append(
                {"type": "-", "column": col, "detail": "(removed)"}
            )

        key_join = " AND ".join(f"o.{_qi(k)} = n.{_qi(k)}" for k in keys_list)
        key_join += f" AND o.{_qi(label_time)} = n.{_qi(label_time)}"

        for col in sorted(common):
            qc = _qi(col)
            try:
                # Try tolerance-aware numeric comparison first
                try:
                    change_sql = (
                        f"SELECT COUNT(*) FROM __old o JOIN __new n ON {key_join} "
                        f"WHERE o.{qc} IS NOT NULL AND n.{qc} IS NOT NULL "
                        f"AND ABS(CAST(o.{qc} AS DOUBLE) - CAST(n.{qc} AS DOUBLE)) "
                        f"> {atol} + {rtol} * ABS(CAST(n.{qc} AS DOUBLE))"
                    )
                    changed = conn.execute(change_sql).fetchone()[0]
                    # Also count null-vs-non-null differences
                    null_diff_sql = (
                        f"SELECT COUNT(*) FROM __old o JOIN __new n ON {key_join} "
                        f"WHERE (o.{qc} IS NULL) != (n.{qc} IS NULL)"
                    )
                    changed += conn.execute(null_diff_sql).fetchone()[0]
                except (duckdb.Error, duckdb.ConversionException):
                    # Non-numeric: fall back to exact comparison
                    change_sql = (
                        f"SELECT COUNT(*) FROM __old o JOIN __new n ON {key_join} "
                        f"WHERE o.{qc} IS DISTINCT FROM n.{qc}"
                    )
                    changed = conn.execute(change_sql).fetchone()[0]

                if changed > 0:
                    joined = min(old_count, new_count)
                    pct = changed / joined if joined > 0 else 0

                    stats_entry: dict[str, Any] = {
                        "changed_count": changed,
                        "changed_pct": pct,
                    }

                    # Compute numeric delta stats when possible
                    try:
                        delta_sql = (
                            f"SELECT "
                            f"AVG(CAST(n.{qc} AS DOUBLE) - CAST(o.{qc} AS DOUBLE)), "
                            f"MAX(ABS(CAST(n.{qc} AS DOUBLE) - CAST(o.{qc} AS DOUBLE))) "
                            f"FROM __old o JOIN __new n ON {key_join} "
                            f"WHERE o.{qc} IS DISTINCT FROM n.{qc}"
                        )
                        delta_row = conn.execute(delta_sql).fetchone()
                        if delta_row and delta_row[0] is not None:
                            stats_entry["mean_delta"] = float(delta_row[0])
                            stats_entry["max_delta"] = float(delta_row[1])
                    except (duckdb.Error, TypeError):
                        pass  # Non-numeric column, delta stats not applicable

                    result.value_changes[col] = stats_entry
                    result.schema_changes.append(
                        {
                            "type": "~",
                            "column": col,
                            "detail": f"{changed} values changed ({pct:.1%})",
                        }
                    )
                else:
                    result.schema_changes.append(
                        {"type": "=", "column": col, "detail": "unchanged"}
                    )
            except (duckdb.Error, TypeError) as exc:
                logger.warning("Column comparison failed for %s: %s", col, exc)
                result.schema_changes.append(
                    {"type": "?", "column": col, "detail": "comparison failed"}
                )

        return result
    finally:
        conn.close()
