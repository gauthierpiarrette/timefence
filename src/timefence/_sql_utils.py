"""Shared SQL quoting and sanitization helpers for DuckDB."""

from __future__ import annotations

from pathlib import Path


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
