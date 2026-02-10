"""Tests for SQL safety helpers against adversarial inputs."""

from __future__ import annotations

import pytest

from timefence.engine import _qi, _ql, _safe_name


class TestQuoteIdentifier:
    """_qi() must produce valid DuckDB identifiers for any input."""

    def test_simple_name(self):
        assert _qi("user_id") == '"user_id"'

    def test_name_with_spaces(self):
        assert _qi("my column") == '"my column"'

    def test_name_with_double_quotes(self):
        assert _qi('col"name') == '"col""name"'

    def test_sql_reserved_word(self):
        assert _qi("select") == '"select"'

    def test_empty_string(self):
        assert _qi("") == '""'

    def test_name_with_single_quotes(self):
        assert _qi("it's") == '"it\'s"'

    def test_semicolon_injection(self):
        result = _qi('"; DROP TABLE users; --')
        assert result == '"""; DROP TABLE users; --"'
        # The leading " is escaped to "", so the identifier stays intact

    def test_unicode(self):
        assert _qi("名前") == '"名前"'

    def test_newlines(self):
        result = _qi("col\nname")
        assert '"' in result  # Still wrapped in quotes


class TestQuoteLiteral:
    """_ql() must produce valid DuckDB string literals for any input."""

    def test_simple_path(self):
        assert _ql("/data/users.parquet") == "'/data/users.parquet'"

    def test_path_with_single_quote(self):
        assert _ql("/data/it's.parquet") == "'/data/it''s.parquet'"

    def test_multiple_single_quotes(self):
        assert _ql("a'b'c") == "'a''b''c'"

    def test_sql_injection_attempt(self):
        result = _ql("'; DROP TABLE users; --")
        assert result == "'''; DROP TABLE users; --'"
        # The escaped quote prevents breaking out of the literal

    def test_empty_string(self):
        assert _ql("") == "''"

    def test_path_object(self):
        from pathlib import Path

        result = _ql(Path("/tmp/data.parquet"))
        assert result.startswith("'")
        assert result.endswith("'")
        assert "data.parquet" in result

    def test_backslashes(self):
        result = _ql("C:\\Users\\data.parquet")
        assert result == "'C:\\Users\\data.parquet'"

    def test_unicode_path(self):
        result = _ql("/données/résultats.parquet")
        assert result.startswith("'")


class TestSafeName:
    """_safe_name() must produce valid SQL identifiers without collisions on reasonable input."""

    def test_simple_name(self):
        assert _safe_name("users") == "users"

    def test_preserves_underscores(self):
        assert _safe_name("user_country") == "user_country"

    def test_replaces_hyphens(self):
        assert _safe_name("my-feature") == "my_feature"

    def test_replaces_dots(self):
        assert _safe_name("schema.table") == "schema_table"

    def test_replaces_spaces(self):
        assert _safe_name("my feature") == "my_feature"

    def test_empty_string(self):
        assert _safe_name("") == "_unnamed"

    def test_special_characters(self):
        result = _safe_name("feat@#$%")
        assert all(c.isalnum() or c == "_" for c in result)

    def test_preserves_alphanumeric(self):
        assert _safe_name("feature123") == "feature123"

    @pytest.mark.parametrize(
        "a, b",
        [
            ("my-feat", "my_feat"),
            ("feat.v1", "feat_v1"),
            ("a b", "a_b"),
        ],
    )
    def test_known_collisions(self, a, b):
        """Document that certain name pairs collide after sanitization."""
        assert _safe_name(a) == _safe_name(b)
