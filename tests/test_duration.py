"""Tests for duration parsing utilities."""

from datetime import timedelta

import pytest

from timefence._duration import (
    duration_to_sql_interval,
    format_duration,
    parse_duration,
)


class TestParseDuration:
    def test_none(self):
        assert parse_duration(None) is None

    def test_timedelta_passthrough(self):
        td = timedelta(days=5)
        assert parse_duration(td) is td

    def test_days(self):
        assert parse_duration("30d") == timedelta(days=30)

    def test_hours(self):
        assert parse_duration("6h") == timedelta(hours=6)

    def test_combined(self):
        assert parse_duration("1d12h") == timedelta(days=1, hours=12)

    def test_minutes(self):
        assert parse_duration("30m") == timedelta(minutes=30)

    def test_seconds(self):
        assert parse_duration("45s") == timedelta(seconds=45)

    def test_zero(self):
        assert parse_duration("0") == timedelta(0)

    def test_zero_d(self):
        assert parse_duration("0d") == timedelta(0)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_whitespace(self):
        assert parse_duration("  30d  ") == timedelta(days=30)

    def test_full_combo(self):
        assert parse_duration("2d3h15m30s") == timedelta(
            days=2, hours=3, minutes=15, seconds=30
        )


class TestFormatDuration:
    def test_none(self):
        assert format_duration(None) is None

    def test_zero(self):
        assert format_duration(timedelta(0)) == "0d"

    def test_days(self):
        assert format_duration(timedelta(days=30)) == "30d"

    def test_hours(self):
        assert format_duration(timedelta(hours=6)) == "6h"

    def test_combined(self):
        assert format_duration(timedelta(days=1, hours=12)) == "1d12h"


class TestDurationToSqlInterval:
    def test_zero(self):
        assert duration_to_sql_interval(timedelta(0)) == "INTERVAL '0' SECOND"

    def test_days(self):
        assert duration_to_sql_interval(timedelta(days=30)) == "INTERVAL '30 DAY'"

    def test_hours(self):
        assert duration_to_sql_interval(timedelta(hours=6)) == "INTERVAL '6 HOUR'"

    def test_combined(self):
        result = duration_to_sql_interval(timedelta(days=1, hours=12))
        assert "1 DAY" in result
        assert "12 HOUR" in result
