"""Tests for error formatting and helpers."""

from __future__ import annotations

from timefence.errors import (
    TimefenceConfigError,
    TimefenceDuplicateError,
    TimefenceError,
    TimefenceLeakageError,
    TimefenceSchemaError,
    TimefenceTimezoneError,
    TimefenceValidationError,
    config_error_embargo_lookback,
    duplicate_error,
    schema_error_missing_key,
    timezone_error,
)


class TestErrorHierarchy:
    def test_all_errors_inherit_from_timefence_error(self):
        for cls in [
            TimefenceSchemaError,
            TimefenceDuplicateError,
            TimefenceTimezoneError,
            TimefenceConfigError,
            TimefenceLeakageError,
            TimefenceValidationError,
        ]:
            assert issubclass(cls, TimefenceError)


class TestSchemaErrorMissingKey:
    def test_basic(self):
        err = schema_error_missing_key(
            "my_feature",
            ["user_id"],
            ["customer_id", "feature_time", "value"],
        )
        assert isinstance(err, TimefenceSchemaError)
        assert "my_feature" in str(err)
        assert "user_id" in str(err)

    def test_finds_similar(self):
        err = schema_error_missing_key(
            "my_feature",
            ["user_id"],
            ["customer_id", "userid", "value"],
        )
        msg = str(err)
        assert "similar" in msg.lower() or "key_mapping" in msg


class TestDuplicateError:
    def test_basic(self):
        err = duplicate_error(
            "spending",
            47,
            [{"keys": "user_42", "ts": "2024-03-15", "count": 2}],
        )
        assert isinstance(err, TimefenceDuplicateError)
        assert "47" in str(err)
        assert "spending" in str(err)

    def test_shows_fix(self):
        err = duplicate_error("feat", 10, [])
        msg = str(err)
        assert "keep_any" in msg


class TestTimezoneError:
    def test_basic(self):
        err = timezone_error(
            "daily_spend",
            "UTC",
            None,
            "2024-03-15 09:00:00+00:00",
            "2024-03-15 00:00:00",
        )
        assert isinstance(err, TimefenceTimezoneError)
        assert "timezone" in str(err).lower()


class TestConfigErrorEmbargoLookback:
    def test_basic(self):
        err = config_error_embargo_lookback("30d", "30d")
        assert isinstance(err, TimefenceConfigError)
        assert "embargo" in str(err)
        assert "max_lookback" in str(err)
