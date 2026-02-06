"""Timefence error hierarchy.

Every error follows the format:
  WHAT happened → WHY it matters → WHERE (data) → HOW to fix
"""

from __future__ import annotations


class TimefenceError(Exception):
    """Base error for all Timefence operations."""


class TimefenceSchemaError(TimefenceError):
    """Schema validation failure (missing columns, type mismatches)."""


class TimefenceDuplicateError(TimefenceError):
    """Duplicate (key, feature_time) pairs detected."""


class TimefenceTimezoneError(TimefenceError):
    """Mixed timezone-aware and timezone-naive timestamps."""


class TimefenceConfigError(TimefenceError):
    """Invalid parameter combination or configuration."""


class TimefenceLeakageError(TimefenceError):
    """Temporal leakage detected (raised by report.assert_clean())."""


class TimefenceValidationError(TimefenceError):
    """General validation failure on inputs."""


def schema_error_missing_key(
    feature_name: str,
    expected_keys: list[str],
    actual_columns: list[str],
) -> TimefenceSchemaError:
    """Build a helpful schema error for missing key columns."""
    missing = [k for k in expected_keys if k not in actual_columns]
    similar = _find_similar(missing, actual_columns)

    msg = f"Feature '{feature_name}' is missing required key column(s): {missing}.\n\n"
    msg += "  Point-in-time joins require matching keys between labels and features.\n"
    msg += f"  Without {missing}, Timefence can't determine which feature rows belong to which entity.\n\n"
    msg += f"  Expected keys: {expected_keys}\n"
    msg += f"  Actual columns: {actual_columns}\n"

    if similar:
        for m, s in similar.items():
            msg += f"  '{s}' is similar to '{m}' — possible rename?\n"
        msg += "\n  Fix: Add key_mapping to your feature definition:\n"
        mapping = {k: similar[k] for k in missing if k in similar}
        msg += f"    key_mapping={mapping}\n"

    return TimefenceSchemaError(msg)


def duplicate_error(
    feature_name: str,
    count: int,
    examples: list[dict],
) -> TimefenceDuplicateError:
    """Build a helpful duplicate error with examples."""
    msg = (
        f"Feature '{feature_name}' has {count} duplicate (key, feature_time) pairs.\n\n"
    )
    msg += "  When multiple feature rows have the same key and timestamp, the\n"
    msg += (
        "  point-in-time join becomes non-deterministic. Timefence cannot guarantee\n"
    )
    msg += "  which row would be selected.\n\n"
    msg += f"  Example duplicates (showing first {min(3, len(examples))}):\n"
    for ex in examples[:3]:
        msg += f"    {ex}\n"
    msg += "\n  Fix (pick one):\n"
    msg += "    1. Deduplicate in your source data or SQL\n"
    msg += '    2. Set: timefence.Feature(..., on_duplicate="keep_any")\n'
    return TimefenceDuplicateError(msg)


def timezone_error(
    feature_name: str,
    label_tz: str | None,
    feature_tz: str | None,
    label_sample: str,
    feature_sample: str,
) -> TimefenceTimezoneError:
    """Build a helpful timezone mismatch error."""
    label_desc = f"timezone-aware ({label_tz})" if label_tz else "timezone-naive"
    feature_desc = f"timezone-aware ({feature_tz})" if feature_tz else "timezone-naive"

    msg = f"Mixed timezones between labels and feature '{feature_name}'.\n\n"
    msg += f"  Labels 'label_time' is {label_desc}.\n"
    msg += f"  Feature '{feature_name}' timestamp is {feature_desc}.\n\n"
    msg += "  Comparing these directly could shift joins by hours.\n\n"
    msg += "  Sample values:\n"
    msg += f"    label_time:   {label_sample}\n"
    msg += f"    feature_time: {feature_sample}\n"
    return TimefenceTimezoneError(msg)


def config_error_embargo_lookback(
    embargo: str, max_lookback: str
) -> TimefenceConfigError:
    """Build error for embargo >= max_lookback."""
    msg = f"embargo ({embargo}) must be less than max_lookback ({max_lookback}).\n\n"
    msg += "  When embargo equals or exceeds max_lookback, the join window is empty —\n"
    msg += (
        "  no feature can ever match. This is almost certainly a misconfiguration.\n\n"
    )
    msg += f"  Current: max_lookback={max_lookback}, embargo={embargo} → empty window\n"
    msg += f"  Likely intent: max_lookback=365d, embargo={embargo}\n\n"
    msg += "  Fix: Increase max_lookback or decrease embargo.\n"
    return TimefenceConfigError(msg)


def _find_similar(missing: list[str], candidates: list[str]) -> dict[str, str]:
    """Find candidates similar to missing names using simple substring matching."""
    result = {}
    for m in missing:
        m_lower = m.lower().replace("_", "")
        for c in candidates:
            c_lower = c.lower().replace("_", "")
            if m_lower in c_lower or c_lower in m_lower:
                result[m] = c
                break
    return result
