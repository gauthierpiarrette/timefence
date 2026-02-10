"""Duration parsing utilities for human-readable time intervals."""

from __future__ import annotations

import re
from datetime import timedelta

_PATTERN = re.compile(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_duration(value: str | timedelta | None) -> timedelta | None:
    """Parse a duration string like '30d', '1d12h', '365d' into a timedelta.

    Accepts:
        - None → None
        - timedelta → pass through
        - "0" or "0d" → timedelta(0)
        - "30d" → 30 days
        - "1d12h" → 1 day 12 hours
        - "6h" → 6 hours
        - "30m" → 30 minutes
    """
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value

    value = value.strip()
    if value == "0":
        return timedelta(0)

    match = _PATTERN.match(value)
    if not match:
        raise ValueError(
            f"Invalid duration '{value}'. "
            "Expected format like '30d', '6h', '1d12h', '365d'."
        )

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def format_duration(td: timedelta | None) -> str | None:
    """Format a timedelta back to a human-readable string."""
    if td is None:
        return None
    total_seconds = int(td.total_seconds())
    if total_seconds == 0:
        return "0d"
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    return "".join(parts)


def duration_to_sql_interval(td: timedelta) -> str:
    """Convert a timedelta to a DuckDB INTERVAL expression."""
    total_seconds = int(td.total_seconds())
    if total_seconds == 0:
        return "INTERVAL '0' SECOND"
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} DAY")
    if hours:
        parts.append(f"{hours} HOUR")
    if minutes:
        parts.append(f"{minutes} MINUTE")
    if seconds:
        parts.append(f"{seconds} SECOND")
    # DuckDB supports compound intervals via addition
    return " + ".join(f"INTERVAL '{p}'" for p in parts)
