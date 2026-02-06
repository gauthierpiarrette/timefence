"""Shared constants for the Timefence engine."""

# Numeric comparison tolerances (numpy.allclose-style)
DEFAULT_ATOL: float = 1e-10
DEFAULT_RTOL: float = 1e-7

# Temporal defaults
DEFAULT_MAX_LOOKBACK: str = "365d"
DEFAULT_MAX_LOOKBACK_DAYS: int = 365

# Default join and missing strategies
DEFAULT_JOIN_MODE: str = "strict"
DEFAULT_ON_MISSING: str = "null"

# Severity classification thresholds
SEVERITY_HIGH_PCT: float = 0.05
SEVERITY_MEDIUM_PCT: float = 0.01
SEVERITY_HIGH_DAYS: int = 7
SEVERITY_MEDIUM_DAYS: int = 1

# Store defaults
DEFAULT_STORE_PATH: str = ".timefence"

# Cache key hash truncation length
CACHE_KEY_LENGTH: int = 16
