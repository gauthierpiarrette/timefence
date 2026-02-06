"""Timefence: Temporal correctness layer for ML training data."""

from timefence._version import __version__
from timefence.core import (
    CSVSource,
    Feature,
    FeatureSet,
    Labels,
    ParquetSource,
    Source,
    SQLSource,
)
from timefence.engine import audit, build, diff, explain
from timefence.store import Store


def from_dbt(
    manifest_path: str = "target/manifest.json",
    **kwargs,
) -> list[Feature]:
    """Import feature definitions from a dbt project.

    Requires the ``timefence[dbt]`` extra::

        pip install timefence[dbt]

    Args:
        manifest_path: Path to the dbt ``manifest.json`` file.
        **kwargs: Additional options forwarded to the dbt adapter.

    Raises:
        NotImplementedError: Always, until the dbt integration is shipped.
    """
    raise NotImplementedError(
        "dbt integration is not yet available. "
        "Install the optional extra with: pip install timefence[dbt]"
    )


__all__ = [
    "CSVSource",
    "Feature",
    "FeatureSet",
    "Labels",
    "ParquetSource",
    "SQLSource",
    "Source",
    "Store",
    "__version__",
    "audit",
    "build",
    "diff",
    "explain",
    "from_dbt",
]
