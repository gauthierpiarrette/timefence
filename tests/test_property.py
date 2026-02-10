"""Property-based tests for the core temporal invariant.

These tests use Hypothesis to verify that for ANY randomly generated data
and ANY join parameters, the core invariant always holds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import timefence
from timefence.engine import build


def _create_test_data(
    tmp_path: Path,
    n_entities: int,
    feature_times: list[datetime],
    label_times: list[datetime],
) -> tuple[Path, Path]:
    """Create test Parquet files with specific timestamps."""
    conn = duckdb.connect()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    try:
        # Create features
        rows = []
        for i, ft in enumerate(feature_times):
            entity = (i % n_entities) + 1
            rows.append((entity, ft, float(i)))

        if rows:
            conn.execute(
                "CREATE TEMP TABLE feat_data (user_id INTEGER, ts TIMESTAMP, val DOUBLE)"
            )
            conn.executemany("INSERT INTO feat_data VALUES (?, ?, ?)", rows)
            conn.execute(
                f"COPY feat_data TO '{data_dir}/features.parquet' (FORMAT PARQUET)"
            )
        else:
            conn.execute(f"""
                COPY (SELECT 1 AS user_id, TIMESTAMP '2020-01-01' AS ts, 0.0 AS val WHERE false)
                TO '{data_dir}/features.parquet' (FORMAT PARQUET)
            """)

        # Create labels
        label_rows = []
        for i, lt in enumerate(label_times):
            entity = (i % n_entities) + 1
            label_rows.append((entity, lt, i % 2 == 0))

        if label_rows:
            conn.execute(
                "CREATE TEMP TABLE label_data (user_id INTEGER, label_time TIMESTAMP, target BOOLEAN)"
            )
            conn.executemany("INSERT INTO label_data VALUES (?, ?, ?)", label_rows)
            conn.execute(
                f"COPY label_data TO '{data_dir}/labels.parquet' (FORMAT PARQUET)"
            )
        else:
            conn.execute(f"""
                COPY (SELECT 1 AS user_id, TIMESTAMP '2020-01-01' AS label_time, true AS target WHERE false)
                TO '{data_dir}/labels.parquet' (FORMAT PARQUET)
            """)

    finally:
        conn.close()

    return data_dir / "features.parquet", data_dir / "labels.parquet"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@pytest.mark.slow
class TestTemporalInvariant:
    """Property tests for the core temporal correctness invariant."""

    @given(
        n_features=st.integers(min_value=1, max_value=20),
        n_labels=st.integers(min_value=1, max_value=20),
        embargo_hours=st.integers(min_value=0, max_value=168),
        inclusive=st.booleans(),
    )
    @settings(max_examples=5000, deadline=60000)
    def test_no_leakage_ever(
        self, tmp_path_factory, n_features, n_labels, embargo_hours, inclusive
    ):
        """For ANY randomly generated data and ANY join mode, the invariant holds."""
        tmp_path = tmp_path_factory.mktemp("prop")
        embargo = timedelta(hours=embargo_hours)
        join_mode = "inclusive" if inclusive else "strict"

        # Generate deterministic timestamps
        base = datetime(2024, 1, 1)
        feature_times = [base + timedelta(hours=i * 7) for i in range(n_features)]
        label_times = [
            base + timedelta(days=30) + timedelta(hours=i * 11) for i in range(n_labels)
        ]

        feat_path, label_path = _create_test_data(
            tmp_path,
            n_entities=5,
            feature_times=feature_times,
            label_times=label_times,
        )

        src = timefence.Source(
            path=str(feat_path), keys=["user_id"], timestamp="ts", name="feat_src"
        )
        feat = timefence.Feature(
            source=src,
            columns=["val"],
            name="test_val",
            embargo=embargo,
            on_duplicate="keep_any",
        )
        labels = timefence.Labels(
            path=str(label_path),
            keys=["user_id"],
            label_time="label_time",
            target=["target"],
        )

        # Ensure embargo < lookback
        lookback = timedelta(days=365)
        assume(embargo < lookback)

        output = str(tmp_path / "out.parquet")
        build(
            labels=labels,
            features=[feat],
            output=output,
            join=join_mode,
            max_lookback="365d",
        )

        # THE INVARIANT: verify on the output
        conn = duckdb.connect()
        try:
            conn.execute(
                f"CREATE TEMP TABLE __result AS SELECT * FROM read_parquet('{output}')"
            )
            df = conn.execute("SELECT * FROM __result").fetchall()
            cols = [d[0] for d in conn.execute("DESCRIBE __result").fetchall()]

            ft_idx = None
            for i, c in enumerate(cols):
                if "feature_time" in c:
                    ft_idx = i
                    break

            lt_idx = cols.index("label_time")

            for row in df:
                ft = row[ft_idx] if ft_idx is not None else None
                lt = row[lt_idx]
                if ft is not None and lt is not None:
                    if inclusive:
                        assert ft <= lt - embargo, (
                            f"Invariant violated: feature_time={ft} should be <= "
                            f"label_time={lt} - embargo={embargo}"
                        )
                    else:
                        assert ft < lt - embargo, (
                            f"Invariant violated: feature_time={ft} should be < "
                            f"label_time={lt} - embargo={embargo}"
                        )
        finally:
            conn.close()

    @given(
        n_rows=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=5000, deadline=60000)
    def test_build_then_audit_roundtrip(self, tmp_path_factory, n_rows):
        """A dataset built by Timefence always passes its own temporal audit."""
        tmp_path = tmp_path_factory.mktemp("roundtrip")

        base = datetime(2024, 1, 1)
        feature_times = [base + timedelta(hours=i * 5) for i in range(n_rows)]
        label_times = [
            base + timedelta(days=60) + timedelta(hours=i * 3) for i in range(n_rows)
        ]

        feat_path, label_path = _create_test_data(
            tmp_path,
            n_entities=max(1, n_rows // 3),
            feature_times=feature_times,
            label_times=label_times,
        )

        src = timefence.Source(
            path=str(feat_path), keys=["user_id"], timestamp="ts", name="src"
        )
        feat = timefence.Feature(
            source=src,
            columns=["val"],
            name="val",
            on_duplicate="keep_any",
        )
        labels = timefence.Labels(
            path=str(label_path),
            keys=["user_id"],
            label_time="label_time",
            target=["target"],
        )

        output = str(tmp_path / "out.parquet")
        result = build(labels=labels, features=[feat], output=output)

        # Audit should pass
        assert result.manifest["audit"]["passed"] is True
