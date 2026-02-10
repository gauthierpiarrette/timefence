"""Timefence benchmark: measure build performance across data scales.

Usage:
    python benchmarks/bench.py              # Run all benchmarks
    python benchmarks/bench.py --quick      # Run small benchmarks only (<=1M, <=10 feats)
    python benchmarks/bench.py --include-pandas  # Also benchmark naive pandas joins
    python benchmarks/bench.py --runs 5     # Number of iterations per scenario (default: 3)

Results are printed as a table and saved to benchmarks/results.json.

Methodology:
    - Each scenario runs N iterations (default 3); we report median and stddev.
    - A warmup run is executed before the first timed iteration.
    - GC is disabled during timed runs to reduce noise.
    - Data generation uses a fixed seed for reproducibility.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import duckdb

# Ensure timefence is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import timefence

SEED = 42


# ---------------------------------------------------------------------------
# Data generation (pure DuckDB, fast even at 10M+)
# ---------------------------------------------------------------------------

def generate_data(
    tmp: Path,
    n_labels: int,
    n_features: int,
    n_entities: int | None = None,
) -> tuple[Path, list[Path]]:
    """Generate synthetic labels + feature parquet files.

    Uses a fixed seed for reproducible data generation.
    Returns (labels_path, [feature_paths...]).
    """
    if n_entities is None:
        n_entities = max(1000, n_labels // 5)

    conn = duckdb.connect()
    conn.execute(f"SELECT setseed({SEED / 100})")

    labels_path = tmp / "labels.parquet"
    conn.execute(f"""
        COPY (
            SELECT
                (i % {n_entities}) AS user_id,
                TIMESTAMP '2024-01-01' + INTERVAL (i * 86400 / {n_labels}) SECOND AS label_time,
                (i % 2 = 0) AS churned
            FROM generate_series(1, {n_labels}) t(i)
        ) TO '{labels_path.as_posix()}' (FORMAT PARQUET)
    """)

    feature_paths = []
    n_feature_rows = n_labels * 2
    for f_idx in range(n_features):
        fpath = tmp / f"feature_{f_idx}.parquet"
        conn.execute(f"""
            COPY (
                SELECT
                    (i % {n_entities}) AS user_id,
                    TIMESTAMP '2023-01-01' + INTERVAL (i * 86400 / {n_feature_rows}) SECOND AS updated_at,
                    RANDOM() AS val_{f_idx}
                FROM generate_series(1, {n_feature_rows}) t(i)
            ) TO '{fpath.as_posix()}' (FORMAT PARQUET)
        """)
        feature_paths.append(fpath)

    conn.close()
    return labels_path, feature_paths


# ---------------------------------------------------------------------------
# Timefence benchmark
# ---------------------------------------------------------------------------

def bench_timefence(
    labels_path: Path,
    feature_paths: list[Path],
    output_path: Path,
    *,
    embargo: str = "0d",
    max_lookback: str = "365d",
    max_staleness: str | None = None,
    splits: dict | None = None,
    n_runs: int = 3,
) -> dict:
    """Run timefence.build() with warmup + N timed runs. Return median timing."""
    sources = []
    features = []
    for i, fpath in enumerate(feature_paths):
        src = timefence.Source(path=str(fpath), keys=["user_id"], timestamp="updated_at")
        sources.append(src)
        feat = timefence.Feature(
            source=src,
            columns=[f"val_{i}"],
            embargo=embargo,
            name=f"feature_{i}",
        )
        features.append(feat)

    labels = timefence.Labels(
        path=str(labels_path),
        keys=["user_id"],
        label_time="label_time",
        target=["churned"],
    )

    build_kwargs = dict(
        labels=labels,
        features=features,
        output=str(output_path),
        max_lookback=max_lookback,
        max_staleness=max_staleness,
        splits=splits,
    )

    # Warmup run (not timed)
    result = timefence.build(**build_kwargs)

    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        result = timefence.build(**build_kwargs)
        elapsed = time.perf_counter() - t0
        gc.enable()
        times.append(elapsed)

    return {
        "engine": "timefence",
        "median_seconds": round(statistics.median(times), 2),
        "stddev_seconds": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
        "runs": n_runs,
        "rows_out": result.stats.row_count,
        "cols_out": result.stats.column_count,
    }


# ---------------------------------------------------------------------------
# Naive pandas benchmark (for comparison)
# ---------------------------------------------------------------------------

def bench_pandas(
    labels_path: Path,
    feature_paths: list[Path],
    output_path: Path,
    *,
    n_runs: int = 3,
) -> dict:
    """Naive pandas: merge_asof per feature, sequential. No embargo/lookback.

    Import is done outside timing to measure only the join work.
    """
    import pandas as pd  # Import once outside timing

    def _run():
        labels_df = pd.read_parquet(labels_path).sort_values("label_time")
        result_df = labels_df.copy()
        for i, fpath in enumerate(feature_paths):
            feat_df = pd.read_parquet(fpath).sort_values("updated_at")
            result_df = pd.merge_asof(
                result_df.sort_values("label_time"),
                feat_df.rename(columns={"updated_at": "label_time"}),
                on="label_time",
                by="user_id",
                direction="backward",
            )
        result_df.to_parquet(output_path)
        return result_df

    # Warmup
    result_df = _run()

    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter()
        result_df = _run()
        elapsed = time.perf_counter() - t0
        gc.enable()
        times.append(elapsed)

    return {
        "engine": "pandas",
        "median_seconds": round(statistics.median(times), 2),
        "stddev_seconds": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
        "runs": n_runs,
        "rows_out": len(result_df),
        "cols_out": len(result_df.columns),
    }


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

CONFIGS = [
    # (label, n_labels, n_features, kwargs)
    ("100K labels, 1 feat", 100_000, 1, {}),
    ("100K labels, 10 feats", 100_000, 10, {}),
    ("100K labels, 50 feats", 100_000, 50, {}),
    ("1M labels, 1 feat", 1_000_000, 1, {}),
    ("1M labels, 10 feats", 1_000_000, 10, {}),
    ("1M labels, 50 feats", 1_000_000, 50, {}),
    ("10M labels, 1 feat", 10_000_000, 1, {}),
    ("10M labels, 10 feats", 10_000_000, 10, {}),
    # Embargo + staleness + splits
    ("1M labels, 10 feats, embargo=1d", 1_000_000, 10, {"embargo": "1d"}),
    ("1M labels, 10 feats, staleness=30d", 1_000_000, 10, {"max_staleness": "30d"}),
    ("1M labels, 10 feats, splits", 1_000_000, 10, {
        "splits": {
            "train": ("2024-01-01", "2024-07-01"),
            "test": ("2024-07-01", "2025-01-01"),
        }
    }),
]

# Quick: safe for 16GB machines
QUICK_CONFIGS = [c for c in CONFIGS if c[1] <= 1_000_000 and c[2] <= 10]


def format_table(results: list[dict]) -> str:
    """Format results as an aligned ASCII table."""
    headers = ["Scenario", "Engine", "Median", "Stddev", "Rows", "Cols"]
    rows = []
    for r in results:
        rows.append([
            r["scenario"],
            r["engine"],
            f"{r['median_seconds']:.2f}s",
            f"\u00b1{r['stddev_seconds']:.2f}s",
            f"{r['rows_out']:,}",
            str(r["cols_out"]),
        ])

    widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(val.ljust(w) for val, w in zip(row, widths)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Timefence benchmarks")
    parser.add_argument("--quick", action="store_true", help="Safe for 16GB machines (<=1M, <=10 feats)")
    parser.add_argument("--include-pandas", action="store_true", help="Also benchmark naive pandas joins")
    parser.add_argument("--runs", type=int, default=3, help="Timed iterations per scenario (default: 3)")
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else CONFIGS
    all_results = []

    print(f"\nTimefence Benchmark Suite")
    print(f"========================")
    print(f"  Python {platform.python_version()} | DuckDB {duckdb.__version__} | Timefence {timefence.__version__}")
    print(f"  {args.runs} runs per scenario (median reported) | seed={SEED}")
    print()

    for label, n_labels, n_feats, kwargs in configs:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)

            print(f"  Generating data: {label} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            labels_path, feat_paths = generate_data(tmp, n_labels, n_feats)
            print(f"({time.perf_counter() - t0:.1f}s)")

            # Timefence
            print(f"  Running timefence: {label} ...", end=" ", flush=True)
            tf_out = tmp / "tf_out.parquet"
            tf_result = bench_timefence(
                labels_path, feat_paths, tf_out,
                embargo=kwargs.get("embargo", "0d"),
                max_lookback=kwargs.get("max_lookback", "365d"),
                max_staleness=kwargs.get("max_staleness"),
                splits=kwargs.get("splits"),
                n_runs=args.runs,
            )
            tf_result["scenario"] = label
            all_results.append(tf_result)
            print(f"{tf_result['median_seconds']:.2f}s (\u00b1{tf_result['stddev_seconds']:.2f}s)")

            # Pandas (optional)
            if (
                args.include_pandas
                and n_labels <= 1_000_000
                and "embargo" not in kwargs
                and "splits" not in kwargs
                and "max_staleness" not in kwargs
            ):
                print(f"  Running pandas:    {label} ...", end=" ", flush=True)
                pd_out = tmp / "pd_out.parquet"
                try:
                    pd_result = bench_pandas(labels_path, feat_paths, pd_out, n_runs=args.runs)
                    pd_result["scenario"] = label
                    all_results.append(pd_result)
                    ratio = pd_result["median_seconds"] / max(tf_result["median_seconds"], 0.01)
                    print(f"{pd_result['median_seconds']:.2f}s ({ratio:.1f}x vs timefence)")
                except Exception as e:
                    print(f"FAILED: {e}")

            print()

    print(format_table(all_results))

    # Save results
    output = {
        "methodology": "1 warmup + N timed runs, GC disabled, median reported.",
        "seed": SEED,
        "runs_per_scenario": args.runs,
        "versions": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "timefence": timefence.__version__,
        },
        "results": all_results,
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
