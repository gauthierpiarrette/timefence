"""Timefence CLI: command-line interface powered by click and rich."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from timefence._constants import (
    DEFAULT_ATOL,
    DEFAULT_MAX_LOOKBACK,
    DEFAULT_ON_MISSING,
    DEFAULT_RTOL,
)
from timefence._version import __version__

logger = logging.getLogger(__name__)


def _ql(value: str | Path) -> str:
    """Quote a value as a SQL single-quoted string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _qi(name: str) -> str:
    """Quote a SQL identifier (column name, table name) for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _safe_name(name: str) -> str:
    """Sanitize a string for use in SQL table/alias names."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "_unnamed"

console = Console()
err_console = Console(stderr=True)


def _parse_features_arg(features_arg: str) -> tuple[str, str | None]:
    """Parse 'path.py:feature_name' into (path, filter_name).

    Returns (file_path, None) when no filter is specified.
    """
    if ":" in features_arg:
        path, _, filter_name = features_arg.rpartition(":")
        # Guard against Windows-style paths like C:\path
        if path and not Path(path).suffix:
            return features_arg, None
        return path, filter_name
    return features_arg, None


def _load_features_from_file(path: str) -> list:
    """Load Feature objects from a Python file.

    Supports 'path.py:feature_name' to filter to a single feature.
    """
    from timefence.core import Feature, FeatureSet

    file_path, filter_name = _parse_features_arg(path)

    path_obj = Path(file_path)
    if not path_obj.exists():
        err_console.print(f"[red]Error: Feature file '{file_path}' not found.[/red]")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("features_module", path_obj)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    features = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, (Feature, FeatureSet)):
            features.append(obj)

    if filter_name is not None:
        # Flatten and filter to matching feature name
        flat = []
        for f in features:
            if isinstance(f, FeatureSet):
                flat.extend(f.features)
            elif isinstance(f, Feature):
                flat.append(f)
        matched = [f for f in flat if f.name == filter_name]
        if not matched:
            available = [f.name for f in flat]
            err_console.print(
                f"[red]Error: Feature '{filter_name}' not found in {file_path}.[/red]\n"
                f"  Available: {', '.join(available)}"
            )
            sys.exit(1)
        return matched

    return features


def _load_config() -> dict[str, Any]:
    """Load timefence.yaml if it exists.

    Tries pyyaml first, falls back to a simple key-value parser for
    basic fields. This avoids requiring pyyaml as a hard dependency.
    """
    for name in ("timefence.yaml", "timefence.yml"):
        if not Path(name).exists():
            continue
        text = Path(name).read_text()
        try:
            import yaml

            return yaml.safe_load(text) or {}
        except ImportError:
            # Minimal YAML parser for the subset we generate in timefence init/quickstart
            return _parse_simple_yaml(text)
    return {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple subset of YAML that timefence.yaml uses.

    Handles: scalars, lists (inline [a, b] and indented - items),
    and one level of nested mappings. Comments are stripped.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_dict: dict[str, Any] | None = None
    current_list: list[str] | None = None

    for raw_line in text.split("\n"):
        # Strip comments
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())

        # Top-level key: value
        if indent == 0 and ":" in line:
            # Flush any pending nested structure
            if current_key and current_dict is not None:
                result[current_key] = current_dict
                current_dict = None
            if current_key and current_list is not None:
                result[current_key] = current_list
                current_list = None

            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if not val:
                current_key = key
            elif val.startswith("[") and val.endswith("]"):
                items = [
                    v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()
                ]
                result[key] = items
                current_key = None
            elif (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                result[key] = val[1:-1]
            else:
                result[key] = val
                current_key = None
            continue

        # Indented list item
        if indent > 0 and line.strip().startswith("- "):
            item = line.strip()[2:].strip().strip("'\"")
            if current_list is None:
                current_list = []
            current_list.append(item)
            continue

        # Indented key: value (nested dict)
        if indent > 0 and ":" in line and current_key:
            if current_dict is None:
                current_dict = {}
            k, _, v = line.strip().partition(":")
            k = k.strip()
            v = v.strip().strip("'\"")
            if v.startswith("[") and v.endswith("]"):
                current_dict[k] = [
                    x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()
                ]
            elif v:
                current_dict[k] = v
            continue

    # Flush final
    if current_key and current_dict is not None:
        result[current_key] = current_dict
    if current_key and current_list is not None:
        result[current_key] = current_list

    return result


def _try_load_config() -> dict[str, Any]:
    """Try to load config, return empty dict on failure."""
    try:
        return _load_config()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.debug("Could not load timefence.yaml: %s", exc)
        return {}


def _resolve_features_file(features_file: str | None, config: dict) -> str | None:
    """Resolve features file from CLI arg, config, or convention."""
    if features_file is not None:
        return features_file
    feat_files = config.get("features", [])
    if isinstance(feat_files, list) and feat_files:
        return feat_files[0]
    if isinstance(feat_files, str):
        return feat_files
    if Path("features.py").exists():
        return "features.py"
    return None


def _resolve_labels_info(
    config: dict,
) -> tuple[str | None, list[str] | None, str, list[str]]:
    """Extract labels path, keys, label_time, target from config.

    Returns (path, keys, label_time, target).
    """
    label_config = config.get("labels", {})
    path = label_config.get("path")
    keys = label_config.get("keys")
    if isinstance(keys, str):
        keys = [keys]
    label_time = label_config.get("label_time", "label_time")
    target = label_config.get("target", [])
    if isinstance(target, str):
        target = [target]
    return path, keys, label_time, target


def _resolve_defaults(config: dict) -> dict[str, Any]:
    """Extract default parameters from config."""
    return config.get("defaults", {})


def _resolve_store(config: dict):
    """Create Store from config or default."""
    from timefence.store import Store

    store_path = config.get("store", ".timefence")
    return Store(store_path)


def _infer_keys_from_features(features: list) -> list[str] | None:
    """Try to infer key columns from feature definitions."""
    from timefence.core import Feature, FeatureSet

    for f in features:
        if isinstance(f, Feature):
            return list(f.source.keys)
        if isinstance(f, FeatureSet) and f.features:
            return list(f.features[0].source.keys)
    return None


@click.group()
@click.version_option(__version__, prog_name="timefence")
def cli():
    """Timefence: temporal correctness layer for ML training data."""


@cli.command()
@click.argument("project_name", default="churn-example")
@click.option("--template", default="churn", help="Example template (default: churn)")
@click.option("--minimal", is_flag=True, help="Generate minimal example")
def quickstart(project_name: str, template: str, minimal: bool):
    """Generate a self-contained example project."""
    from timefence.quickstart import generate_quickstart

    output_dir = generate_quickstart(project_name, minimal=minimal)
    console.print(f"\n[green]Created project: {output_dir}[/green]")
    console.print(f"\n  cd {project_name}")
    console.print("  timefence audit data/train_LEAKY.parquet")
    console.print("  timefence build -o data/train_CLEAN.parquet")
    console.print("  timefence audit data/train_CLEAN.parquet")
    console.print()


@cli.command()
@click.argument("path")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def inspect(path: str, json_output: bool):
    """Suggest keys and timestamps for a data file."""
    import duckdb

    path_obj = Path(path)
    if not path_obj.exists():
        err_console.print(f"[red]Error: File '{path}' not found.[/red]")
        sys.exit(1)

    conn = duckdb.connect()
    try:
        if path_obj.suffix in (".parquet", ".pq"):
            conn.execute(
                f"CREATE TEMP TABLE __data AS SELECT * FROM read_parquet({_ql(path)})"
            )
        elif path_obj.suffix == ".csv":
            conn.execute(
                f"CREATE TEMP TABLE __data AS SELECT * FROM read_csv({_ql(path)})"
            )
        else:
            err_console.print(f"[red]Unsupported format: {path_obj.suffix}[/red]")
            sys.exit(1)

        row_count = conn.execute("SELECT COUNT(*) FROM __data").fetchone()[0]
        columns = conn.execute("DESCRIBE __data").fetchall()

        likely_key = None
        likely_timestamp = None
        col_info = []

        for col_name, col_type, *_ in columns:
            unique_count = conn.execute(
                f'SELECT COUNT(DISTINCT "{col_name}") FROM __data'
            ).fetchone()[0]
            pct = unique_count / row_count * 100 if row_count > 0 else 0

            notes = ""
            if pct > 99 and "int" in col_type.lower():
                notes = "likely entity key"
                if likely_key is None:
                    likely_key = col_name
            elif "timestamp" in col_type.lower() or "date" in col_type.lower():
                if unique_count > row_count * 0.1:
                    notes = "likely timestamp"
                    if likely_timestamp is None:
                        likely_timestamp = col_name

            col_info.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "unique_count": unique_count,
                    "unique_pct": round(pct, 1),
                    "notes": notes,
                }
            )

        if json_output:
            click.echo(
                json.dumps(
                    {
                        "path": path,
                        "row_count": row_count,
                        "column_count": len(columns),
                        "columns": col_info,
                        "suggestion": {
                            "key": likely_key,
                            "timestamp": likely_timestamp,
                        }
                        if likely_key or likely_timestamp
                        else None,
                    },
                    indent=2,
                )
            )
        else:
            console.print(
                f"\n[bold]FILE:[/bold] {path} ({row_count:,} rows, {len(columns)} columns)\n"
            )
            table = Table(show_header=True, header_style="bold")
            table.add_column("Column")
            table.add_column("Type")
            table.add_column("Unique")
            table.add_column("Notes")

            for ci in col_info:
                table.add_row(
                    ci["name"],
                    ci["type"],
                    f"{ci['unique_count']:,} ({ci['unique_pct']:.0f}%)",
                    ci["notes"],
                )
            console.print(table)

            if likely_key and likely_timestamp:
                console.print("\n[bold]Suggestion:[/bold]")
                console.print("  timefence.Source(")
                console.print(f'      path="{path}",')
                console.print(f'      keys=["{likely_key}"],')
                console.print(f'      timestamp="{likely_timestamp}",')
                console.print("  )")
                console.print()
    finally:
        conn.close()


@cli.command()
@click.argument("data")
@click.option(
    "--features", "features_file", default=None, help="Path to features Python file"
)
@click.option("--keys", default=None, help="Key column name(s), comma-separated")
@click.option("--label-time", default=None, help="Label time column name")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--html", "html_path", default=None, help="Export HTML report")
@click.option("--strict", is_flag=True, help="Exit code 1 if leakage found (CI mode)")
def audit(
    data: str,
    features_file: str | None,
    keys: str | None,
    label_time: str | None,
    json_output: bool,
    html_path: str | None,
    strict: bool,
):
    """Audit a dataset for temporal leakage."""
    from timefence.engine import audit as do_audit

    config = _try_load_config()
    defaults = _resolve_defaults(config)

    # Resolve features file from config
    features_file = _resolve_features_file(features_file, config)
    if features_file is None:
        err_console.print(
            "[red]Error: --features is required (no timefence.yaml or features.py found).[/red]"
        )
        sys.exit(1)

    # Resolve keys and label_time from config
    _, config_keys, config_label_time, _ = _resolve_labels_info(config)
    if label_time is None:
        label_time = config_label_time

    features = _load_features_from_file(features_file)

    if keys is None:
        if config_keys:
            keys = ",".join(config_keys)
        else:
            inferred = _infer_keys_from_features(features)
            if inferred:
                keys = ",".join(inferred)
    if keys is None:
        err_console.print("[red]Error: --keys is required.[/red]")
        sys.exit(1)

    keys_list = [k.strip() for k in keys.split(",")]

    report = do_audit(
        data,
        features,
        keys=keys_list,
        label_time=label_time,
        max_lookback=defaults.get("max_lookback", DEFAULT_MAX_LOOKBACK),
        max_staleness=defaults.get("max_staleness"),
        join=defaults.get("join", "strict"),
    )

    if json_output:
        click.echo(
            json.dumps(
                {
                    "has_leakage": report.has_leakage,
                    "total_rows": report.total_rows,
                    "mode": report.mode,
                    "features": {
                        name: {
                            "clean": d.clean,
                            "leaky_row_count": d.leaky_row_count,
                            "leaky_row_pct": d.leaky_row_pct,
                            "severity": d.severity,
                        }
                        for name, d in report.features.items()
                    },
                },
                indent=2,
            )
        )
    else:
        _print_audit_report(report)

    if html_path:
        report.to_html(html_path)
        console.print(f"\n[dim]HTML report written to {html_path}[/dim]")

    if strict and report.has_leakage:
        sys.exit(1)


def _print_audit_report(report) -> None:
    """Rich-formatted audit report for the terminal."""
    console.print()
    console.print("[bold]TEMPORAL AUDIT REPORT[/bold]")
    console.print(f"Scanned {report.total_rows:,} rows")
    console.print()

    if report.has_leakage:
        leaky = len(report.leaky_features)
        total = len(report.features)
        console.print(
            f"[bold yellow]WARNING[/bold yellow]  LEAKAGE DETECTED in {leaky} of {total} features\n"
        )
    else:
        console.print(
            "[bold green]ALL CLEAN[/bold green] — no temporal leakage detected\n"
        )

    for name, detail in report.features.items():
        if detail.clean:
            null_info = f", {detail.null_rows} null" if detail.null_rows else ""
            console.print(
                f"[green]  OK[/green]    {name} — clean ({detail.total_rows:,} rows{null_info})"
            )
        else:
            console.print(f"[red]  LEAK[/red]  {name}")
            console.print(
                f"        {detail.leaky_row_count:,} rows ({detail.leaky_row_pct:.1%}) use feature data from the future"
            )
            if detail.max_leakage:
                console.print(f"        Max leakage: {detail.max_leakage}")
            if detail.median_leakage:
                console.print(f"        Median leakage: {detail.median_leakage}")
            console.print(f"        Severity: {detail.severity}")
            console.print()

    if report.has_leakage:
        console.print(
            "\n[dim]Next step: run `timefence build` to rebuild without leakage[/dim]"
        )
    console.print()


@cli.command()
@click.option("--labels", default=None, help="Path to labels file")
@click.option(
    "--features", "features_file", default=None, help="Path to features Python file"
)
@click.option("--output", "-o", required=True, help="Output path for training set")
@click.option("--max-lookback", default=None, help="Maximum lookback window")
@click.option("--max-staleness", default=None, help="Maximum feature staleness")
@click.option("--on-missing", default=None, type=click.Choice(["null", "skip"]))
@click.option(
    "--join-mode",
    "join_mode",
    default=None,
    type=click.Choice(["strict", "inclusive"]),
)
@click.option("--split", multiple=True, help="Time split: name:start:end")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.option("--flatten", is_flag=True, help="Strip feature name prefix from columns")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def build(
    labels: str | None,
    features_file: str | None,
    output: str,
    max_lookback: str | None,
    max_staleness: str | None,
    on_missing: str | None,
    join_mode: str | None,
    split: tuple[str, ...],
    dry_run: bool,
    flatten: bool,
    json_output: bool,
):
    """Build a point-in-time correct training set."""
    from timefence.core import Labels as LabelsClass
    from timefence.engine import build as do_build
    from timefence.engine import explain as do_explain

    config = _try_load_config()
    defaults = _resolve_defaults(config)

    # Resolve features file
    features_file = _resolve_features_file(features_file, config)
    if features_file is None:
        err_console.print(
            "[red]Error: --features is required (no timefence.yaml or features.py found).[/red]"
        )
        sys.exit(1)
    features = _load_features_from_file(features_file)

    # Resolve labels from config
    config_label_path, config_keys, config_label_time, config_target = (
        _resolve_labels_info(config)
    )
    if labels is None:
        labels = config_label_path
    if labels is None:
        err_console.print(
            "[red]Error: --labels is required (not found in timefence.yaml).[/red]"
        )
        sys.exit(1)

    label_keys = config_keys
    label_time = config_label_time
    label_target = config_target

    # If keys not in config, infer from features or detect from file
    if label_keys is None:
        label_keys = _infer_keys_from_features(features)
    if label_keys is None:
        import duckdb

        conn = duckdb.connect()
        try:
            cols = [
                c[0]
                for c in conn.execute(
                    f"DESCRIBE (SELECT * FROM read_parquet({_ql(labels)}))"
                ).fetchall()
            ]
            label_keys = [cols[0]]
        finally:
            conn.close()

    # Detect target columns if not in config
    if not label_target:
        import duckdb

        conn = duckdb.connect()
        try:
            cols = [
                c[0]
                for c in conn.execute(
                    f"DESCRIBE (SELECT * FROM read_parquet({_ql(labels)}))"
                ).fetchall()
            ]
            label_target = [c for c in cols if c not in label_keys and c != label_time]
        finally:
            conn.close()

    labels_obj = LabelsClass(
        path=labels,
        keys=label_keys,
        label_time=label_time,
        target=label_target if label_target else ["target"],
    )

    # Resolve output path against config output.dir
    output_config = config.get("output", {})
    if isinstance(output_config, dict):
        output_dir = output_config.get("dir")
        if output_dir and not Path(output).is_absolute():
            output = str(Path(output_dir) / output)

    # Apply defaults: CLI flags > config defaults > built-in defaults
    max_lookback = max_lookback or defaults.get("max_lookback", DEFAULT_MAX_LOOKBACK)
    max_staleness = max_staleness or defaults.get("max_staleness")
    join_mode = join_mode or defaults.get("join", "strict")
    on_missing = on_missing or defaults.get("on_missing", DEFAULT_ON_MISSING)

    if dry_run:
        plan = do_explain(
            labels_obj, features, max_lookback=max_lookback, join=join_mode
        )
        console.print(str(plan))
        return

    # Parse splits
    splits_dict = None
    if split:
        splits_dict = {}
        for s in split:
            parts = s.split(":")
            if len(parts) != 3:
                err_console.print(
                    f"[red]Invalid split format '{s}'. Expected name:start:end[/red]"
                )
                sys.exit(1)
            splits_dict[parts[0]] = (parts[1], parts[2])

    store = _resolve_store(config)

    result = do_build(
        labels=labels_obj,
        features=features,
        output=output,
        max_lookback=max_lookback,
        max_staleness=max_staleness,
        join=join_mode,
        on_missing=on_missing,
        splits=splits_dict,
        store=store,
        flatten_columns=flatten,
    )

    if json_output:
        click.echo(json.dumps(result.manifest, indent=2, default=str))
    else:
        _print_build_result(result, labels_obj, features)


def _print_build_result(result, labels_obj, features) -> None:
    """Rich-formatted build result for the terminal."""
    console.print()
    console.print("[bold]Building training set...[/bold]\n")
    console.print(
        f"  Labels     {result.stats.row_count:,} rows from {labels_obj.path}"
    )
    console.print(f"  Features   {len(result.stats.feature_stats)} features\n")

    join_mode = result.manifest.get("parameters", {}).get("join", "strict")
    op = "<" if join_mode == "strict" else "<="
    console.print(
        f"  Joining with point-in-time correctness (feature_time {op} label_time):\n"
    )

    for fname, fstats in result.stats.feature_stats.items():
        matched = fstats.get("matched", 0)
        missing = fstats.get("missing", 0)
        total = matched + missing
        cache_status = fstats.get("cached", False)
        cache_tag = " [dim](cached)[/dim]" if cache_status else ""
        if missing:
            console.print(
                f"  [green]OK[/green]  {fname:20s} {matched:,} / {total:,} matched ({missing:,} missing -> null){cache_tag}"
            )
        else:
            console.print(
                f"  [green]OK[/green]  {fname:20s} {matched:,} / {total:,} matched{cache_tag}"
            )

    console.print()
    if result.output_path:
        console.print(
            f"  Written   {result.output_path} ({result.stats.row_count:,} rows, {result.stats.column_count} cols)"
        )
    manifest_path = result.manifest.get("manifest_path")
    if manifest_path:
        console.print(f"  Manifest  {manifest_path}")
    console.print(f"  Time      {result.stats.duration_seconds:.1f}s")
    console.print()


@cli.command(name="explain")
@click.option("--labels", default=None, help="Path to labels file")
@click.option(
    "--features", "features_file", default=None, help="Path to features Python file"
)
@click.option("--max-lookback", default=None)
@click.option(
    "--join-mode",
    "join_mode",
    default=None,
    type=click.Choice(["strict", "inclusive"]),
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def explain_cmd(
    labels: str | None,
    features_file: str | None,
    max_lookback: str | None,
    join_mode: str | None,
    json_output: bool,
):
    """Preview join logic without executing."""
    from timefence.core import Labels as LabelsClass
    from timefence.engine import explain as do_explain

    config = _try_load_config()
    defaults = _resolve_defaults(config)

    features_file = _resolve_features_file(features_file, config)
    if features_file is None:
        err_console.print("[red]Error: --features is required.[/red]")
        sys.exit(1)
    features = _load_features_from_file(features_file)

    # Resolve labels
    config_label_path, config_keys, config_label_time, config_target = (
        _resolve_labels_info(config)
    )
    if labels is None:
        labels = config_label_path
    if labels is None:
        err_console.print("[red]Error: --labels is required.[/red]")
        sys.exit(1)

    label_keys = config_keys or _infer_keys_from_features(features)
    if label_keys is None:
        import duckdb

        conn = duckdb.connect()
        try:
            cols = [
                c[0]
                for c in conn.execute(
                    f"DESCRIBE (SELECT * FROM read_parquet({_ql(labels)}))"
                ).fetchall()
            ]
            label_keys = [cols[0]]
        finally:
            conn.close()

    label_time = config_label_time
    target = config_target or ["target"]

    labels_obj = LabelsClass(
        path=labels,
        keys=label_keys,
        label_time=label_time,
        target=target,
    )

    max_lookback = max_lookback or defaults.get("max_lookback", DEFAULT_MAX_LOOKBACK)
    join_mode = join_mode or defaults.get("join", "strict")

    plan = do_explain(labels_obj, features, max_lookback=max_lookback, join=join_mode)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "label_count": plan.label_count,
                    "plan": plan.plan,
                },
                indent=2,
                default=str,
            )
        )
    else:
        console.print(str(plan))


@cli.command(name="diff")
@click.argument("old_path")
@click.argument("new_path")
@click.option("--keys", required=True, help="Key column name(s), comma-separated")
@click.option("--label-time", required=True, help="Label time column")
@click.option(
    "--atol",
    default=DEFAULT_ATOL,
    type=float,
    help="Absolute tolerance for numeric comparison",
)
@click.option(
    "--rtol",
    default=DEFAULT_RTOL,
    type=float,
    help="Relative tolerance for numeric comparison",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def diff_cmd(
    old_path: str,
    new_path: str,
    keys: str,
    label_time: str,
    atol: float,
    rtol: float,
    json_output: bool,
):
    """Compare two training datasets."""
    from timefence.engine import diff as do_diff

    keys_list = [k.strip() for k in keys.split(",")]
    result = do_diff(
        old_path, new_path, keys=keys_list, label_time=label_time, atol=atol, rtol=rtol
    )

    if json_output:
        click.echo(
            json.dumps(
                {
                    "old_rows": result.old_rows,
                    "new_rows": result.new_rows,
                    "schema_changes": result.schema_changes,
                    "value_changes": result.value_changes,
                },
                indent=2,
                default=str,
            )
        )
    else:
        console.print(str(result))


@cli.command()
@click.option(
    "--features", "features_file", default=None, help="Path to features Python file"
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def catalog(features_file: str | None, json_output: bool):
    """List all features defined in the project."""
    from timefence._duration import format_duration
    from timefence.core import Feature, FeatureSet

    config = _try_load_config()
    features_file = _resolve_features_file(features_file, config)
    if features_file is None:
        err_console.print("[red]Error: --features is required.[/red]")
        sys.exit(1)

    features = _load_features_from_file(features_file)

    flat = []
    for f in features:
        if isinstance(f, FeatureSet):
            flat.extend(f.features)
        elif isinstance(f, Feature):
            flat.append(f)

    if json_output:
        items = []
        for feat in flat:
            embargo = (
                format_duration(feat.embargo)
                if feat.embargo.total_seconds() > 0
                else None
            )
            items.append(
                {
                    "name": feat.name,
                    "source": feat.source.name,
                    "keys": list(feat.source.keys),
                    "embargo": embargo,
                    "mode": feat.mode,
                }
            )
        click.echo(json.dumps({"features": items}, indent=2))
    else:
        table = Table(title="FEATURE CATALOG", show_header=True, header_style="bold")
        table.add_column("Name")
        table.add_column("Source")
        table.add_column("Keys")
        table.add_column("Embargo")
        table.add_column("Mode")

        for feat in flat:
            src_name = feat.source.name
            keys_str = ", ".join(feat.source.keys)
            embargo = (
                format_duration(feat.embargo)
                if feat.embargo.total_seconds() > 0
                else "—"
            )
            table.add_row(feat.name, src_name, keys_str, embargo, feat.mode)

        console.print()
        console.print(table)
        console.print()


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def doctor(json_output: bool):
    """Diagnose project setup and common issues."""
    checks: list[dict[str, Any]] = []

    def check(status: str, message: str, detail: str = ""):
        checks.append({"status": status, "message": message, "detail": detail})

    # Check for config
    config_found = False
    for name in ("timefence.yaml", "timefence.yml"):
        if Path(name).exists():
            config_found = True
            check("OK", f"{name} found and valid")
            break
    if not config_found:
        check("WARN", "No timefence.yaml found (optional)")

    # Check DuckDB
    try:
        import duckdb

        check("OK", f"DuckDB v{duckdb.__version__} available")
    except ImportError:
        check("FAIL", "DuckDB not installed")

    # Check for feature files
    config = _try_load_config()
    feature_files = config.get("features", [])
    if isinstance(feature_files, str):
        feature_files = [feature_files]
    if not feature_files:
        for pattern in ["features.py", "features/*.py"]:
            matches = list(Path(".").glob(pattern))
            feature_files.extend(str(m) for m in matches)

    all_flat_features = []
    for ff in feature_files:
        if Path(ff).exists():
            try:
                features = _load_features_from_file(ff)
                from timefence.core import Feature, FeatureSet

                flat = []
                for f in features:
                    if isinstance(f, FeatureSet):
                        flat.extend(f.features)
                    elif isinstance(f, Feature):
                        flat.append(f)
                all_flat_features.extend(flat)

                check("OK", f"{len(flat)} features defined in {ff}")

                # Check sources exist
                for feat in flat:
                    if feat.source.path and not feat.source.path.exists():
                        check("FAIL", f"Source file not found: {feat.source.path}")

            except Exception as e:
                check("FAIL", f"Error loading {ff}: {e}")
        else:
            check("WARN", f"Feature file not found: {ff}")

    # Check label file schema matches feature keys
    label_config = config.get("labels", {})
    label_path = label_config.get("path")
    if label_path and Path(label_path).exists():
        try:
            import duckdb as ddb

            conn = ddb.connect()
            try:
                cols = [
                    c[0]
                    for c in conn.execute(
                        f"DESCRIBE (SELECT * FROM read_parquet({_ql(label_path)}))"
                    ).fetchall()
                ]
                label_keys = label_config.get("keys", [])
                if isinstance(label_keys, str):
                    label_keys = [label_keys]
                missing_keys = [k for k in label_keys if k not in cols]
                if missing_keys:
                    check("FAIL", f"Label file missing key columns: {missing_keys}")
                else:
                    check("OK", "Label file schema matches feature keys")
            finally:
                conn.close()
        except Exception as e:
            check("WARN", f"Could not validate label file: {e}")

    # Check for duplicate (key, timestamp) in sources
    if all_flat_features:
        try:
            import duckdb as ddb

            conn = ddb.connect()
            try:
                seen_sources: set[str] = set()
                for feat in all_flat_features:
                    if feat.source.path is None or not feat.source.path.exists():
                        continue
                    src_key = str(feat.source.path)
                    if src_key in seen_sources:
                        continue
                    seen_sources.add(src_key)

                    tbl = f"__doc_{_safe_name(feat.source.name)}"
                    conn.execute(
                        f"CREATE OR REPLACE TEMP TABLE {tbl} AS "
                        f"SELECT * FROM read_parquet({_ql(feat.source.path)})"
                    )
                    key_cols = ", ".join(_qi(k) for k in feat.source.keys)
                    ts = _qi(feat.source.timestamp)
                    dup_count = conn.execute(
                        f"SELECT COUNT(*) FROM ("
                        f"  SELECT {key_cols}, {ts}, COUNT(*) as cnt"
                        f"  FROM {tbl} GROUP BY {key_cols}, {ts} HAVING cnt > 1"
                        f") t"
                    ).fetchone()[0]
                    if dup_count > 0:
                        check(
                            "WARN",
                            f"Source '{feat.source.name}' has {dup_count} duplicate ({key_cols}, {ts}) rows",
                            "Consider deduplication or set on_duplicate='keep_any' on affected features.",
                        )
                    else:
                        check("OK", "All source files exist and are readable")
            finally:
                conn.close()
        except (ImportError, OSError, ValueError) as exc:
            logger.debug("Doctor source check failed: %s", exc)

    # Check for column name conflicts between features
    if len(all_flat_features) > 1:
        seen_cols: dict[str, str] = {}
        conflicts = []
        for feat in all_flat_features:
            if hasattr(feat, "_columns") and feat._columns:
                for col in feat._columns.values():
                    namespaced = f"{feat.name}__{col}"
                    if namespaced in seen_cols:
                        conflicts.append(
                            f"{namespaced} from {feat.name} and {seen_cols[namespaced]}"
                        )
                    seen_cols[namespaced] = feat.name
        if conflicts:
            check("WARN", f"Column name conflicts: {', '.join(conflicts)}")
        else:
            check("OK", "No column name conflicts between features")

    if json_output:
        click.echo(json.dumps({"checks": checks}, indent=2))
    else:
        console.print("\n[bold]PROJECT HEALTH CHECK[/bold]\n")
        for c in checks:
            status = c["status"]
            if status == "OK":
                console.print(f"  [green]OK[/green]    {c['message']}")
            elif status == "WARN":
                console.print(f"  [yellow]WARN[/yellow]  {c['message']}")
                if c.get("detail"):
                    console.print(f"        {c['detail']}")
            elif status == "FAIL":
                console.print(f"  [red]FAIL[/red]  {c['message']}")
                if c.get("detail"):
                    console.print(f"        {c['detail']}")
        console.print()


@cli.command()
@click.argument("path", default=".")
def init(path: str):
    """Initialize a project with a timefence.yaml config file."""
    project_dir = Path(path)
    project_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_dir / "timefence.yaml"
    if config_path.exists():
        console.print(
            f"[yellow]timefence.yaml already exists in {project_dir}[/yellow]"
        )
        return

    config_content = """# timefence.yaml
name: my-project
version: "1.0"

# data_dir: data/

# features:
#   - features.py

# labels:
#   path: data/labels.parquet
#   keys: [entity_id]
#   label_time: label_time
#   target: [target]

defaults:
  max_lookback: 365d
  join: strict
  on_missing: "null"

store: .timefence/
"""
    config_path.write_text(config_content)
    console.print(f"[green]Created {config_path}[/green]")


if __name__ == "__main__":
    cli()
