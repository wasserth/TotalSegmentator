#!/usr/bin/env python
import sys
import csv
import json
import argparse
from pathlib import Path


def flatten(prefix, value):
    """Flatten a (possibly nested list) metric value into {column: scalar} pairs.

    Scalars stay as one column; lists are expanded with index suffixes, e.g.
    centroid_vox -> centroid_vox_0/_1/_2 and bbox_vox -> bbox_vox_0_0/_0_1/...
    This keeps the output tidy (one scalar per cell) for pandas / R.
    """
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(flatten(f"{prefix}_{i}", item))
        return out
    return {prefix: value}


def rows_from_stats(stats, subject):
    """Turn one statistics.json dict ({structure: {metric: value}}) into tidy rows."""
    rows = []
    for structure, metrics in stats.items():
        row = {"subject": subject, "structure": structure}
        for metric, value in metrics.items():
            row.update(flatten(metric, value))
        rows.append(row)
    return rows


def collect_rows(input_dir, filename, quiet=False):
    """Find every `filename` under input_dir and return (rows, n_subjects)."""
    stats_files = sorted(input_dir.rglob(filename))
    rows = []
    n_subjects = 0
    for f in stats_files:
        try:
            with open(f) as fh:
                stats = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skipping {f} ({e})")
            continue
        rel = f.parent.relative_to(input_dir)
        subject = str(rel) if str(rel) != "." else f.parent.name
        rows.extend(rows_from_stats(stats, subject))
        n_subjects += 1
        if not quiet:
            print(f"  + {subject}: {len(stats)} structures")
    return rows, n_subjects


def ordered_columns(rows):
    """Stable column order: subject, structure, then metric columns in first-seen order."""
    columns = ["subject", "structure"]
    seen = set(columns)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def write_table(rows, columns, output):
    """Write rows to CSV, parquet or JSON, chosen by the output file extension."""
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet output (`pip install pyarrow`).")
        table = pa.table({col: [row.get(col) for row in rows] for col in columns})
        pq.write_table(table, output)
    elif suffix == ".json":
        with open(output, "w") as f:
            json.dump(rows, f, indent=2)
    else:  # default: CSV
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def main():
    """
    Aggregate the per-image statistics.json files of a whole cohort into one table.

    TotalSegmentator's `--statistics` writes one statistics.json per image. For a
    study you usually want a single analysis-ready table across all subjects. This
    command scans a directory tree for statistics.json files and merges them into
    one tidy (long) table: one row per (subject, structure), with a column per
    metric. Output format is chosen by the extension of -o (.csv, .parquet, .json).

    Usage:
    totalseg_aggregate_stats -i cohort_dir -o cohort_stats.csv
    totalseg_aggregate_stats -i cohort_dir -o cohort_stats.parquet
    totalseg_aggregate_stats -i cohort_dir -o radiomics.csv --filename statistics_radiomics.json

    Subject ids are derived from the path of each statistics file relative to the
    input directory (e.g. cohort_dir/subj001/statistics.json -> "subj001").
    """
    parser = argparse.ArgumentParser(
        description="Aggregate per-image statistics.json files of a cohort into one table.",
        epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="directory", dest="input",
                        help="Directory containing TotalSegmentator outputs (searched recursively for the statistics file).",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output",
                        help="Output table path. Format from extension: .csv (default), .parquet or .json.",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-f", "--filename", type=str, default="statistics.json",
                        help="Name of the per-image statistics file to look for (default: statistics.json).")

    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="Print no per-subject output.")

    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")

    rows, n_subjects = collect_rows(args.input, args.filename, quiet=args.quiet)
    if not rows:
        print(f"ERROR: no '{args.filename}' files found under {args.input}")
        sys.exit(2)

    columns = ordered_columns(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_table(rows, columns, args.output)
    print(f"Aggregated {n_subjects} subjects ({len(rows)} rows) -> {args.output}")


if __name__ == "__main__":
    main()
