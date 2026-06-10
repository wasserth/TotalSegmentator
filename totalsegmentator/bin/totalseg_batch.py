#!/usr/bin/env python
import sys
import time
import argparse
import traceback
from pathlib import Path

# Heavy imports (torch / nnunet) are done lazily inside main() so the small helper
# functions below stay importable (and testable) without a full runtime.


def find_images(input_dir, pattern=None):
    """Find the input images to process in input_dir.

    By default all .nii / .nii.gz files directly inside input_dir. With --pattern,
    a glob relative to input_dir (e.g. "*/ct.nii.gz") is used instead.
    """
    if pattern:
        return sorted(input_dir.glob(pattern))
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")))


def case_id(path):
    """Derive a case id from an image filename (strip the .nii/.nii.gz suffix)."""
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def output_target(output_dir, cid, ml):
    """Per-case output location. A directory of masks, or a multilabel file when ml=True.

    In both cases statistics.json ends up inside output_dir/<cid>/, so the cohort
    aggregator can find one statistics file per case.
    """
    case_dir = output_dir / cid
    return case_dir / "segmentation.nii.gz" if ml else case_dir


def main():
    """
    Run TotalSegmentator over a whole folder of images in one process.

    The model weights are loaded once and reused across all images (instead of being
    reloaded for every image as in a shell loop), which substantially reduces the
    runtime for a cohort. Per-image failures are logged and the run continues. If
    --statistics is set, the per-image statistics are also aggregated into one table.

    Each case is written to <output_dir>/<case_id>/ (case_id is the image filename
    without its .nii/.nii.gz suffix).

    Usage:
    totalseg_batch -i ct_folder -o output_folder
    totalseg_batch -i ct_folder -o output_folder --statistics -ta total -d gpu
    totalseg_batch -i study_folder -o output_folder --pattern "*/ct.nii.gz"
    """
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator over a folder of images, loading the model only once.",
        epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="directory", dest="input",
                        help="Directory containing the input images (.nii/.nii.gz).",
                        type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory. Each case is written to <output>/<case_id>/.",
                        type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob (relative to -i) selecting the input images, e.g. '*/ct.nii.gz'. "
                             "Default: all .nii/.nii.gz files directly inside -i.")

    # Pass-through TotalSegmentator options (the common ones for cohort processing).
    from totalsegmentator.registry import TASKS
    parser.add_argument("-ta", "--task", choices=TASKS, metavar="task", default="total",
                        help="Task to run (default: total). See 'totalseg_info --list-tasks'.")
    parser.add_argument("-d", "--device", type=str, default="gpu",
                        help="Device: 'gpu', 'cpu', 'mps' or 'gpu:X'.")
    parser.add_argument("-f", "--fast", action="store_true", default=False, help="Run faster 3mm model.")
    parser.add_argument("-ff", "--fastest", action="store_true", default=False, help="Run faster 6mm model.")
    parser.add_argument("-ml", "--ml", action="store_true", default=False, help="Save one multilabel image per case.")
    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+", default=None,
                        help="Only predict this subset of classes.")
    parser.add_argument("-bs", "--body_seg", action="store_true", default=False,
                        help="Crop to body region before processing.")
    parser.add_argument("-s", "--statistics", action="store_true", default=False,
                        help="Compute statistics.json per case (and aggregate them, see --no_aggregate).")
    parser.add_argument("-sx", "--statistics_extra", action="store_true", default=False,
                        help="Add extra per-structure metrics to the statistics.")
    parser.add_argument("-p", "--preview", action="store_true", default=False, help="Generate a preview png per case.")
    parser.add_argument("--no_aggregate", action="store_true", default=False,
                        help="Do not aggregate the per-case statistics into one table.")
    parser.add_argument("--stop_on_error", action="store_true", default=False,
                        help="Abort on the first failing image instead of skipping it.")
    parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Less output per image.")

    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")

    images = find_images(args.input, args.pattern)
    if not images:
        sel = f"pattern '{args.pattern}'" if args.pattern else ".nii/.nii.gz files"
        print(f"ERROR: no {sel} found under {args.input}")
        sys.exit(2)

    # Lazy heavy imports.
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.nnunet import set_predictor_cache_enabled, clear_predictor_cache

    print(f"Found {len(images)} images. Running task '{args.task}' on device '{args.device}'.")
    args.output.mkdir(parents=True, exist_ok=True)

    errors = []
    n_ok = 0
    t0 = time.time()
    # Load the model once and reuse it for every image.
    set_predictor_cache_enabled(True)
    try:
        for i, image in enumerate(images, start=1):
            cid = case_id(image)
            target = output_target(args.output, cid, args.ml)
            (args.output / cid).mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{len(images)}] {cid}")
            try:
                totalsegmentator(image, target, task=args.task, device=args.device,
                                 fast=args.fast, fastest=args.fastest, ml=args.ml,
                                 roi_subset=args.roi_subset, body_seg=args.body_seg,
                                 statistics=args.statistics, statistics_extra=args.statistics_extra,
                                 preview=args.preview, quiet=args.quiet)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - one bad image should not abort the cohort
                errors.append((cid, repr(e)))
                print(f"  ERROR on {cid}: {e}")
                if args.stop_on_error:
                    raise
                traceback.print_exc()
    finally:
        set_predictor_cache_enabled(False)  # also clears the cache and frees GPU memory
        clear_predictor_cache()

    # Write an error log if anything failed.
    if errors:
        error_log = args.output / "batch_errors.log"
        with open(error_log, "w") as f:
            for cid, msg in errors:
                f.write(f"{cid}\t{msg}\n")
        print(f"{len(errors)} image(s) failed. See {error_log}")

    # Aggregate the per-case statistics into one table.
    if args.statistics and not args.no_aggregate and n_ok > 0:
        from totalsegmentator.bin.totalseg_aggregate_stats import collect_rows, ordered_columns, write_table
        rows, n_subjects = collect_rows(args.output, "statistics.json", quiet=True)
        if rows:
            out_table = args.output / "cohort_statistics.csv"
            write_table(rows, ordered_columns(rows), out_table)
            print(f"Aggregated statistics for {n_subjects} cases -> {out_table}")

    print(f"Done: {n_ok}/{len(images)} succeeded in {time.time() - t0:.0f}s.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
