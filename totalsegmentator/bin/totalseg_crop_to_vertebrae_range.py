#!/usr/bin/env python
import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

from totalsegmentator.nifti_ext_header import load_multilabel_nifti, save_multilabel_nifti


VERTEBRAE_ORDER = ([f"vertebrae_C{i}" for i in range(1, 8)] +
                   [f"vertebrae_T{i}" for i in range(1, 13)] +
                   [f"vertebrae_L{i}" for i in range(1, 7)] +
                   ["vertebrae_S1", "sacrum"])

# Rare anatomical variants which are normally empty; do not warn if they are missing
OPTIONAL_VERTEBRAE = ["vertebrae_L6", "vertebrae_S1"]

BORDER_MARGIN = 3  # same margin as totalsegmentator.statistics.touches_border


def parse_vertebra_name(name):
    """Accepts e.g. 'l1', 'L1', 'vertebrae_L1', 'sacrum'."""
    if name.strip().lower() == "sacrum":
        return "sacrum"
    m = re.fullmatch(r"(?:vertebrae_)?([ctls])(\d+)", name.strip(), flags=re.IGNORECASE)
    if m is not None:
        vert = f"vertebrae_{m.group(1).upper()}{int(m.group(2))}"
        if vert in VERTEBRAE_ORDER:
            return vert
    raise argparse.ArgumentTypeError(f"Unknown vertebra '{name}'. Use e.g. c7, t12, l1, l5, sacrum.")


def get_vertebrae_in_range(vert_a, vert_b):
    """Returns all vertebrae from upper to lower (inclusive), independent of the argument order."""
    i, j = sorted([VERTEBRAE_ORDER.index(vert_a), VERTEBRAE_ORDER.index(vert_b)])
    return VERTEBRAE_ORDER[i:j + 1]


def get_label_map(img):
    try:
        _, label_map = load_multilabel_nifti(img)
        return label_map
    except Exception:
        return None


def load_seg_data(img):
    return np.rint(np.asanyarray(img.dataobj)).astype(np.int16)


def get_si_axis(affine):
    """Index of the voxel axis which is closest to the superior-inferior direction."""
    return [c in ("S", "I") for c in nib.aff2axcodes(affine)].index(True)


def get_extent(mask, axis):
    """First and last slice index (inclusive) of the mask along axis. None if mask is empty."""
    other_axes = tuple(a for a in range(mask.ndim) if a != axis)
    idx = np.where(mask.any(axis=other_axes))[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1])


def slab(axis, lo, hi):
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi + 1)
    return tuple(sl)


def keep_slices(data, axis, ranges):
    """Set everything outside of the slice ranges [(lo, hi), ...] (inclusive) along axis to 0."""
    out = np.zeros_like(data)
    for lo, hi in ranges:
        out[slab(axis, lo, hi)] = data[slab(axis, lo, hi)]
    return out


def touches_border_slab(mask, axis, lo, n_slices, margin=BORDER_MARGIN):
    """
    Check if a mask touches the border of the original image. mask is the sub-array of the image
    starting at slice lo along axis (the image has n_slices along axis).
    """
    for a in range(3):
        size = mask.shape[a]
        if a == axis:
            first_end = max(margin - lo, 0)
            last_start = max(n_slices - margin - lo, 0)
        else:
            first_end = margin
            last_start = size - margin
        if first_end > 0 and np.take(mask, range(min(first_end, size)), axis=a).any():
            return True
        if last_start < size and np.take(mask, range(last_start, size), axis=a).any():
            return True
    return False


def calc_mask_stats(mask, ct, axis, lo, n_slices, vox_vol, spacing_si, include_incomplete):
    complete = not touches_border_slab(mask, axis, lo, n_slices)
    stats = {"complete": complete, "volume_mm3": None, "intensity_mean": None,
             "intensity_median": None, "height_mm": None}
    if not complete and not include_incomplete:
        return stats

    n_voxels = int(mask.sum())
    stats["volume_mm3"] = round(float(n_voxels * vox_vol), 3)
    extent = get_extent(mask, axis)
    stats["height_mm"] = 0.0 if extent is None else round(float((extent[1] - extent[0] + 1) * spacing_si), 3)
    if ct is not None and n_voxels > 0:
        values = ct[mask]
        stats["intensity_mean"] = round(float(values.mean()), 5)
        stats["intensity_median"] = round(float(np.median(values)), 5)
    return stats


def calc_region_stats(seg, label_map, body_trunc, ct, axis, lo, hi, vox_vol, spacing_si,
                      include_incomplete):
    n_slices = seg.shape[axis]
    sl = slab(axis, lo, hi)
    seg_r = seg[sl]
    ct_r = ct[sl] if ct is not None else None
    masks = {name: seg_r == label_id for label_id, name in label_map.items()}
    masks["body_trunc"] = body_trunc[sl]
    return {name: calc_mask_stats(mask, ct_r, axis, lo, n_slices, vox_vol, spacing_si, include_incomplete)
            for name, mask in masks.items()}


def create_preview(seg_out, label_map, vert_data, vert_ids, ct_data, affine, crop_regions,
                   range_vertebrae, output_path):
    """
    Sagittal slice through the middle of the vertebrae showing the cropped segmentation, the
    vertebrae defining the crop regions and the upper (white) and lower (red) end of each region.

    crop_regions: list of regions, each a list of the vertebrae which define its upper and lower end
    range_vertebrae: names of all vertebrae in the range which are present (to find the sagittal slice)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def to_canonical(data):
        return np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(data, affine)).dataobj)

    seg_c = to_canonical(seg_out)
    vert_c = to_canonical(vert_data)
    spacing = nib.as_closest_canonical(nib.Nifti1Image(seg_out, affine)).header.get_zooms()

    # RAS: axis 0 = left->right, 1 = posterior->anterior, 2 = inferior->superior
    range_vert_mask = np.isin(vert_c, [vert_ids[v] for v in range_vertebrae])
    x = int(round(np.argwhere(range_vert_mask)[:, 0].mean()))

    def sagittal(data):
        return data[x, ::-1, :].T  # rows: inferior->superior (origin lower), cols: anterior->posterior

    seg_s = sagittal(seg_c)
    vert_s = sagittal(vert_c)

    aspect = spacing[2] / spacing[1]
    fig_h = 10
    fig_w = min(max(fig_h * seg_s.shape[1] / (seg_s.shape[0] * aspect), 4), 16)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("black")

    if ct_data is not None:
        ct_s = sagittal(to_canonical(ct_data))
        ax.imshow(ct_s, cmap="gray", vmin=-160, vmax=240, origin="lower", aspect=aspect,
                  interpolation="nearest")
    else:
        ax.imshow(np.zeros(seg_s.shape), cmap="gray", vmin=0, vmax=1, origin="lower", aspect=aspect)

    overlay = np.zeros(seg_s.shape + (4,))
    legend = []
    class_colors = plt.get_cmap("tab10")
    for i, (label_id, name) in enumerate(sorted(label_map.items())):
        color = class_colors(i % 10)
        overlay[seg_s == label_id] = (*color[:3], 0.5)
        legend.append(Patch(color=color, label=name))

    vert_colors = ["#ff00ff", "#ffff00"]
    shown_vertebrae = list(dict.fromkeys(v for region in crop_regions for v in region))
    for i, vert_name in enumerate(shown_vertebrae):
        color = matplotlib.colors.to_rgb(vert_colors[i % 2])
        overlay[vert_s == vert_ids[vert_name]] = (*color, 0.8)
        legend.append(Patch(color=color, label=vert_name))

    ax.imshow(overlay, origin="lower", aspect=aspect, interpolation="nearest")

    # dashes are offset so that coinciding upper/lower lines of adjacent vertebrae are both visible
    upper_style = (0, (4, 4))
    lower_style = (4, (4, 4))
    for region in crop_regions:
        z_lo, z_hi = get_extent(np.isin(vert_c, [vert_ids[v] for v in region]), axis=2)
        ax.axhline(z_hi + 0.5, color="white", linestyle=upper_style, linewidth=0.8)
        ax.axhline(z_lo - 0.5, color="red", linestyle=lower_style, linewidth=0.8)
    legend.append(Line2D([], [], color="white", linestyle="--", label="upper end"))
    legend.append(Line2D([], [], color="red", linestyle="--", label="lower end"))

    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="black")
    plt.close(fig)

"""
Run for one test case:

cd /mnt/nvme/data/test_data/totalseg_crop_to_vertebrae_range
totalseg_crop_to_vertebrae_range \
  -i tissue_types.nii.gz \
  -o tissue_types_cropped.nii.gz \
  -v vertebrae.nii.gz \
  -b body.nii.gz \
  -r l1 l5 \
  -c ct.nii.gz \
  -s statistics.json \
  -p preview.png
"""
def main():
    parser = argparse.ArgumentParser(
        description="Crop a multilabel segmentation to a vertebrae range (from the upper end of the "
                    "upper vertebra to the lower end of the lower vertebra) and to body_trunc. "
                    "Optionally calculate statistics and a preview image.",
        epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="Multilabel segmentation to crop (e.g. tissue_types.nii.gz)",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output",
                        help="Cropped multilabel segmentation",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-v", metavar="filepath", dest="vertebrae",
                        help="Multilabel vertebrae segmentation (e.g. output of '-ta vertebrae_pp --ml')",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-b", metavar="filepath", dest="body",
                        help="Body segmentation (output of '-ta body --ml'). Everything outside of "
                             "body_trunc is removed.",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-r", "--range", nargs=2, metavar=("VERT1", "VERT2"), dest="vert_range",
                        type=parse_vertebra_name, required=True,
                        help="Vertebrae range, e.g. 'l1 l5'. Order does not matter.")

    parser.add_argument("-c", "--ct", metavar="filepath", dest="ct",
                        help="CT image. Needed for intensity (HU) statistics and as preview background.",
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-s", "--statistics", metavar="filepath", dest="statistics",
                        help="Output json with volume, mean/median HU and height per class "
                             "(and for body_trunc) inside of the cropped region.",
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("--stats_include_incomplete", action="store_true", default=False,
                        help="Also calculate statistics for classes touching the image border "
                             "(by default they are set to null since they are incomplete).")

    parser.add_argument("--per_vertebrae", action="store_true", default=False,
                        help="Report statistics separately for each vertebra in the range (each "
                             "cropped to the upper and lower end of this vertebra).")

    parser.add_argument("-p", "--preview", metavar="filepath", dest="preview",
                        help="Output png with a sagittal slice through the middle of the vertebrae.",
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    args = parser.parse_args()
    quiet = args.quiet

    range_vertebrae = get_vertebrae_in_range(*args.vert_range)
    vert_upper, vert_lower = range_vertebrae[0], range_vertebrae[-1]

    seg_img = nib.load(args.input)
    vert_img = nib.load(args.vertebrae)
    body_img = nib.load(args.body)
    ct_img = nib.load(args.ct) if args.ct is not None else None

    for name, img in [("vertebrae", vert_img), ("body", body_img), ("ct", ct_img)]:
        if img is None:
            continue
        if img.shape[:3] != seg_img.shape[:3]:
            print(f"ERROR: shape of {name} image {img.shape} does not match shape of input "
                  f"segmentation {seg_img.shape}.")
            sys.exit(1)
        if not np.allclose(img.affine, seg_img.affine, atol=1e-3):
            print(f"WARNING: affine of {name} image does not match affine of input segmentation.")

    seg_label_map = get_label_map(seg_img)
    seg = load_seg_data(seg_img)
    if seg_label_map is None:
        seg_label_map = {int(v): f"label_{int(v)}" for v in np.unique(seg) if v != 0}

    vert_label_map = get_label_map(vert_img)
    if vert_label_map is None:
        print("ERROR: vertebrae segmentation contains no label names in the nifti header. Use a multilabel "
              "output of TotalSegmentator (e.g. 'TotalSegmentator -i ct.nii.gz -o vertebrae.nii.gz -ta vertebrae_pp --ml').")
        sys.exit(1)
    vert_name_to_id = {name: label_id for label_id, name in vert_label_map.items()}
    vert_data = load_seg_data(vert_img)

    body_label_map = get_label_map(body_img)
    body_trunc_id = 1
    if body_label_map is not None:
        body_trunc_id = {name: label_id for label_id, name in body_label_map.items()}.get("body_trunc", 1)
    body_trunc = load_seg_data(body_img) == body_trunc_id

    axis = get_si_axis(seg_img.affine)

    # extent (first, last slice along superior-inferior axis) of each vertebra in the range
    vert_extents = {}
    for vert_name in range_vertebrae:
        extent = None
        if vert_name in vert_name_to_id:
            extent = get_extent(vert_data == vert_name_to_id[vert_name], axis)
        if extent is None:
            if args.per_vertebrae and vert_name not in OPTIONAL_VERTEBRAE and not quiet:
                print(f"WARNING: {vert_name} not found in vertebrae segmentation. Skipping.")
            continue
        vert_extents[vert_name] = extent

    if args.per_vertebrae:
        if len(vert_extents) == 0:
            print(f"ERROR: none of the vertebrae {vert_upper}-{vert_lower} found in vertebrae segmentation.")
            sys.exit(1)
        crop_vertebrae = [v for v in range_vertebrae if v in vert_extents]
        crop_vertebrae = [crop_vertebrae[0], crop_vertebrae[-1]]
    else:
        missing = [v for v in [vert_upper, vert_lower] if v not in vert_extents]
        if len(missing) > 0:
            print(f"ERROR: {', '.join(missing)} not found in vertebrae segmentation. Can not determine crop range.")
            sys.exit(1)
        crop_vertebrae = [vert_upper, vert_lower]

    lo = min(vert_extents[v][0] for v in crop_vertebrae)
    hi = max(vert_extents[v][1] for v in crop_vertebrae)

    # per_vertebrae: only keep the slices of each vertebra (no gaps between vertebrae)
    if args.per_vertebrae:
        crop_regions = [[v] for v in range_vertebrae if v in vert_extents]
    else:
        crop_regions = [crop_vertebrae]
    crop_ranges = [(min(vert_extents[v][0] for v in region), max(vert_extents[v][1] for v in region))
                   for region in crop_regions]

    seg[~body_trunc] = 0
    seg_out = keep_slices(seg, axis, crop_ranges)

    out_dtype = np.uint8 if seg_out.max() < 256 else np.int16
    header = seg_img.header.copy()
    header.extensions.clear()
    header.set_data_dtype(out_dtype)
    out_img = nib.Nifti1Image(seg_out.astype(out_dtype), seg_img.affine, header)
    save_multilabel_nifti(out_img, args.output, seg_label_map)
    if not quiet:
        slices_str = ", ".join(f"{r_lo}-{r_hi}" for r_lo, r_hi in crop_ranges)
        print(f"Cropped to {crop_vertebrae[0]}-{crop_vertebrae[1]} (slices {slices_str} along axis {axis}). "
              f"Saved to {args.output}")

    ct = ct_img.get_fdata(dtype=np.float32) if ct_img is not None else None

    if args.statistics is not None:
        if ct is None and not quiet:
            print("WARNING: no CT image provided (-c). Intensity statistics will be null.")
        spacing = seg_img.header.get_zooms()[:3]
        vox_vol = float(np.prod(spacing))
        spacing_si = float(spacing[axis])

        if args.per_vertebrae:
            regions = {v: vert_extents[v] for v in range_vertebrae if v in vert_extents}
        else:
            regions = {f"{vert_upper}-{vert_lower}": (lo, hi)}

        stats = {}
        for region_name, (r_lo, r_hi) in regions.items():
            region_vertebrae = [region_name] if args.per_vertebrae else crop_vertebrae
            vert_mask = np.isin(vert_data[slab(axis, r_lo, r_hi)], [vert_name_to_id[v] for v in region_vertebrae])
            stats[region_name] = {
                "slice_range": [r_lo, r_hi],
                "height_mm": round((r_hi - r_lo + 1) * spacing_si, 3),
                "vertebrae_complete": not touches_border_slab(vert_mask, axis, r_lo, seg.shape[axis]),
                "classes": calc_region_stats(seg, seg_label_map, body_trunc, ct, axis, r_lo, r_hi,
                                             vox_vol, spacing_si, args.stats_include_incomplete),
            }

        with open(args.statistics, "w") as f:
            json.dump(stats, f, indent=4)
        if not quiet:
            print(f"Saved statistics to {args.statistics}")

    if args.preview is not None:
        create_preview(seg_out, seg_label_map, vert_data, vert_name_to_id, ct, seg_img.affine,
                       crop_regions, list(vert_extents.keys()), args.preview)
        if not quiet:
            print(f"Saved preview to {args.preview}")


if __name__ == "__main__":
    main()
