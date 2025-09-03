#!/usr/bin/env python3
import argparse
import platform
from pathlib import Path
import warnings
import json
import time


def _auto_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    print("🍎 Enhanced TotalSegmentator for Mac")
    print(f"🖥️ Running on: {platform.platform()}")

    parser = argparse.ArgumentParser(description="Enhanced liver vessel pipeline with optional portal/hepatic splitting.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input CT NIfTI")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--mode", choices=["standard", "enhanced_liver", "liver_vessels_only"],
                        default="enhanced_liver")
    parser.add_argument("--liver_mask", type=Path, help="Optional precomputed liver mask")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--robust_crop", action="store_true", help="Use robust crop in base runs")
    parser.add_argument("--min_component_size", type=int, default=40, help="Min vessel component retention size")

    # Fallback behavior
    parser.add_argument("--no_fallback_full_liver", action="store_true",
                        help="Disable full-volume fallback if liver mask missing.")

    # Portal / hepatic splitting flags
    parser.add_argument("--split_portal_hepatic", action="store_true",
                        help="Split liver vessels into portal vs hepatic branches.")
    parser.add_argument("--split_support_dir", type=Path,
                        help="Directory containing portal_vein_and_splenic_vein.nii.gz & inferior_vena_cava.nii.gz. "
                             "If absent and --generate_split_support is set, a focused segmentation is run here.")
    parser.add_argument("--generate_split_support", action="store_true",
                        help="If support labels are missing, generate them via partial segmentation.")
    parser.add_argument("--no_skeleton", action="store_true",
                        help="Disable skeleton-based refinement in splitting.")
    parser.add_argument("--min_split_component_size", type=int, default=20,
                        help="Minimum voxel size to keep for each class after splitting.")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output}")

    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.enhanced_liver_vessels import process_enhanced_liver_vessels

    device = _auto_device(args.device)
    print(f"⚙️ Device: {device}")

    start_time = time.time()

    if args.mode == "enhanced_liver":
        liver_std_dir = args.output / "liver_standard"
        liver_std_dir.mkdir(exist_ok=True)
        print("▶ Baseline liver segmentation (roi_subset=['liver']) ...")
        try:
            totalsegmentator(
                str(args.input),
                str(liver_std_dir),
                roi_subset=["liver"],
                robust_crop=args.robust_crop,
                ml=True,
                device=device
            )
        except Exception as e:
            warnings.warn(f"[EnhancedCLI] Liver baseline segmentation failed: {e}")

        liver_mask_path = args.liver_mask if args.liver_mask else (liver_std_dir / "liver.nii.gz")
        if not liver_mask_path.exists():
            warnings.warn("[EnhancedCLI] liver.nii.gz not found. Enhancement will fallback if allowed.")

        print("▶ Enhanced vessel processing...")
        enhanced_out = args.output / "enhanced_liver_vessels.nii.gz"
        process_enhanced_liver_vessels(
            args.input,
            enhanced_out,
            liver_mask_path=liver_mask_path if liver_mask_path.exists() else None,
            device=device,
            allow_fallback_full_liver=not args.no_fallback_full_liver,
            min_component_size=args.min_component_size
        )
        print(f"✅ Enhanced vessels saved: {enhanced_out}")

    elif args.mode == "liver_vessels_only":
        print("▶ Running liver_vessels task only...")
        try:
            totalsegmentator(
                str(args.input),
                str(args.output),
                task="liver_vessels",
                robust_crop=args.robust_crop,
                device=device
            )
            print("✅ Done (liver_vessels_only).")
        except Exception as e:
            warnings.warn(f"[EnhancedCLI] liver_vessels task failed: {e}")
    else:
        print("▶ Standard TotalSegmentator run...")
        totalsegmentator(
            str(args.input),
            str(args.output),
            robust_crop=args.robust_crop,
            device=device
        )
        print("✅ Standard run complete.")

    # ------------------------------------------------------------------
    # Portal / Hepatic splitting (only if enhanced_liver mode executed)
    # ------------------------------------------------------------------
    if args.mode == "enhanced_liver" and args.split_portal_hepatic:
        try:
            from totalsegmentator.vessel_split import split_portal_hepatic
            print("▶ Splitting portal vs hepatic vessels...")
            support_dir = args.split_support_dir or args.output

            portal_label = support_dir / "portal_vein_and_splenic_vein.nii.gz"
            ivc_label = support_dir / "inferior_vena_cava.nii.gz"

            need_generation = any(not p.exists() for p in [portal_label, ivc_label])
            if need_generation and args.generate_split_support:
                print("  ↳ Generating missing support labels (portal vein + IVC + liver)...")
                # We always include liver to ensure a mask
                roi_subset = ["portal_vein_and_splenic_vein", "inferior_vena_cava", "liver"]
                support_dir.mkdir(parents=True, exist_ok=True)
                try:
                    totalsegmentator(
                        str(args.input),
                        str(support_dir),
                        roi_subset=roi_subset,
                        robust_crop=args.robust_crop,
                        ml=True,
                        device=device
                    )
                except Exception as e:
                    warnings.warn(f"[EnhancedCLI] Support segmentation failed: {e}")

            liver_path_for_split = None
            # Prefer liver in enhanced output
            primary_liver = args.output / "liver_standard" / "liver.nii.gz"
            if primary_liver.exists():
                liver_path_for_split = primary_liver
            else:
                alt_liver = support_dir / "liver.nii.gz"
                if alt_liver.exists():
                    liver_path_for_split = alt_liver

            qc = split_portal_hepatic(
                liver_vessels_path=args.output / "enhanced_liver_vessels.nii.gz",
                liver_path=liver_path_for_split,
                portal_trunk_path=portal_label if portal_label.exists() else None,
                ivc_path=ivc_label if ivc_label.exists() else None,
                output_dir=args.output,
                use_skeleton=not args.no_skeleton,
                min_component_voxels=args.min_split_component_size
            )
            print("✅ Portal/Hepatic split QC:", qc)

            meta_path = args.output / "enhanced_liver_vessels_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["portal_hepatic_split_qc"] = qc
                meta_path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            warnings.warn(f"Portal/hepatic splitting failed: {e}")

    elapsed = time.time() - start_time
    print(f"⏱️ Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()