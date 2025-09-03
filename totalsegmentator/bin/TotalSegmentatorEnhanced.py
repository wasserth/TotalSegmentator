#!/usr/bin/env python3
import argparse
import platform
from pathlib import Path
import warnings
import json
import time


# ---------------- Device Helpers ----------------

def _resolve_display_device(user_choice: str) -> str:
    import torch
    if user_choice and user_choice.lower() not in ("auto", ""):
        d = user_choice.lower()
        if d.startswith("cuda"): return d
        if d.startswith("gpu"): return d.replace("gpu", "cuda", 1)
        if d in ("cpu", "mps"): return d
        return "cpu"
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _map_display_to_backend(display_device: str) -> str:
    d = display_device.lower()
    if d == "cuda": return "gpu"
    if d.startswith("cuda:"): return "gpu:" + d.split(":", 1)[1]
    if d in ("cpu", "mps"): return d
    return "cpu"


def _validate_backend_device(backend_device: str) -> str:
    import torch
    if backend_device.startswith("gpu") and not torch.cuda.is_available():
        warnings.warn("[Device] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if backend_device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            warnings.warn("[Device] MPS requested but not available. Falling back to CPU.")
            return "cpu"
    return backend_device


def _print_gpu_info(backend_device: str):
    if backend_device.startswith("gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                idx = 0
                if ":" in backend_device:
                    try:
                        idx = int(backend_device.split(":", 1)[1])
                    except Exception:
                        pass
                name = torch.cuda.get_device_name(idx)
                prop = torch.cuda.get_device_properties(idx)
                mem = prop.total_memory / (1024**3)
                print(f"🧠 GPU: {name} (index {idx}, {mem:.1f} GB)")
        except Exception as e:
            print(f"[Device] GPU info error: {e}")


# ---------------- Main Script ----------------

def main():
    print("🍎 Enhanced TotalSegmentator")
    print(f"🖥️ Platform: {platform.platform()}")

    parser = argparse.ArgumentParser(
        description="Enhanced liver vessel pipeline (inline liver + main body veins, optional portal/hepatic labeling)."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input CT NIfTI")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--mode", choices=["standard", "enhanced_liver", "liver_vessels_only"],
                        default="enhanced_liver")
    parser.add_argument("--liver_mask", type=Path, help="Optional precomputed liver mask")

    parser.add_argument("--device", default="auto",
                        help="Device: auto|cpu|mps|cuda|cuda:N|gpu|gpu:N")
    parser.add_argument("--robust_crop", action="store_true", help="Use robust crop models")
    parser.add_argument("--min_component_size", type=int, default=40,
                        help="Min vessel component size to retain in enhancement")
    parser.add_argument("--no_fallback_full_liver", action="store_true",
                        help="Disable full-volume fallback if liver mask missing")

    # Splitting
    parser.add_argument("--split_portal_hepatic", action="store_true",
                        help="Produce portal_vein_branches.nii.gz, hepatic_veins.nii.gz and liver_vessels_labeled.nii.gz")
    parser.add_argument("--split_support_dir", type=Path,
                        help="(Optional) alternative directory containing portal trunk & IVC & liver")
    parser.add_argument("--generate_split_support", action="store_true",
                        help="Generate support labels if missing (portal trunk + IVC + liver)")
    parser.add_argument("--no_skeleton", action="store_true",
                        help="Disable skeleton refinement in splitting")
    parser.add_argument("--min_split_component_size", type=int, default=20,
                        help="Minimum voxel size per class after splitting")

    # Convenience flags
    parser.add_argument("--labeled_only", action="store_true",
                        help="Keep ONLY liver_vessels_labeled.nii.gz (remove portal_vein_branches & hepatic_veins). Implies --split_portal_hepatic")
    parser.add_argument("--remove_enhanced_binary", action="store_true",
                        help="After successful labeling (with --labeled_only), remove enhanced_liver_vessels.nii.gz")

    # NEW: ability to skip automatic trunk vein generation if not desired
    parser.add_argument("--skip_main_body_veins", action="store_true",
                        help="Do not auto-generate portal_vein_and_splenic_vein / inferior_vena_cava.")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output}")

    if args.labeled_only:
        args.split_portal_hepatic = True

    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.enhanced_liver_vessels import process_enhanced_liver_vessels

    display_device = _resolve_display_device(args.device)
    backend_device = _validate_backend_device(_map_display_to_backend(display_device))
    print(f"⚙️ Device request: {args.device} -> display: {display_device} -> backend: {backend_device}")
    _print_gpu_info(backend_device)

    start_time = time.time()

    # ---------------- Enhancement Mode ----------------
    if args.mode == "enhanced_liver":
        # 1. Ensure liver mask (direct output)
        liver_mask_path = None
        if args.liver_mask:
            liver_mask_path = args.liver_mask if args.liver_mask.exists() else None
            if liver_mask_path is None:
                warnings.warn(f"[EnhancedCLI] Provided liver_mask {args.liver_mask} missing; will attempt generation.")

        if liver_mask_path is None:
            liver_file = args.output / "liver.nii.gz"
            if liver_file.exists():
                liver_mask_path = liver_file

        # 2. Auto-generate liver + main trunk veins if needed and not skipped
        if not args.skip_main_body_veins:
            need_any = False
            needed_rois = []
            # Always ensure liver present
            if liver_mask_path is None or not liver_mask_path.exists():
                need_any = True
            # Portal trunk
            if not (args.output / "portal_vein_and_splenic_vein.nii.gz").exists():
                need_any = True
            # IVC
            if not (args.output / "inferior_vena_cava.nii.gz").exists():
                need_any = True

            if need_any:
                print("▶ Generating liver + main body veins (portal trunk, IVC)...")
                try:
                    totalsegmentator(
                        str(args.input),
                        str(args.output),
                        roi_subset=["liver", "portal_vein_and_splenic_vein", "inferior_vena_cava"],
                        robust_crop=args.robust_crop,
                        ml=False,
                        device=backend_device
                    )
                    liver_mask_path = args.output / "liver.nii.gz"
                except Exception as e:
                    warnings.warn(f"[EnhancedCLI] Liver/main vein generation failed: {e}")
        else:
            # If skipping main body veins but liver still missing, at least generate liver
            if (liver_mask_path is None) or (not liver_mask_path.exists()):
                print("▶ Generating liver only (skip_main_body_veins set)...")
                try:
                    totalsegmentator(
                        str(args.input),
                        str(args.output),
                        roi_subset=["liver"],
                        robust_crop=args.robust_crop,
                        ml=False,
                        device=backend_device
                    )
                    liver_mask_path = args.output / "liver.nii.gz"
                except Exception as e:
                    warnings.warn(f"[EnhancedCLI] Liver-only generation failed: {e}")

        if liver_mask_path is None or not liver_mask_path.exists():
            warnings.warn("[EnhancedCLI] liver.nii.gz not found. Enhancement may fallback to full volume.")

        # 3. Enhanced vessel segmentation
        print("▶ Enhanced vessel processing...")
        enhanced_out = args.output / "enhanced_liver_vessels.nii.gz"
        process_enhanced_liver_vessels(
            args.input,
            enhanced_out,
            liver_mask_path=liver_mask_path if (liver_mask_path and liver_mask_path.exists()) else None,
            device=backend_device,
            allow_fallback_full_liver=not args.no_fallback_full_liver,
            min_component_size=args.min_component_size
        )
        print(f"✅ Enhanced vessels saved: {enhanced_out}")

    # ---------------- Liver Vessels Only Mode ----------------
    elif args.mode == "liver_vessels_only":
        print("▶ Running liver_vessels task only...")
        try:
            totalsegmentator(
                str(args.input),
                str(args.output),
                task="liver_vessels",
                robust_crop=args.robust_crop,
                device=backend_device
            )
            print("✅ Done (liver_vessels_only).")
        except Exception as e:
            warnings.warn(f"[EnhancedCLI] liver_vessels task failed: {e}")

    # ---------------- Standard Mode ----------------
    else:
        print("▶ Standard TotalSegmentator run...")
        totalsegmentator(
            str(args.input),
            str(args.output),
            robust_crop=args.robust_crop,
            device=backend_device
        )
        print("✅ Standard run complete.")

    # ---------------- Splitting (optional / implied) ----------------
    if args.mode == "enhanced_liver" and args.split_portal_hepatic:
        try:
            from totalsegmentator.vessel_split import split_portal_hepatic
            print("▶ Splitting portal vs hepatic vessels...")

            support_dir = args.split_support_dir or args.output
            portal_label = support_dir / "portal_vein_and_splenic_vein.nii.gz"
            ivc_label = support_dir / "inferior_vena_cava.nii.gz"

            # If user asked to generate and they are missing (should normally be present unless skip_main_body_veins)
            need_generation = any(not p.exists() for p in [portal_label, ivc_label])
            if need_generation and args.generate_split_support:
                print("  ↳ Generating support labels (portal trunk + IVC + liver) for splitting...")
                try:
                    totalsegmentator(
                        str(args.input),
                        str(args.output),
                        roi_subset=["portal_vein_and_splenic_vein", "inferior_vena_cava", "liver"],
                        robust_crop=args.robust_crop,
                        ml=False,
                        device=backend_device
                    )
                except Exception as e:
                    warnings.warn(f"[EnhancedCLI] Support label generation failed: {e}")

            liver_for_split = args.output / "liver.nii.gz" if (args.output / "liver.nii.gz").exists() else None

            qc = split_portal_hepatic(
                liver_vessels_path=args.output / "enhanced_liver_vessels.nii.gz",
                liver_path=liver_for_split,
                portal_trunk_path=portal_label if portal_label.exists() else None,
                ivc_path=ivc_label if ivc_label.exists() else None,
                output_dir=args.output,
                use_skeleton=not args.no_skeleton,
                min_component_voxels=args.min_split_component_size
            )
            print("✅ Split QC:", qc)

            # Update metadata
            meta_path = args.output / "enhanced_liver_vessels_metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = {}
                meta["portal_hepatic_split_qc"] = qc
                if args.labeled_only:
                    meta["labeled_only"] = True
                meta_path.write_text(json.dumps(meta, indent=2))

            if args.labeled_only:
                removed_any = False
                for fname in ["portal_vein_branches.nii.gz", "hepatic_veins.nii.gz"]:
                    p = args.output / fname
                    if p.exists():
                        try:
                            p.unlink()
                            removed_any = True
                        except Exception as e:
                            warnings.warn(f"Could not remove {fname}: {e}")
                if removed_any:
                    print("🧹 Kept only liver_vessels_labeled.nii.gz (multi-label).")

                if args.remove_enhanced_binary:
                    enhanced_bin = args.output / "enhanced_liver_vessels.nii.gz"
                    if enhanced_bin.exists():
                        try:
                            enhanced_bin.unlink()
                            print("🧹 Removed enhanced_liver_vessels.nii.gz after labeling.")
                        except Exception as e:
                            warnings.warn(f"Could not remove enhanced_liver_vessels.nii.gz: {e}")

        except Exception as e:
            warnings.warn(f"[EnhancedCLI] Splitting failed: {e}")

    elapsed = time.time() - start_time
    print(f"⏱️ Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()