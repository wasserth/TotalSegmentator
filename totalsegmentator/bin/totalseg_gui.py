#!/usr/bin/env python3
"""
totalseg_gui

Improved pipeline with precise spatial alignment and web viewer:
  1) DICOM -> PNG slices with precise affine (totalseg_dicom_to_png.py)
  2) DICOM -> NIfTI (dcm2niix, for segmentation only)
  3) Segmentation + STL export (TotalSegmentatorImproved)
  4) Generate stl_list.json for web viewer
  5) Auto-open web viewer in browser

Windows usage:
  python -m totalsegmentator.bin.totalseg_gui
"""

from __future__ import annotations

import sys
import threading
import queue
import subprocess
from pathlib import Path
import os
import shutil
import webbrowser
import json
import platform

# -----------------------------
# Vessel label definitions
# -----------------------------
VESSEL_KEYWORDS = (
    "artery",
    "vein",
    "vena",
    "vessel",
    "aorta",
    "cava",
    "trunk",
)


# -----------------------------

from tkinter import filedialog
if hasattr(sys, "stdout") and not sys.stdout.isatty():
    sys.stdout = os.fdopen(1, "w", buffering=1)
if hasattr(sys, "stderr") and not sys.stderr.isatty():
    sys.stderr = os.fdopen(2, "w", buffering=1)
print("✅ stdout/stderr restored, logs will now appear in Terminal")


# Use ttkbootstrap for a modern look
try:
    import ttkbootstrap as b
    from ttkbootstrap.constants import *
    from ttkbootstrap.scrolled import ScrolledText
    from ttkbootstrap.dialogs import Messagebox
except ImportError:
    print("ttkbootstrap not found. Please install it with: pip install ttkbootstrap", file=sys.stderr)
    sys.exit(1)


import vtk

def slicer_like_mesh_postprocess(
        
    in_stl,
    out_stl,
    *,
    is_vessel: bool,
    # ===== Vessel defaults (safe start) =====
    vessel_subdiv: int = 1,              # ✅ 你要求：subdivision 1 次
    vessel_geom_iters: int = 25,         # 轻量几何重排（接近Slicer观感的关键）
    vessel_geom_relax: float = 0.12,
    vessel_sinc_iters: int = 8,          # 轻度非收缩平滑
    vessel_sinc_passband: float = 0.24,
    vessel_hole_size_m: float = 0.003,   # 3mm（血管尽量少填洞，避免封口/搭桥）
    # ===== Organ defaults =====
    organ_subdiv: int = 0,
    organ_geom_iters: int = 10,
    organ_geom_relax: float = 0.10,
    organ_sinc_iters: int = 20,
    organ_sinc_passband: float = 0.12,
    organ_hole_size_m: float = 0.02,     # 2cm
):
    """
    Slicer-like STL postprocess (VTK only, no extra deps).

    Key idea (why this looks like 3D Slicer Surface smoothing):
      - Subdivision (optional) gives geometric freedom (rounder tubes)
      - Light vertex-based smoothing (vtkSmoothPolyDataFilter) redistributes triangles
        and breaks voxel-aligned "staircase" patterns
      - WindowedSinc (vtkWindowedSincPolyDataFilter) smooths without strong shrink
      - Normals rebuild improves shading continuity

    IMPORTANT:
      - Your STL units are meters (--units m). Hole sizes are in meters.
      - For vessels, keep hole filling tiny or disable by setting vessel_hole_size_m=0.
    """

    # ---- select parameters by type ----
    if is_vessel:
        subdiv_iters = int(vessel_subdiv)
        geom_iters = int(vessel_geom_iters)
        geom_relax = float(vessel_geom_relax)
        sinc_iters = int(vessel_sinc_iters)
        sinc_passband = float(vessel_sinc_passband)
        hole_size = float(vessel_hole_size_m)
    else:
        subdiv_iters = int(organ_subdiv)
        geom_iters = int(organ_geom_iters)
        geom_relax = float(organ_geom_relax)
        sinc_iters = int(organ_sinc_iters)
        sinc_passband = float(organ_sinc_passband)
        hole_size = float(organ_hole_size_m)

    # --- Load STL ---
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(in_stl))
    reader.Update()

    # --- Ensure triangles ---
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(reader.GetOutputPort())

    # --- Clean (merge duplicate points / tiny cracks) ---
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())

    upstream = clean.GetOutputPort()

    # --- Fill holes (size-limited) ---
    # For vessels, keep this very small to avoid "capping" ends / bridging.
    if hole_size and hole_size > 0:
        hole = vtk.vtkFillHolesFilter()
        hole.SetInputConnection(upstream)
        hole.SetHoleSize(hole_size)   # unit = meters
        upstream = hole.GetOutputPort()

    # --- Subdivision (Loop) ---
    if subdiv_iters and subdiv_iters > 0:
        subdiv = vtk.vtkLoopSubdivisionFilter()
        subdiv.SetInputConnection(upstream)
        subdiv.SetNumberOfSubdivisions(subdiv_iters)
        upstream = subdiv.GetOutputPort()

    # --- Geometry redistribution (LIGHT) ---
    # This is the missing piece that often makes VTK output still look "blocky".
    # Keep it light to preserve thin branches.
    geom = vtk.vtkSmoothPolyDataFilter()
    geom.SetInputConnection(upstream)
    geom.SetNumberOfIterations(geom_iters)
    geom.SetRelaxationFactor(geom_relax)
    geom.FeatureEdgeSmoothingOff()
    geom.BoundarySmoothingOff()
    # If you ever see shrink, reduce geom_iters or geom_relax.
    upstream = geom.GetOutputPort()

    # --- Windowed Sinc smoothing (non-shrinking-ish) ---
    sinc = vtk.vtkWindowedSincPolyDataFilter()
    sinc.SetInputConnection(upstream)
    sinc.SetNumberOfIterations(sinc_iters)
    sinc.SetPassBand(sinc_passband)
    sinc.FeatureEdgeSmoothingOff()
    sinc.BoundarySmoothingOff()
    sinc.NonManifoldSmoothingOn()
    sinc.NormalizeCoordinatesOn()
    upstream = sinc.GetOutputPort()

    # --- Normals (better shading = looks smoother) ---
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(upstream)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()      # keep smooth shading, avoid hard edges
    normals.Update()

    # --- Write back ---
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(out_stl))
    writer.SetInputData(normals.GetOutput())
    writer.Write()



def run_cmd(cmd: list[str], log, cwd: Path | None = None) -> int:
    log(f"$ {' '.join(cmd)}\n")
    try:
        env = os.environ.copy()
        # Force UTF-8 in child processes to avoid Windows cp1252 Unicode errors
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            log(line)
        return p.wait()
    except FileNotFoundError:
        log(f"Command not found: {cmd[0]}\n")
        return 127
    except Exception as e:
        log(f"Command error: {e}\n")
        return 1


def generate_stl_list_json(stl_dir: Path) -> Path:
    """Generate stl_list.json for web viewer"""
    stl_files = sorted([f.name for f in stl_dir.glob("*.stl")])
    
    list_json = {
        "files": stl_files,
        "count": len(stl_files)
    }
    
    output_path = stl_dir / "stl_list.json"
    with open(output_path, "w") as f:
        json.dump(list_json, f, indent=2)
    
    print(f"✅ Generated {output_path} with {len(stl_files)} STL files")
    return output_path

def is_vessel(stl_name: str) -> bool:
    """
    Determine whether an STL belongs to vessel category
    based on filename keywords.
    """
    name = stl_name.lower()
    return any(k in name for k in VESSEL_KEYWORDS)

def list_label_stems(task_dir: Path) -> list[str]:
    """
    Return label stems from *.nii.gz under task_dir
    e.g. aorta from aorta.nii.gz
    """
    stems = []
    for f in task_dir.glob("*.nii.gz"):
        stems.append(f.name[:-7])
    return sorted(stems)


class PipelineThread(threading.Thread):
    def __init__(self, q: queue.Queue[str], cfg: dict, mode: str = "all"):
        super().__init__(daemon=True)
        self.q = q
        self.cfg = cfg
        self.rc = 1
        self.mode = mode

    def log(self, msg: str):
        self.q.put(msg)

    def run(self):
        try:
            self.rc = self._run()
        except Exception as e:
            self.log(f"Unexpected error: {e}\n")
            self.rc = 1
        finally:
            self.q.put("__DONE__")

    def _run(self) -> int:
        dicom_dir = Path(self.cfg["dicom_dir"]).expanduser()
        out_root = Path(self.cfg["out_root"]).expanduser()
        case_name = self.cfg["case_name"].strip() or "case01"
        scale = float(self.cfg["scale"]) if self.cfg["scale"] else 0.01
        selected_tasks = self.cfg.get("tasks", "total_all")
        blender_hint = (self.cfg.get("blender_path") or "").strip()
        dcm2niix_hint = (self.cfg.get("dcm2niix_path") or "").strip()

        out_root.mkdir(parents=True, exist_ok=True)
        out_nii = out_root / "out_nii"
        out_png = out_root / "dicom_slices"
        out_seg = out_root / "out_total_all"
        out_blend_dir = out_root / "out"
        out_nii.mkdir(exist_ok=True)
        out_blend_dir.mkdir(exist_ok=True)

        # Resolver helpers for external tools
        def _resolve_blender():
            if blender_hint:
                p = Path(blender_hint)
                if p.exists():
                    return str(p)
            hit = shutil.which('blender')
            if hit:
                return hit
            base = Path('C:/Program Files/Blender Foundation')
            if base.exists():
                for c in sorted(base.glob('Blender */blender.exe'), reverse=True):
                    if c.exists():
                        return str(c)
            return None

        def _resolve_dcm2niix():
            if dcm2niix_hint:
                p = Path(dcm2niix_hint)
                if p.exists():
                    return str(p)
            hit = shutil.which('dcm2niix')
            if hit:
                return hit
            return None

        def _progress(p):
            try:
                self.q.put(f"__PROG__:{int(p)}")
            except Exception:
                pass

        # STEP 1: DICOM -> PNG
        if self.mode in ("all", "step1"):
            self.log("\n" + "="*60 + "\n")
            self.log("📸 Step 1: Exporting PNG slices with precise affine from DICOM\n")
            self.log("="*60 + "\n")
            
            rc = run_cmd([
                sys.executable,
                "-m", "totalsegmentator.bin.totalseg_dicom_to_png",
                "-i", str(dicom_dir),
                "-o", str(out_png),
                "--multi-views",
                "--metadata",
                "--window", "custom", 
                "--wl", "40",
                "--ww", "1200", 
            ], self.log)
            
            if rc != 0:
                self.log("❌ Failed to export PNG slices from DICOM\n")
                return rc
            
            self.log("✅ PNG slices exported with affine metadata\n")
            _progress(20)
            if self.mode == 'step1':
                return 0

        # STEP 2: DICOM -> NIfTI
        nii_path = out_nii / f"{case_name}.nii.gz"
        if self.mode in ("all", "step2"):
            self.log("\n" + "="*60 + "\n")
            self.log("📄 Step 2: Converting DICOM to NIfTI (for segmentation)\n")
            self.log("="*60 + "\n")
            
            if not nii_path.exists():
                dcm2 = _resolve_dcm2niix()
                if not dcm2:
                    self.log("dcm2niix not found. Set path in GUI or add to PATH.\n")
                    return 127
                rc = run_cmd([
                    dcm2, '-z', 'y', '-o', str(out_nii), '-f', case_name, str(dicom_dir)
                ], self.log)
                if rc != 0:
                    return rc
                if not nii_path.exists():
                    gz = list(out_nii.glob('*.nii.gz'))
                    if gz:
                        nii_path = gz[0]
            else:
                self.log(f"Reusing existing NIfTI: {nii_path}\n")
            
            self.log("✅ NIfTI ready for segmentation\n")
            _progress(40)
            if self.mode == 'step2':
                return 0
        else:
            if not nii_path.exists():
                self.log(f"⚠️  NIfTI not found, looking for existing file...\n")
                gz = list(out_nii.glob('*.nii.gz'))
                if gz:
                    nii_path = gz[0]
                    self.log(f"Found: {nii_path}\n")
                else:
                    self.log(f"❌ No NIfTI found in {out_nii}\n")
                    return 1
                
        stl_task_dir = out_seg / selected_tasks

        if stl_task_dir.exists():
            shutil.rmtree(stl_task_dir)

        stl_task_dir.mkdir(parents=True, exist_ok=True)


        # STEP 3: Segmentation + STL export (UNIFIED RULE)

        if self.mode in ("all", "step3"):
            self.log("\n" + "=" * 60 + "\n")
            self.log("🧠 Step 3: Segmentation + mesh export (task-independent rules)\n")
            self.log("=" * 60 + "\n")

            selected_tasks = self.cfg.get("tasks", "total_all")
            stl_task_dir = out_seg / selected_tasks
            stl_task_dir.mkdir(parents=True, exist_ok=True)

            # ---- overwrite old STL only ----
            self.log(f"🧹 Overwrite enabled: removing old STL in {stl_task_dir}\n")
            for p in stl_task_dir.glob("*.stl"):
                try:
                    p.unlink()
                except Exception:
                    pass

            cmd = [
                sys.executable,
                "-m",
                "totalsegmentator.bin.TotalSegmentatorImproved",
                "-i", str(nii_path),
                "-o", str(out_seg),
                "--tasks", selected_tasks,
                "--export-mesh",
                "--export-format", "stl",
                "--units", "m",
                "--device", "gpu",
            ]

            if selected_tasks == "total_all":
                cmd.append("--with-liver-vessels")

            rc = run_cmd(cmd, self.log)
            if rc != 0:
                return rc

            self.log("✅ Mesh export completed (raw STL)\n")
            _progress(70)

            if self.mode == "step3":
                return 0



        # STEP 4: Slicer-style mesh post-processing (in-place)

        if self.mode in ("all", "step4"):
            self.log("\n" + "=" * 60 + "\n")
            self.log("🩸 Step 4: Slicer-style mesh smoothing (topology-preserving)\n")
            self.log("=" * 60 + "\n")

            selected_tasks = self.cfg.get("tasks", "total_all")
            task_dir = out_seg / selected_tasks

            stl_files = list(task_dir.glob("*.stl"))
            if not stl_files:
                self.log("⚠️  No STL files found, skipping Step 4.\n")
            else:
                for stl in stl_files:
                    label = stl.stem
                    vessel = is_vessel(label)

                    # ===============================
                    # 🔧 UNIFIED PARAMETER POLICY
                    # ===============================
                    if vessel:
                        subdiv_iters = 1
                        geom_iters = 25
                        geom_relax = 0.12
                        sinc_iters = 8
                        passband = 0.25
                        hole_size = 0.005
                    else:
                        subdiv_iters = 0
                        geom_iters = 10
                        geom_relax = 0.10
                        sinc_iters = 20
                        passband = 0.12
                        hole_size = 0.02

                    self.log(
                        f"   🔧 {'VESSEL' if vessel else 'ORGAN'} | {stl.name} | "
                        f"subdiv={subdiv_iters} geom={geom_iters} sinc={sinc_iters} hole={hole_size}\n"
                    )

                    tmp = stl.with_suffix(".tmp.stl")

                    try:
                        if vessel:
                            slicer_like_mesh_postprocess(
                                in_stl=stl,
                                out_stl=tmp,
                                is_vessel=True,
                                vessel_subdiv=subdiv_iters,
                                vessel_geom_iters=geom_iters,
                                vessel_geom_relax=geom_relax,
                                vessel_sinc_iters=sinc_iters,
                                vessel_sinc_passband=passband,
                                vessel_hole_size_m=hole_size,
                            )
                        else:
                            slicer_like_mesh_postprocess(
                                in_stl=stl,
                                out_stl=tmp,
                                is_vessel=False,
                                organ_subdiv=subdiv_iters,
                                organ_geom_iters=geom_iters,
                                organ_geom_relax=geom_relax,
                                organ_sinc_iters=sinc_iters,
                                organ_sinc_passband=passband,
                                organ_hole_size_m=hole_size,
                            )

                        tmp.replace(stl)

                    except Exception as e:
                        self.log(f"   ❌ Failed: {stl.name} → {e}\n")
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass

            _progress(75)

            if self.mode == "step4":
                return 0

       
        # STEP 5: Blender import (all meshes together)
        scene_setup = out_blend_dir / "scene-setup.blend"
        if self.mode in ("all", "step5"):
            self.log("\n" + "="*60 + "\n")
            self.log("🎨 Step 5: Importing meshes to Blender\n")
            self.log("="*60 + "\n")

            blender_exe = _resolve_blender()
            if not blender_exe:
                self.log("Blender not found. Set path in GUI or add to PATH.\n")
                return 127

            stl_dir = out_seg / selected_tasks
            blender_script = Path(__file__).with_name("totalseg_blender_import.py")

            cmd = [
                blender_exe,
                "-b",
                "-P", str(blender_script),
                "--",
                "--stl-dir", str(stl_dir),
                "--units", "m",
                "--collection", "Organs",
                "--group-categories",
                "--palette", "exact",
                "--scale", str(scale),
                "--mirror-x", "true",
                "--save", str(scene_setup),
            ]

            rc = run_cmd(cmd, self.log)
            if rc != 0:
                return rc

            self.log("✅ Meshes imported to Blender\n")
            _progress(85)
            if self.mode == "step5":
                return 0



        # STEP 6: Apply materials
        if self.mode in ("all", "step6"):
            self.log("\n" + "="*60 + "\n")
            self.log("🎨 Step 6: Applying exact anatomical materials\n")
            self.log("="*60 + "\n")
            
            blender_exe = _resolve_blender()
            if not blender_exe:
                self.log("Blender not found. Set path in GUI or add to PATH.\n")
                return 127
            colored = out_blend_dir / "scene-colored.blend"
            rc = run_cmd(
                [
                    blender_exe,
                    "-b",    
                    str(scene_setup),
                    "-P",
                    str(Path(__file__).with_name("totalseg_material.py")),
                    "--",
                    str(colored),
                ],
                self.log,
            )
            if rc != 0:
                return rc
            
            # Configure CT data folder path
            self.log("\n" + "="*60 + "\n")
            self.log("🔧 Configuring CT data folder path in Blender scene...\n")
            self.log("="*60 + "\n")
            
            import tempfile
            
            temp_script_fd, temp_script_path = tempfile.mkstemp(suffix='.py', text=True)
            
            try:
                dicom_path = out_png.resolve()
                
                self.log(f"📂 Data folder path: {dicom_path}\n")
                
                script_content = f'''
import bpy
import sys

print("="*60)
print("🔧 Configuring Blender Scene")
print("="*60)

data_folder = r"{str(dicom_path)}"
print(f"Setting ct_data_folder to: {{data_folder}}")

bpy.context.scene.ct_data_folder = data_folder
bpy.context.scene.png_folder = data_folder

import os
if os.path.exists(data_folder):
    print("✅ Data folder exists")
    meta_file = os.path.join(data_folder, "affine_metadata.json")
    if os.path.exists(meta_file):
        print("✅ affine_metadata.json found")
    else:
        print("⚠️  affine_metadata.json not found")
else:
    print("❌ Data folder does not exist!")

bpy.ops.wm.save_mainfile()
print("✅ Saved .blend file with configured paths")
print("="*60)
'''
                
                with os.fdopen(temp_script_fd, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                
                rc = run_cmd(
                    [
                        blender_exe,
                        str(colored),
                        "-b",
                        "-P",
                        temp_script_path
                    ],
                    self.log
                )
                
                if rc == 0:
                    self.log("✅ Successfully configured CT data folder path\n")
                else:
                    self.log("⚠️  Warning: Failed to configure data folder path\n")
                    self.log("   You may need to set it manually in Blender\n")
            
            finally:
                try:
                    os.unlink(temp_script_path)
                except Exception:
                    pass
            
            _progress(100)
            self.log(f"\n" + "="*60 + "\n")
            self.log(f"✅ Pipeline complete!\n")
            self.log(f"📁 Output scene: {colored}\n")
            self.log(f"📂 CT slices (with precise affine): {dicom_path}\n")
            self.log("="*60 + "\n")

        # -------------------------------------------------
        # POST: Launch Web Viewer via HTTP server (portable, robust)
        # -------------------------------------------------
        if self.mode == "all":
            self.log("\n" + "=" * 60 + "\n")
            self.log("🌐 Step 7: Launching Web Viewer (HTTP, portable)...\n")
            self.log("=" * 60 + "\n")

            selected_tasks = self.cfg.get("tasks", "total_all")
            stl_dir = out_seg / selected_tasks

            # -------------------------------------------------
            # 1. Generate stl_list.json
            # -------------------------------------------------
            try:
                generate_stl_list_json(stl_dir)
                self.log(f"✅ Generated stl_list.json in {stl_dir}\n")
            except Exception as e:
                self.log(f"❌ Failed to generate stl_list.json: {e}\n")
                return 1

            # -------------------------------------------------
            # 2. Resolve project root
            # -------------------------------------------------
            project_root = Path(__file__).resolve().parents[2]
            self.log(f"📁 Project root: {project_root}\n")

            # -------------------------------------------------
            # 3. Check if server port is already in use
            # -------------------------------------------------
            def _port_in_use(port: int) -> bool:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(("127.0.0.1", port)) == 0

            server_script = Path(__file__).with_name("start_viewer_server.py")

            if not server_script.exists():
                self.log(f"❌ start_viewer_server.py not found at {server_script}\n")
                return 1

            if _port_in_use(8000):
                self.log("ℹ️  Web Viewer server already running on port 8000\n")
            else:
                self.log("🚀 Starting Web Viewer server...\n")
                subprocess.Popen(
                    [sys.executable, str(server_script)],
                    cwd=str(server_script.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                import time
                time.sleep(1)

            # -------------------------------------------------
            # 4. Compute STL directory relative to project root
            # -------------------------------------------------
            try:
                rel_stl_dir = stl_dir.resolve().relative_to(project_root)
            except ValueError:
                self.log("❌ STL directory is not inside project root\n")
                self.log(f"   STL dir: {stl_dir.resolve()}\n")
                self.log(f"   Project root: {project_root}\n")
                return 1

            # Safety check (prevent future regressions)
            assert rel_stl_dir.as_posix().startswith("data/")

            # -------------------------------------------------
            # 5. Construct viewer URL (absolute URL path!)
            # -------------------------------------------------
            viewer_url = (
                "http://localhost:8000/"
                "totalsegmentator/bin/web_viewer/viewer_enhanced.html"
                f"?dir=/{rel_stl_dir.as_posix()}"
            )

            self.log(f"🌐 Opening Web Viewer:\n{viewer_url}\n")

            # -------------------------------------------------
            # 6. Open browser (robust, cross-platform)
            # -------------------------------------------------
            self.log("🚀 Opening Web Viewer in browser...\n")

            try:
                if platform.system() == "Darwin":      # macOS
                    subprocess.Popen(["open", viewer_url])
                elif platform.system() == "Linux":
                    subprocess.Popen(["xdg-open", viewer_url])
                elif platform.system() == "Windows":
                    subprocess.Popen(["cmd", "/c", "start", viewer_url])
                else:
                    webbrowser.open(viewer_url)

                self.log("✅ Browser launch command executed\n")

            except Exception as e:
                self.log(f"⚠️ Failed to auto-open browser: {e}\n")
                self.log(f"👉 Please open manually:\n{viewer_url}\n")

            self.log("=" * 60 + "\n")

        return 0



class App(b.Window):
    def __init__(self):
        # Use modern light theme
        super().__init__(themename="flatly")
        self.title("TotalSegmentator - Medical Imaging Pipeline")
        
        # Set minimum size for usability
        self.minsize(950, 900)
        
        # Start with a good default size that scales well
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        window_width = max(950, min(1600, int(screen_width * 0.7)))
        window_height = max(900, min(1000, int(screen_height * 0.75)))
        
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.resizable(True, True)
        
        self.tools_visible = False
        self.log_visible = False
        
        self._build_ui()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: PipelineThread | None = None
        self._is_processing = False


    def _build_ui(self):
        main_frame = b.Frame(self, padding=30)
        main_frame.pack(fill=BOTH, expand=YES)
        
        main_frame.grid_rowconfigure(6, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # HEADER
        header_frame = b.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 25))
        
        title_label = b.Label(
            header_frame,
            text="TotalSegmentator Pipeline",
            font=("Helvetica", 28, "bold"),
            bootstyle="primary"
        )
        title_label.pack(anchor=W)
        
        subtitle_label = b.Label(
            header_frame,
            text="Medical imaging with precise spatial alignment + Web 3D Viewer",
            font=("Helvetica", 13),
            bootstyle="secondary"
        )
        subtitle_label.pack(anchor=W, pady=(8, 0))

        b.Separator(main_frame, bootstyle="secondary").pack(fill=X, pady=(0, 25))

        # INPUT/OUTPUT
        io_frame = b.LabelFrame(
            main_frame,
            text="  Input & Output  ",
            padding=25,
            bootstyle="primary"
        )
        io_frame.pack(fill=X, expand=NO, pady=(0, 20))
        io_frame.grid_columnconfigure(1, weight=1)

        b.Label(io_frame, text="DICOM Folder").grid(
            row=0, column=0, sticky="w", pady=12, padx=(0, 20)
        )
        self.e_dicom = b.Entry(io_frame, )
        self.e_dicom.grid(row=0, column=1, sticky="we", padx=(0, 15), ipady=4)
        b.Button(
            io_frame,
            text="Browse...",
            bootstyle="primary-outline",
            command=self._pick_dicom,
            width=15
        ).grid(row=0, column=2)

        b.Label(io_frame, text="Output Folder").grid(
            row=1, column=0, sticky="w", pady=12, padx=(0, 20)
        )
        self.e_out = b.Entry(io_frame, )
        self.e_out.grid(row=1, column=1, sticky="we", padx=(0, 15), ipady=4)
        b.Button(
            io_frame,
            text="Browse...",
            bootstyle="primary-outline",
            command=self._pick_out,
            width=15
        ).grid(row=1, column=2)

        # CONFIGURATION
        config_frame = b.LabelFrame(
            main_frame,
            text="  Configuration  ",
            padding=25,
            bootstyle="info"
        )
        config_frame.pack(fill=X, expand=NO, pady=(0, 20))

        config_inner = b.Frame(config_frame)
        config_inner.pack(fill=X)
        config_inner.grid_columnconfigure(1, weight=1)
        config_inner.grid_columnconfigure(3, weight=1)
        
        b.Label(config_inner, text="Project Name").grid(
            row=0, column=0, sticky="w", pady=12, padx=(0, 20)
        )
        self.e_case = b.Entry(config_inner, width=25, )
        self.e_case.insert(0, "Project-01")
        self.e_case.grid(row=0, column=1, sticky="w", ipady=4)

        b.Label(config_inner, text="Blender Scale").grid(
            row=0, column=2, sticky="w", pady=12, padx=(60, 20)
        )
        self.v_scale = b.StringVar(value="0.01")
        scale_entry = b.Entry(config_inner, textvariable=self.v_scale, width=12, )
        scale_entry.grid(row=0, column=3, sticky="w", ipady=4)
        
        b.Label(config_inner, text="Segmentation Tasks").grid(
            row=1, column=0, sticky="w", pady=12, padx=(0, 20)
        )
        self.v_tasks = b.StringVar(value="total_all")
        tasks_combo = b.Combobox(
            config_inner,
            textvariable=self.v_tasks,
            values=["total_all", "liver_segments", "liver_vessels", "total_vessels"],
            state="readonly",
            width=22
        )
        tasks_combo.grid(row=1, column=1, sticky="w", ipady=4)
        
    

        # TOOL PATHS
        tools_header_frame = b.Frame(main_frame)
        tools_header_frame.pack(fill=X, pady=(0, 8))
        
        self.tools_toggle_btn = b.Button(
            tools_header_frame,
            text="▶ Advanced: Tool Paths (Optional)",
            bootstyle="link",
            command=self._toggle_tools,
            cursor="hand2"
        )
        self.tools_toggle_btn.pack(anchor=W)

        self.tools_frame = b.Frame(main_frame)
        
        tools_content = b.LabelFrame(
            self.tools_frame,
            text="  Tool Paths  ",
            padding=25,
            bootstyle="secondary"
        )
        tools_content.pack(fill=X)
        tools_content.grid_columnconfigure(1, weight=1)

        b.Label(tools_content, text="Blender Executable", ).grid(
            row=0, column=0, sticky="w", pady=10, padx=(0, 20)
        )
        self.e_blender = b.Entry(tools_content, font=("Helvetica", 10))
        self.e_blender.grid(row=0, column=1, sticky="we", padx=(0, 15), ipady=4)
        b.Button(
            tools_content,
            text="Browse...",
            bootstyle="secondary-outline",
            command=self._pick_blender,
            width=15
        ).grid(row=0, column=2)

        b.Label(tools_content, text="dcm2niix Executable", ).grid(
            row=1, column=0, sticky="w", pady=10, padx=(0, 20)
        )
        self.e_dcm2niix = b.Entry(tools_content, font=("Helvetica", 10))
        self.e_dcm2niix.grid(row=1, column=1, sticky="we", padx=(0, 15), ipady=4)
        b.Button(
            tools_content,
            text="Browse...",
            bootstyle="secondary-outline",
            command=self._pick_dcm2niix,
            width=15
        ).grid(row=1, column=2)

        # RUN PIPELINE
        run_frame = b.LabelFrame(
            main_frame,
            text="  Run Pipeline  ",
            padding=25,
            bootstyle="success"
        )
        run_frame.pack(fill=X, expand=NO, pady=(0, 20))

        button_frame = b.Frame(run_frame)
        button_frame.pack(fill=X, pady=(0, 20))
        
        self.run_all_btn = b.Button(
            button_frame,
            text="▶ Run Full Pipeline",
            bootstyle="success",
            command=lambda: self._start('all')
        )
        self.run_all_btn.pack(side=LEFT, fill=X, expand=YES, padx=(0, 8), ipady=8)

        step_frame = b.Frame(run_frame)
        step_frame.pack(fill=X)
        
        b.Label(step_frame, text="Individual Steps:").pack(anchor=W, pady=(0, 10))
        
        steps_container = b.Frame(step_frame)
        steps_container.pack(fill=X)
        
        step_buttons = [
            ("1. DICOM→PNG", "step1"),
            ("2. DICOM→NIfTI", "step2"),
            ("3. Segment", "step3"),
            ("4. Vessel Refinement", "step4"),
            ("5. Import", "step5"),
            ("6. Materials", "step6")
        ]
        
        for i, (text, mode) in enumerate(step_buttons):
            btn = b.Button(
                steps_container,
                text=text,
                bootstyle="info-outline",
                command=lambda m=mode: self._start(m)
            )
            btn.grid(row=0, column=i, sticky='ew', padx=4, ipady=5)
            steps_container.grid_columnconfigure(i, weight=1)

        # PROGRESS
        progress_frame = b.Frame(main_frame)
        progress_frame.pack(fill=X, pady=(0, 15))
        
        self.status_label = b.Label(
            progress_frame,
            text="● Ready to start",
            font=("Helvetica", 14, "bold"),
            bootstyle="secondary"
        )
        self.status_label.pack(anchor=W, pady=(0, 12))
        
        self.pb = b.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=100,
            bootstyle="success-striped"
        )
        self.pb.pack(fill=X, ipady=8)

        # LOG
        log_header_frame = b.Frame(main_frame)
        log_header_frame.pack(fill=X, pady=(20, 8))
        
        self.log_toggle_btn = b.Button(
            log_header_frame,
            text="▶ Show Process Log",
            bootstyle="link",
            command=self._toggle_log,
            cursor="hand2",
        )
        self.log_toggle_btn.pack(anchor=W)

        self.log_frame = b.Frame(main_frame)
        
        log_content = b.LabelFrame(
            self.log_frame,
            text="  Process Log  ",
            padding=20,
            bootstyle="secondary"
        )
        log_content.pack(fill=BOTH, expand=YES)
        
        log_inner = b.Frame(log_content)
        log_inner.pack(fill=BOTH, expand=YES)
        log_inner.grid_rowconfigure(0, weight=1)
        log_inner.grid_columnconfigure(0, weight=1)
        
        self.txt = ScrolledText(
            log_inner,
            height=15,
            autohide=True,
            bootstyle="secondary"
        )
        self.txt.grid(row=0, column=0, sticky="nsew")
        
        self.txt.text.configure(
            bg="#f8f9fa",
            fg="#212529",
            insertbackground="#000000",
            font=("Consolas", 10)
        )

    def _detect_vmtk(self):
        if not vmtk_detector:
            return
        
        try:
            self.vmtk_config = vmtk_detector.VMTKDetector.auto_detect()
        except Exception as e:
            print(f"⚠️  VMTK detection failed: {e}")
            self.vmtk_config = None
            return
        
        if getattr(self.vmtk_config, "available", False):
            msg = getattr(self.vmtk_config, "message", "VMTK detected")
            print(f"✅ {msg}")
        else:
            msg = getattr(self.vmtk_config, "message", "VMTK not available")
            print(f"⚠️  {msg}")
            if hasattr(self, "vessel_enhance_check"):
                try:
                    self.vessel_enhance_check.config(state="disabled")
                except Exception:
                    pass

    def _toggle_tools(self):
        if self.tools_visible:
            self.tools_frame.pack_forget()
            self.tools_toggle_btn.config(text="▶ Advanced: Tool Paths (Optional)")
            self.tools_visible = False
        else:
            self.tools_frame.pack(fill=X, pady=(0, 20), after=self.tools_toggle_btn.master)
            self.tools_toggle_btn.config(text="▼ Advanced: Tool Paths (Optional)")
            self.tools_visible = True

    def _toggle_log(self):
        if self.log_visible:
            self.log_frame.pack_forget()
            self.log_toggle_btn.config(text="▶ Show Process Log")
            self.log_visible = False
        else:
            self.log_frame.pack(fill=BOTH, expand=YES, after=self.log_toggle_btn.master)
            self.log_toggle_btn.config(text="▼ Hide Process Log")
            self.log_visible = True

    def _pick_dicom(self):
        path = filedialog.askdirectory(title="Select DICOM folder")
        if path:
            self.e_dicom.delete(0, END)
            self.e_dicom.insert(0, path)

    def _pick_out(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.e_out.delete(0, END)
            self.e_out.insert(0, path)

    def _pick_blender(self):
        path = filedialog.askopenfilename(
            title="Select Blender executable",
            filetypes=[("Executables", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.e_blender.delete(0, END)
            self.e_blender.insert(0, path)

    def _pick_dcm2niix(self):
        path = filedialog.askopenfilename(
            title="Select dcm2niix executable",
            filetypes=[("Executables", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.e_dcm2niix.delete(0, END)
            self.e_dcm2niix.insert(0, path)

    def _append_log(self, msg: str):
        self.txt.insert(END, msg)
        self.txt.see(END)

    def _start(self, mode: str):
        if self._is_processing:
            Messagebox.show_warning("Already running", "A pipeline is already in progress.")
            return
            
        dicom_dir = self.e_dicom.get().strip()
        out_root = self.e_out.get().strip()
        if not dicom_dir or not out_root:
            Messagebox.show_error(
                "Please select both DICOM folder and output folder.", "Missing Paths"
            )
            return
        cfg = {
            "dicom_dir": dicom_dir,
            "out_root": out_root,
            "case_name": self.e_case.get().strip() or "Project-01",
            "scale": self.v_scale.get().strip() or "0.01",
            "blender_path": self.e_blender.get().strip(),
            "dcm2niix_path": self.e_dcm2niix.get().strip(),
            "tasks": self.v_tasks.get(),
            "vmtk_config": getattr(self, "vmtk_config", None),
        }
        
        if not self.log_visible:
            self._toggle_log()
        
        self.txt.delete("1.0", END)
        self.pb['value'] = 0
        self._is_processing = True
        self.status_label.config(text="● Processing...", bootstyle="warning")
        #self.pb.start()
        self.worker = PipelineThread(self.log_queue, cfg, mode=mode)
        self.worker.start()
        self.after(50, self._drain)

    def _drain(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__DONE__":
                    self.pb.stop()
                    self._is_processing = False
                    if self.worker and self.worker.rc == 0:
                        self.pb['value'] = 100
                        self.status_label.config(text="✓ Complete!", bootstyle="success")
                        Messagebox.ok("Pipeline completed successfully!\n\nWeb viewer should open automatically.", "Success")
                    else:
                        self.pb['value'] = 0
                        self.status_label.config(text="✗ Failed", bootstyle="danger")
                        Messagebox.show_error(
                            "Pipeline failed. Check the log for details.", "Error"
                        )
                    return
                elif isinstance(msg, str) and msg.startswith("__PROG__:"):
                    try:
                        progress_val = int(msg.split(":", 1)[1])
                        self.pb.stop()
                        self.pb['value'] = progress_val
                        
                        if progress_val == 20:
                            self.status_label.config(text="● Exporting PNG slices... (1/6)", bootstyle="info")
                        elif progress_val == 40:
                            self.status_label.config(text="● Converting to NIfTI... (2/6)", bootstyle="info")
                        elif progress_val == 70:
                            self.status_label.config(text="● Segmenting organs... (3/6)", bootstyle="info")
                        elif progress_val == 75:
                            self.status_label.config(text="● Refining vessels... (4/6)", bootstyle="info")
                        elif progress_val == 85:
                            self.status_label.config(text="● Importing to Blender... (5/6)", bootstyle="info")
                        elif progress_val == 100:
                            self.status_label.config(text="● Applying materials... (6/6)", bootstyle="info")
                    except Exception:
                        pass
                self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain)


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"GUI failed to start: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()