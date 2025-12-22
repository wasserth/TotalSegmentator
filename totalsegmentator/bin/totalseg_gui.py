#!/usr/bin/env python3
"""
totalseg_gui

A simple Tkinter GUI to run the six steps:
  1) DICOM -> NIfTI (dcm2niix)
  2) NIfTI -> PNG slices (axial/coronal/sagittal)
  3) Segmentation + STL export (TotalSegmentatorImproved)
  4) Import to Blender with exact palette & collections (totalseg_blender_import.py)
  5) Apply exact materials again (totalseg_material.py)
  6) Add interactive DICOM slice viewer (totalseg_blender_slider.py)

Windows/Linux usage:
  python -m totalsegmentator.bin.totalseg_gui

macOS CLI mode (if tkinter unavailable):
  python -m totalsegmentator.bin.totalseg_gui --cli \
    --dicom /path/to/dicom --output /path/to/output
"""

from __future__ import annotations

import sys
import threading
import queue
import subprocess
from pathlib import Path
import os
import shutil
import argparse

# Try to import GUI dependencies, but allow fallback to CLI mode
GUI_AVAILABLE = False
TKINTER_ERROR = None
try:
    from tkinter import filedialog, END
    import ttkbootstrap as b
    from ttkbootstrap.constants import *
    from ttkbootstrap.widgets.scrolled import ScrolledText
    from ttkbootstrap.dialogs import Messagebox
    GUI_AVAILABLE = True
except ImportError as e:
    TKINTER_ERROR = str(e)
    # Only exit if user didn't request CLI mode
    if '--cli' not in sys.argv and '--help' not in sys.argv and '-h' not in sys.argv:
        print("=" * 70, file=sys.stderr)
        print("ERROR: GUI dependencies not available", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"\nMissing: {e}", file=sys.stderr)
        print("\nTo use GUI mode, install dependencies:", file=sys.stderr)
        print("  pip install ttkbootstrap", file=sys.stderr)
        print("\nOn macOS, tkinter may also require:", file=sys.stderr)
        print("  - Official Python from python.org (includes tkinter)")
        print("  - Or: brew install python-tk@3.11", file=sys.stderr)
        print("\nAlternatively, use CLI mode:", file=sys.stderr)
        print("  python -m totalsegmentator.bin.totalseg_gui --cli \\", file=sys.stderr)
        print("    --dicom /path/to/dicom --output /path/to/output", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)


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


def export_png_slices(nii_path: Path, out_dir: Path, log):
    log(f"Exporting PNG slices from {nii_path} to {out_dir}\n")
    try:
        import numpy as np
        import nibabel as nib
        import imageio.v2 as iio
    except Exception as e:
        log(
            f"Missing Python deps for slice export: {e}. Install with: pip install nibabel imageio\n"
        )
        return 1

    try:
        img = nib.load(str(nii_path))
        data = img.get_fdata()
        out_dir.mkdir(parents=True, exist_ok=True)

        def write_axis(arr, subfolder):
            d = out_dir / subfolder
            d.mkdir(parents=True, exist_ok=True)
            vmin, vmax = np.percentile(arr, (1, 99))
            for i, sl in enumerate(arr, start=1):
                s = np.clip((sl - vmin) / (vmax - vmin + 1e-9), 0, 1)
                iio.imwrite(d / f"{i:03d}.png", (s * 255).astype(np.uint8))

        write_axis(np.moveaxis(data, 2, 0), "axial")
        write_axis(np.moveaxis(data, 1, 0), "coronal")
        write_axis(np.moveaxis(data, 0, 0), "sagittal")
        return 0
    except Exception as e:
        log(f"Slice export failed: {e}\n")
        return 1


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
        # Always enable mirroring to flip organs left-to-right
        mirrored = True
        scale = float(self.cfg["scale"]) if self.cfg["scale"] else 0.01
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

        # 1) DICOM -> NIfTI
        nii_path = out_nii / f"{case_name}.nii.gz"
        if self.mode in ("all", "step1"):
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
            _progress(15)
            if self.mode == 'step1':
                return 0
        else:
            self.log(f"Reusing existing NIfTI: {nii_path}\n")

        # 2) NIfTI -> PNG slices
        if self.mode in ("all", "step2"):
            rc = export_png_slices(nii_path, out_png, self.log)
            if rc != 0:
                return rc
            _progress(30)
            if self.mode == 'step2':
                return 0

        # 3) Segmentation + STL export
        if self.mode in ("all", "step3"):
            rc = run_cmd(
                [
                    sys.executable,
                    "-m",
                    "totalsegmentator.bin.TotalSegmentatorImproved",
                    "-i",
                    str(nii_path),
                    "-o",
                    str(out_seg),
                    "--tasks",
                    "all",
                    "--with-liver-vessels",
                    "--smoothing",
                    "heavy",
                    "--export-mesh",
                    "--export-format",
                    "stl",
                    "--units",
                    "m",
                    "--mesh-smooth-iters",
                    "30",
                    "--device",
                    "gpu",
                ],
                self.log,
            )
            if rc != 0:
                return rc
            _progress(60)
            if self.mode == 'step3':
                return 0

        # Locate mesh dir
        stl_dir = out_seg / "total_all"
        if not stl_dir.exists():
            candidates = [
                p for p in out_seg.glob("**/*") if p.is_dir() and list(p.glob("*.stl"))
            ]
            if candidates:
                stl_dir = candidates[0]
        self.log(f"Using mesh directory: {stl_dir}\n")

        # 4) Blender import with exact palette
        scene_setup = out_blend_dir / "scene-setup.blend"
        if self.mode in ("all", "step4"):
            blender_exe = _resolve_blender()
            if not blender_exe:
                self.log("Blender not found. Set path in GUI or add to PATH.\n")
                return 127
            blender_import_cmd = [
                blender_exe,
                "-b",
                "-P",
                str(Path(__file__).with_name("totalseg_blender_import.py")),
                "--",
                "--stl-dir",
                str(stl_dir),
                "--units",
                "m",
                "--collection",
                "Organs",
                "--group-categories",
                "--palette",
                "exact",
                "--scale",
                str(scale),
                "--mirror-x",
                "true",
            ]
            blender_import_cmd += ["--save", str(scene_setup)]
            rc = run_cmd(blender_import_cmd, self.log)
            if rc != 0:
                return rc
            _progress(75)
            if self.mode == 'step4':
                return 0

        # 5) Apply exact materials again (idempotent)
        colored = out_blend_dir / "scene-colored.blend"
        if self.mode in ("all", "step5"):
            blender_exe = _resolve_blender()
            if not blender_exe:
                self.log("Blender not found. Set path in GUI or add to PATH.\n")
                return 127
            rc = run_cmd(
                [
                    blender_exe,
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
            _progress(90)
            if self.mode == 'step5':
                return 0

        # Step 6 removed - user can manually add addon in Blender
        
        _progress(100)
        self.log(f"\nPipeline complete! Final scene: {colored}\n")
        self.log(f"\nTo add DICOM slider:\n")
        self.log(f"1. Open the .blend file in Blender\n")
        self.log(f"2. Open Text Editor, create new text block\n")
        self.log(f"3. Copy dicom_slider_addon.py into it\n")
        self.log(f"4. Run the script (Alt+P)\n")
        self.log(f"5. Configure path in DICOM tab: {out_png}\n")
        return 0


# Only define GUI class if dependencies are available
if GUI_AVAILABLE:
    class App(b.Window):
        def __init__(self):
            # Use modern light theme
            super().__init__(themename="flatly")
            self.title("TotalSegmentator - Medical Imaging Pipeline")
            
            # Set minimum size for usability
            self.minsize(950, 900)
            
            # Start with a good default size that scales well
            # Get screen dimensions
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            
            # Calculate responsive window size (70% of screen, min 950x900, max 1600x1100)
            window_width = max(950, min(1600, int(screen_width * 0.7)))
            window_height = max(900, min(1100, int(screen_height * 0.75)))
            
            # Center the window
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            self.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.resizable(True, True)
            
            # State for collapsible sections
            self.tools_visible = False
            self.log_visible = False
            
            self._build_ui()
            self.log_queue: queue.Queue[str] = queue.Queue()
            self.worker: PipelineThread | None = None
            self._is_processing = False

        def _build_ui(self):
            # Main container with responsive padding
            main_frame = b.Frame(self, padding=30)
            main_frame.pack(fill=BOTH, expand=YES)
            
            # Make main_frame responsive
            main_frame.grid_rowconfigure(6, weight=1)  # Log section gets extra space
            main_frame.grid_columnconfigure(0, weight=1)

            # --- HEADER SECTION ---
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
                text="Automated medical imaging segmentation with Blender 3D visualization",
                font=("Helvetica", 13),
                bootstyle="secondary"
            )
            subtitle_label.pack(anchor=W, pady=(8, 0))

            # Separator
            b.Separator(main_frame, bootstyle="secondary").pack(fill=X, pady=(0, 25))

            # --- SECTION 1: INPUT/OUTPUT ---
            io_frame = b.Labelframe(
                main_frame,
                text="  Input & Output  ",
                padding=25,
                bootstyle="primary"
            )
            io_frame.pack(fill=X, expand=NO, pady=(0, 20))
            io_frame.grid_columnconfigure(1, weight=1)

            # DICOM folder
            b.Label(io_frame, text="DICOM Folder").grid(
                row=0, column=0, sticky="w", pady=12, padx=(0, 20)
            )
            self.e_dicom = b.Entry(io_frame)
            self.e_dicom.grid(row=0, column=1, sticky="we", padx=(0, 15), ipady=4)
            b.Button(
                io_frame,
                text="Browse...",
                bootstyle="primary-outline",
                command=self._pick_dicom,
                width=15
            ).grid(row=0, column=2)

            # Output folder
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

            # --- SECTION 2: CONFIGURATION ---
            config_frame = b.Labelframe(
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
            
            # Case name
            b.Label(config_inner, text="Project Name").grid(
                row=0, column=0, sticky="w", pady=12, padx=(0, 20)
            )
            self.e_case = b.Entry(config_inner, width=25)
            self.e_case.insert(0, "Project-01")
            self.e_case.grid(row=0, column=1, sticky="w", ipady=4)

            # Scale
            b.Label(config_inner, text="Blender Scale").grid(
                row=0, column=2, sticky="w", pady=12, padx=(60, 20)
            )
            self.v_scale = b.StringVar(value="0.01")
            scale_entry = b.Entry(config_inner, textvariable=self.v_scale, width=12)
            scale_entry.grid(row=0, column=3, sticky="w", ipady=4)

            # --- SECTION 3: TOOL PATHS (COLLAPSIBLE) ---
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
            # Don't pack it yet - it's hidden by default
            
            tools_content = b.Labelframe(
                self.tools_frame,
                text="  Tool Paths  ",
                padding=25,
                bootstyle="secondary"
            )
            tools_content.pack(fill=X)
            tools_content.grid_columnconfigure(1, weight=1)

            b.Label(tools_content, text="Blender Executable").grid(
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

            b.Label(tools_content, text="dcm2niix Executable").grid(
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

            # --- SECTION 4: EXECUTION ---
            run_frame = b.Labelframe(
                main_frame,
                text="  Run Pipeline  ",
                padding=25,
                bootstyle="success"
            )
            run_frame.pack(fill=X, expand=NO, pady=(0, 20))

            # Main action button
            button_frame = b.Frame(run_frame)
            button_frame.pack(fill=X, pady=(0, 20))
            
            self.run_all_btn = b.Button(
                button_frame,
                text="▶ Run Full Pipeline",
                bootstyle="success",
                command=lambda: self._start('all')
            )
            self.run_all_btn.pack(side=LEFT, fill=X, expand=YES, padx=(0, 8), ipady=8)

            # Individual step buttons
            step_frame = b.Frame(run_frame)
            step_frame.pack(fill=X)
            
            b.Label(step_frame, text="Individual Steps:").pack(anchor=W, pady=(0, 10))
            
            steps_container = b.Frame(step_frame)
            steps_container.pack(fill=X)
            
            step_buttons = [
                ("1. DICOM→NIfTI", "step1"),
                ("2. NIfTI→PNG", "step2"),
                ("3. Segment", "step3"),
                ("4. Import", "step4"),
                ("5. Materials", "step5"),
                ("6. Slice Viewer", "step6")  # NEW
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

            # --- SECTION 5: PROGRESS ---
            progress_frame = b.Frame(main_frame)
            progress_frame.pack(fill=X, pady=(0, 15))
            
            # Status label
            self.status_label = b.Label(
                progress_frame,
                text="● Ready to start",
                font=("Helvetica", 14, "bold"),
                bootstyle="secondary"
            )
            self.status_label.pack(anchor=W, pady=(0, 12))
            
            # Progress bar - make it taller on larger screens
            self.pb = b.Progressbar(
                progress_frame,
                mode="determinate",
                maximum=100,
                bootstyle="success-striped"
            )
            self.pb.pack(fill=X, ipady=8)

            # --- SECTION 6: LOG (COLLAPSIBLE) ---
            log_header_frame = b.Frame(main_frame)
            log_header_frame.pack(fill=X, pady=(20, 8))
            
            self.log_toggle_btn = b.Button(
                log_header_frame,
                text="▶ Show Process Log",
                bootstyle="link",
                command=self._toggle_log,
                cursor="hand2"
            )
            self.log_toggle_btn.pack(anchor=W)

            self.log_frame = b.Frame(main_frame)
            # Don't pack it yet - it's hidden by default
            
            log_content = b.Labelframe(
                self.log_frame,
                text="  Process Log  ",
                padding=20,
                bootstyle="secondary"
            )
            log_content.pack(fill=BOTH, expand=YES)
            
            # Make log area responsive - height adjusts based on window size
            log_inner = b.Frame(log_content)
            log_inner.pack(fill=BOTH, expand=YES)
            log_inner.grid_rowconfigure(0, weight=1)
            log_inner.grid_columnconfigure(0, weight=1)
            
            self.txt = ScrolledText(
                log_inner,
                height=15,  # Increased default height
                autohide=True,
                bootstyle="secondary"
            )
            self.txt.grid(row=0, column=0, sticky="nsew")
            
            # Configure text widget
            self.txt.text.configure(
                bg="#f8f9fa",
                fg="#212529",
                insertbackground="#000000",
                font=("Consolas", 10)
            )

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
            }
            
            # Auto-show log when starting
            if not self.log_visible:
                self._toggle_log()
            
            self.txt.delete("1.0", END)
            self.pb['value'] = 0
            self._is_processing = True
            self.status_label.config(text="● Processing...", bootstyle="warning")
            self.pb.start()
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
                            self.status_label.config(text="✓ Complete! (6/6)", bootstyle="success")
                            Messagebox.ok("Pipeline completed successfully!", "Success")
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
                            
                            # Update status text based on progress with step numbers
                            if progress_val == 15:
                                self.status_label.config(text="● Converting DICOM... (1/6)", bootstyle="info")
                            elif progress_val == 30:
                                self.status_label.config(text="● Exporting slices... (2/6)", bootstyle="info")
                            elif progress_val == 60:
                                self.status_label.config(text="● Segmenting organs... (3/6)", bootstyle="info")
                            elif progress_val == 75:
                                self.status_label.config(text="● Importing to Blender... (4/6)", bootstyle="info")
                            elif progress_val == 90:
                                self.status_label.config(text="● Applying materials... (5/6)", bootstyle="info")
                            elif progress_val == 100:
                                self.status_label.config(text="● Adding slice viewer... (6/6)", bootstyle="info")
                        except Exception:
                            pass
                    self._append_log(msg)
            except queue.Empty:
                pass
            self.after(100, self._drain)


def run_cli_mode(args):
    """Run pipeline in CLI mode without GUI."""
    print("=" * 70)
    print("TotalSegmentator Pipeline - CLI Mode")
    print("=" * 70)
    
    if not args.dicom or not args.output:
        print("\nError: --dicom and --output are required", file=sys.stderr)
        print("\nUsage:", file=sys.stderr)
        print("  python -m totalsegmentator.bin.totalseg_gui --cli \\", file=sys.stderr)
        print("    --dicom /path/to/dicom \\", file=sys.stderr)
        print("    --output /path/to/output \\", file=sys.stderr)
        print("    [--case-name Project-01] \\", file=sys.stderr)
        print("    [--scale 0.01]", file=sys.stderr)
        return 1
    
    cfg = {
        "dicom_dir": args.dicom,
        "out_root": args.output,
        "case_name": args.case_name or "Project-01",
        "scale": str(args.scale or 0.01),
        "blender_path": args.blender or "",
        "dcm2niix_path": args.dcm2niix or "",
    }
    
    print(f"\nConfiguration:")
    print(f"  DICOM folder: {cfg['dicom_dir']}")
    print(f"  Output folder: {cfg['out_root']}")
    print(f"  Project name: {cfg['case_name']}")
    print(f"  Scale: {cfg['scale']}")
    print()
    
    log_queue: queue.Queue[str] = queue.Queue()
    
    def print_log():
        while True:
            try:
                msg = log_queue.get(timeout=0.1)
                if msg == "__DONE__":
                    break
                elif isinstance(msg, str) and msg.startswith("__PROG__:"):
                    progress = msg.split(":", 1)[1]
                    print(f"\rProgress: {progress}%", end="", flush=True)
                else:
                    print(msg, end="", flush=True)
            except queue.Empty:
                continue
    
    log_thread = threading.Thread(target=print_log, daemon=True)
    log_thread.start()
    
    worker = PipelineThread(log_queue, cfg, mode='all')
    worker.start()
    worker.join()
    
    print()
    if worker.rc == 0:
        print("\n" + "=" * 70)
        print("✓ Pipeline completed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Pipeline failed. Check the log above for details.")
        print("=" * 70)
        return worker.rc


def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator Pipeline GUI/CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no GUI)")
    parser.add_argument("--dicom", help="DICOM input folder")
    parser.add_argument("--output", help="Output folder")
    parser.add_argument("--case-name", default="Project-01", help="Project name")
    parser.add_argument("--scale", type=float, default=0.01, help="Blender scale factor")
    parser.add_argument("--blender", help="Path to Blender executable (optional)")
    parser.add_argument("--dcm2niix", help="Path to dcm2niix executable (optional)")
    
    args = parser.parse_args()
    
    # Use CLI mode if requested or if GUI is unavailable
    if args.cli or not GUI_AVAILABLE:
        if not GUI_AVAILABLE and not args.cli:
            print(f"Note: GUI unavailable ({TKINTER_ERROR}), using CLI mode\n", file=sys.stderr)
        return run_cli_mode(args)
    
    # Launch GUI
    try:
        app = App()
        app.mainloop()
        return 0
    except Exception as e:
        print(f"GUI failed to start: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
