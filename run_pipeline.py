import argparse
import subprocess
import sys
import os
from pathlib import Path

# Force UTF-8 encoding and unbuffered output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

def log(msg):
    """Print with immediate flush for real-time updates."""
    print(msg, flush=True)

def progress(percent, step_name=""):
    """Send progress update in parseable format."""
    log(f"__PROGRESS__:{percent}:{step_name}")

def main():
    parser = argparse.ArgumentParser(description='TotalSegmentator Pipeline')
    parser.add_argument('--dicom-dir', required=True, help='DICOM directory name')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--project-name', required=True, help='Blender project name')
    parser.add_argument('--scale', default='0.01', help='Blender scale factor')
    parser.add_argument('--mode', default='all', help='Pipeline mode: all, step1-step6')
    
    args = parser.parse_args()
    
    # Define paths
    web_app_dir = Path(__file__).parent / 'web-app'
    uploads_dir = web_app_dir / 'uploads'
    dicom_path = uploads_dir / args.dicom_dir
    
    # Handle output path - check if it's absolute or relative
    output_dir_arg = Path(args.output_dir)
    
    # If path contains folder name from browser (e.g., "MyOutputFolder")
    # treat it as a folder name in uploads unless it's clearly absolute
    if output_dir_arg.is_absolute():
        # Absolute path like C:\Users\...\Downloads\Output
        output_path = output_dir_arg
    elif '/' in args.output_dir or '\\' in args.output_dir:
        # Contains path separators - likely from folder selection
        # Resolve relative to web-app directory
        output_path = (web_app_dir / args.output_dir).resolve()
    else:
        # Simple folder name - use in uploads directory
        output_path = uploads_dir / args.output_dir
    
    log(f"\n{'='*70}")
    log(f"PIPELINE CONFIGURATION")
    log(f"{'='*70}")
    log(f"DICOM Directory: {dicom_path}")
    log(f"Output Directory: {output_path}")
    log(f"Project Name: {args.project_name}")
    log(f"Scale: {args.scale}")
    log(f"Mode: {args.mode}")
    log(f"{'='*70}\n")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    progress(0, "Initializing")
    
    # Use the existing GUI's pipeline in CLI mode
    log("\n[*] Starting TotalSegmentator pipeline...\n")
    
    gui_cmd = [
        sys.executable,
        '-u',  # Unbuffered output
        '-m', 'totalsegmentator.bin.totalseg_gui',
        '--cli',
        '--dicom', str(dicom_path),
        '--output', str(output_path),
        '--case-name', args.project_name,
        '--scale', args.scale
    ]
    
    log(f"Command: {' '.join(gui_cmd)}\n")
    progress(5, "Starting pipeline")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            gui_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
        )
        
        # Track progress based on log messages
        current_progress = 5
        
        # Stream output line by line
        for line in process.stdout:
            log(line.rstrip())
            
            # Update progress based on log content
            line_lower = line.lower()
            if 'dicom' in line_lower and 'nifti' in line_lower:
                progress(15, "Converting DICOM to NIfTI")
            elif 'png' in line_lower or 'slice' in line_lower:
                progress(30, "Exporting PNG slices")
            elif 'segment' in line_lower or 'totalsegmentator' in line_lower:
                progress(50, "Running segmentation")
            elif 'mesh' in line_lower or 'stl' in line_lower:
                progress(65, "Generating 3D meshes")
            elif 'blender' in line_lower and 'import' in line_lower:
                progress(75, "Importing to Blender")
            elif 'material' in line_lower or 'color' in line_lower:
                progress(90, "Applying materials")
            elif 'complete' in line_lower or 'done' in line_lower:
                progress(95, "Finalizing")
        
        return_code = process.wait()
        
        if return_code == 0:
            progress(100, "Complete")
            log(f"\n{'='*70}")
            log("[SUCCESS] Pipeline completed successfully!")
            log(f"{'='*70}")
            log(f"Output location: {output_path}")
            log(f"Blender project: {output_path / 'out' / 'scene-colored.blend'}")
            return 0
        else:
            log(f"\n{'='*70}")
            log("[ERROR] Pipeline failed!")
            log(f"{'='*70}")
            log(f"Exit code: {return_code}")
            return return_code
            
    except Exception as e:
        log(f"\n[ERROR] Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
