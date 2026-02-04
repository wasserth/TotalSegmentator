# TotalSegmentator Setup Guide

Complete setup instructions for running TotalSegmentator GUI and Web App from scratch on macOS, Linux, and Windows.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Python Environment Setup](#python-environment-setup)
   - [Option A: Python venv](#option-a-python-venv-recommended)
   - [Option B: Conda](#option-b-conda)
4. [Install TotalSegmentator](#install-totalsegmentator)
5. [Install External Tools](#install-external-tools)
6. [Running TotalSegmentator GUI](#running-totalsegmentator-gui)
7. [Running the Web App](#running-the-web-app)
8. [Verification Steps](#verification-steps)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### All Platforms
- **Git** (for cloning the repository)
- **Python 3.9+** (Python 3.9, 3.10, 3.11, or 3.12 recommended)
- **Node.js v18+** (for web app only)
- **Blender 3.0+** (for 3D visualization)
- **dcm2niix** (for DICOM conversion)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed or version is < 3.9, download from [python.org](https://www.python.org/downloads/)

---

## Clone the Repository

```bash
# Navigate to your desired directory
cd ~/Documents/GitHub  # macOS/Linux
# or
cd C:\Users\YourUsername\Documents\GitHub  # Windows

# Clone the repository
git clone https://github.com/wasserth/TotalSegmentator.git
cd TotalSegmentator
```

---

## Python Environment Setup

Choose **either** Option A (venv) **or** Option B (conda). We recommend venv for simplicity.

### Option A: Python venv (Recommended)

#### macOS / Linux

```bash
# Navigate to project root
cd ~/Documents/GitHub/TotalSegmentator

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Windows (PowerShell)

```powershell
# Navigate to project root
cd C:\Users\YourUsername\Documents\GitHub\TotalSegmentator

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Windows (Command Prompt)

```cmd
cd C:\Users\YourUsername\Documents\GitHub\TotalSegmentator
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
```

---

### Option B: Conda

#### All Platforms

```bash
# Navigate to project root
cd ~/Documents/GitHub/TotalSegmentator  # macOS/Linux
# or
cd C:\Users\YourUsername\Documents\GitHub\TotalSegmentator  # Windows

# Create conda environment with Python 3.13
conda create -n totalseg python=3.13 -y

# Activate conda environment
conda activate totalseg

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## Install TotalSegmentator

**Make sure your virtual environment is activated** before running these commands!

### Step 1: Install Core Package

```bash
# Install TotalSegmentator in development mode with all dependencies
pip install -e .
```

This installs all core dependencies from setup.py:
- torch (PyTorch for deep learning)
- numpy (numerical computing)
- SimpleITK (medical image processing)
- nibabel (NIfTI file handling)
- tqdm (progress bars)
- xvfbwrapper (virtual display)
- nnunetv2 (segmentation framework)
- requests (HTTP library)
- dicom2nifti (DICOM conversion)
- pyarrow (data serialization)
- Pillow (image processing)
- pydicom (DICOM handling)

### Step 2: Install GUI Dependencies

```bash
# Required for GUI
pip install ttkbootstrap nibabel imageio

# Additional GUI-related packages
pip install matplotlib  # For plotting and visualization
```

### Step 3: Install Enhanced Features (Optional)

```bash
# Install enhanced features for advanced processing
pip install -e ".[enhanced]"
```

This adds:
- scikit-image (smoothing and mesh processing)
- trimesh (STL export to Blender)
- scipy (advanced image processing)

### Step 4: Install Web Development Tools (Optional - for web app development)

```bash
# Only needed if you're developing/modifying the web app backend
pip install flask fastapi uvicorn
```

### Complete Installation (All at Once)

If you want to install everything in one go:

```bash
# One-line installation for all features
pip install -e . && pip install ttkbootstrap nibabel imageio matplotlib && pip install -e ".[enhanced]"
```

### Verify Installation

```bash
# Check if TotalSegmentator commands are available
TotalSegmentator --version

# Check if GUI command exists (may not be in PATH, but module should be importable)
python -c "from totalsegmentator.bin import totalseg_gui; print('GUI module: OK')"
```

If these commands work, installation was successful! ✅

---

## Install External Tools

### Blender

#### macOS
```bash
# Option 1: Download from website
# Visit https://www.blender.org/download/ and install .dmg

# Option 2: Using Homebrew
brew install --cask blender

# Verify installation
blender --version
```

#### Linux (Ubuntu/Debian)
```bash
# Option 1: Snap (recommended)
sudo snap install blender --classic

# Option 2: Apt (older version)
sudo apt update
sudo apt install blender

# Verify installation
blender --version
```

#### Windows
```powershell
# Download installer from https://www.blender.org/download/
# Run the .msi installer

# Verify installation (add Blender to PATH during install)
blender --version
```

### dcm2niix

#### macOS
```bash
# Using Homebrew
brew install dcm2niix

# Verify installation
dcm2niix -h
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install dcm2niix

# Verify installation
dcm2niix -h
```

#### Windows
```powershell
# Download latest release from:
# https://github.com/rordenlab/dcm2niix/releases

# Extract to a folder (e.g., C:\Program Files\dcm2niix)
# Add to PATH:
# System Properties → Environment Variables → Path → Add folder location

# Verify installation
dcm2niix -h
```

---

## Running TotalSegmentator GUI

### Step 1: Activate Your Environment

#### venv (macOS/Linux)
```bash
cd ~/Documents/GitHub/TotalSegmentator
source .venv/bin/activate
```

#### venv (Windows PowerShell)
```powershell
cd C:\Users\YourUsername\Documents\GitHub\TotalSegmentator
.\.venv\Scripts\Activate.ps1
```

#### Conda (All Platforms)
```bash
conda activate totalseg
```

### Step 2: Launch the GUI

```bash
# Run the GUI script directly
python totalsegmentator/bin/totalseg_gui.py

# Or use the installed command (if available)
totalseg_gui
```

### GUI Usage

1. **Select Input**: Choose your DICOM folder or NIfTI file
2. **Select Output**: Choose where to save results
3. **Configure Settings**: 
   - Task: Select segmentation task (total, lung_vessels, body, etc.)
   - Fast mode: Enable for faster processing
   - Statistics: Calculate volume statistics
4. **Run Segmentation**: Click "Run TotalSegmentator"
5. **View Results**: Results saved to output folder

---

## Running the Web App

### Prerequisites
- Node.js v18+ installed
- TotalSegmentator Python environment set up (see above)

### Step 1: Install Node.js

#### macOS
```bash
# Option 1: Using Homebrew
brew install node@18

# Option 2: Using nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

#### Linux
```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Or using apt (Ubuntu/Debian)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### Windows
```powershell
# Option 1: Download installer from https://nodejs.org/
# Install the LTS version (v18.x)

# Option 2: Using nvm-windows
# Download from: https://github.com/coreybutler/nvm-windows/releases
# Then run:
nvm install 18.20.0
nvm use 18.20.0
```

### Step 2: Verify Node Installation

```bash
node --version  # Should show v18.x.x or higher
npm --version   # Should show 9.x.x or higher
```

### Step 3: Setup Web App

```bash
# Navigate to web-app directory
cd web-app

# Install Node.js dependencies
npm install

# Create environment file
cp .env.example .env.local  # macOS/Linux
# or
copy .env.example .env.local  # Windows
```

### Step 4: Configure Environment Variables

Edit `.env.local` with your Python path:

#### macOS/Linux
```bash
# Find your Python path
which python  # if venv activated
# or
which python3

# Example .env.local:
PYTHON_PATH=/Users/abeez/Documents/GitHub/TotalSegmentator/.venv/bin/python
```

#### Windows
```powershell
# Find your Python path
where python

# Example .env.local:
PYTHON_PATH=C:/Users/abeez/Documents/GitHub/TotalSegmentator/.venv/Scripts/python.exe
```

### Step 5: Start Development Server

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

### Web App Workflow

1. **Upload DICOM files** through the web interface
2. **Configure** project name, scale, and output settings
3. **Run Pipeline** - executes these steps:
   - DICOM → NIfTI conversion
   - NIfTI → PNG slices export
   - Segmentation + STL mesh export
   - Import meshes into Blender
   - Apply materials and setup viewer

---

## Verification Steps

### Test Python Environment

**Make sure your environment is activated first!**

```bash
# Core dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import SimpleITK; print(f'SimpleITK: {SimpleITK.__version__}')"
python -c "import nibabel; print(f'Nibabel: {nibabel.__version__}')"
python -c "import nnunetv2; print(f'nnUNetv2: {nnunetv2.__version__}')"

# GUI dependencies
python -c "import ttkbootstrap; print(f'ttkbootstrap: {ttkbootstrap.__version__}')"
python -c "import tkinter; print('tkinter: OK')"
python -c "import imageio; print(f'imageio: {imageio.__version__}')"
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

# DICOM/Medical imaging
python -c "import pydicom; print(f'pydicom: {pydicom.__version__}')"
python -c "import dicom2nifti; print('dicom2nifti: OK')"

# Utility libraries
python -c "import tqdm; print(f'tqdm: {tqdm.__version__}')"
python -c "import requests; print(f'requests: {requests.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')"

# Enhanced features (if installed)
python -c "import skimage; print(f'scikit-image: {skimage.__version__}')" 2>/dev/null || echo "scikit-image: Not installed (optional)"
python -c "import trimesh; print(f'trimesh: {trimesh.__version__}')" 2>/dev/null || echo "trimesh: Not installed (optional)"
python -c "import scipy; print(f'scipy: {scipy.__version__}')" 2>/dev/null || echo "scipy: Not installed (optional)"
```

### Test TotalSegmentator Commands

```bash
# Check version
TotalSegmentator --version

# List available tasks
TotalSegmentator --help

# Test other commands
totalseg_get_phase --help
totalseg_get_modality --help
totalseg_download_weights --help
```

### Test TotalSegmentator Modules

```bash
# Test main module
python -c "import totalsegmentator; print('TotalSegmentator module: OK')"

# Test GUI module
python -c "from totalsegmentator.bin import totalseg_gui; print('GUI module: OK')"

# Test libs
python -c "from totalsegmentator.libs import download_pretrained_weights; print('Download module: OK')"
python -c "from totalsegmentator.libs import nostdout; print('Nostdout module: OK')"
```

### Test Web App

```bash
cd web-app
npm run build  # Should complete without errors
```

### Complete Verification Script

Save this as `verify_installation.sh` (macOS/Linux) or `verify_installation.ps1` (Windows):

```bash
#!/bin/bash
# filepath: verify_installation.sh

echo "=== TotalSegmentator Installation Verification ==="
echo ""

echo "1. Checking Python environment..."
python --version
echo ""

echo "2. Checking core dependencies..."
python -c "import torch; import numpy; import SimpleITK; import nibabel; import nnunetv2; print('✓ Core dependencies OK')"
echo ""

echo "3. Checking GUI dependencies..."
python -c "import ttkbootstrap; import tkinter; import imageio; import matplotlib; print('✓ GUI dependencies OK')"
echo ""

echo "4. Checking medical imaging libraries..."
python -c "import pydicom; import dicom2nifti; print('✓ Medical imaging libraries OK')"
echo ""

echo "5. Checking TotalSegmentator commands..."
TotalSegmentator --version
echo ""

echo "6. Checking external tools..."
blender --version 2>/dev/null && echo "✓ Blender installed" || echo "✗ Blender not found"
dcm2niix -h 2>/dev/null | head -1 && echo "✓ dcm2niix installed" || echo "✗ dcm2niix not found"
echo ""

echo "=== Verification Complete ==="
```

Make it executable and run:
```bash
chmod +x verify_installation.sh
./verify_installation.sh
```

---

## Troubleshooting

### Python Environment Issues

#### "command not found: python"
```bash
# Try python3 instead
python3 --version

# Or create an alias (add to ~/.bashrc or ~/.zshrc)
alias python=python3
```

#### "No module named 'torch'"
```bash
# Make sure environment is activated
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows

# Check if venv is activated - you should see (.venv) in prompt
# If still not working, reinstall
pip install torch
```

#### "ImportError: cannot import name 'packaging'"
```bash
pip install --upgrade setuptools packaging
```

#### "ModuleNotFoundError: No module named 'ttkbootstrap'"
```bash
# Reinstall GUI dependencies
pip install ttkbootstrap nibabel imageio matplotlib
```

#### All imports fail after installation
```bash
# Make absolutely sure you're in the right environment
which python  # Should show path to .venv or conda env

# If it shows system python, activate environment:
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows
conda activate totalseg  # Conda

# Then verify again
which python
```

---

### GUI Issues

#### "tkinter not found" (Linux)
```bash
sudo apt-get install python3-tk
```

#### "No display found" (Linux SSH)
```bash
# Enable X11 forwarding
export DISPLAY=:0
# or install virtual display
sudo apt-get install xvfb
```

#### GUI opens but crashes immediately
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall GUI dependencies
pip install --force-reinstall ttkbootstrap nibabel imageio
```

#### "AttributeError: module 'ttkbootstrap' has no attribute 'Style'"
```bash
# Update ttkbootstrap to latest version
pip install --upgrade ttkbootstrap
```

---

### Web App Issues

#### "node: command not found"
- Restart terminal after installing Node.js
- Or restart computer
- Check PATH: `echo $PATH` (macOS/Linux) or `echo %PATH%` (Windows)

#### "npm ERR! code ENOENT"
```bash
# Make sure you're in web-app directory
cd web-app
pwd  # Should end with /web-app

# Remove and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### "Port 3000 already in use"
```bash
# macOS/Linux
lsof -ti:3000 | xargs kill

# Windows
netstat -ano | findstr :3000
# Note the PID, then:
taskkill /PID <PID> /F

# Or use different port
PORT=3001 npm run dev  # macOS/Linux
$env:PORT=3001; npm run dev  # Windows PowerShell
```

#### "Cannot find module '@next/...' "
```bash
# Clear Next.js cache
rm -rf .next
npm run dev
```

---

### External Tools Issues

#### "blender: command not found"
```bash
# Add Blender to PATH
# macOS: Add to ~/.zshrc or ~/.bash_profile
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"

# Linux: Usually installed correctly via package manager
which blender

# Windows: Add during installation or manually add to PATH
# C:\Program Files\Blender Foundation\Blender 3.6
```

#### "dcm2niix: command not found"
```bash
# Verify installation
which dcm2niix  # macOS/Linux
where dcm2niix  # Windows

# If not found, reinstall (see Install External Tools section)
```

---

### Installation Issues

#### "pip install -e . fails"
```bash
# Make sure you're in the TotalSegmentator root directory
pwd  # Should show path to TotalSegmentator

# Check if setup.py exists
ls setup.py

# Update pip first
pip install --upgrade pip setuptools wheel

# Try again
pip install -e .
```

#### "ERROR: Could not build wheels for ..."
```bash
# Install build dependencies
pip install --upgrade pip setuptools wheel build

# For specific packages that need compilers:
# macOS: Install Xcode Command Line Tools
xcode-select --install

# Linux: Install build essentials
sudo apt-get install build-essential python3-dev

# Windows: Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### torch installation fails
```bash
# Install CPU-only version (smaller, faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA support (for NVIDIA GPUs)
# Visit https://pytorch.org/get-started/locally/ for specific CUDA version
```

---

### Permission Issues

#### macOS: "Operation not permitted"
```bash
# Grant terminal full disk access:
# System Preferences → Security & Privacy → Privacy → Full Disk Access
# Add Terminal.app or your terminal emulator
```

#### Linux: "Permission denied"
```bash
# Make scripts executable
chmod +x totalsegmentator/bin/*.py

# If pip install fails with permissions:
pip install --user -e .
```

#### Windows: "Access denied"
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → Run as Administrator
```

---

### Still Having Issues?

1. **Check Python version**: Must be 3.9+
   ```bash
   python --version
   ```

2. **Verify environment is activated**: You should see `(.venv)` or `(totalseg)` in prompt

3. **Check disk space**: Models require ~5GB
   ```bash
   df -h  # macOS/Linux
   ```

4. **Check internet connection**: First run downloads models

5. **Look at error logs with verbose output**: 
   ```bash
   TotalSegmentator --verbose -i input.nii.gz -o output/
   ```

6. **Create a clean environment**:
   ```bash
   # Delete old environment
   rm -rf .venv  # or: conda env remove -n totalseg
   
   # Follow setup steps again from scratch
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -e .
   pip install ttkbootstrap nibabel imageio matplotlib
   ```

7. **Check for conflicting Python installations**:
   ```bash
   which -a python  # macOS/Linux - shows all python in PATH
   where python  # Windows - shows all python in PATH
   ```

---

## Quick Reference

### Activate Environment

```bash
# venv (macOS/Linux)
source .venv/bin/activate

# venv (Windows)
.\.venv\Scripts\Activate.ps1

# conda (All platforms)
conda activate totalseg
```

### Deactivate Environment

```bash
# venv
deactivate

# conda
conda deactivate
```

### Update TotalSegmentator

```bash
# Activate environment first
cd ~/Documents/GitHub/TotalSegmentator
git pull origin master
pip install -e . --upgrade
```

### Reinstall All Dependencies

```bash
# Activate environment first
pip install --upgrade pip setuptools wheel
pip install -e . --upgrade
pip install --upgrade ttkbootstrap nibabel imageio matplotlib
pip install -e ".[enhanced]" --upgrade
```

### Common Commands

```bash
# Run GUI
python totalsegmentator/bin/totalseg_gui.py

# Run CLI segmentation
TotalSegmentator -i input.nii.gz -o output/

# Get contrast phase
totalseg_get_phase -i input.nii.gz

# Get modality
totalseg_get_modality -i input.nii.gz

# Download weights manually
totalseg_download_weights -t total

# Web app dev server
cd web-app && npm run dev
```

---

## Platform-Specific Notes

### macOS
- Use `python3` and `pip3` instead of `python` and `pip`
- Blender path: `/Applications/Blender.app/Contents/MacOS/blender`
- May need to allow Terminal in Security & Privacy settings
- Install Xcode Command Line Tools: `xcode-select --install`

### Linux
- Install `python3-tk` for GUI: `sudo apt-get install python3-tk`
- May need `xvfb` for headless environments
- Use package manager for external tools
- Install build tools: `sudo apt-get install build-essential python3-dev`

### Windows
- Use PowerShell (not CMD) for better compatibility
- May need to adjust execution policy for scripts
- Use forward slashes in .env.local paths: `C:/Users/...`
- Blender path: `C:\Program Files\Blender Foundation\Blender 3.6\blender.exe`
- Install Microsoft C++ Build Tools if compilation is needed

---

## Success Checklist

- [ ] Python 3.9+ installed and verified (`python --version`)
- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Virtual environment activated (see `(.venv)` or `(totalseg)` in prompt)
- [ ] Pip upgraded (`pip install --upgrade pip setuptools wheel`)
- [ ] TotalSegmentator core installed (`pip install -e .`)
- [ ] GUI dependencies installed (`pip install ttkbootstrap nibabel imageio matplotlib`)
- [ ] TotalSegmentator command works (`TotalSegmentator --version`)
- [ ] All core imports successful (run verification script)
- [ ] Blender installed and in PATH (`blender --version`)
- [ ] dcm2niix installed and in PATH (`dcm2niix -h`)
- [ ] (For web app) Node.js 18+ installed (`node --version`)
- [ ] (For web app) npm install completed in web-app directory
- [ ] (For web app) .env.local configured with correct Python path

**If all items are checked, you're ready to use TotalSegmentator!** 🎉

---

## Additional Resources

- **Main Repository**: https://github.com/wasserth/TotalSegmentator
- **Documentation**: Check repository README and Wiki
- **Issues**: Report bugs on GitHub Issues
- **Models**: Automatically downloaded on first run (requires ~5GB)
- **PyTorch**: https://pytorch.org/get-started/locally/
- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet

---

**Version**: 2.11.0  
**Last Updated**: 2025-02-24
