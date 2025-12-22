# TotalSegmentator Web App

> **Note**: This web app is a frontend interface that uses the existing `totalseg_gui` pipeline from the parent TotalSegmentator project.

## How It Works

1. **Web App (Next.js)**: Provides a modern UI for uploading DICOM files
2. **Pipeline Script**: Bridges the web app to the Python `totalseg_gui` 
3. **TotalSegmentator GUI**: Executes the actual segmentation pipeline in CLI mode

## Prerequisites

### Required Software

- **Node.js v18+** (for the web application)
- **Python 3.8+** with virtual environment (for TotalSegmentator)
- **Blender 3.0+** (for 3D visualization)
- **dcm2niix** (for DICOM conversion)

### Python Virtual Environment Setup

```bash
# Navigate to the main project directory
cd c:\Users\abeez\Documents\GitHub\TotalSegmentator

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or (Windows CMD)
.\.venv\Scripts\activate.bat

# Install TotalSegmentator with dependencies
pip install -e .
pip install ttkbootstrap nibabel imageio
```

### Install External Tools

1. **Blender**: Download from https://www.blender.org/download/
2. **dcm2niix**: Download from https://github.com/rordenlab/dcm2niix/releases

Add both to your system PATH or configure paths in the web app.

---

## Getting Started with the Web App

```bash
# Navigate to web-app directory
cd c:\Users\abeez\Documents\GitHub\TotalSegmentator\web-app

# Install Node.js dependencies
npm install

# Create environment file
copy .env.example .env.local

# Edit .env.local with your Python path:
# PYTHON_PATH=C:/Users/abeez/Documents/GitHub/TotalSegmentator/.venv/Scripts/python.exe

# Start the development server
npm run dev
```

Open http://localhost:3000 in your browser.

---

## Pipeline Workflow

1. **Upload DICOM files** through the web interface
2. **Configure** project name, scale, and output settings
3. **Run Pipeline** - executes these steps via `totalseg_gui`:
   - Step 1: DICOM → NIfTI conversion (dcm2niix)
   - Step 2: NIfTI → PNG slices export
   - Step 3: Segmentation + STL mesh export (TotalSegmentator)
   - Step 4: Import meshes into Blender
   - Step 5: Apply materials and colors
   - Step 6: Setup DICOM slice viewer addon

---

## Common Issues

**"node is not recognized"**
- Restart your terminal/VS Code after installing Node.js
- If still failing, restart your computer

**"npm run dev" doesn't work**
- Make sure you ran `npm install` first
- Check you're in the correct folder: `c:\Users\abeez\Documents\GitHub\TotalSegmentator\web-app`

**Port 3000 already in use**
- Close other apps using that port, or run: `$env:PORT=3001; npm run dev` (PowerShell)

---

## Run on a new computer (VS Code)

1. Install prerequisites
   - Install Node.js v18 or newer (https://nodejs.org/) and Git.
     - Verify: `node -v` and `git --version`.
2. Open the project in VS Code
   - File → Open Folder → select `c:\Users\abeez\Documents\GitHub\TotalSegmentator\web-app`.
   - Open the integrated terminal: Ctrl+` (or View → Terminal).
3. Install dependencies
   - npm: `npm install`
   - pnpm: `pnpm install`
   - yarn: `yarn`
4. (Optional) Create environment file if your app needs one:
   - `copy .env.example .env.local` and edit values if `.env.example` exists.
5. Start dev server
   - `npm run dev`
   - The app should be available at http://localhost:3000 by default.

Install Node on Windows (two options)
Option A — Recommended: use nvm-windows (multiple Node versions)
1. Download nvm-windows installer from: https://github.com/coreybutler/nvm-windows/releases
2. Run the installer, then open a new PowerShell/Cmd.
3. Install and use an LTS Node version, for example:
   - `nvm install 18.20.0`
   - `nvm use 18.20.0`
4. Verify: `node -v` and `npm -v`

Option B — Direct Node installer
1. Download the Windows installer (LTS) from https://nodejs.org/ and run it.
2. After install open a new terminal and verify:
   - `node -v`
   - `npm -v`

Notes
- If `node` or `npm` are still not found after install, restart VS Code or open a new terminal so the PATH refreshes.
- Prefer Node v18+ for Next.js apps. Use nvm to switch versions easily.
- After Node is installed: run `npm install` in the project root, then `npm run dev`.

Troubleshooting (common errors)
- "npm run dev" fails with "next: command not found" or similar:
  - Run `npm install` in the project root to install dev dependencies.
- MODULE_NOT_FOUND / missing packages:
  - Remove node_modules and lockfile, then reinstall:
    - `rm -rf node_modules package-lock.json && npm install`
    - (Windows) `rmdir /s /q node_modules` and `del package-lock.json`
- Node version issues:
  - Use Node v18+ (switch with nvm or reinstall).
- Port already in use:
  - Kill the process using the port or run `PORT=3001 npm run dev` (Windows PowerShell: `$env:PORT=3001; npm run dev`).
- Still failing:
  - Copy the first 20 lines of the terminal error and share them; that helps diagnose the exact issue.

VS Code tips
- Install recommended extensions (ESLint, Prettier, TypeScript).
- Use the integrated terminal to run commands.
- If debugging, set breakpoints in the Next.js dev server or use the browser devtools.
