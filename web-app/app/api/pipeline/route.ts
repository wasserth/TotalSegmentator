import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';

const execAsync = promisify(exec);

const TOTALSEG_ROOT = path.resolve(process.cwd(), '..');
const PYTHON_VENV = path.join(TOTALSEG_ROOT, '.venv', 'bin', 'python');
const UPLOAD_DIR = path.join(process.cwd(), 'uploads');
const OUTPUT_DIR = path.join(process.cwd(), 'outputs');

async function ensureDirectories() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
}

export async function POST(request: NextRequest) {
  try {
    await ensureDirectories();

    const body = await request.json();
    const { 
      dicomDir, 
      outputDir, 
      projectName = 'Project-01', 
      scale = '0.01',
      mode = 'all' // 'all', 'step1', 'step2', etc.
    } = body;

    if (!dicomDir || !outputDir) {
      return NextResponse.json(
        { error: 'Missing required parameters: dicomDir and outputDir' },
        { status: 400 }
      );
    }

    const inputPath = path.join(UPLOAD_DIR, dicomDir);
    const outputPath = path.join(OUTPUT_DIR, outputDir);

    if (!existsSync(inputPath)) {
      const uploadContents = existsSync(UPLOAD_DIR) 
        ? await fs.readdir(UPLOAD_DIR) 
        : [];
      
      return NextResponse.json(
        { 
          error: 'Input directory does not exist. Please upload DICOM files first.',
          detail: {
            expectedPath: inputPath,
            uploadsFolder: UPLOAD_DIR,
            uploadsExists: existsSync(UPLOAD_DIR),
            uploadsContents: uploadContents
          }
        },
        { status: 400 }
      );
    }

    if (!existsSync(PYTHON_VENV)) {
      return NextResponse.json(
        { error: `Python virtual environment not found at: ${PYTHON_VENV}` },
        { status: 500 }
      );
    }

    // Use the GUI script's CLI mode - it does everything!
    const cliCmd = [
      PYTHON_VENV,
      '-m', 'totalsegmentator.bin.totalseg_gui',
      '--cli',
      '--dicom', `"${inputPath}"`,
      '--output', `"${outputPath}"`,
      '--case-name', projectName,
      '--scale', scale
    ].join(' ');

    const logs: string[] = [];
    logs.push('='.repeat(70) + '\n');
    logs.push('TotalSegmentator Pipeline\n');
    logs.push('='.repeat(70) + '\n\n');
    logs.push(`📁 Input:   ${inputPath}\n`);
    logs.push(`📂 Output:  ${outputPath}\n`);
    logs.push(`📋 Project: ${projectName}\n`);
    logs.push(`📏 Scale:   ${scale}\n\n`);
    logs.push('Starting pipeline...\n\n');

    try {
      const { stdout, stderr } = await execAsync(cliCmd, {
        maxBuffer: 50 * 1024 * 1024,
        timeout: 90 * 60 * 1000,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1'
        },
        cwd: TOTALSEG_ROOT
      });
      
      logs.push(stdout);
      if (stderr && !stderr.includes('FutureWarning') && !stderr.includes('DeprecationWarning')) {
        logs.push('\n⚠️  Warnings:\n');
        logs.push(stderr);
      }

      logs.push('\n' + '='.repeat(70) + '\n');
      logs.push('✅ PIPELINE COMPLETED SUCCESSFULLY\n');
      logs.push('='.repeat(70) + '\n\n');

      // Check outputs
      const outputFiles: any = {};
      const checks = [
        ['nifti', path.join(outputPath, 'out_nii'), 'NIfTI files'],
        ['slices', path.join(outputPath, 'dicom_slices'), 'PNG slices'],
        ['segmentation', path.join(outputPath, 'out_total_all'), 'Segmentation'],
        ['blender', path.join(outputPath, 'out'), 'Blender scenes']
      ];

      for (const [key, dir, label] of checks) {
        if (existsSync(dir)) {
          const files = await fs.readdir(dir);
          outputFiles[key] = { path: dir, count: files.length };
          logs.push(`  ${label}: ${dir} (${files.length} items)\n`);
        }
      }

      return NextResponse.json({
        success: true,
        message: 'Pipeline completed successfully',
        outputPath,
        outputFiles,
        logs
      });

    } catch (error: any) {
      logs.push(`\n❌ Pipeline execution failed:\n`);
      logs.push(`Error: ${error.message}\n`);
      if (error.stdout) logs.push('\nStdout:\n' + error.stdout + '\n');
      if (error.stderr) logs.push('\nStderr:\n' + error.stderr + '\n');
      
      return NextResponse.json({ 
        error: 'Pipeline execution failed', 
        details: error.message,
        logs 
      }, { status: 500 });
    }

  } catch (error: any) {
    console.error('Pipeline error:', error);
    return NextResponse.json(
      { error: error.message || 'Pipeline execution failed', stack: error.stack },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const outputDir = searchParams.get('outputDir');

  if (!outputDir) {
    return NextResponse.json({ error: 'Missing outputDir parameter' }, { status: 400 });
  }

  const outputPath = path.join(OUTPUT_DIR, outputDir);
  
  try {
    const exists = existsSync(outputPath);
    if (!exists) {
      return NextResponse.json({ exists: false });
    }

    const status: any = { exists: true };
    const checks = [
      ['nifti', 'out_nii'],
      ['slices', 'dicom_slices'],
      ['segmentation', 'out_total_all'],
      ['blender', 'out']
    ];

    for (const [key, folder] of checks) {
      const dir = path.join(outputPath, folder);
      if (existsSync(dir)) {
        const files = await fs.readdir(dir);
        status[key] = { exists: true, fileCount: files.length };
      } else {
        status[key] = { exists: false };
      }
    }
    
    return NextResponse.json(status);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}