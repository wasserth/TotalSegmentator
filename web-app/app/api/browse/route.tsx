import { NextRequest, NextResponse } from 'next/server';
import { PythonExecutor } from '@/lib/python-executor';
import path from 'path';
import fs from 'fs';

// Configure path to your TotalSegmentator repository (parent directory)
const TOTALSEGMENTATOR_PATH = process.env.TOTALSEGMENTATOR_PATH || 
  path.join(process.cwd(), '..');

const executor = new PythonExecutor({
  totalsegmentatorPath: TOTALSEGMENTATOR_PATH,
  pythonPath: process.env.PYTHON_PATH || 'python3'
});

/**
 * Convert relative paths to absolute paths
 */
function resolveAbsolutePath(inputPath: string): string {
  // If already absolute, return as-is
  if (path.isAbsolute(inputPath)) {
    return inputPath;
  }
  
  // Try resolving from TotalSegmentator root directory
  const absolutePath = path.resolve(TOTALSEGMENTATOR_PATH, inputPath);
  
  return absolutePath;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      dicomDir,
      outputDir,
      projectName,
      scale,
      blenderPath,
      dcm2niixPath,
      mode,
      task = 'total',
      fast = false,
      ml = true,
      statistics = false,
      device = 'cpu'
    } = body;

    // Validate required fields
    if (!dicomDir || !outputDir) {
      return NextResponse.json(
        { error: 'DICOM and output directories are required' },
        { status: 400 }
      );
    }

    // Convert to absolute paths
    const absoluteDicomDir = resolveAbsolutePath(dicomDir);
    const absoluteOutputDir = resolveAbsolutePath(outputDir);

    console.log('Starting pipeline with params:', {
      dicomDir: absoluteDicomDir,
      outputDir: absoluteOutputDir,
      projectName,
      mode,
      task
    });

    // Check if input directory exists
    if (!fs.existsSync(absoluteDicomDir)) {
      return NextResponse. json(
        { 
          error: `Input directory does not exist: ${absoluteDicomDir}`,
          hint: 'Please provide the full path, e.g., /Users/abeez/Documents/GitHub/TotalSegmentator/Patient'
        },
        { status: 400 }
      );
    }

    // Create output directory if it doesn't exist
    if (!fs.existsSync(absoluteOutputDir)) {
      fs.mkdirSync(absoluteOutputDir, { recursive: true });
      console.log('Created output directory:', absoluteOutputDir);
    }

    // Execute TotalSegmentator
    const result = await executor.runTotalSegmentator({
      input: absoluteDicomDir,
      output: absoluteOutputDir,
      ml,
      fast,
      task,
      statistics,
      device,
      verbose: true
    });

    return NextResponse.json({
      success: true,
      output: result.stdout,
      errors: result.stderr || null,
      paths: {
        input: absoluteDicomDir,
        output: absoluteOutputDir
      }
    });

  } catch (error: any) {
    console.error('Pipeline execution error:', error);
    return NextResponse.json(
      { 
        error: error.message || 'Pipeline execution failed', 
        details: error.stderr,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}