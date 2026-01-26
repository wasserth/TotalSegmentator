import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      dicomDir, 
      outputDir,
      projectName = 'Project-01', 
      scale = '0.01',
      mode = 'all' // 'all', 'step1', 'step2', etc.
    } = body;

    // Get paths from environment variables
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const totalSegPath = process.env.TOTALSEGMENTATOR_PATH || '..';

    // Resolve the actual Python path (support both relative and absolute)
    const resolvedPythonPath = path.isAbsolute(pythonPath) 
      ? pythonPath 
      : path.resolve(process.cwd(), pythonPath);

    // Validate Python path exists
    if (!fs.existsSync(resolvedPythonPath)) {
      return NextResponse.json(
        { 
          error: 'Python virtual environment not found',
          details: `Python path not found at: ${resolvedPythonPath}`,
          hint: 'Please check your .env.local file and ensure PYTHON_PATH is correct. After updating, restart the dev server.'
        },
        { status: 500 }
      );
    }

    // Build the Python script path (go up one level from web-app)
    const scriptPath = path.resolve(process.cwd(), '..', 'run_pipeline.py');
    
    console.log('Looking for script at:', scriptPath);
    
    if (!fs.existsSync(scriptPath)) {
      return NextResponse.json(
        { 
          error: 'Pipeline script not found',
          details: `Script not found at: ${scriptPath}`,
          hint: 'Make sure run_pipeline.py exists in the TotalSegmentator directory (parent of web-app)'
        },
        { status: 500 }
      );
    }

    // Prepare arguments
    const args = [
      scriptPath,
      '--dicom-dir', dicomDir,
      '--output-dir', outputDir,
      '--project-name', projectName,
      '--scale', scale
    ];

    if (mode && mode !== 'all') {
      args.push('--mode', mode);
    }

    console.log('Running pipeline:', resolvedPythonPath, args.join(' '));

    // Run the Python pipeline with streaming
    const logs: string[] = [];
    let lastProgress = 0;
    
    const result = await new Promise((resolve, reject) => {
      const env = { ...process.env };
      if (process.platform === 'win32') {
        env.PYTHONIOENCODING = 'utf-8';
        env.PYTHONUTF8 = '1';
      }
      env.PYTHONUNBUFFERED = '1';
      
      const pythonProcess = spawn(resolvedPythonPath, args, {
        cwd: path.resolve(process.cwd(), '..'),
        env: env,
      });

      pythonProcess.stdout.on('data', (data) => {
        const lines = data.toString('utf-8').split('\n');
        
        for (const line of lines) {
          if (!line.trim()) continue;
          
          // Parse progress updates
          if (line.includes('__PROGRESS__:')) {
            const match = line.match(/__PROGRESS__:(\d+):(.+)/);
            if (match) {
              const percent = parseInt(match[1]);
              const stepName = match[2];
              logs.push(`__PROGRESS__:${percent}:${stepName}\n`);
              lastProgress = percent;
              console.log(`Progress: ${percent}% - ${stepName}`);
            }
          } else {
            console.log(line);
            logs.push(line + '\n');
          }
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        const log = data.toString('utf-8');
        console.error(log);
        logs.push(log);
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve({ success: true, logs });
        } else {
          reject(new Error(`Pipeline exited with code ${code}`));
        }
      });

      pythonProcess.on('error', (err) => {
        reject(err);
      });
    });

    return NextResponse.json({ 
      success: true, 
      logs,
      message: 'Pipeline completed successfully'
    });

  } catch (error: any) {
    console.error('Pipeline error:', error);
    return NextResponse.json(
      { 
        error: error.message || 'Pipeline failed',
        detail: error.stack
      },
      { status: 500 }
    );
  }
}