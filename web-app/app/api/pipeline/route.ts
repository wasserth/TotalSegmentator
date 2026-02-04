import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function POST(request: NextRequest) {
  let logs: string[] = [];
  try {
    const body = await request.json();
    const { 
      dicomDir, 
      outputDir,
      projectName = 'Project-01', 
      scale = '0.01',
      task = 'total_all',
      mode = 'all' // 'all', 'step1', 'step2', etc.
    } = body;

    // Resolve Python executable robustly (absolute path, relative path, or PATH executable).
    const cwd = process.cwd();
    const envPython = process.env.PYTHON_PATH;
    const candidates = [
      envPython,
      path.resolve(cwd, '.venv', 'bin', 'python'),
      path.resolve(cwd, '..', '.venv', 'bin', 'python'),
      path.resolve(cwd, '.venv', 'Scripts', 'python.exe'),
      path.resolve(cwd, '..', '.venv', 'Scripts', 'python.exe'),
      'python3',
      'python',
    ].filter(Boolean) as string[];

    let resolvedPythonPath = '';
    for (const candidate of candidates) {
      // Keep plain executable names for PATH lookup by spawn.
      if (!candidate.includes('/') && !candidate.includes('\\')) {
        resolvedPythonPath = candidate;
        break;
      }
      if (fs.existsSync(candidate)) {
        resolvedPythonPath = candidate;
        break;
      }
    }

    if (!resolvedPythonPath) {
      return NextResponse.json(
        {
          error: 'Python executable not found',
          details: `Tried: ${candidates.join(', ')}`,
          hint: 'Set PYTHON_PATH in web-app/.env.local (or project .env.local loaded into web-app) and restart dev server.',
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
      '--scale', scale,
      '--task', task,
    ];

    if (mode && mode !== 'all') {
      args.push('--mode', mode);
    }

    console.log('Running pipeline:', resolvedPythonPath, args.join(' '));

    // Run the Python pipeline with streaming
    logs = [];
    let lastProgress = 0;
    
    await new Promise((resolve, reject) => {
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
          const err = new Error(`Pipeline exited with code ${code}`);
          // @ts-expect-error attach collected logs for API response
          err.logs = logs;
          reject(err);
        }
      });

      pythonProcess.on('error', (err) => {
        reject(err);
      });
    });

    const keyLogs = logs.filter((line) => {
      const t = line.trim();
      return (
        t.includes('__PROGRESS__:') ||
        t.includes('Step ') ||
        t.includes('[ERROR]') ||
        t.includes('❌') ||
        t.includes('Pipeline complete')
      );
    });

    return NextResponse.json({
      success: true, 
      logs,
      keyLogs,
      message: 'Pipeline completed successfully'
    });

  } catch (error: any) {
    console.error('Pipeline error:', error);
    const errorLogs = (error?.logs as string[] | undefined) || logs;
    const keyLogs = (errorLogs || []).filter((line) => {
      const t = line.trim();
      return t.includes('__PROGRESS__:') || t.includes('[ERROR]') || t.includes('❌') || t.includes('Error');
    }).slice(-30);
    return NextResponse.json(
      { 
        error: error.message || 'Pipeline failed',
        detail: error.stack,
        logs: errorLogs,
        keyLogs,
      },
      { status: 500 }
    );
  }
}
