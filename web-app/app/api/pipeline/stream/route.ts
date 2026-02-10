import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

function resolvePythonExecutable(cwd: string): string {
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

  for (const candidate of candidates) {
    if (!candidate.includes('/') && !candidate.includes('\\')) return candidate;
    if (fs.existsSync(candidate)) return candidate;
  }
  return '';
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const {
    dicomDir,
    outputDir,
    projectName = 'Project-01',
    scale = '0.01',
    task = 'total_all',
    mode = 'all',
  } = body;

  const encoder = new TextEncoder();
  const cwd = process.cwd();
  const projectRoot = path.resolve(cwd, '..');
  const pythonPath = resolvePythonExecutable(cwd);
  const scriptPath = path.resolve(projectRoot, 'run_pipeline.py');

  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      const safeClose = () => {
        if (!closed) {
          closed = true;
          try {
            controller.close();
          } catch {
            // ignore
          }
        }
      };
      const sendMessage = (type: string, message: string, progress?: number) => {
        if (closed) return;
        const data = `data: ${JSON.stringify({
          type,
          message,
          progress,
          timestamp: new Date().toISOString(),
        })}\n\n`;
        try {
          controller.enqueue(encoder.encode(data));
        } catch {
          closed = true;
        }
      };

      if (!pythonPath) {
        sendMessage('error', 'Python executable not found. Check PYTHON_PATH/.venv.');
        safeClose();
        return;
      }
      if (!fs.existsSync(scriptPath)) {
        sendMessage('error', `run_pipeline.py not found: ${scriptPath}`);
        safeClose();
        return;
      }

      const args = [
        scriptPath,
        '--dicom-dir', dicomDir,
        '--output-dir', outputDir,
        '--project-name', projectName,
        '--scale', String(scale),
        '--task', task,
      ];
      if (mode && mode !== 'all') args.push('--mode', mode);

      sendMessage('info', 'Pipeline started', 0);

      const proc = spawn(pythonPath, args, {
        cwd: projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      });

      const abortHandler = () => {
        sendMessage('info', 'Client disconnected, stopping pipeline');
        try {
          proc.kill('SIGTERM');
        } catch {
          // ignore
        }
        safeClose();
      };
      request.signal?.addEventListener('abort', abortHandler, { once: true });

      const handleLine = (line: string) => {
        const t = line.trim();
        if (!t) return;

        if (t.includes('__PROGRESS__:')) {
          const m = t.match(/__PROGRESS__:(\d+):(.+)/);
          if (m) {
            sendMessage('progress', m[2].trim(), parseInt(m[1], 10));
            return;
          }
        }

        if (t.includes('[ERROR]') || t.startsWith('❌') || t.startsWith('Traceback')) {
          sendMessage('error', t);
          return;
        }

        if (t.includes('Step ') || t.includes('Pipeline complete') || t.includes('✅')) {
          sendMessage('info', t);
        }
      };

      let stdoutBuffer = '';
      proc.stdout.on('data', (data) => {
        stdoutBuffer += data.toString('utf-8');
        const parts = stdoutBuffer.split('\n');
        stdoutBuffer = parts.pop() || '';
        for (const line of parts) handleLine(line);
      });

      let stderrBuffer = '';
      proc.stderr.on('data', (data) => {
        stderrBuffer += data.toString('utf-8');
        const parts = stderrBuffer.split('\n');
        stderrBuffer = parts.pop() || '';
        for (const line of parts) handleLine(line);
      });

      proc.on('close', (code) => {
        if (stdoutBuffer.trim()) handleLine(stdoutBuffer);
        if (stderrBuffer.trim()) handleLine(stderrBuffer);
        if (code === 0) {
          sendMessage('complete', 'Pipeline completed successfully', 100);
        } else {
          sendMessage('error', `Pipeline exited with code ${code}`);
        }
        safeClose();
      });

      proc.on('error', (err) => {
        sendMessage('error', err.message || 'Failed to spawn pipeline process');
        safeClose();
      });
    },
    cancel() {
      // client closed stream
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
