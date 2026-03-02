import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import fs from 'fs';

function openTarget(target: string) {
  if (process.platform === 'darwin') {
    return spawn('open', [target], { detached: true, stdio: 'ignore' });
  }
  if (process.platform === 'win32') {
    return spawn('cmd', ['/c', 'start', '', target], { detached: true, stdio: 'ignore' });
  }
  return spawn('xdg-open', [target], { detached: true, stdio: 'ignore' });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const target = String(body?.target || '').trim();
    const type = body?.type === 'url' ? 'url' : 'path';

    if (!target) {
      return NextResponse.json({ error: 'Missing target' }, { status: 400 });
    }

    if (type === 'path' && !fs.existsSync(target)) {
      return NextResponse.json({ error: `Path not found: ${target}` }, { status: 404 });
    }

    const proc = openTarget(target);
    proc.unref();
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Open failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
