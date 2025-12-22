import { NextResponse } from 'next/server';
import os from 'os';
import path from 'path';

export async function GET() {
  try {
    const homeDir = os.homedir();
    let downloadsPath: string;

    // Determine Downloads folder based on OS
    if (process.platform === 'win32') {
      // Windows: C:\Users\Username\Downloads
      downloadsPath = path.join(homeDir, 'Downloads');
    } else if (process.platform === 'darwin') {
      // macOS: /Users/Username/Downloads
      downloadsPath = path.join(homeDir, 'Downloads');
    } else {
      // Linux: /home/username/Downloads
      downloadsPath = path.join(homeDir, 'Downloads');
    }

    return NextResponse.json({ 
      path: downloadsPath,
      platform: process.platform
    });
  } catch (error: any) {
    console.error('Error getting downloads path:', error);
    return NextResponse.json(
      { error: 'Failed to get downloads path' },
      { status: 500 }
    );
  }
}
