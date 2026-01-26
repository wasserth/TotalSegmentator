import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import os from 'os';

/**
 * API endpoint for browsing directories
 * Returns list of subdirectories for a given path
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { currentPath } = body;

    // Determine the starting path
    let browsePath: string;
    
    if (!currentPath || currentPath === '') {
      // Start from home directory or common locations on Windows
      if (process.platform === 'win32') {
        browsePath = os.homedir(); // e.g., C:\Users\abeez
      } else {
        browsePath = os.homedir();
      }
    } else {
      browsePath = currentPath;
    }

    // Resolve to absolute path
    browsePath = path.resolve(browsePath);

    // Check if path exists
    if (!fs.existsSync(browsePath)) {
      return NextResponse.json({
        error: 'Path does not exist',
        currentPath: browsePath,
        parentPath: '',
        directories: []
      });
    }

    // Read directory contents
    const entries = fs.readdirSync(browsePath, { withFileTypes: true });
    
    // Filter only directories and sort alphabetically
    const directories = entries
      .filter(entry => entry.isDirectory())
      .filter(entry => !entry.name.startsWith('.')) // Hide hidden folders
      .map(entry => ({
        name: entry.name,
        path: path.join(browsePath, entry.name)
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    // Get parent path
    const parentPath = path.dirname(browsePath);
    const hasParent = parentPath !== browsePath;

    return NextResponse.json({
      currentPath: browsePath,
      parentPath: hasParent ? parentPath : '',
      directories
    });

  } catch (error: any) {
    console.error('Browse error:', error);
    return NextResponse.json(
      { 
        error: error.message || 'Failed to browse directory',
        currentPath: '',
        parentPath: '',
        directories: []
      },
      { status: 500 }
    );
  }
}