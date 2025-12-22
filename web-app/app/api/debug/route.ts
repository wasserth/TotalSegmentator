import { NextRequest, NextResponse } from 'next/server';
import { existsSync } from 'fs';
import { readdir } from 'fs/promises';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');
const OUTPUT_DIR = path.join(process.cwd(), 'outputs');

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const folder = searchParams.get('folder');

  try {
    const info: any = {
      uploadDir: UPLOAD_DIR,
      outputDir: OUTPUT_DIR,
      uploadDirExists: existsSync(UPLOAD_DIR),
      outputDirExists: existsSync(OUTPUT_DIR),
    };

    if (folder) {
      const folderPath = path.join(UPLOAD_DIR, folder);
      info.requestedFolder = folderPath;
      info.requestedFolderExists = existsSync(folderPath);
      
      if (existsSync(folderPath)) {
        const files = await readdir(folderPath, { recursive: true });
        info.filesInFolder = files;
        info.fileCount = files.length;
      }
    }

    if (existsSync(UPLOAD_DIR)) {
      const uploads = await readdir(UPLOAD_DIR);
      info.uploadedFolders = uploads;
    }

    return NextResponse.json(info);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
