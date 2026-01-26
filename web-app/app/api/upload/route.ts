import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

const UPLOAD_DIR = path.join(process.cwd(), 'uploads');

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files') as File[];
    const folderName = formData.get('folderName') as string || 'dicom_upload';

    console.log('Upload request:', {
      filesCount: files.length,
      folderName,
      uploadDir: UPLOAD_DIR
    });

    if (files.length === 0) {
      return NextResponse.json(
        { error: 'No files uploaded' },
        { status: 400 }
      );
    }

    // Create upload directory
    const uploadPath = path.join(UPLOAD_DIR, folderName);
    await mkdir(uploadPath, { recursive: true });

    console.log('Created upload path:', uploadPath);

    // Save all files
    const savedFiles = [];
    for (const file of files) {
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);
      
      // Get relative path from file.webkitRelativePath or use file.name
      // @ts-ignore - webkitRelativePath exists on File in browsers
      const relativePath = file.webkitRelativePath || file.name;
      
      // Remove the first folder component (it's the selected folder name)
      const parts = relativePath.split('/');
      const fileName = parts.length > 1 ? parts.slice(1).join('/') : parts[0];
      
      const filePath = path.join(uploadPath, fileName);
      const fileDir = path.dirname(filePath);
      
      await mkdir(fileDir, { recursive: true });
      await writeFile(filePath, buffer);
      
      savedFiles.push(fileName);
    }

    console.log('Saved files:', savedFiles.length);
    console.log('Upload path exists:', existsSync(uploadPath));

    return NextResponse.json({
      success: true,
      message: `Uploaded ${savedFiles.length} files`,
      folderName,
      uploadPath,
      files: savedFiles
    });

  } catch (error: any) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: error.message || 'Upload failed', stack: error.stack },
      { status: 500 }
    );
  }
}
