'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Upload, 
  Play, 
  FolderOpen, 
  CheckCircle, 
  XCircle, 
  FileText,
  Settings,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

export default function TotalSegmentatorApp() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [folderName, setFolderName] = useState('Patient');
  const [outputPath, setOutputPath] = useState('');
  const [projectName, setProjectName] = useState('Project-01');
  const [scale, setScale] = useState('0.01');
  
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showLog, setShowLog] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const outputFolderRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Get default Downloads folder path on component mount
  useEffect(() => {
    // Set default to Downloads folder
    const getDownloadsPath = async () => {
      try {
        const response = await fetch('/api/get-downloads-path');
        const data = await response.json();
        if (data.path) {
          setOutputPath(data.path);
        }
      } catch (err) {
        console.error('Failed to get downloads path:', err);
      }
    };
    getDownloadsPath();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      setFiles(selectedFiles);
      setError(null);
      setUploadComplete(false);
      setSuccess(null);
    }
  };

  const handleOutputFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      // Extract the full path from the first file
      const firstFile = selectedFiles[0];
      
      // Get the full directory path
      if (firstFile.webkitRelativePath) {
        const parts = firstFile.webkitRelativePath.split('/');
        parts.pop(); // Remove filename
        const folderPath = parts.join('/');
        
        // Try to get absolute path from file
        // @ts-ignore - path property exists in some browsers
        const filePath = firstFile.path || '';
        
        if (filePath) {
          // Extract directory from full file path
          const pathParts = filePath.split(/[/\\]/);
          pathParts.pop(); // Remove filename
          const absolutePath = pathParts.join('/');
          setOutputPath(absolutePath);
        } else {
          // Fallback to relative path
          setOutputPath(folderPath);
        }
      }
      
      // Clear the input so user can select the same folder again if needed
      e.target.value = '';
    }
  };

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setError('Please select files to upload');
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccess(null);
    setUploadComplete(false);
    setLogs([`📁 Preparing to upload ${files.length} files...\n\n`]);
    if (!showLog) setShowLog(true);

    try {
      const formData = new FormData();
      formData.append('folderName', folderName);
      
      let addedCount = 0;
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
        addedCount++;
        if (addedCount % 10 === 0) {
          setLogs(prev => [...prev, `  Added ${addedCount}/${files.length} files...\n`]);
        }
      }

      setLogs(prev => [...prev, `\n📤 Uploading ${addedCount} files to server...\n`]);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setLogs(prev => [...prev, `\n✅ Upload Complete!\n`]);
      setLogs(prev => [...prev, `   Files: ${data.files.length}\n`]);
      setLogs(prev => [...prev, `   Location: ${data.uploadPath}\n\n`]);
      setUploadComplete(true);
      setSuccess(`Successfully uploaded ${data.files.length} files`);

    } catch (err: any) {
      console.error('Upload error:', err);
      setError(`Upload failed: ${err.message}`);
      setLogs(prev => [...prev, `\n❌ Upload failed: ${err.message}\n`]);
      setUploadComplete(false);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRunPipeline = async (mode: string = 'all') => {
    if (!uploadComplete) {
      setError('Please upload files first');
      return;
    }

    if (!outputPath.trim()) {
      setError('Please select an output folder');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setSuccess(null);
    setProgress(0);
    setCurrentStep('Starting...');
    setLogs(prev => [...prev, '\n' + '='.repeat(70) + '\n']);
    setLogs(prev => [...prev, '🚀 STARTING PIPELINE\n']);
    setLogs(prev => [...prev, '='.repeat(70) + '\n\n']);
    setLogs(prev => [...prev, `📂 Output will be saved to: ${outputPath}\n\n`]);
    if (!showLog) setShowLog(true);

    try {
      setLogs(prev => [...prev, '📡 Sending request to /api/pipeline...\n']);
      
      const response = await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dicomDir: folderName,
          outputDir: outputPath,
          projectName,
          scale,
          mode
        }),
      });

      setLogs(prev => [...prev, `📥 Response status: ${response.status}\n`]);

      const data = await response.json();
      
      setLogs(prev => [...prev, `📦 Response data received\n\n`]);

      if (!response.ok) {
        setLogs(prev => [...prev, '❌ Pipeline failed!\n']);
        if (data.details) {
          setLogs(prev => [...prev, `Details: ${data.details}\n`]);
        }
        if (data.hint) {
          setLogs(prev => [...prev, `Hint: ${data.hint}\n`]);
        }
        throw new Error(data.error || 'Pipeline failed');
      }

      // Process logs and extract progress
      if (data.logs && Array.isArray(data.logs)) {
        for (const log of data.logs) {
          // Parse progress updates
          if (log.includes('__PROGRESS__:')) {
            const match = log.match(/__PROGRESS__:(\d+):(.+)/);
            if (match) {
              const percent = parseInt(match[1]);
              const stepName = match[2].trim();
              setProgress(percent);
              setCurrentStep(stepName);
              setLogs(prev => [...prev, `\n[${percent}%] ${stepName}\n`]);
            }
          } else {
            setLogs(prev => [...prev, log]);
          }
        }
      }
      
      setProgress(100);
      setCurrentStep('Complete');
      setSuccess('Pipeline completed successfully!');
      
    } catch (err: any) {
      console.error('Pipeline error:', err);
      setError(`Pipeline failed: ${err.message}`);
      setLogs(prev => [...prev, `\n❌ Error: ${err.message}\n`]);
      setCurrentStep('Failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const scrollToBottom = () => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  };

  return (
    <div className="container mx-auto p-8 max-w-6xl">
      <Card className="shadow-lg">
        {/* Header */}
        <CardHeader className="pb-6">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                TotalSegmentator Pipeline
              </CardTitle>
              <CardDescription className="text-lg mt-2">
                Automated medical imaging segmentation with Blender 3D visualization
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <div className="border-t border-gray-200 my-4" />

        <CardContent className="pt-6 space-y-6">
          {/* Section 1: Input & Output */}
          <div className="space-y-4 p-6 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
            <h3 className="text-xl font-semibold text-blue-900 flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              Input & Output
            </h3>
            
            <div className="space-y-4">
              {/* DICOM Input */}
              <div>
                <Label htmlFor="folder-upload" className="text-base">DICOM Input Folder</Label>
                <p className="text-sm text-muted-foreground mb-2">
                  Select the folder containing your DICOM files
                </p>
                <div className="flex gap-2 mt-2">
                  <Input
                    id="folder-upload"
                    type="file"
                    ref={fileInputRef}
                    // @ts-ignore
                    webkitdirectory=""
                    directory=""
                    multiple
                    onChange={handleFileChange}
                    disabled={isUploading || isProcessing}
                    className="cursor-pointer"
                  />
                </div>
                {files && (
                  <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
                    <FileText className="h-4 w-4" />
                    <span>{files.length} file(s) selected</span>
                    {uploadComplete && <CheckCircle className="h-4 w-4 text-green-600" />}
                  </div>
                )}
              </div>

              {/* Divider */}
              <div className="border-t border-blue-200 my-4"></div>

              {/* Output Folder Selection */}
              <div>
                <Label htmlFor="output-folder" className="text-base">Output Destination</Label>
                <p className="text-sm text-muted-foreground mb-2">
                  Choose where the processed results will be saved
                </p>
                <div className="flex gap-2 mt-2">
                  <Input
                    id="output-folder"
                    type="file"
                    ref={outputFolderRef}
                    // @ts-ignore
                    webkitdirectory=""
                    directory=""
                    onChange={handleOutputFolderChange}
                    disabled={isUploading || isProcessing}
                    className="cursor-pointer"
                    accept=""
                    title="Select output folder"
                  />
                </div>
                <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
                  <div className="flex items-start gap-2">
                    <FolderOpen className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-blue-900 mb-1">Results will be saved to:</p>
                      <code className="text-xs text-blue-800 break-all block bg-white px-2 py-1 rounded border border-blue-200">
                        {outputPath || 'Loading default (Downloads folder)...'}
                      </code>
                      <p className="text-xs text-blue-600 mt-2">
                        💡 Click above to browse and select your preferred output location
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Configuration */}
          <div className="space-y-4 p-6 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
            <h3 className="text-xl font-semibold text-purple-900 flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Configuration
            </h3>
            
            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label htmlFor="folder-name">Input Folder Name</Label>
                <Input
                  id="folder-name"
                  value={folderName}
                  onChange={(e) => setFolderName(e.target.value)}
                  placeholder="Patient"
                  disabled={isProcessing || isUploading}
                />
                <p className="text-xs text-muted-foreground mt-1">Used for organizing uploads</p>
              </div>
              <div>
                <Label htmlFor="project-name">Project Name</Label>
                <Input
                  id="project-name"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="Project-01"
                  disabled={isProcessing || isUploading}
                />
                <p className="text-xs text-muted-foreground mt-1">Name for Blender scene</p>
              </div>
              <div>
                <Label htmlFor="scale">Blender Scale</Label>
                <Input
                  id="scale"
                  value={scale}
                  onChange={(e) => setScale(e.target.value)}
                  placeholder="0.01"
                  disabled={isProcessing || isUploading}
                />
                <p className="text-xs text-muted-foreground mt-1">Scale factor for 3D models</p>
              </div>
            </div>
          </div>

          {/* Section 3: Run Pipeline */}
          <div className="space-y-4 p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
            <h3 className="text-xl font-semibold text-green-900 flex items-center gap-2">
              <Play className="h-5 w-5" />
              Run Pipeline
            </h3>

            {/* Main Actions */}
            <div className="grid grid-cols-2 gap-4">
              <Button
                onClick={handleUpload}
                disabled={!files || isProcessing || isUploading}
                variant="outline"
                size="lg"
                className="h-14"
              >
                <Upload className="mr-2 h-5 w-5" />
                {isUploading ? 'Uploading...' : uploadComplete ? 'Re-upload Files' : 'Upload Files'}
              </Button>

              <Button
                onClick={() => handleRunPipeline('all')}
                disabled={!uploadComplete || isProcessing || isUploading}
                size="lg"
                className="h-14 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
              >
                <Play className="mr-2 h-5 w-5" />
                {isProcessing ? 'Processing...' : 'Run Full Pipeline'}
              </Button>
            </div>

            {/* Individual Steps */}
            <div>
              <Label className="text-sm text-muted-foreground mb-2 block">Individual Steps:</Label>
              <div className="grid grid-cols-6 gap-2">
                {[
                  { label: '1. DICOM→NIfTI', step: 'step1' },
                  { label: '2. NIfTI→PNG', step: 'step2' },
                  { label: '3. Segment', step: 'step3' },
                  { label: '4. Import', step: 'step4' },
                  { label: '5. Materials', step: 'step5' },
                  { label: '6. Viewer', step: 'step6' }
                ].map(({ label, step }) => (
                  <Button
                    key={step}
                    onClick={() => handleRunPipeline(step)}
                    disabled={!uploadComplete || isProcessing}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                  >
                    {label}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          {/* Progress Section */}
          {(isProcessing || isUploading || progress > 0) && (
            <div className="space-y-3 p-6 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200">
              <div className="flex items-center justify-between">
                <Label className="text-lg font-semibold">
                  {currentStep || 'Processing...'}
                </Label>
                <span className="text-sm text-muted-foreground">{progress}%</span>
              </div>
              <Progress value={progress} className="h-3" />
            </div>
          )}

          {/* Status Messages */}
          {success && (
            <Alert className="border-green-200 bg-green-50">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">{success}</AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {!uploadComplete && files && files.length > 0 && !isUploading && (
            <Alert className="border-yellow-200 bg-yellow-50">
              <AlertDescription className="text-yellow-800">
                ⚠️ Files selected but not uploaded. Click "Upload Files" first.
              </AlertDescription>
            </Alert>
          )}

          {/* Log Section (Collapsible) */}
          <div className="space-y-2">
            <Button
              variant="ghost"
              onClick={() => setShowLog(!showLog)}
              className="w-full justify-start hover:bg-gray-100"
            >
              {showLog ? <ChevronDown className="h-4 w-4 mr-2" /> : <ChevronRight className="h-4 w-4 mr-2" />}
              <span className="font-semibold">Process Log</span>
            </Button>

            {showLog && logs.length > 0 && (
              <div 
                ref={logRef}
                className="bg-gray-900 text-green-400 p-6 rounded-lg h-96 overflow-auto font-mono text-sm"
                style={{ scrollBehavior: 'smooth' }}
              >
                {logs.map((log, i) => (
                  <div key={i} className="whitespace-pre-wrap">{log}</div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
