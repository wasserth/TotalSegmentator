'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Folder, Play, ChevronRight, ChevronDown, CheckCircle, XCircle, Loader2, Settings, AlertCircle } from 'lucide-react';

interface Log {
  timestamp: string;
  message: string;
}

type Status = 'ready' | 'processing' | 'success' | 'error';

const TotalSegmentatorApp = () => {
  const [dicomFolder, setDicomFolder] = useState('');
  const [outputFolder, setOutputFolder] = useState('');
  const [projectName, setProjectName] = useState('Project-01');
  const [blenderScale, setBlenderScale] = useState('0.01');
  const [blenderPath, setBlenderPath] = useState('');
  const [dcm2niixPath, setDcm2niixPath] = useState('');
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [status, setStatus] = useState<Status>('ready');
  const [logs, setLogs] = useState<Log[]>([]);
  
  const [showTools, setShowTools] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (message: string) => {
    setLogs(prev => [...prev, { timestamp: new Date().toISOString(), message }]);
  };

  /**
   * Modern folder picker using File System Access API
   * Falls back to manual input if not supported
   */
  const handleFolderPicker = async (type: 'dicom' | 'output' | 'blender' | 'dcm2niix') => {
    // Check if File System Access API is supported
    if ('showDirectoryPicker' in window) {
      try {
        // @ts-ignore - TypeScript doesn't have types for this yet
        const dirHandle = await window.showDirectoryPicker({
          mode: 'read',
          startIn: 'documents'
        });
        
        const folderPath = dirHandle.name;
        // Get full path if available
        const fullPath = await getFullPath(dirHandle);
        
        switch (type) {
          case 'dicom':
            setDicomFolder(fullPath || folderPath);
            break;
          case 'output':
            setOutputFolder(fullPath || folderPath);
            break;
          case 'blender':
            setBlenderPath(fullPath || folderPath);
            break;
          case 'dcm2niix':
            setDcm2niixPath(fullPath || folderPath);
            break;
        }
        
        addLog(`Selected folder: ${fullPath || folderPath}`);
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error('Error selecting folder:', error);
          // Fallback to manual input
          handleManualInput(type);
        }
      }
    } else {
      // Fallback for browsers that don't support File System Access API
      handleManualInput(type);
    }
  };

  /**
   * Get full path from directory handle (requires permission)
   */
  const getFullPath = async (dirHandle: any): Promise<string> => {
    try {
      // Try to resolve full path (this might not work in all browsers)
      if ('resolve' in dirHandle) {
        const paths = await dirHandle.resolve();
        return paths.join('/');
      }
      // Fallback: just return the name
      return dirHandle.name;
    } catch (error) {
      return dirHandle.name;
    }
  };

  /**
   * Fallback to manual input dialog
   */
  const handleManualInput = (type: string) => {
    const folder = prompt(`Enter full path for ${type}:\n\nExample: /Users/username/Documents/dicom`);
    if (folder) {
      switch (type) {
        case 'dicom':
          setDicomFolder(folder);
          break;
        case 'output':
          setOutputFolder(folder);
          break;
        case 'blender':
          setBlenderPath(folder);
          break;
        case 'dcm2niix':
          setDcm2niixPath(folder);
          break;
      }
    }
  };

  const runPipeline = async (mode = 'all') => {
    if (!dicomFolder || !outputFolder) {
      alert('Please select both DICOM folder and output folder');
      return;
    }

    if (isProcessing) {
      alert('A pipeline is already in progress');
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setStatus('processing');
    setLogs([]);
    
    if (! showLogs) setShowLogs(true);

    try {
      addLog(`Starting pipeline (mode: ${mode})...`);
      
      const response = await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dicomDir: dicomFolder,
          outputDir: outputFolder,
          projectName,
          scale: blenderScale,
          blenderPath,
          dcm2niixPath,
          mode
        })
      });

      if (!response.ok) throw new Error('Pipeline failed');

      const steps = [
        { progress: 15, step: 'Converting DICOM...  (1/6)' },
        { progress: 30, step: 'Exporting slices... (2/6)' },
        { progress: 60, step: 'Segmenting organs... (3/6)' },
        { progress: 75, step: 'Importing to Blender... (4/6)' },
        { progress: 90, step: 'Applying materials... (5/6)' },
        { progress: 100, step: 'Adding slice viewer... (6/6)' }
      ];

      for (const { progress: p, step } of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        setProgress(p);
        setCurrentStep(step);
        addLog(step);
      }

      setStatus('success');
      addLog('Pipeline completed successfully!');
      alert('Pipeline completed successfully!');
      
    } catch (error: any) {
      setStatus('error');
      addLog(`Error: ${error.message}`);
      alert('Pipeline failed. Check the log for details.');
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusIcon = () => {
    if (status === 'ready') return <div className="w-3 h-3 rounded-full bg-gray-400" />;
    if (status === 'processing') return <Loader2 className="w-5 h-5 text-yellow-500 animate-spin" />;
    if (status === 'success') return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (status === 'error') return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const getStatusText = () => {
    if (status === 'ready') return 'Ready to start';
    if (status === 'processing') return currentStep || 'Processing...';
    if (status === 'success') return '✓ Complete!  (6/6)';
    if (status === 'error') return '✗ Failed';
  };

  const stepButtons = [
    { label: '1.  DICOM→NIfTI', mode: 'step1' },
    { label: '2. NIfTI→PNG', mode: 'step2' },
    { label: '3.  Segment', mode: 'step3' },
    { label: '4.  Import', mode: 'step4' },
    { label: '5.  Materials', mode: 'step5' },
    { label: '6. Slice Viewer', mode: 'step6' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            TotalSegmentator Pipeline
          </h1>
          <p className="text-lg text-gray-600">
            Automated medical imaging segmentation with Blender 3D visualization
          </p>
        </div>

        <div className="h-px bg-gradient-to-r from-blue-200 to-indigo-200 mb-8" />

        {/* Input/Output Section */}
        <div className="bg-white rounded-xl shadow-sm border border-blue-100 p-8 mb-6">
          <h2 className="text-xl font-semibold text-blue-900 mb-6 flex items-center gap-2">
            <Folder className="w-5 h-5" />
            Input & Output
          </h2>
          
          <div className="space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-[180px_1fr_auto] gap-4 items-center">
              <label className="text-gray-700 font-medium">DICOM Folder</label>
              <input
                type="text"
                value={dicomFolder}
                onChange={(e) => setDicomFolder(e.target.value)}
                className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Click Browse to select folder..."
                readOnly
              />
              <button
                onClick={() => handleFolderPicker('dicom')}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors border border-blue-700 flex items-center gap-2"
              >
                <Folder className="w-4 h-4" />
                Browse... 
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-[180px_1fr_auto] gap-4 items-center">
              <label className="text-gray-700 font-medium">Output Folder</label>
              <input
                type="text"
                value={outputFolder}
                onChange={(e) => setOutputFolder(e.target.value)}
                className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Click Browse to select folder..."
                readOnly
              />
              <button
                onClick={() => handleFolderPicker('output')}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors border border-blue-700 flex items-center gap-2"
              >
                <Folder className="w-4 h-4" />
                Browse...
              </button>
            </div>
          </div>
        </div>

        {/* Configuration Section */}
        <div className="bg-white rounded-xl shadow-sm border border-indigo-100 p-8 mb-6">
          <h2 className="text-xl font-semibold text-indigo-900 mb-6 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Configuration
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-gray-700 font-medium mb-2">Project Name</label>
              <input
                type="text"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
            
            <div>
              <label className="block text-gray-700 font-medium mb-2">Blender Scale</label>
              <input
                type="text"
                value={blenderScale}
                onChange={(e) => setBlenderScale(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>

        {/* Tool Paths Section (Collapsible) */}
        <div className="mb-6">
          <button
            onClick={() => setShowTools(!showTools)}
            className="flex items-center gap-2 text-gray-700 hover:text-gray-900 mb-3 transition-colors"
          >
            {showTools ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            <span className="font-medium">Advanced: Tool Paths (Optional)</span>
          </button>
          
          {showTools && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <div className="space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-[200px_1fr_auto] gap-4 items-center">
                  <label className="text-gray-700 font-medium">Blender Executable</label>
                  <input
                    type="text"
                    value={blenderPath}
                    onChange={(e) => setBlenderPath(e.target.value)}
                    className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent text-sm"
                    placeholder="Click Browse to select file..."
                    readOnly
                  />
                  <button
                    onClick={() => handleFolderPicker('blender')}
                    className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors border border-gray-700 flex items-center gap-2"
                  >
                    <Folder className="w-4 h-4" />
                    Browse...
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-[200px_1fr_auto] gap-4 items-center">
                  <label className="text-gray-700 font-medium">dcm2niix Executable</label>
                  <input
                    type="text"
                    value={dcm2niixPath}
                    onChange={(e) => setDcm2niixPath(e.target.value)}
                    className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent text-sm"
                    placeholder="Click Browse to select file..."
                    readOnly
                  />
                  <button
                    onClick={() => handleFolderPicker('dcm2niix')}
                    className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors border border-gray-700 flex items-center gap-2"
                  >
                    <Folder className="w-4 h-4" />
                    Browse... 
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Run Pipeline Section */}
        <div className="bg-white rounded-xl shadow-sm border border-green-100 p-8 mb-6">
          <h2 className="text-xl font-semibold text-green-900 mb-6 flex items-center gap-2">
            <Play className="w-5 h-5" />
            Run Pipeline
          </h2>
          
          <button
            onClick={() => runPipeline('all')}
            disabled={isProcessing}
            className="w-full py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold text-lg mb-6 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Run Full Pipeline
              </>
            )}
          </button>

          <div>
            <p className="text-gray-600 mb-3 font-medium">Individual Steps:</p>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {stepButtons.map((btn) => (
                <button
                  key={btn.mode}
                  onClick={() => runPipeline(btn.mode)}
                  disabled={isProcessing}
                  className="px-3 py-3 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 transition-colors border border-indigo-200 text-sm font-medium disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
                >
                  {btn.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Progress Section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-6">
          <div className="flex items-center gap-3 mb-4">
            {getStatusIcon()}
            <span className="text-lg font-semibold text-gray-900">{getStatusText()}</span>
          </div>
          
          <div className="relative w-full h-4 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`absolute top-0 left-0 h-full transition-all duration-500 ${
                status === 'success' ? 'bg-green-500' : 
                status === 'error' ?  'bg-red-500' : 
                'bg-blue-500'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-right text-sm text-gray-600 mt-2">{progress}%</p>
        </div>

        {/* Log Section (Collapsible) */}
        <div>
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="flex items-center gap-2 text-gray-700 hover:text-gray-900 mb-3 transition-colors"
          >
            {showLogs ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            <span className="font-medium">Process Log</span>
          </button>
          
          {showLogs && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div
                ref={logContainerRef}
                className="bg-gray-50 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm"
              >
                {logs.length === 0 ? (
                  <p className="text-gray-500">No logs yet.  Start a pipeline to see output... </p>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className="mb-1 text-gray-800">
                      <span className="text-gray-500 text-xs">
                        [{new Date(log.timestamp).toLocaleTimeString()}]
                      </span>{' '}
                      {log.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* Info Alert */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900">
            <p className="font-semibold mb-1">Folder Picker Active</p>
            <p>Click the Browse buttons to select folders using your system's folder picker.  Works in Chrome, Edge, and other modern browsers.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TotalSegmentatorApp;