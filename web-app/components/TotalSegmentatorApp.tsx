'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Folder, Play, ChevronRight, ChevronDown, CheckCircle, XCircle, Loader2, Settings, AlertCircle } from 'lucide-react';
import { useLocale } from '@/app/contexts/LocaleContext';

interface Log {
  timestamp: string;
  message: string;
}

type Status = 'ready' | 'processing' | 'success' | 'error';

const TotalSegmentatorApp = () => {
  const { locale, setLocale, t } = useLocale();
  const [dicomFolder, setDicomFolder] = useState('');
  const [outputFolder, setOutputFolder] = useState('');
  const [projectName, setProjectName] = useState('Project-01');
  const [blenderScale, setBlenderScale] = useState(20.0);
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
        
        // Try to get the full path
        let fullPath = dirHandle.name;
        
        // For file handles (blender/dcm2niix executables), we need different handling
        if (type === 'blender' || type === 'dcm2niix') {
          // These should be files, not directories
          // Use the directory path selected
          fullPath = dirHandle.name;
        }
        
        switch (type) {
          case 'dicom':
            setDicomFolder(fullPath);
            break;
          case 'output':
            setOutputFolder(fullPath);
            break;
          case 'blender':
            setBlenderPath(fullPath);
            break;
          case 'dcm2niix':
            setDcm2niixPath(fullPath);
            break;
        }
        
        addLog(`Selected: ${fullPath}`);
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error('Error selecting folder:', error);
          addLog(`Folder picker cancelled or not supported. Please enter path manually.`);
          // Fallback to manual input
          handleManualInput(type);
        }
      }
    } else {
      // Fallback for browsers that don't support File System Access API
      addLog('File System Access API not supported. Please enter path manually.');
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

    // Check for spaces in folder names
    if (dicomFolder.includes(' ') || outputFolder.includes(' ')) {
      alert('⚠️ Folder names cannot contain spaces!\n\nPlease rename your folders to use underscores (_) or hyphens (-) instead of spaces.\n\nExample: "Testing_Output" instead of "Testing Output"');
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
    { label: t.buttons.step1, mode: 'step1' },
    { label: t.buttons.step2, mode: 'step2' },
    { label: t.buttons.step3, mode: 'step3' },
    { label: t.buttons.step4, mode: 'step4' },
    { label: t.buttons.step5, mode: 'step5' },
    { label: t.buttons.step6, mode: 'step6' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Floating Language Toggle Button - Top Right */}
      <div className="fixed top-8 right-8 z-50">
        <button
          onClick={() => setLocale(locale === 'en' ? 'jp' : 'en')}
          className="w-16 h-16 rounded-full shadow-lg bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 transition-all duration-300 flex items-center justify-center text-white font-bold text-lg border-2 border-white"
          title={locale === 'en' ? 'Switch to Japanese' : 'Switch to English'}
        >
          {locale === 'en' ? 'JP' : 'EN'}
        </button>
      </div>

      <div className="w-full">
        {/* Header */}
        <div className="px-8 py-6 border-b border-blue-200/50">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            {t.header.title}
          </h1>
          <p className="text-lg text-gray-600">
            {t.header.subtitle}
          </p>
        </div>

        {/* Three Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
          {/* Column 1: Input & Output */}
          <div className="p-6 bg-white/30 backdrop-blur-sm rounded-xl">
            <h2 className="text-xl font-semibold text-blue-900 mb-6 flex items-center gap-2">
              <Folder className="w-5 h-5" />
              {t.sections.inputOutput}
            </h2>
            
            <div className="space-y-5">
              <div>
                <label className="block text-gray-700 font-medium mb-2">{t.labels.dicomFolder}</label>
                <input
                  type="text"
                  value={dicomFolder}
                  onChange={(e) => setDicomFolder(e.target.value)}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-2 shadow-sm"
                  placeholder="Click Browse to select folder..."
                  readOnly
                />
                <button
                  onClick={() => handleFolderPicker('dicom')}
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors border border-blue-700 flex items-center justify-center gap-2 shadow-sm"
                >
                  <Folder className="w-4 h-4" />
                  {t.labels.browse}
                </button>
              </div>

              <div>
                <label className="block text-gray-700 font-medium mb-2">{t.labels.outputFolder}</label>
                <input
                  type="text"
                  value={outputFolder}
                  onChange={(e) => setOutputFolder(e.target.value)}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-2 shadow-sm"
                  placeholder="Click Browse to select folder..."
                  readOnly
                />
                <button
                  onClick={() => handleFolderPicker('output')}
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors border border-blue-700 flex items-center justify-center gap-2 shadow-sm"
                >
                  <Folder className="w-4 h-4" />
                  {t.labels.browse}
                </button>
              </div>
            </div>

            {/* Tool Paths Section */}
            <div className="mt-6">
              <button
                onClick={() => setShowTools(!showTools)}
                className="flex items-center gap-2 text-gray-700 hover:text-gray-900 mb-3 transition-colors"
              >
                {showTools ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                <span className="font-medium">{t.sections.toolPaths}</span>
              </button>

              {showTools && (
                <div className="space-y-4 pl-7">
                  <div>
                    <label className="block text-gray-700 font-medium mb-2 text-sm">{t.labels.blenderExe}</label>
                    <input
                      type="text"
                      value={blenderPath}
                      onChange={(e) => setBlenderPath(e.target.value)}
                      className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent text-sm mb-2 shadow-sm"
                      placeholder="Click Browse..."
                      readOnly
                    />
                    <button
                      onClick={() => handleFolderPicker('blender')}
                      className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm flex items-center justify-center gap-2 shadow-sm"
                    >
                      <Folder className="w-4 h-4" />
                      {t.labels.browse}
                    </button>
                  </div>

                  <div>
                    <label className="block text-gray-700 font-medium mb-2 text-sm">{t.labels.dcm2niixExe}</label>
                    <input
                      type="text"
                      value={dcm2niixPath}
                      onChange={(e) => setDcm2niixPath(e.target.value)}
                      className="w-full px-3 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent text-sm mb-2 shadow-sm"
                      placeholder="Click Browse..."
                      readOnly
                    />
                    <button
                      onClick={() => handleFolderPicker('dcm2niix')}
                      className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm flex items-center justify-center gap-2 shadow-sm"
                    >
                      <Folder className="w-4 h-4" />
                      {t.labels.browse}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Column 2: Configuration */}
          <div className="p-6 bg-white/30 backdrop-blur-sm rounded-xl">
            <h2 className="text-xl font-semibold text-indigo-900 mb-6 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              {t.sections.configuration}
            </h2>
            
            <div className="space-y-6">
              <div>
                <label className="block text-gray-700 font-medium mb-2">{t.labels.projectName}</label>
                <input
                  type="text"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent shadow-sm"
                />
              </div>
              
              <div>
                <label className="block text-gray-700 font-medium mb-2">
                  {t.labels.blenderScale} <span className="text-indigo-600 font-semibold">{blenderScale.toFixed(3)}</span>
                </label>
                <input
                  type="range"
                  min="10.0"
                  max="50.0"
                  step="5.0"
                  value={blenderScale}
                  onChange={(e) => setBlenderScale(parseFloat(e.target.value))}
                  className="w-full h-2 bg-white border border-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600 shadow-sm"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>10.0</span>
                  <span>50.0</span>
                </div>
              </div>
            </div>

            {/* Info Alert */}
            <div className="mt-6 bg-blue-50/80 border border-blue-200 rounded-lg p-4 flex gap-3 shadow-sm">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-900">
                <p className="font-semibold mb-1">{t.alerts.notesTitle}</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>{t.alerts.note1}</li>
                  <li><strong>{t.alerts.note2}</strong></li>
                  <li>{t.alerts.note3}</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Column 3: Run Pipeline */}
          <div className="p-6">
            <h2 className="text-xl font-semibold text-green-900 mb-6 flex items-center gap-2">
              <Play className="w-5 h-5" />
              {t.sections.runPipeline}
            </h2>
            
            <button
              onClick={() => runPipeline('all')}
              disabled={isProcessing}
              className="w-full py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold text-lg mb-6 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-sm"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  {t.buttons.processing}
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  {t.buttons.runFull}
                </>
              )}
            </button>

            <div>
              <p className="text-gray-700 mb-3 font-medium">{t.labels.individualSteps}</p>
              <div className="grid grid-cols-1 gap-2">
                {stepButtons.map((btn) => (
                  <button
                    key={btn.mode}
                    onClick={() => runPipeline(btn.mode)}
                    disabled={isProcessing}
                    className="px-3 py-3 bg-indigo-50/80 text-indigo-700 rounded-lg hover:bg-indigo-100 transition-colors border border-indigo-200 text-sm font-medium disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed shadow-sm"
                  >
                    {btn.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Progress Section - Full Width Below */}
        <div className="mx-6 mb-6 p-6 border-t border-blue-200/50">
          <div className="flex items-center gap-3 mb-4">
            {getStatusIcon()}
            <span className="text-lg font-semibold text-gray-900">{getStatusText()}</span>
          </div>
          
          <div className="relative w-full h-4 bg-white/50 rounded-full overflow-hidden shadow-sm border border-gray-200">
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

        {/* Log Section - Full Width Below */}
        <div className="mx-6 mb-6">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="flex items-center gap-2 text-gray-700 hover:text-gray-900 mb-3 transition-colors"
          >
            {showLogs ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            <span className="font-medium">{t.sections.processLog}</span>
          </button>
          
          {showLogs && (
            <div className="p-6">
              <div
                ref={logContainerRef}
                className="bg-white/80 backdrop-blur-sm rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm border border-gray-200 shadow-sm"
              >
                {logs.length === 0 ? (
                  <p className="text-gray-500">{t.logs.noLogs}</p>
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
      </div>
    </div>
  );
};

export default TotalSegmentatorApp;