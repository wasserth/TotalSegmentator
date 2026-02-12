'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Folder, Play, ChevronRight, ChevronDown, CheckCircle, XCircle, Loader2, Settings, AlertCircle } from 'lucide-react';
import { useLocale } from '@/app/contexts/LocaleContext';
import AlertModal from './AlertModal';
import FolderBrowserModal from './FolderBrowserModal';

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
  const [tasks, setTasks] = useState<string[]>(['total_all']);
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
  const [showTaskDropdown, setShowTaskDropdown] = useState(false);
  
  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState<'dicom' | 'output' | 'blender' | 'dcm2niix'>('dicom');
  const [modalTitle, setModalTitle] = useState('');
  
  // Alert modal states
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertTitle, setAlertTitle] = useState('');
  const [alertMessage, setAlertMessage] = useState('');
  const [alertType, setAlertType] = useState<'success' | 'error' | 'warning' | 'info'>('info');
  
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (message: string) => {
    setLogs(prev => [...prev, { timestamp: new Date().toISOString(), message }]);
  };

  const showAlert = (title: string, message: string, type: 'success' | 'error' | 'warning' | 'info') => {
    setAlertTitle(title);
    setAlertMessage(message);
    setAlertType(type);
    setAlertOpen(true);
  };

  /**
   * Open custom folder picker modal
   */
  const handleFolderPicker = (type: 'dicom' | 'output' | 'blender' | 'dcm2niix') => {
    setModalType(type);
    
    // Set modal title based on type
    const titles = {
      dicom: 'Select DICOM Folder',
      output: 'Select Output Folder',
      blender: 'Select Blender Executable Folder',
      dcm2niix: 'Select dcm2niix Executable Folder'
    };
    
    setModalTitle(titles[type]);
    setModalOpen(true);
  };

  /**
   * Handle folder selection from custom modal
   */
  const handleFolderSelect = (path: string) => {
    switch (modalType) {
      case 'dicom':
        setDicomFolder(path);
        break;
      case 'output':
        setOutputFolder(path);
        break;
      case 'blender':
        setBlenderPath(path);
        break;
      case 'dcm2niix':
        setDcm2niixPath(path);
        break;
    }
    
    addLog(`Selected ${modalType}: ${path}`);
  };

  const runPipeline = async (mode = 'all') => {
    if (!dicomFolder || !outputFolder) {
      showAlert('Missing Folders', 'Please select both DICOM folder and output folder', 'warning');
      return;
    }

    // Check for spaces in folder names
    if (dicomFolder.includes(' ') || outputFolder.includes(' ')) {
      showAlert(
        'Invalid Folder Names', 
        'Folder names cannot contain spaces!\n\nPlease rename your folders to use underscores (_) or hyphens (-) instead of spaces.\n\nExample: "Testing_Output" instead of "Testing Output"',
        'warning'
      );
      return;
    }

    if (isProcessing) {
      showAlert('Pipeline Running', 'A pipeline is already in progress', 'info');
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setStatus('processing');
    setLogs([]);
    setCurrentStep('Initializing pipeline...');
    
    if (! showLogs) setShowLogs(true);

    try {
      const selectedTasks = tasks.length ? tasks : ['total_all'];
      addLog(`Starting pipeline (mode: ${mode}, tasks: ${selectedTasks.join(', ')})...`);
      const response = await fetch('/api/pipeline/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dicomDir: dicomFolder,
          outputDir: outputFolder,
          projectName,
          task: selectedTasks.join(','),
          scale: blenderScale,
          blenderPath,
          dcm2niixPath,
          mode
        })
      });

      if (!response.ok) {
        let detail = 'Pipeline failed';
        try {
          const errJson = await response.json();
          detail = errJson?.details || errJson?.detail || errJson?.error || detail;
        } catch {}
        throw new Error(detail);
      }
      if (!response.body) {
        throw new Error('No stream body from server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let completed = false;
      let seenError = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const evt of events) {
          const line = evt
            .split('\n')
            .find((l) => l.startsWith('data: '));
          if (!line) continue;
          try {
            const payload = JSON.parse(line.slice(6)) as {
              type: string;
              message: string;
              progress?: number;
            };

            if (payload.type === 'progress') {
              if (typeof payload.progress === 'number') {
                setProgress(payload.progress);
              }
              setCurrentStep(payload.message || 'Processing...');
              addLog(`Progress ${payload.progress ?? ''}%: ${payload.message}`);
            } else if (payload.type === 'info') {
              addLog(payload.message);
            } else if (payload.type === 'error') {
              seenError = true;
              addLog(`Error: ${payload.message}`);
            } else if (payload.type === 'complete') {
              completed = true;
              setStatus('success');
              setProgress(100);
              setCurrentStep('Complete');
              addLog(payload.message || 'Pipeline completed successfully!');
            }
          } catch {
            // Ignore malformed SSE event payloads.
          }
        }
      }

      if (completed && !seenError) {
        showAlert('Success!', 'Pipeline completed successfully!', 'success');
      } else {
        throw new Error('Pipeline failed. See log for details.');
      }
      
    } catch (error: any) {
      setStatus('error');
      addLog(`Error: ${error.message}`);
      showAlert('Pipeline Failed', 'Check the log for details.', 'error');
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

              <div className="relative">
                <label className="block text-gray-700 font-medium mb-2">{t.labels.segmentationTask}</label>
                {(() => {
                  const preferred = ['total_all', 'liver_segments', 'liver_vessels'];
                  const allOptions = [
                    'total_all',
                    'liver_segments',
                    'liver_vessels',
                    'total_vessels',
                    'body',
                    'abdominal_muscles',
                    'lung_vessels',
                    'pleural_pericard_effusion',
                    'ventricle_parts',
                  ];
                  const rest = allOptions.filter((t) => !preferred.includes(t));
                  const options = [...preferred, ...rest];
                  const summary = tasks.length ? tasks.join(', ') : 'Select tasks';
                  return (
                    <>
                      <button
                        type="button"
                        onClick={() => setShowTaskDropdown((v) => !v)}
                        className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg shadow-sm text-left text-gray-800 flex items-center justify-between"
                      >
                        <span className="text-sm truncate">{summary}</span>
                        <ChevronDown className={`w-4 h-4 transition-transform ${showTaskDropdown ? 'rotate-180' : ''}`} />
                      </button>
                      {showTaskDropdown && (
                        <div className="absolute z-10 mt-2 w-full max-h-64 overflow-auto bg-white border border-gray-300 rounded-lg shadow-lg p-2">
                          {options.map((opt) => (
                            <label key={opt} className="flex items-center gap-3 py-1 text-gray-800">
                              <input
                                type="checkbox"
                                checked={tasks.includes(opt)}
                                onChange={() =>
                                  setTasks((prev) =>
                                    prev.includes(opt) ? prev.filter((t) => t !== opt) : [...prev, opt]
                                  )
                                }
                                className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                              />
                              <span className="text-sm">{opt}</span>
                            </label>
                          ))}
                        </div>
                      )}
                    </>
                  );
                })()}
                <p className="text-xs text-gray-500 mt-2">Click to open and select multiple tasks.</p>
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

      {/* Custom Folder Browser Modal */}
      <FolderBrowserModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSelect={handleFolderSelect}
        title={modalTitle}
      />

      {/* Custom Alert Modal */}
      <AlertModal
        isOpen={alertOpen}
        onClose={() => setAlertOpen(false)}
        title={alertTitle}
        message={alertMessage}
        type={alertType}
      />
    </div>
  );
};

export default TotalSegmentatorApp;
