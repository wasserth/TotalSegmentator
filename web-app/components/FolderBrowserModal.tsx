'use client';

import React, { useState, useEffect } from 'react';
import { Folder, ChevronRight, X, Home } from 'lucide-react';

interface FolderBrowserModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  title: string;
}

export default function FolderBrowserModal({ isOpen, onClose, onSelect, title }: FolderBrowserModalProps) {
  const [currentPath, setCurrentPath] = useState('');
  const [directories, setDirectories] = useState<any[]>([]);
  const [parentPath, setParentPath] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadDirectories('');
    }
  }, [isOpen]);

  const loadDirectories = async (path: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/browse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ currentPath: path })
      });

      const data = await response.json();
      setCurrentPath(data.currentPath);
      setParentPath(data.parentPath);
      setDirectories(data. directories);
    } catch (error) {
      console.error('Error loading directories:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = () => {
    onSelect(currentPath);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Current Path */}
        <div className="p-4 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center gap-2 text-sm">
            <Home className="w-4 h-4 text-gray-500" />
            <span className="text-gray-700 font-mono">{currentPath || 'Home'}</span>
          </div>
        </div>

        {/* Directory List */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center py-8 text-gray-500">Loading...</div>
          ) : (
            <div className="space-y-1">
              {/* Parent Directory */}
              {parentPath && (
                <button
                  onClick={() => loadDirectories(parentPath)}
                  className="w-full flex items-center gap-3 p-3 hover:bg-gray-100 rounded-lg transition-colors text-left"
                >
                  <Folder className="w-5 h-5 text-gray-400" />
                  <span className="text-gray-600">..</span>
                </button>
              )}

              {/* Directories */}
              {directories.map((dir) => (
                <button
                  key={dir. path}
                  onClick={() => loadDirectories(dir.path)}
                  className="w-full flex items-center gap-3 p-3 hover:bg-blue-50 rounded-lg transition-colors text-left group"
                >
                  <Folder className="w-5 h-5 text-blue-500" />
                  <span className="flex-1 text-gray-900">{dir.name}</span>
                  <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-blue-500" />
                </button>
              ))}

              {directories.length === 0 && ! parentPath && (
                <div className="text-center py-8 text-gray-500">
                  No accessible directories found
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 flex items-center justify-between gap-4">
          <button
            onClick={onClose}
            className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSelect}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Select This Folder
          </button>
        </div>
      </div>
    </div>
  );
}