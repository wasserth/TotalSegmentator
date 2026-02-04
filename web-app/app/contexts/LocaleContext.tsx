'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

type Locale = 'en' | 'jp';

interface Translations {
  header: {
    title: string;
    subtitle: string;
  };
  sections: {
    inputOutput: string;
    configuration: string;
    runPipeline: string;
    processLog: string;
    toolPaths: string;
  };
  labels: {
    dicomFolder: string;
    outputFolder: string;
    browse: string;
    projectName: string;
    segmentationTask: string;
    blenderScale: string;
    blenderExe: string;
    dcm2niixExe: string;
    individualSteps: string;
  };
  buttons: {
    runFull: string;
    processing: string;
    step1: string;
    step2: string;
    step3: string;
    step4: string;
    step5: string;
    step6: string;
  };
  status: {
    ready: string;
    processing: string;
    complete: string;
    failed: string;
  };
  alerts: {
    notesTitle: string;
    note1: string;
    note2: string;
    note3: string;
  };
  logs: {
    noLogs: string;
  };
}

const en: Translations = {
  header: {
    title: "TotalSegmentator Pipeline",
    subtitle: "Automated medical imaging segmentation with Blender 3D visualization"
  },
  sections: {
    inputOutput: "Input & Output",
    configuration: "Configuration",
    runPipeline: "Run Pipeline",
    processLog: "Process Log",
    toolPaths: "Tool Paths (Optional)"
  },
  labels: {
    dicomFolder: "DICOM Folder",
    outputFolder: "Output Folder",
    browse: "Browse...",
    projectName: "Project Name",
    segmentationTask: "Segmentation Task",
    blenderScale: "Blender Scale:",
    blenderExe: "Blender Executable",
    dcm2niixExe: "dcm2niix Executable",
    individualSteps: "Individual Steps:"
  },
  buttons: {
    runFull: "Run Full Pipeline",
    processing: "Processing...",
    step1: "1. DICOM→NIfTI",
    step2: "2. NIfTI→PNG",
    step3: "3. Segment",
    step4: "4. Import",
    step5: "5. Materials",
    step6: "6. Slice Viewer"
  },
  status: {
    ready: "Ready to start",
    processing: "Processing...",
    complete: "✓ Complete! (6/6)",
    failed: "✗ Failed"
  },
  alerts: {
    notesTitle: "Important Notes",
    note1: "Use Chrome/Edge browser for folder picker",
    note2: "Folder names must not contain spaces",
    note3: "Use underscores (_) or hyphens (-) instead"
  },
  logs: {
    noLogs: "No logs yet. Start a pipeline to see output..."
  }
};

const jp: Translations = {
  header: {
    title: "TotalSegmentator パイプライン",
    subtitle: "Blender 3D可視化による自動医療画像セグメンテーション"
  },
  sections: {
    inputOutput: "入力と出力",
    configuration: "設定",
    runPipeline: "パイプライン実行",
    processLog: "処理ログ",
    toolPaths: "ツールパス（オプション）"
  },
  labels: {
    dicomFolder: "DICOMフォルダ",
    outputFolder: "出力フォルダ",
    browse: "参照...",
    projectName: "プロジェクト名",
    segmentationTask: "セグメンテーションタスク",
    blenderScale: "Blenderスケール：",
    blenderExe: "Blender実行ファイル",
    dcm2niixExe: "dcm2niix実行ファイル",
    individualSteps: "個別ステップ："
  },
  buttons: {
    runFull: "完全パイプライン実行",
    processing: "処理中...",
    step1: "1. DICOM→NIfTI",
    step2: "2. NIfTI→PNG",
    step3: "3. セグメント",
    step4: "4. インポート",
    step5: "5. マテリアル",
    step6: "6. スライスビューアー"
  },
  status: {
    ready: "開始準備完了",
    processing: "処理中...",
    complete: "✓ 完了！（6/6）",
    failed: "✗ 失敗"
  },
  alerts: {
    notesTitle: "重要な注意事項",
    note1: "フォルダピッカーにはChrome/Edgeブラウザを使用してください",
    note2: "フォルダ名にスペースを含めることはできません",
    note3: "代わりにアンダースコア（_）またはハイフン（-）を使用してください"
  },
  logs: {
    noLogs: "ログはまだありません。パイプラインを開始すると出力が表示されます..."
  }
};

interface LocaleContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: Translations;
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined);

const translations: Record<Locale, Translations> = { en, jp };

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>('en');

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t: translations[locale] }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const context = useContext(LocaleContext);
  if (!context) throw new Error('useLocale must be used within LocaleProvider');
  return context;
}
