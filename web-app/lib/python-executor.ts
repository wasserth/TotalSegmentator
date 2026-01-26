import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

export interface PythonExecutorOptions {
  pythonPath?: string;
  totalsegmentatorPath: string;
}

export class PythonExecutor {
  private pythonPath: string;
  private totalsegmentatorPath: string;
  private binPath: string;

  constructor(options: PythonExecutorOptions) {
    this.pythonPath = options.pythonPath || 'python3';
    this.totalsegmentatorPath = options.totalsegmentatorPath;
    this.binPath = path.join(this.totalsegmentatorPath, 'totalsegmentator', 'bin');
  }

  /**
   * Execute TotalSegmentator. py - Main segmentation script
   */
  async runTotalSegmentator(args: {
    input: string;
    output: string;
    ml?: boolean;
    nr_thr_resamp?: number;
    nr_thr_saving?: number;
    fast?: boolean;
    nora_tag?: string;
    preview?: boolean;
    task?: string;
    roi_subset?: string[];
    statistics?: boolean;
    radiomics?: boolean;
    crop_path?: string;
    body_seg?: boolean;
    force_split?: boolean;
    output_type?: string;
    quiet?: boolean;
    verbose?: boolean;
    test?: number;
    skip_saving?: boolean;
    device?: string;
    license_number?: string;
    statistics_exclude_masks_at_border?: boolean;
    no_derived_masks?: boolean;
    v1_order?: boolean;
  }) {
    const scriptPath = path.join(this. binPath, 'TotalSegmentator.py');
    
    let command = `${this.pythonPath} "${scriptPath}" -i "${args.input}" -o "${args.output}"`;
    
    if (args.ml) command += ' --ml';
    if (args. nr_thr_resamp) command += ` --nr_thr_resamp ${args.nr_thr_resamp}`;
    if (args.nr_thr_saving) command += ` --nr_thr_saving ${args. nr_thr_saving}`;
    if (args.fast) command += ' --fast';
    if (args.task) command += ` --task ${args.task}`;
    if (args.roi_subset) command += ` --roi_subset ${args.roi_subset.join(' ')}`;
    if (args.statistics) command += ' --statistics';
    if (args.radiomics) command += ' --radiomics';
    if (args.body_seg) command += ' --body_seg';
    if (args.device) command += ` --device ${args.device}`;
    if (args.verbose) command += ' --verbose';
    if (args.quiet) command += ' --quiet';

    console.log('Executing command:', command);

    return this.executeCommand(command);
  }

  /**
   * Execute crop_to_body. py - Crop images to body region
   */
  async cropToBody(args: {
    input: string;
    output: string;
    bodyMaskInput?: string;
  }) {
    const scriptPath = path.join(this.binPath, 'crop_to_body.py');
    
    let command = `${this.pythonPath} "${scriptPath}" -i "${args.input}" -o "${args.output}"`;
    if (args.bodyMaskInput) command += ` --body_seg "${args.bodyMaskInput}"`;

    return this.executeCommand(command);
  }

  /**
   * Execute totalseg_combine_masks.py - Combine multiple masks
   */
  async combineMasks(args: {
    input: string;
    output: string;
    masks: string[];
  }) {
    const scriptPath = path.join(this.binPath, 'totalseg_combine_masks. py');
    
    const command = `${this.pythonPath} "${scriptPath}" -i "${args.input}" -o "${args.output}" -m ${args.masks.join(' ')}`;

    return this.executeCommand(command);
  }

  /**
   * Execute totalseg_download_weights.py - Download model weights
   */
  async downloadWeights(task?: string) {
    const scriptPath = path.join(this.binPath, 'totalseg_download_weights.py');
    
    let command = `${this.pythonPath} "${scriptPath}"`;
    if (task) command += ` -t ${task}`;

    return this.executeCommand(command);
  }

  /**
   * Execute command and return output
   */
  private executeCommand(command: string): Promise<{ stdout: string; stderr: string }> {
    return execAsync(command, {
      maxBuffer: 1024 * 1024 * 50, // 50MB buffer for large outputs
      env: { ...process.env }
    });
  }

  /**
   * Execute command with real-time streaming
   */
  streamCommand(
    command: string, 
    onData: (data: string) => void, 
    onError: (error: string) => void
  ): Promise<number> {
    return new Promise((resolve, reject) => {
      console.log('Streaming command:', command);
      
      const childProcess = spawn(command, {
        shell: true,
        env: { ...process.env }
      });

      childProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log('STDOUT:', output);
        onData(output);
      });

      childProcess.stderr.on('data', (data) => {
        const output = data.toString();
        console.log('STDERR:', output);
        onError(output);
      });

      childProcess.on('close', (code) => {
        console.log('Process exited with code:', code);
        resolve(code || 0);
      });

      childProcess.on('error', (error) => {
        console. error('Process error:', error);
        reject(error);
      });
    });
  }

  /**
   * Run TotalSegmentator with streaming output
   */
  async runTotalSegmentatorStream(
    args: Parameters<typeof this.runTotalSegmentator>[0],
    onProgress: (message: string) => void,
    onError: (message: string) => void
  ): Promise<number> {
    const scriptPath = path.join(this.binPath, 'TotalSegmentator.py');
    
    let command = `${this.pythonPath} "${scriptPath}" -i "${args.input}" -o "${args.output}"`;
    
    if (args.ml) command += ' --ml';
    if (args.fast) command += ' --fast';
    if (args.task) command += ` --task ${args. task}`;
    if (args.statistics) command += ' --statistics';
    if (args.verbose) command += ' --verbose';
    if (args.device) command += ` --device ${args.device}`;

    return this.streamCommand(command, onProgress, onError);
  }
}