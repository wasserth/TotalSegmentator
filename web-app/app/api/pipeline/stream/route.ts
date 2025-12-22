import { NextRequest } from 'next/server';
import { PythonExecutor } from '@/lib/python-executor';
import path from 'path';

const TOTALSEGMENTATOR_PATH = process.env.TOTALSEGMENTATOR_PATH || 
  path.join(process.cwd(), '..');

export async function POST(request: NextRequest) {
  const body = await request.json();
  const {
    dicomDir,
    outputDir,
    projectName,
    task = 'total',
    fast = false,
    ml = true,
    statistics = false,
    device = 'cpu'
  } = body;

  const encoder = new TextEncoder();
  
  const stream = new ReadableStream({
    async start(controller) {
      const executor = new PythonExecutor({
        totalsegmentatorPath: TOTALSEGMENTATOR_PATH,
        pythonPath: process.env.PYTHON_PATH || 'python3'
      });

      const sendMessage = (type: string, message: string, progress?: number) => {
        const data = `data: ${JSON.stringify({ 
          type, 
          message, 
          progress,
          timestamp: new Date().toISOString() 
        })}\n\n`;
        controller.enqueue(encoder.encode(data));
      };

      try {
        sendMessage('info', `Starting TotalSegmentator pipeline...`, 0);
        sendMessage('info', `Input: ${dicomDir}`, 0);
        sendMessage('info', `Output: ${outputDir}`, 0);

        let currentProgress = 0;

        const exitCode = await executor.runTotalSegmentatorStream(
          {
            input: dicomDir,
            output: outputDir,
            ml,
            fast,
            task,
            statistics,
            device,
            verbose: true
          },
          (message) => {
            // Parse progress from TotalSegmentator output
            const progressMatch = message. match(/(\d+)%/);
            if (progressMatch) {
              currentProgress = parseInt(progressMatch[1]);
            }
            sendMessage('progress', message. trim(), currentProgress);
          },
          (error) => {
            sendMessage('error', error. trim());
          }
        );

        if (exitCode === 0) {
          sendMessage('complete', 'Pipeline completed successfully!', 100);
        } else {
          sendMessage('error', `Pipeline exited with code ${exitCode}`);
        }

      } catch (error: any) {
        sendMessage('error', error.message);
      } finally {
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}