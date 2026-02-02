/**
 * FFmpeg RTSP to raw frames
 * Pulls RTSP stream and outputs raw video frames
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { createLogger } from './logger';

const logger = createLogger('ffmpeg');

export interface FFmpegReaderOptions {
  rtspUrl: string;
  width: number;
  height: number;
  fps: number;
}

export class FFmpegReader extends EventEmitter {
  private options: FFmpegReaderOptions;
  private process: ChildProcess | null = null;
  private running = false;
  private frameBuffer: Buffer[] = [];
  private frameSize: number;
  private currentBuffer: Buffer = Buffer.alloc(0);

  constructor(options: FFmpegReaderOptions) {
    super();
    this.options = options;
    this.frameSize = options.width * options.height * 3; // RGB24
  }

  start(): void {
    if (this.running) return;

    this.running = true;
    this.spawnProcess();
  }

  stop(): void {
    this.running = false;
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
  }

  private spawnProcess(): void {
    const { rtspUrl, width, height, fps } = this.options;

    const args = [
      '-rtsp_transport', 'tcp',
      '-i', rtspUrl,
      '-f', 'rawvideo',
      '-pix_fmt', 'rgb24',
      '-s', `${width}x${height}`,
      '-r', String(fps),
      '-an', // No audio
      '-'
    ];

    const ffmpegBin = process.env.FFMPEG_PATH && process.env.FFMPEG_PATH.trim().length > 0
      ? process.env.FFMPEG_PATH.trim()
      : 'ffmpeg';

    logger.info(`Starting FFmpeg: ${ffmpegBin} ${args.join(' ')}`);

    this.process = spawn(ffmpegBin, args, {
      stdio: ['ignore', 'pipe', 'pipe']
    });

    this.process.stdout!.on('data', (data: Buffer) => {
      this.handleData(data);
    });

    this.process.stderr!.on('data', (data: Buffer) => {
      const line = data.toString().trim();
      if (line && !line.includes('frame=')) {
        logger.debug(`FFmpeg: ${line}`);
      }
    });

    this.process.on('close', (code) => {
      logger.warn(`FFmpeg exited with code ${code}`);
      this.process = null;

      if (this.running) {
        // Auto-reconnect after delay
        setTimeout(() => {
          if (this.running) {
            logger.info('Reconnecting FFmpeg...');
            this.spawnProcess();
          }
        }, 2000);
      }
    });

    this.process.on('error', (err) => {
      logger.error(`FFmpeg error: ${err.message}`);
    });
  }

  private handleData(data: Buffer): void {
    // Append to current buffer
    this.currentBuffer = Buffer.concat([this.currentBuffer, data]);

    // Extract complete frames
    while (this.currentBuffer.length >= this.frameSize) {
      const frame = this.currentBuffer.subarray(0, this.frameSize);
      this.currentBuffer = this.currentBuffer.subarray(this.frameSize);
      this.emit('frame', frame);
    }
  }
}
