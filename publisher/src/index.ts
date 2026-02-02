/**
 * Multi-Camera LiveKit Publisher
 * Main entry point
 * 
 * Pulls annotated RTSP streams and publishes to LiveKit room
 */

import 'dotenv/config';
import { loadConfig, AppConfig } from './config';
import { FFmpegReader } from './ffmpeg';
import { LiveKitPublisher } from './livekit';
import { createLogger } from './logger';

const logger = createLogger('main');

interface CameraPublisher {
  camId: string;
  reader: FFmpegReader;
  publisher: LiveKitPublisher;
}

async function main() {
  logger.info('Starting Multi-Camera LiveKit Publisher');

  // Load configuration
  const config = loadConfig();
  logger.info(`Loaded config with ${config.cameras.length} cameras`);

  const publishers: CameraPublisher[] = [];

  // Create publisher for each camera
  for (const camera of config.cameras) {
    const annotatedRtsp = `${config.restream.rtsp_base}/ann_${camera.id}`;
    logger.info(`Setting up publisher for ${camera.id}: ${annotatedRtsp}`);

    // Create FFmpeg reader
    const reader = new FFmpegReader({
      rtspUrl: annotatedRtsp,
      width: config.restream.resolution[0],
      height: config.restream.resolution[1],
      fps: config.restream.fps,
    });

    // Create LiveKit publisher
    const publisher = new LiveKitPublisher({
      config: config.livekit,
      trackName: `ann_${camera.id}`,
      width: config.restream.resolution[0],
      height: config.restream.resolution[1],
      fps: config.restream.fps,
    });

    // Connect frames from reader to publisher
    reader.on('frame', (frame: Buffer) => {
      publisher.publishFrame(frame);
    });

    publishers.push({
      camId: camera.id,
      reader,
      publisher,
    });
  }

  // Connect all publishers to LiveKit
  logger.info('Connecting publishers to LiveKit...');
  for (const pub of publishers) {
    try {
      await pub.publisher.connect();
      logger.info(`Publisher connected for ${pub.camId}`);
    } catch (err) {
      logger.error(`Failed to connect publisher for ${pub.camId}: ${err}`);
    }
  }

  // Start all FFmpeg readers
  logger.info('Starting FFmpeg readers...');
  for (const pub of publishers) {
    pub.reader.start();
    logger.info(`Reader started for ${pub.camId}`);
  }

  logger.info('Publisher running. Press Ctrl+C to stop.');

  // Log stats periodically
  setInterval(() => {
    for (const pub of publishers) {
      const stats = pub.publisher.getStats();
      logger.debug(`${pub.camId}: frames=${stats.frameCount}, connected=${stats.connected}`);
    }
  }, 10000);

  // Handle shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down...');

    for (const pub of publishers) {
      pub.reader.stop();
      await pub.publisher.disconnect();
    }

    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('Shutting down...');

    for (const pub of publishers) {
      pub.reader.stop();
      await pub.publisher.disconnect();
    }

    process.exit(0);
  });
}

main().catch((err) => {
  logger.error(`Fatal error: ${err}`);
  process.exit(1);
});
