/**
 * Configuration loader
 * Loads config from config.yaml
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';

export interface CameraConfig {
  id: string;
  rtsp: string;
}

export interface LiveKitConfig {
  url: string;
  api_key: string;
  api_secret: string;
  room: string;
}

export interface RestreamConfig {
  rtsp_base: string;
  fps: number;
  resolution: [number, number];
}

export interface AppConfig {
  cameras: CameraConfig[];
  livekit: LiveKitConfig;
  restream: RestreamConfig;
}

export function loadConfig(configPath?: string): AppConfig {
  const resolvedPath = configPath || process.env.CONFIG_PATH || '../config.yaml';
  
  // Resolve relative to publisher directory
  const publisherDir = path.dirname(__dirname);
  let configFile = path.resolve(publisherDir, resolvedPath);
  
  if (!fs.existsSync(configFile)) {
    // Try from project root
    const projectRoot = path.dirname(publisherDir);
    configFile = path.join(projectRoot, 'config.yaml');
  }
  
  if (!fs.existsSync(configFile)) {
    throw new Error(`Config file not found: ${configFile}`);
  }
  
  const content = fs.readFileSync(configFile, 'utf-8');
  const raw = yaml.parse(content);
  
  return {
    cameras: raw.cameras || [],
    livekit: {
      url: raw.livekit?.url || 'ws://127.0.0.1:7880',
      api_key: raw.livekit?.api_key || 'devkey',
      api_secret: raw.livekit?.api_secret || 'devsecret',
      room: raw.livekit?.room || 'multicam',
    },
    restream: {
      rtsp_base: raw.restream?.rtsp_base || 'rtsp://127.0.0.1:8554',
      fps: raw.restream?.fps || 15,
      resolution: raw.restream?.resolution || [1280, 720],
    },
  };
}
