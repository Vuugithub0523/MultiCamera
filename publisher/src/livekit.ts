/**
 * LiveKit publisher
 * Publishes raw video frames to a LiveKit room (rtc-node v0.5.x)
 */

import {
  Room,
  RoomEvent,
  VideoFrame,
  VideoSource,
  LocalVideoTrack,
} from '@livekit/rtc-node';

// NOTE: rtc-node v0.5.x keeps many enums/messages under dist/proto
import { TrackPublishOptions, VideoEncoding } from '@livekit/rtc-node/dist/proto/room_pb.js';
import { VideoCodec, VideoBufferType } from '@livekit/rtc-node/dist/proto/video_frame_pb.js';

import { AccessToken } from 'livekit-server-sdk';
import { LiveKitConfig } from './config';
import { createLogger } from './logger';

const logger = createLogger('livekit');

export interface PublisherOptions {
  config: LiveKitConfig;
  trackName: string;
  width: number;
  height: number;
  fps: number;
}

export class LiveKitPublisher {
  private options: PublisherOptions;
  private room: Room | null = null;
  private videoSource: VideoSource | null = null;
  private videoTrack: LocalVideoTrack | null = null;
  private publishedTrackSid: string | null = null;
  private connected = false;
  private frameCount = 0;

  constructor(options: PublisherOptions) {
    this.options = options;
  }

  async connect(): Promise<void> {
    const { config, trackName, width, height, fps } = this.options;

    // Generate access token
    const token = new AccessToken(config.api_key, config.api_secret, {
      identity: `publisher-${trackName}`,
    });
    token.addGrant({
      room: config.room,
      roomJoin: true,
      canPublish: true,
      canSubscribe: false,
    });

    const jwt = await token.toJwt();

    // Create room
    this.room = new Room();

    this.room.on(RoomEvent.Connected, () => {
      logger.info(`Connected to room: ${config.room}`);
      this.connected = true;
    });

    this.room.on(RoomEvent.Disconnected, (reason?: string) => {
      logger.warn(`Disconnected from room${reason ? `: ${reason}` : ''}`);
      this.connected = false;
      this.publishedTrackSid = null;
    });

    // Connect to room
    logger.info(`Connecting to LiveKit: ${config.url}`);
    await this.room.connect(config.url, jwt);

    // Create video source and track
    this.videoSource = new VideoSource(width, height);
    this.videoTrack = LocalVideoTrack.createVideoTrack(trackName, this.videoSource);

    // Publish track
    const publishOptions = new TrackPublishOptions({
      videoCodec: VideoCodec.H264,
      videoEncoding: new VideoEncoding({
        // protobuf expects uint64 => bigint
        maxBitrate: 2_000_000n,
        maxFramerate: fps,
      }),
    });

    const publication = await this.room.localParticipant!.publishTrack(this.videoTrack, publishOptions);
    this.publishedTrackSid = publication.sid;

    logger.info(`Published track: ${trackName} (sid=${publication.sid})`);
  }

  async disconnect(): Promise<void> {
    try {
      if (this.room?.localParticipant && this.publishedTrackSid) {
        await this.room.localParticipant.unpublishTrack(this.publishedTrackSid);
      }
    } catch (err) {
      logger.warn(`Unpublish error: ${err}`);
    }

    await this.room?.disconnect();
    this.connected = false;
    this.publishedTrackSid = null;
    logger.info('Disconnected from LiveKit');
  }

  publishFrame(frameData: Buffer): void {
    if (!this.connected || !this.videoSource) return;

    try {
      const { width, height } = this.options;

      // FFmpegReader outputs RGB24, so declare buffer type accordingly.
      const frame = new VideoFrame(new Uint8Array(frameData), width, height, VideoBufferType.RGB24);
      this.videoSource.captureFrame(frame);
      this.frameCount++;
    } catch (err) {
      logger.error(`Error publishing frame: ${err}`);
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  getStats(): { frameCount: number; connected: boolean } {
    return {
      frameCount: this.frameCount,
      connected: this.connected,
    };
  }
}
