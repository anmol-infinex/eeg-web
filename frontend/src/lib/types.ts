/**
 * Domain types and interface definitions for Seizure-Sentry.
 */

export type ConnectionStatus = 'Disconnected' | 'Connecting' | 'Streaming' | 'Simulated'

export type ChannelQuality = 'good' | 'noisy' | 'flat' | 'railed'

export interface SampleChunk {
  seq: number
  t0_ms: number
  samples: Float32Array // flat sample-major buffer [sample * n_channels + ch]
  channels: string[]
  fs: number
  flags: number
}

export interface EegSource {
  readonly id: string
  readonly profileId: string
  isSupported(): boolean
  connect(): Promise<void>
  disconnect(): Promise<void>
  onChunk(cb: (chunk: SampleChunk) => void): void
  onStatus(cb: (status: ConnectionStatus) => void): void
}

export interface EegPrediction {
  session_id?: string
  window_start: number
  window_end: number
  probs: {
    normal: number
    pre_seizure: number
    seizure: number
  }
  label: 'Normal' | 'Pre-Seizure' | 'Seizure'
  state: string
  model_version?: string
  simulated?: boolean
}

// Backwards-compatible alias
export type Prediction = EegPrediction

export interface EegAlert {
  id: string
  session_id?: string
  level: 'WARNING' | 'CRITICAL' | 'warning' | 'critical'
  started_at: string | number
  ended_at?: string | number | null
  peak_prob: number
  muted?: boolean
  acknowledged?: boolean
  ui_banner?: boolean
  red_border_pulse?: boolean
  sound_alarm?: boolean
  simulated?: boolean
}

// Backwards-compatible alias
export type Alert = EegAlert

export interface WsLinkStatus {
  type?: string
  session_id?: string
  link_state?: string
  channel_quality?: ChannelQuality[]
  risk_level?: string
  dropped_frames?: number
  artifact_guard_active?: boolean
  predictions_paused?: boolean
  banner_message?: string
  quality?: ChannelQuality[]
  timestamp?: number
}

export type WsStatus = WsLinkStatus

export interface DeviceProfile {
  id: string
  name: string
  fs: number
  n_channels: number
  channels: string[]
  status: string
  connection: string
  compatibility: 'compatible' | 'adapted' | 'incompatible'
  compatibilityReason: string
}

export interface PipelineDiagnostics {
  session_id?: string
  hop1_device_packets: number
  hop2_chunks_sent: number
  hop3_frames_received: number
  hop4_windows_inferred: number
  hop5_predictions_displayed: number
  dropped_chunks: number
  health: 'ok' | 'degraded'
  last_latency_ms: number
  timestamp?: number
}

export interface SessionRecord {
  session_id: string
  created_at: string
  status: 'active' | 'completed' | 'stopped'
  source: string
  n_samples?: number
  duration_sec?: number
  alert_count?: number
}
