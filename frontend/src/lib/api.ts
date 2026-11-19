/**
 * REST API client for Seizure-Sentry backend.
 */

import type { PipelineDiagnostics, SessionRecord } from './types'

const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

let _accessKey = sessionStorage.getItem('sentry_access_key') ?? ''

export function getAccessKey() { return _accessKey }
export function setAccessKey(key: string) {
  _accessKey = key
  if (key) sessionStorage.setItem('sentry_access_key', key)
  else sessionStorage.removeItem('sentry_access_key')
}

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (_accessKey) headers['X-Access-Key'] = _accessKey
  return headers
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_name: string
  version: string
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

export interface PredictResponse {
  prediction: 'Normal' | 'Pre-Seizure' | 'Seizure'
  probabilities: { normal: number; pre_seizure: number; seizure: number }
}

export async function predict(
  eeg_data: number[][],
  fs = 256,
  channels?: string[]
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ eeg_data, fs, channels })
  })
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`)
  return res.json()
}

export interface SyntheticWindowResponse {
  state: string
  fs: number
  n_samples: number
  n_channels: number
  data: number[][]
}

export async function getSyntheticWindow(
  state: 'normal' | 'pre_seizure' | 'seizure' = 'normal'
): Promise<SyntheticWindowResponse> {
  const res = await fetch(`${BASE}/synthetic/window?state=${state}`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Synthetic window failed: ${res.status}`)
  return res.json()
}

export interface DemoStartResponse {
  session_id: string
  status: string
  preset: string
  synthetic: boolean
  simulated: boolean
  badge: string
}

export async function startDemo(sessionId?: string): Promise<DemoStartResponse> {
  const res = await fetch(`${BASE}/demo/start`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ session_id: sessionId })
  })
  if (!res.ok) throw new Error(`Start demo failed: ${res.status}`)
  return res.json()
}

export interface ReplayRecording {
  id: string
  name: string
  duration_sec: number
  fs: number
  n_channels: number
}

export async function getReplayRecordings(): Promise<ReplayRecording[]> {
  const res = await fetch(`${BASE}/replay/recordings`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Get recordings failed: ${res.status}`)
  return res.json()
}

export async function startReplay(
  recording_id: string,
  speed = 1.0,
  session_id?: string
): Promise<{ status: string; session_id: string; recording_id: string; speed: number }> {
  const res = await fetch(`${BASE}/replay/start`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ recording_id, speed, session_id })
  })
  if (!res.ok) throw new Error(`Start replay failed: ${res.status}`)
  return res.json()
}

export async function stopReplay(session_id?: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/replay/stop`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ session_id })
  })
  if (!res.ok) throw new Error(`Stop replay failed: ${res.status}`)
  return res.json()
}

export async function listSessions(limit = 50, offset = 0): Promise<SessionRecord[]> {
  const res = await fetch(`${BASE}/sessions?limit=${limit}&offset=${offset}`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`List sessions failed: ${res.status}`)
  return res.json()
}

export async function getSessionDetails(sessionId: string): Promise<SessionRecord> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Get session failed: ${res.status}`)
  return res.json()
}

export async function stopSession(sessionId: string): Promise<SessionRecord> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/stop`, {
    method: 'POST',
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Stop session failed: ${res.status}`)
  return res.json()
}

export async function getPipelineDiagnostics(sessionId: string): Promise<PipelineDiagnostics> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/pipeline`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Get diagnostics failed: ${res.status}`)
  return res.json()
}

export async function createShareToken(
  sessionId: string,
  expires_in_sec = 3600
): Promise<{ session_id: string; share_token: string; viewer_url: string; expires_in: number }> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/share`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ expires_in_sec })
  })
  if (!res.ok) throw new Error(`Create share token failed: ${res.status}`)
  return res.json()
}

export async function revokeShareToken(sessionId: string, token?: string): Promise<{ status: string }> {
  const query = token ? `?token=${encodeURIComponent(token)}` : ''
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/share${query}`, {
    method: 'DELETE',
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Revoke share token failed: ${res.status}`)
  return res.json()
}

export async function getClinicalReport(sessionId: string): Promise<string> {
  const res = await fetch(`${BASE}/sessions/${encodeURIComponent(sessionId)}/report`, {
    headers: authHeaders()
  })
  if (!res.ok) throw new Error(`Get clinical report failed: ${res.status}`)
  return res.text()
}

export function getReportEegPlotUrl(sessionId: string): string {
  const keyParam = _accessKey ? `?key=${encodeURIComponent(_accessKey)}` : ''
  return `${BASE}/sessions/${encodeURIComponent(sessionId)}/report/eeg.png${keyParam}`
}
