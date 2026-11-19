/**
 * LiveMonitor page: real-time EEG waveform monitoring, 3-class ML prediction state,
 * multi-modal alerting (visual banner, screen-edge pulse, synthesized audio),
 * Artifact Guard safety suppression, 5-hop pipeline diagnostics, and Web Serial linking.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { Square, Wifi, WifiOff, Cpu, Volume2, VolumeX, Pause, Play } from 'lucide-react'
import EegCanvas from '../components/EegCanvas'
import StateCard from '../components/StateCard'
import AlertBanner from '../components/AlertBanner'
import TimelineStrip from '../components/TimelineStrip'
import DiagnosticsPanel from '../components/DiagnosticsPanel'
import DeviceModal from '../components/DeviceModal'
import Button from '../components/Button'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import { EegWsClient } from '../lib/wsClient'
import { WebSerialSource } from '../lib/serial'
import { alertAudio } from '../lib/audio'
import {
  getAccessKey,
  startDemo,
  getSyntheticWindow,
  getPipelineDiagnostics,
} from '../lib/api'
import type {
  ConnectionStatus,
  ChannelQuality,
  EegPrediction,
  EegAlert,
  PipelineDiagnostics,
  WsLinkStatus,
} from '../lib/types'

const CHANNEL_LABELS = [
  'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
  'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
]

const QUALITY_COLOR_MAP: Record<ChannelQuality, string> = {
  good: 'var(--ok)',
  noisy: 'var(--warn)',
  flat: 'var(--crit)',
  railed: 'var(--crit)',
}

export default function LiveMonitor() {
  const [connStatus, setConnStatus] = useState<ConnectionStatus>('Disconnected')
  const [qualities, setQualities] = useState<ChannelQuality[]>(Array(16).fill('good'))
  const [currentPrediction, setCurrentPrediction] = useState<EegPrediction | null>(null)
  const [predictions, setPredictions] = useState<EegPrediction[]>([])
  const [activeAlert, setActiveAlert] = useState<EegAlert | null>(null)
  const [artifactGuardActive, setArtifactGuardActive] = useState(false)
  const [diagnostics, setDiagnostics] = useState<PipelineDiagnostics | null>(null)

  const [highlightCh, setHighlightCh] = useState<number | null>(null)
  const [windowSec, setWindowSec] = useState<number>(10)
  const [amplitudeScale, setAmplitudeScale] = useState<number>(50)
  const [isPaused, setIsPaused] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [isDeviceModalOpen, setIsDeviceModalOpen] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  // Decoupled out-of-tree typed ring buffer for 60fps rendering without React GC overhead
  const ringBufferRef = useRef<Float32Array>(new Float32Array(0))
  const wsRef = useRef<EegWsClient | null>(null)
  const serialRef = useRef<WebSerialSource | null>(null)
  const activeSessionIdRef = useRef<string>('')
  const elapsedInterval = useRef<ReturnType<typeof setInterval> | null>(null)
  const diagInterval = useRef<ReturnType<typeof setInterval> | null>(null)
  const syntheticPollingInterval = useRef<ReturnType<typeof setInterval> | null>(null)

  const appendToRingBuffer = useCallback((samples: Float32Array) => {
    if (isPaused) return
    const maxSamples = 256 * 16 * 60 // 60 seconds buffer capacity
    const old = ringBufferRef.current
    const combined = new Float32Array(old.length + samples.length)
    combined.set(old)
    combined.set(samples, old.length)
    ringBufferRef.current = combined.slice(Math.max(0, combined.length - maxSamples))
  }, [isPaused])

  const formatTime = (s: number) => {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  // Handle incoming alerts with audio synthesis & muting
  const handleIncomingAlert = useCallback((alert: EegAlert) => {
    setActiveAlert(alert)
    if (artifactGuardActive || isMuted) {
      alertAudio.stopCritical()
      return
    }
    const isCrit = String(alert.level).toUpperCase() === 'CRITICAL'
    if (isCrit) {
      alertAudio.startCritical()
    } else {
      alertAudio.playWarning()
    }
  }, [artifactGuardActive, isMuted])

  // Acknowledge or Mute alert
  const handleAcknowledgeAlert = useCallback((_alertId?: string) => {
    alertAudio.stopCritical()
    setActiveAlert(prev => prev ? { ...prev, acknowledged: true, muted: true } : null)
  }, [])

  const handleToggleMute = useCallback(() => {
    setIsMuted(m => {
      const next = !m
      alertAudio.setMuted(next)
      return next
    })
  }, [])

  // Start One-Click Preset Seizure Demo
  const startDemoSession = useCallback(async () => {
    alertAudio.unlock()
    setIsDemoMode(true)
    setConnStatus('Simulated')
    setElapsed(0)
    ringBufferRef.current = new Float32Array(0)

    elapsedInterval.current = setInterval(() => setElapsed(e => e + 1), 1000)

    try {
      const demoRes = await startDemo().catch(() => null)
      const sessionId = demoRes?.session_id ?? `demo-${Date.now()}`
      activeSessionIdRef.current = sessionId

      const ws = new EegWsClient(sessionId, 'viewer', 'synthetic')
      wsRef.current = ws

      ws.onStatusChange = (status) => setConnStatus(status === 'Streaming' ? 'Simulated' : status)
      ws.onPrediction = (pred) => {
        setCurrentPrediction(pred)
        setPredictions(prev => [...prev.slice(-300), pred])
      }
      ws.onAlert = (alert) => handleIncomingAlert(alert)
      ws.onStatusUpdate = (status: WsLinkStatus) => {
        if (status.artifact_guard_active !== undefined) {
          setArtifactGuardActive(status.artifact_guard_active)
          if (status.artifact_guard_active) alertAudio.stopCritical()
        }
        if (status.quality && Array.isArray(status.quality)) {
          setQualities(status.quality as ChannelQuality[])
        }
      }
      ws.onEegChunk = (chunk) => appendToRingBuffer(chunk)

      ws.connect(getAccessKey())

      // Periodically fetch diagnostics
      diagInterval.current = setInterval(async () => {
        try {
          const diag = await getPipelineDiagnostics(sessionId)
          setDiagnostics(diag)
        } catch {}
      }, 2000)

      // Fallback synthetic generator in case WebSocket is offline in local preview
      const states: Array<'normal' | 'pre_seizure' | 'seizure'> = [
        'normal', 'normal', 'normal', 'pre_seizure', 'seizure', 'seizure', 'pre_seizure', 'normal'
      ]
      let idx = 0
      syntheticPollingInterval.current = setInterval(async () => {
        const state = states[idx % states.length]
        idx++
        try {
          const win = await getSyntheticWindow(state)
          const flat = new Float32Array(win.data.flat())
          appendToRingBuffer(flat)
          const label = state === 'normal' ? 'Normal' : state === 'pre_seizure' ? 'Pre-Seizure' : 'Seizure'
          const probs = {
            normal: state === 'normal' ? 0.88 : 0.08,
            pre_seizure: state === 'pre_seizure' ? 0.78 : 0.12,
            seizure: state === 'seizure' ? 0.84 : 0.04,
          }
          const pred: EegPrediction = {
            window_start: Date.now(),
            window_end: Date.now() + 5000,
            label: label as 'Normal' | 'Pre-Seizure' | 'Seizure',
            probs,
            state,
            model_version: '1.0.0',
            simulated: true,
          }
          setCurrentPrediction(pred)
          setPredictions(prev => [...prev.slice(-300), pred])
        } catch {}
      }, 5000)
    } catch {
      // Fallback handled cleanly
    }
  }, [appendToRingBuffer, handleIncomingAlert])

  // Start Hardware Web Serial Session
  const handleDeviceLinked = useCallback((source: WebSerialSource) => {
    alertAudio.unlock()
    serialRef.current = source
    setIsDemoMode(false)
    setConnStatus('Streaming')
    setElapsed(0)
    ringBufferRef.current = new Float32Array(0)

    elapsedInterval.current = setInterval(() => setElapsed(e => e + 1), 1000)

    const sessionId = `serial-${Date.now()}`
    activeSessionIdRef.current = sessionId

    // Connect producer WebSocket to stream hardware frames to ML backend
    const ws = new EegWsClient(sessionId, 'producer', 'cyton-serial')
    wsRef.current = ws

    ws.onStatusChange = (status) => setConnStatus(status)
    ws.onPrediction = (pred) => {
      setCurrentPrediction(pred)
      setPredictions(prev => [...prev.slice(-300), pred])
    }
    ws.onAlert = (alert) => handleIncomingAlert(alert)
    ws.onStatusUpdate = (status: WsLinkStatus) => {
      if (status.artifact_guard_active !== undefined) {
        setArtifactGuardActive(status.artifact_guard_active)
        if (status.artifact_guard_active) alertAudio.stopCritical()
      }
      if (status.quality && Array.isArray(status.quality)) {
        setQualities(status.quality as ChannelQuality[])
      }
    }

    ws.connect(getAccessKey())

    // Stream serial chunks into ring buffer and forward to WebSocket
    source.onChunk((chunk) => {
      appendToRingBuffer(chunk.samples)
      ws.sendBinaryChunk(chunk.seq, chunk.t0_ms, chunk.samples, 16, chunk.flags)
    })

    source.onStatus((status) => {
      if (status === 'Disconnected') {
        setConnStatus('Disconnected')
      }
    })

    diagInterval.current = setInterval(async () => {
      try {
        const diag = await getPipelineDiagnostics(sessionId)
        setDiagnostics(diag)
      } catch {}
    }, 2000)
  }, [appendToRingBuffer, handleIncomingAlert])

  // Stop active session cleanly
  const stopSession = useCallback(() => {
    if (elapsedInterval.current) {
      clearInterval(elapsedInterval.current)
      elapsedInterval.current = null
    }
    if (diagInterval.current) {
      clearInterval(diagInterval.current)
      diagInterval.current = null
    }
    if (syntheticPollingInterval.current) {
      clearInterval(syntheticPollingInterval.current)
      syntheticPollingInterval.current = null
    }

    alertAudio.stopAll()

    wsRef.current?.disconnect()
    wsRef.current = null

    serialRef.current?.disconnect().catch(() => {})
    serialRef.current = null

    setConnStatus('Disconnected')
    setIsDemoMode(false)
    setActiveAlert(null)
    setArtifactGuardActive(false)
    setElapsed(0)
  }, [])

  useEffect(() => () => stopSession(), [stopSession])

  // Universal Keyboard Shortcuts (Space, M, A, 1, 2, 3)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return

      if (e.code === 'Space') {
        e.preventDefault()
        setIsPaused(p => !p)
      } else if (e.key === 'm' || e.key === 'M') {
        handleToggleMute()
      } else if (e.key === 'a' || e.key === 'A') {
        handleAcknowledgeAlert()
      } else if (e.key === '1') {
        setWindowSec(5)
      } else if (e.key === '2') {
        setWindowSec(10)
      } else if (e.key === '3') {
        setWindowSec(20)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleAcknowledgeAlert, handleToggleMute])

  const isCriticalAlertActive =
    activeAlert &&
    String(activeAlert.level).toUpperCase() === 'CRITICAL' &&
    !activeAlert.acknowledged &&
    !artifactGuardActive

  return (
    <div
      className={isCriticalAlertActive ? 'border-pulse-critical' : ''}
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: 'calc(100vh - 48px - 36px - 48px)',
        gap: 0,
        position: 'relative',
        borderRadius: 6,
        padding: isCriticalAlertActive ? 4 : 0,
        transition: 'padding 150ms ease',
      }}
    >
      <PageHeader
        title="Live monitor"
        actions={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              onClick={handleToggleMute}
              title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '6px 10px',
                cursor: 'pointer',
                color: isMuted ? 'var(--warn)' : 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                fontSize: 12,
              }}
            >
              {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
              {isMuted ? 'Muted' : 'Sound'}
            </button>

            {connStatus !== 'Disconnected' && (
              <button
                onClick={() => setIsPaused(p => !p)}
                title={isPaused ? 'Resume Waveform' : 'Pause Waveform'}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '6px 10px',
                  cursor: 'pointer',
                  color: isPaused ? 'var(--warn)' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 12,
                }}
              >
                {isPaused ? <Play size={14} /> : <Pause size={14} />}
                {isPaused ? 'Resume' : 'Pause'}
              </button>
            )}

            {connStatus === 'Disconnected' ? (
              <>
                <Button variant="secondary" onClick={() => setIsDeviceModalOpen(true)} size="sm">
                  <Cpu size={14} /> Link device
                </Button>
                <Button onClick={startDemoSession} size="sm">
                  <Wifi size={14} /> Start demo (simulated)
                </Button>
              </>
            ) : (
              <Button variant="destructive" onClick={stopSession} size="sm">
                <Square size={14} /> Stop session
              </Button>
            )}
          </div>
        }
      />

      {/* Alert Banner / Artifact Guard Banner */}
      <AlertBanner
        alert={activeAlert}
        artifactGuardActive={artifactGuardActive}
        onAcknowledge={handleAcknowledgeAlert}
        onMute={handleToggleMute}
        onDismiss={() => setActiveAlert(null)}
      />

      {connStatus === 'Disconnected' ? (
        <EmptyState
          icon={<WifiOff size={40} />}
          title="No active session"
          description="Start a demo session to stream synthetic EEG through the ML model, or link your OpenBCI device via Web Serial."
          action={{ label: 'Start demo session', onClick: startDemoSession }}
        />
      ) : (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>
          {/* Top meta tags: Simulated watermark chip + window scale controls */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
            {isDemoMode && (
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '2px 8px',
                  borderRadius: 4,
                  background: 'color-mix(in srgb, var(--sim) 15%, transparent)',
                  color: 'var(--sim)',
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                }}
              >
                SIMULATED DATA
              </div>
            )}

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)' }}>Window:</span>
              {[5, 10, 20].map(sec => (
                <button
                  key={sec}
                  onClick={() => setWindowSec(sec)}
                  style={{
                    padding: '2px 6px',
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: windowSec === sec ? 'var(--surface-2)' : 'transparent',
                    color: windowSec === sec ? 'var(--text)' : 'var(--text-muted)',
                    cursor: 'pointer',
                    fontSize: 11,
                    fontFamily: 'IBM Plex Mono',
                  }}
                >
                  {sec}s
                </button>
              ))}
              <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>Scale:</span>
              {[25, 50, 100].map(scale => (
                <button
                  key={scale}
                  onClick={() => setAmplitudeScale(scale)}
                  style={{
                    padding: '2px 6px',
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: amplitudeScale === scale ? 'var(--surface-2)' : 'transparent',
                    color: amplitudeScale === scale ? 'var(--text)' : 'var(--text-muted)',
                    cursor: 'pointer',
                    fontSize: 11,
                    fontFamily: 'IBM Plex Mono',
                  }}
                >
                  {scale}µV
                </button>
              ))}
            </div>
          </div>

          {/* Main monitoring grid: Channel List | EegCanvas | StateCard */}
          <div style={{ flex: 1, display: 'flex', gap: 12, minHeight: 0 }}>
            {/* Channel List */}
            <div
              style={{
                width: 105,
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: 8,
                overflowY: 'auto',
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  marginBottom: 6,
                  padding: '0 4px',
                }}
              >
                Channels
              </div>
              {CHANNEL_LABELS.map((label, i) => (
                <div
                  key={label}
                  onClick={() => setHighlightCh(h => (h === i ? null : i))}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '3px 4px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    background:
                      highlightCh === i
                        ? 'color-mix(in srgb, var(--accent) 12%, transparent)'
                        : 'transparent',
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: QUALITY_COLOR_MAP[qualities[i] ?? 'good'],
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 11,
                      fontFamily: 'IBM Plex Mono',
                      color: highlightCh === i ? 'var(--accent)' : 'var(--text)',
                      fontWeight: highlightCh === i ? 600 : 400,
                    }}
                  >
                    {label}
                  </span>
                </div>
              ))}
            </div>

            {/* EEG Canvas Waveform Renderer */}
            <div
              style={{
                flex: 1,
                background: 'var(--eeg-bg)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                minHeight: 0,
                minWidth: 0,
                overflow: 'hidden',
              }}
            >
              <EegCanvas
                ringBufferRef={ringBufferRef}
                channelLabels={CHANNEL_LABELS}
                channelQualities={qualities}
                windowSeconds={windowSec}
                amplitudeUvDiv={amplitudeScale}
                highlightChannel={highlightCh}
                onChannelClick={(ch) => setHighlightCh(h => (h === ch ? null : ch))}
                isSimulated={isDemoMode}
                fs={256}
              />
            </div>

            {/* Clinical State & Probability Card */}
            <StateCard currentPrediction={currentPrediction} isWarmup={elapsed < 5} />
          </div>

          {/* Historical Prediction Timeline Strip */}
          <TimelineStrip predictions={predictions} />

          {/* 5-Hop End-to-End Diagnostics Panel */}
          {diagnostics && <DiagnosticsPanel diagnostics={diagnostics} />}

          {/* Status footer bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 24,
              padding: '6px 12px',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              fontSize: 12,
              fontFamily: 'IBM Plex Mono',
              color: 'var(--text-muted)',
              flexShrink: 0,
            }}
          >
            <span>256 Hz Model Rate</span>
            <span>Dropped: {diagnostics?.dropped_chunks ?? 0}</span>
            <span>Latency: {diagnostics?.last_latency_ms?.toFixed(1) ?? '0.0'} ms</span>
            <span style={{ marginLeft: 'auto' }}>Session: {formatTime(elapsed)}</span>
          </div>
        </div>
      )}

      {/* 4-Step Web Serial Connection Wizard Modal */}
      <DeviceModal
        isOpen={isDeviceModalOpen}
        onClose={() => setIsDeviceModalOpen(false)}
        onDeviceLinked={handleDeviceLinked}
      />
    </div>
  )
}
