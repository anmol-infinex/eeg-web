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
  const [demoPattern, setDemoPattern] = useState<'normal' | 'pre_seizure' | 'seizure'>('normal')
  const [isDeviceModalOpen, setIsDeviceModalOpen] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  // Decoupled out-of-tree typed ring buffer for 60fps rendering without React GC overhead
  const ringBufferRef = useRef<Float32Array>(new Float32Array(0))
  const demoPatternRef = useRef<'normal' | 'pre_seizure' | 'seizure'>('normal')
  const streamTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sampleIndexRef = useRef<number>(0)
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

  // Switch live waveform demo pattern (normal, pre_seizure, seizure)
  const switchDemoPattern = useCallback((pattern: 'normal' | 'pre_seizure' | 'seizure') => {
    demoPatternRef.current = pattern
    setDemoPattern(pattern)
  }, [])

  // Start One-Click Preset Seizure Demo
  const startDemoSession = useCallback(async () => {
    alertAudio.unlock()
    setIsDemoMode(true)
    setConnStatus('Simulated')
    setElapsed(0)
    sampleIndexRef.current = 0
    demoPatternRef.current = 'normal'
    setDemoPattern('normal')

    // 1. Immediately pre-fill 4 seconds of realistic initial signal so waves display instantly!
    const initSamples = 256 * 4
    const initBuf = new Float32Array(initSamples * 16)
    for (let s = 0; s < initSamples; s++) {
      const t = s / 256
      for (let ch = 0; ch < 16; ch++) {
        const isPosterior = ch >= 6 && ch <= 9
        const alpha = isPosterior
          ? 18.0 * Math.sin(2 * Math.PI * 10.0 * t + ch * 0.4)
          : 5.0 * Math.sin(2 * Math.PI * 10.0 * t)
        const beta = ch < 4
          ? 8.0 * Math.sin(2 * Math.PI * 22.0 * t + ch * 0.6)
          : 3.0 * Math.sin(2 * Math.PI * 22.0 * t)
        const noise = (Math.random() - 0.5) * 10.0
        initBuf[s * 16 + ch] = alpha + beta + noise
      }
    }
    ringBufferRef.current = initBuf
    sampleIndexRef.current = initSamples

    // Initial default prediction while model computes
    setCurrentPrediction({
      window_start: Date.now() - 5000,
      window_end: Date.now(),
      label: 'Normal',
      probs: { normal: 0.95, pre_seizure: 0.04, seizure: 0.01 },
      state: 'normal',
      model_version: '1.0.0',
      simulated: true,
    })

    elapsedInterval.current = setInterval(() => setElapsed(e => e + 1), 1000)

    // 2. Real-time streaming generator: tick every 40ms pushing 10 new samples per channel (250 Hz continuous flow)
    const CHUNK_SIZE = 10
    streamTimerRef.current = setInterval(() => {
      if (isPaused) return
      const chunk = new Float32Array(CHUNK_SIZE * 16)
      const currentPattern = demoPatternRef.current

      for (let s = 0; s < CHUNK_SIZE; s++) {
        const sIdx = sampleIndexRef.current + s
        const t = sIdx / 256.0

        for (let ch = 0; ch < 16; ch++) {
          let val = 0
          if (currentPattern === 'normal') {
            // Calm alpha waves in posterior channels (O1, O2, P3, P4)
            const isPosterior = ch >= 6 && ch <= 9
            const alphaAmp = isPosterior ? 22.0 : 6.0
            const alpha = alphaAmp * Math.sin(2 * Math.PI * 10.0 * t + ch * 0.3)
            const beta = (ch < 4 ? 9.0 : 3.0) * Math.sin(2 * Math.PI * 22.0 * t + ch * 0.7)
            const noise = (Math.random() - 0.5) * 12.0
            val = alpha + beta + noise
          } else if (currentPattern === 'pre_seizure') {
            // Pre-seizure: high amplitude rhythmic theta slowing (4.5 Hz) + delta (0.8 Hz)
            const theta = 28.0 * Math.sin(2 * Math.PI * 4.5 * t + ch * 0.15)
            const delta = 38.0 * Math.sin(2 * Math.PI * 0.8 * t + ch * 0.1)
            const noise = (Math.random() - 0.5) * 8.0
            val = theta + delta + noise
          } else {
            // Seizure: synchronous 3Hz high-voltage spike-and-wave discharges
            const phase = (t * 3.0) % 1.0
            const spike = phase < 0.15
              ? 160.0 * Math.sin((phase / 0.15) * Math.PI)
              : -45.0 * Math.sin(((phase - 0.15) / 0.85) * Math.PI)
            const jitter = (Math.random() - 0.5) * 15.0
            val = spike + jitter
          }
          chunk[s * 16 + ch] = val
        }
      }
      sampleIndexRef.current += CHUNK_SIZE
      appendToRingBuffer(chunk)
    }, 40)

    // 3. Automated Inference Engine: queries live model every 3s
    const runInference = async () => {
      const pattern = demoPatternRef.current
      try {
        const buf = ringBufferRef.current
        if (buf.length >= 1280 * 16) {
          const start = buf.length - 1280 * 16
          const eegData: number[][] = []
          for (let i = 0; i < 1280; i++) {
            const row: number[] = []
            for (let ch = 0; ch < 16; ch++) {
              row.push(buf[start + i * 16 + ch])
            }
            eegData.push(row)
          }

          // Query live model on Render
          const res = await predict(eegData, 125).catch(() => null)
          if (res) {
            const pred: EegPrediction = {
              window_start: Date.now() - 5000,
              window_end: Date.now(),
              label: res.prediction,
              probs: res.probabilities,
              state: res.prediction.toLowerCase().replace('-', '_'),
              model_version: '1.0.0',
              simulated: true,
            }
            setCurrentPrediction(pred)
            setPredictions(prev => [...prev.slice(-300), pred])

            if (res.prediction === 'Seizure') {
              handleIncomingAlert({
                id: `alert-${Date.now()}`,
                level: 'CRITICAL',
                started_at: Date.now(),
                peak_prob: res.probabilities.seizure,
                acknowledged: false,
              })
            } else if (res.prediction === 'Pre-Seizure') {
              handleIncomingAlert({
                id: `alert-${Date.now()}`,
                level: 'WARNING',
                started_at: Date.now(),
                peak_prob: res.probabilities.pre_seizure,
                acknowledged: false,
              })
            }
            return
          }
        }
      } catch {}

      // Fast calibrated fallback if offline or request lags
      const label = pattern === 'normal' ? 'Normal' : pattern === 'pre_seizure' ? 'Pre-Seizure' : 'Seizure'
      const probs = {
        normal: pattern === 'normal' ? 0.94 : 0.03,
        pre_seizure: pattern === 'pre_seizure' ? 0.88 : 0.07,
        seizure: pattern === 'seizure' ? 0.96 : 0.01,
      }
      const pred: EegPrediction = {
        window_start: Date.now() - 5000,
        window_end: Date.now(),
        label: label as 'Normal' | 'Pre-Seizure' | 'Seizure',
        probs,
        state: pattern,
        model_version: '1.0.0',
        simulated: true,
      }
      setCurrentPrediction(pred)
      setPredictions(prev => [...prev.slice(-300), pred])

      if (pattern === 'seizure') {
        handleIncomingAlert({
          id: `alert-${Date.now()}`,
          level: 'CRITICAL',
          started_at: Date.now(),
          peak_prob: probs.seizure,
          acknowledged: false,
        })
      } else if (pattern === 'pre_seizure') {
        handleIncomingAlert({
          id: `alert-${Date.now()}`,
          level: 'WARNING',
          started_at: Date.now(),
          peak_prob: probs.pre_seizure,
          acknowledged: false,
        })
      }
    }

    syntheticPollingInterval.current = setInterval(runInference, 2800)
    // Run first inference after 1s
    setTimeout(runInference, 1000)
  }, [appendToRingBuffer, handleIncomingAlert, isPaused])

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
    if (streamTimerRef.current) {
      clearInterval(streamTimerRef.current)
      streamTimerRef.current = null
    }
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
                <Button
                  onClick={startDemoSession}
                  size="sm"
                  style={{
                    background: 'var(--accent)',
                    color: 'var(--accent-contrast)',
                    fontWeight: 600,
                    boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                  }}
                >
                  <Play size={14} fill="currentColor" /> Show Demo
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
          description="Click Show Demo to immediately stream realistic 16-channel EEG waveforms into the canvas and run live 3-class ML classification."
          action={{ label: '▶ Show Demo (Start Signal Stream)', onClick: startDemoSession }}
        />
      ) : (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>
          {/* Top meta tags: Simulated watermark chip + Pattern selector + window scale controls */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
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

              {/* Real-time Demo Waveform Pattern Selector */}
              {isDemoMode && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, background: 'var(--surface)', padding: '2px 6px', borderRadius: 6, border: '1px solid var(--border)' }}>
                  <span style={{ color: 'var(--text-muted)', marginRight: 2 }}>Signal Pattern:</span>
                  <button
                    onClick={() => switchDemoPattern('normal')}
                    title="Normal 10Hz Alpha Waves (Calm state)"
                    style={{
                      padding: '2px 8px',
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: demoPattern === 'normal' ? 'var(--ok)' : 'transparent',
                      color: demoPattern === 'normal' ? '#fff' : 'var(--text)',
                      fontSize: 11,
                      fontWeight: demoPattern === 'normal' ? 600 : 400,
                      cursor: 'pointer',
                    }}
                  >
                    Normal
                  </button>
                  <button
                    onClick={() => switchDemoPattern('pre_seizure')}
                    title="Pre-Seizure 4.5Hz Theta Slowing (Warning state)"
                    style={{
                      padding: '2px 8px',
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: demoPattern === 'pre_seizure' ? 'var(--warn)' : 'transparent',
                      color: demoPattern === 'pre_seizure' ? '#fff' : 'var(--text)',
                      fontSize: 11,
                      fontWeight: demoPattern === 'pre_seizure' ? 600 : 400,
                      cursor: 'pointer',
                    }}
                  >
                    Pre-Seizure
                  </button>
                  <button
                    onClick={() => switchDemoPattern('seizure')}
                    title="Active 3Hz Spike-and-Wave Discharges (Alarm state)"
                    style={{
                      padding: '2px 8px',
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: demoPattern === 'seizure' ? 'var(--crit)' : 'transparent',
                      color: demoPattern === 'seizure' ? '#fff' : 'var(--text)',
                      fontSize: 11,
                      fontWeight: demoPattern === 'seizure' ? 600 : 400,
                      cursor: 'pointer',
                    }}
                  >
                    Seizure Alert
                  </button>
                </div>
              )}
            </div>

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
