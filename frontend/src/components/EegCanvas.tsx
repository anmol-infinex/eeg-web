/**
 * High-performance HTML5 Canvas 2D multi-channel EEG waveform renderer (60 fps).
 * Features: High-DPI Retina scaling, 1.0s timebase grid, 50 uV/div scaling,
 * extreme voltage spike clamping to slot bounds, safe division guard,
 * channel labels & quality indicators, and semi-transparent simulated watermark.
 */

import { useRef, useEffect, useCallback } from 'react'
import type { ChannelQuality } from '../lib/types'

interface EegCanvasProps {
  chunks?: Float32Array[]
  ringBufferRef?: React.MutableRefObject<Float32Array>
  channelLabels: string[]
  channelQualities?: ChannelQuality[]
  windowSeconds?: number
  amplitudeUvDiv?: number
  highlightChannel?: number | null
  onChannelClick?: (ch: number) => void
  isSimulated?: boolean
  fs?: number
}

const QUALITY_COLOR_MAP: Record<ChannelQuality, string> = {
  good: '#1B7F4D',
  noisy: '#B26A00',
  flat: '#B42318',
  railed: '#B42318',
}

export default function EegCanvas({
  chunks,
  ringBufferRef: externalRingBufferRef,
  channelLabels,
  channelQualities,
  windowSeconds = 10,
  amplitudeUvDiv = 50,
  highlightChannel = null,
  onChannelClick,
  isSimulated = false,
  fs = 256,
}: EegCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)
  const localRingBufferRef = useRef<Float32Array>(new Float32Array(0))
  const activeRingBuffer = externalRingBufferRef || localRingBufferRef

  const nChannels = channelLabels.length || 16
  const samplesPerWindow = windowSeconds * fs

  // Cached theme colors to avoid getComputedStyle layout thrashing in rAF
  const colorsRef = useRef({
    bg: '#FFFFFF',
    trace: '#1B2A41',
    grid: '#E4E7EC',
    text: '#5B6573',
  })

  const updateColors = useCallback(() => {
    if (typeof document === 'undefined') return
    const s = getComputedStyle(document.documentElement)
    colorsRef.current = {
      bg: s.getPropertyValue('--eeg-bg').trim() || '#FFFFFF',
      trace: s.getPropertyValue('--eeg-trace').trim() || '#1B2A41',
      grid: s.getPropertyValue('--eeg-grid').trim() || '#E4E7EC',
      text: s.getPropertyValue('--text-muted').trim() || '#5B6573',
    }
  }, [])

  useEffect(() => {
    updateColors()
    const observer = new MutationObserver(updateColors)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] })
    return () => observer.disconnect()
  }, [updateColors])

  // Ingest incoming props chunks into local ring buffer if external buffer is not used
  useEffect(() => {
    if (externalRingBufferRef || !chunks || !chunks.length) return
    const lastChunk = chunks[chunks.length - 1]
    const maxBuf = samplesPerWindow * nChannels * 2
    const old = localRingBufferRef.current
    const combined = new Float32Array(old.length + lastChunk.length)
    combined.set(old)
    combined.set(lastChunk, old.length)
    localRingBufferRef.current = combined.slice(Math.max(0, combined.length - maxBuf))
  }, [chunks, samplesPerWindow, nChannels, externalRingBufferRef])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    const cssWidth = canvas.clientWidth
    const cssHeight = canvas.clientHeight

    // Keep backing store in sync with High-DPI display
    if (canvas.width !== Math.floor(cssWidth * dpr) || canvas.height !== Math.floor(cssHeight * dpr)) {
      canvas.width = Math.floor(cssWidth * dpr)
      canvas.height = Math.floor(cssHeight * dpr)
    }

    ctx.save()
    ctx.scale(dpr, dpr)

    const colors = colorsRef.current
    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, cssWidth, cssHeight)

    // Render semi-transparent SIMULATED DATA watermark if simulated
    if (isSimulated) {
      ctx.save()
      ctx.font = 'bold 36px "IBM Plex Sans", sans-serif'
      ctx.fillStyle = 'rgba(91, 75, 138, 0.12)'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('SIMULATED DATA', cssWidth / 2, cssHeight / 2)
      ctx.restore()
    }

    if (nChannels === 0) {
      ctx.restore()
      animFrameRef.current = requestAnimationFrame(draw)
      return
    }

    const slotHeight = cssHeight / nChannels
    const safeUvDiv = Math.max(1.0, amplitudeUvDiv || 50)
    const buf = activeRingBuffer.current
    const totalSamples = buf.length > 0 ? Math.floor(buf.length / nChannels) : 0
    const pxPerSample = cssWidth / samplesPerWindow

    // 1. Horizontal division lines between channels
    ctx.strokeStyle = colors.grid
    ctx.lineWidth = 0.5
    for (let ch = 0; ch <= nChannels; ch++) {
      const y = ch * slotHeight
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(cssWidth, y)
      ctx.stroke()
    }

    // 2. Vertical timebase grid lines every 1.0 second
    const samplesPerSec = fs
    for (let s = 0; s <= windowSeconds; s++) {
      const x = (totalSamples - (windowSeconds - s) * samplesPerSec) * pxPerSample
      if (x >= 0 && x <= cssWidth) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, cssHeight)
        ctx.stroke()
      }
    }

    // 3. Render multi-channel waveform traces
    if (totalSamples > 1) {
      const startSample = Math.max(0, totalSamples - samplesPerWindow)
      const renderCount = totalSamples - startSample

      for (let ch = 0; ch < nChannels; ch++) {
        const centerY = (ch + 0.5) * slotHeight
        const minY = ch * slotHeight
        const maxY = (ch + 1) * slotHeight
        const isHighlighted = highlightChannel === ch

        ctx.strokeStyle = colors.trace
        ctx.lineWidth = isHighlighted ? 1.6 : 1.0
        ctx.globalAlpha = isHighlighted ? 1.0 : 0.85

        ctx.beginPath()
        let started = false

        for (let i = 0; i < renderCount; i++) {
          const sampleIdx = startSample + i
          const valUv = buf[sampleIdx * nChannels + ch]
          const x = i * pxPerSample

          // Calculate Y with safe division and clamp to channel slot bounds
          const rawY = centerY - (valUv / safeUvDiv) * (slotHeight * 0.4)
          // Clamp extreme voltage spikes (>500,000 uV) to slot bounds
          const clampedY = Math.max(minY, Math.min(maxY, rawY))

          if (!started) {
            ctx.moveTo(x, clampedY)
            started = true
          } else {
            ctx.lineTo(x, clampedY)
          }
        }
        ctx.stroke()
        ctx.globalAlpha = 1.0
      }
    }

    // 4. Render channel labels and quality indicators along left edge
    for (let ch = 0; ch < nChannels; ch++) {
      const centerY = (ch + 0.5) * slotHeight
      const label = channelLabels[ch] ?? `Ch${ch + 1}`
      const quality = channelQualities?.[ch] ?? 'good'
      const qualityColor = QUALITY_COLOR_MAP[quality] ?? '#1B7F4D'

      // Quality dot
      ctx.fillStyle = qualityColor
      ctx.beginPath()
      ctx.arc(8, centerY, 3.5, 0, Math.PI * 2)
      ctx.fill()

      // Channel name
      ctx.font = '10px "IBM Plex Mono", monospace'
      ctx.fillStyle = highlightChannel === ch ? colors.trace : colors.text
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'left'
      ctx.fillText(label, 16, centerY)
    }

    ctx.restore()
    animFrameRef.current = requestAnimationFrame(draw)
  }, [
    nChannels,
    samplesPerWindow,
    amplitudeUvDiv,
    windowSeconds,
    highlightChannel,
    isSimulated,
    fs,
    channelLabels,
    channelQualities,
    activeRingBuffer,
  ])

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animFrameRef.current)
  }, [draw])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        cursor: 'crosshair',
      }}
      onClick={(e) => {
        const canvas = canvasRef.current
        if (!canvas || !onChannelClick) return
        const rect = canvas.getBoundingClientRect()
        const y = e.clientY - rect.top
        const ch = Math.floor((y / rect.height) * nChannels)
        if (ch >= 0 && ch < nChannels) {
          onChannelClick(ch)
        }
      }}
    />
  )
}
