/**
 * AlertBanner component: slide-down amber/red clinical notification banner
 * with Acknowledge and Mute buttons, Artifact Guard suppression indication,
 * and peak probability display.
 */

import { X, CheckCircle, VolumeX, AlertTriangle, AlertOctagon, ShieldAlert } from 'lucide-react'
import type { EegAlert } from '../lib/types'

interface AlertBannerProps {
  alert?: EegAlert | null
  artifactGuardActive?: boolean
  onAcknowledge?: (alertId: string) => void
  onMute?: () => void
  onDismiss?: (alertId: string) => void
}

export default function AlertBanner({
  alert,
  artifactGuardActive = false,
  onAcknowledge,
  onMute,
  onDismiss,
}: AlertBannerProps) {
  // Artifact Guard banner takes precedence or shows as safety interlock
  if (artifactGuardActive) {
    return (
      <div
        className="animate-slide-down"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 16px',
          borderRadius: 6,
          borderLeft: '3px solid var(--border)',
          background: 'var(--surface-2)',
          color: 'var(--text)',
          marginBottom: 10,
        }}
      >
        <ShieldAlert size={16} style={{ color: 'var(--warn)', flexShrink: 0 }} />
        <span style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>
          Signal quality too low. Predictions paused.
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'IBM Plex Mono' }}>
          Artifact Guard Active
        </span>
      </div>
    )
  }

  if (!alert) return null

  const isCrit = String(alert.level).toUpperCase() === 'CRITICAL'
  const border = isCrit ? 'var(--crit)' : 'var(--warn)'
  const bg = isCrit
    ? 'color-mix(in srgb, var(--crit) 10%, var(--surface))'
    : 'color-mix(in srgb, var(--warn) 10%, var(--surface))'
  const color = isCrit ? 'var(--crit)' : 'var(--warn)'
  const Icon = isCrit ? AlertOctagon : AlertTriangle

  const message = isCrit
    ? `Critical seizure activity detected — peak probability ${(alert.peak_prob * 100).toFixed(0)}%`
    : `Pre-seizure pattern detected — peak probability ${(alert.peak_prob * 100).toFixed(0)}%`

  return (
    <div
      className="animate-slide-down"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 16px',
        borderRadius: 6,
        borderLeft: `3px solid ${border}`,
        background: bg,
        color: color,
        marginBottom: 10,
      }}
    >
      <Icon size={16} style={{ flexShrink: 0 }} />
      <span style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>{message}</span>

      {onAcknowledge && (
        <button
          onClick={() => onAcknowledge(alert.id)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            background: 'none',
            border: '1px solid currentColor',
            borderRadius: 4,
            color: 'inherit',
            cursor: 'pointer',
            padding: '3px 8px',
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          <CheckCircle size={12} /> Acknowledge
        </button>
      )}

      {onMute && (
        <button
          onClick={onMute}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            background: 'none',
            border: '1px solid currentColor',
            borderRadius: 4,
            color: 'inherit',
            cursor: 'pointer',
            padding: '3px 8px',
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          <VolumeX size={12} /> Mute
        </button>
      )}

      {onDismiss && (
        <button
          onClick={() => onDismiss(alert.id)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'inherit',
            padding: 2,
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <X size={14} />
        </button>
      )}
    </div>
  )
}
