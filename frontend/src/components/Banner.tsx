import AlertBanner from './AlertBanner'
import type { EegAlert } from '../lib/types'

interface LegacyBannerProps {
  level?: 'warning' | 'critical' | 'info' | 'neutral'
  message?: string
  alert?: EegAlert | null
  artifactGuardActive?: boolean
  onAcknowledge?: () => void
  onMute?: () => void
  onDismiss?: () => void
}

export default function Banner(props: LegacyBannerProps) {
  if (props.alert !== undefined || props.artifactGuardActive !== undefined) {
    return (
      <AlertBanner
        alert={props.alert}
        artifactGuardActive={props.artifactGuardActive}
        onAcknowledge={props.onAcknowledge ? () => props.onAcknowledge!() : undefined}
        onMute={props.onMute}
        onDismiss={props.onDismiss ? () => props.onDismiss!() : undefined}
      />
    )
  }

  // Fallback for direct message/level props
  const lvl = props.level ?? 'info'
  const isCrit = lvl === 'critical'
  const isWarn = lvl === 'warning'
  const mockAlert: EegAlert = {
    id: 'legacy-banner',
    level: isCrit ? 'CRITICAL' : isWarn ? 'WARNING' : 'warning',
    started_at: Date.now(),
    peak_prob: 0,
  }

  return (
    <div
      className="animate-slide-down"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 16px',
        borderRadius: 6,
        borderLeft: `3px solid ${isCrit ? 'var(--crit)' : isWarn ? 'var(--warn)' : 'var(--accent)'}`,
        background: isCrit
          ? 'color-mix(in srgb, var(--crit) 10%, var(--surface))'
          : isWarn
          ? 'color-mix(in srgb, var(--warn) 10%, var(--surface))'
          : 'var(--surface-2)',
        color: isCrit ? 'var(--crit)' : isWarn ? 'var(--warn)' : 'var(--text)',
        marginBottom: 10,
      }}
    >
      <span style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>{props.message}</span>
      {props.onAcknowledge && (
        <button
          onClick={props.onAcknowledge}
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
          }}
        >
          Acknowledge
        </button>
      )}
      {props.onMute && (
        <button
          onClick={props.onMute}
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
          }}
        >
          Mute
        </button>
      )}
    </div>
  )
}

export { AlertBanner }
