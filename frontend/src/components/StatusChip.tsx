/**
 * StatusChip: Precision telemetry status indicator badge.
 * Designed with strict clinical workstation aesthetics:
 * - Crisp 4px rectangular shape
 * - Muted status background with 1px hairline border
 * - Live status dot with high contrast
 */

type Status = 'Disconnected' | 'Connecting' | 'Streaming' | 'Simulated' | 'ok' | 'degraded' | 'offline'

interface StatusConfig {
  dot: string
  bg: string
  border: string
  label: string
}

const STATUS_MAP: Record<Status, StatusConfig> = {
  ok: {
    dot: 'var(--ok)',
    bg: 'color-mix(in srgb, var(--ok) 10%, transparent)',
    border: 'color-mix(in srgb, var(--ok) 30%, transparent)',
    label: 'System Online'
  },
  degraded: {
    dot: 'var(--warn)',
    bg: 'color-mix(in srgb, var(--warn) 10%, transparent)',
    border: 'color-mix(in srgb, var(--warn) 30%, transparent)',
    label: 'Degraded'
  },
  offline: {
    dot: 'var(--crit)',
    bg: 'color-mix(in srgb, var(--crit) 10%, transparent)',
    border: 'color-mix(in srgb, var(--crit) 30%, transparent)',
    label: 'Offline'
  },
  Streaming: {
    dot: 'var(--ok)',
    bg: 'color-mix(in srgb, var(--ok) 10%, transparent)',
    border: 'color-mix(in srgb, var(--ok) 30%, transparent)',
    label: 'Streaming'
  },
  Simulated: {
    dot: 'var(--sim)',
    bg: 'color-mix(in srgb, var(--sim) 10%, transparent)',
    border: 'color-mix(in srgb, var(--sim) 30%, transparent)',
    label: 'Simulated'
  },
  Connecting: {
    dot: 'var(--warn)',
    bg: 'color-mix(in srgb, var(--warn) 10%, transparent)',
    border: 'color-mix(in srgb, var(--warn) 30%, transparent)',
    label: 'Connecting'
  },
  Disconnected: {
    dot: 'var(--text-muted)',
    bg: 'var(--surface-2)',
    border: 'var(--border)',
    label: 'Disconnected'
  }
}

export default function StatusChip({ status }: { status: Status }) {
  const conf = STATUS_MAP[status] || STATUS_MAP.offline

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        height: 28,
        padding: '0 10px',
        borderRadius: 4,
        border: `1px solid ${conf.border}`,
        background: conf.bg,
        fontSize: 11,
        fontWeight: 500,
        color: conf.dot,
        letterSpacing: '0.02em',
        userSelect: 'none',
        flexShrink: 0
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: conf.dot,
          boxShadow: status === 'ok' || status === 'Streaming' ? `0 0 4px ${conf.dot}` : 'none',
          flexShrink: 0
        }}
      />
      <span>{conf.label}</span>
    </span>
  )
}
