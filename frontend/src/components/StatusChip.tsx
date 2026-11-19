import { clsx } from 'clsx'

type Status = 'Disconnected' | 'Connecting' | 'Streaming' | 'Simulated' | 'ok' | 'degraded' | 'offline'

const STATUS_STYLES: Record<Status, { dot: string; border: string; label: string }> = {
  Disconnected: { dot: 'var(--text-muted)', border: 'var(--border)', label: 'Disconnected' },
  Connecting:   { dot: 'var(--accent)', border: 'var(--accent)', label: 'Connecting' },
  Streaming:    { dot: 'var(--ok)', border: 'var(--ok)', label: 'Streaming' },
  Simulated:    { dot: 'var(--sim)', border: 'var(--sim)', label: 'Simulated' },
  ok:           { dot: 'var(--ok)', border: 'var(--ok)', label: 'Backend ok' },
  degraded:     { dot: 'var(--warn)', border: 'var(--warn)', label: 'Degraded' },
  offline:      { dot: 'var(--crit)', border: 'var(--crit)', label: 'Offline' },
}

export default function StatusChip({ status }: { status: Status }) {
  const s = STATUS_STYLES[status]
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '2px 8px', borderRadius: 6, border: `1px solid ${s.border}`,
      fontSize: 12, fontWeight: 500, color: s.dot, background: 'transparent'
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%', background: s.dot, flexShrink: 0
      }} />
      {s.label}
    </span>
  )
}
