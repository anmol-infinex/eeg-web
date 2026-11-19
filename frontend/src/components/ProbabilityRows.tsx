interface Props {
  normal: number
  preSeizure: number
  seizure: number
  currentState?: string
}

function ProbRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
      <span style={{ width: 90, fontSize: 12, color: 'var(--text-muted)', flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, background: 'var(--surface-2)', borderRadius: 2, height: 4, overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, background: color, height: '100%', transition: 'width 200ms' }} />
      </div>
      <span style={{ width: 42, fontSize: 12, fontFamily: 'IBM Plex Mono', textAlign: 'right', color: 'var(--text)' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

export default function ProbabilityRows({ normal, preSeizure, seizure, currentState }: Props) {
  const rows = [
    { label: 'P(normal)',      value: normal,     color: currentState === 'Normal'      ? 'var(--ok)'   : 'var(--border)' },
    { label: 'P(pre-seizure)', value: preSeizure, color: currentState === 'Pre-Seizure' ? 'var(--warn)' : 'var(--border)' },
    { label: 'P(seizure)',     value: seizure,    color: currentState === 'Seizure'     ? 'var(--crit)' : 'var(--border)' },
  ]
  return (
    <div style={{ padding: '4px 0' }}>
      {rows.map(r => <ProbRow key={r.label} {...r} />)}
    </div>
  )
}
