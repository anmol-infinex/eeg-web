import { AlertTriangle, Activity, AlertOctagon, HelpCircle } from 'lucide-react'

type EegState = 'Normal' | 'Pre-Seizure' | 'Seizure' | 'Unknown'

const STATE_CONFIG = {
  'Normal':      { color: 'var(--ok)',   Icon: Activity },
  'Pre-Seizure': { color: 'var(--warn)', Icon: AlertTriangle },
  'Seizure':     { color: 'var(--crit)', Icon: AlertOctagon },
  'Unknown':     { color: 'var(--text-muted)', Icon: HelpCircle },
}

export default function StateChip({ state }: { state: EegState }) {
  const cfg = STATE_CONFIG[state] ?? STATE_CONFIG.Unknown
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 10px',
      borderRadius: 6, border: `1px solid ${cfg.color}`,
      background: `color-mix(in srgb, ${cfg.color} 12%, transparent)`,
      color: cfg.color, fontSize: 13, fontWeight: 500
    }}>
      <cfg.Icon size={14} strokeWidth={1.5} />
      {state}
    </span>
  )
}
