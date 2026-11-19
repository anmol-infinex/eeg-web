import { useState } from 'react'
import { Bell, CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import EmptyState from '../components/EmptyState'
import Button from '../components/Button'
import type { EegAlert } from '../lib/types'

type Level = 'All' | 'Warning' | 'Critical'

interface AlertsPageProps {
  alertsList?: EegAlert[]
  onAcknowledge?: (alertId: string) => void
}

export default function AlertsPage({ alertsList = [], onAcknowledge }: AlertsPageProps) {
  const [filter, setFilter] = useState<Level>('All')
  const [localAlerts, setLocalAlerts] = useState<EegAlert[]>(alertsList)

  const filtered = filter === 'All'
    ? localAlerts
    : localAlerts.filter(a => String(a.level).toUpperCase() === filter.toUpperCase())

  const handleAck = (id: string) => {
    setLocalAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true, muted: true } : a))
    onAcknowledge?.(id)
  }

  return (
    <div style={{ maxWidth: 940 }}>
      <PageHeader
        title="Alerts"
        subtitle="Clinical warning and critical seizure episode log"
        actions={
          <div style={{ display: 'flex', gap: 4 }}>
            {(['All', 'Warning', 'Critical'] as Level[]).map(l => (
              <button
                key={l}
                onClick={() => setFilter(l)}
                style={{
                  padding: '4px 12px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: filter === l ? 'var(--surface-2)' : 'transparent',
                  color: filter === l ? 'var(--text)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: 12,
                  fontWeight: filter === l ? 500 : 400,
                }}
              >
                {l}
              </button>
            ))}
          </div>
        }
      />

      {filtered.length === 0 ? (
        <EmptyState
          icon={<Bell size={40} />}
          title="No alerts logged"
          description="No clinical alerts recorded yet. Warning (Pre-Seizure) and Critical (Seizure) alerts will be recorded here."
        />
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, background: 'var(--surface)', borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border)' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface-2)' }}>
              {['TIME', 'LEVEL', 'PEAK PROB', 'STATUS', 'ACTIONS'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 500 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(a => {
              const isCrit = String(a.level).toUpperCase() === 'CRITICAL'
              const Icon = isCrit ? AlertOctagon : AlertTriangle
              const color = isCrit ? 'var(--crit)' : 'var(--warn)'
              return (
                <tr key={a.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 12px', fontFamily: 'IBM Plex Mono', fontSize: 12 }}>
                    {typeof a.started_at === 'number'
                      ? new Date(a.started_at).toLocaleTimeString()
                      : String(a.started_at)}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color, fontWeight: 500, fontSize: 12 }}>
                      <Icon size={13} /> {isCrit ? 'CRITICAL' : 'WARNING'}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', fontFamily: 'IBM Plex Mono', fontSize: 12 }}>
                    {(a.peak_prob * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>
                    {a.acknowledged ? (
                      <span style={{ color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        <CheckCircle2 size={12} /> Acknowledged
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>Pending</span>
                    )}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    {!a.acknowledged && (
                      <Button size="sm" variant="secondary" onClick={() => handleAck(a.id)}>
                        Acknowledge
                      </Button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
