import { useEffect, useState } from 'react'
import { Activity, Cpu, Monitor } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import KeyValueList from '../components/KeyValueList'
import StatusChip from '../components/StatusChip'
import Button from '../components/Button'
import { healthCheck, type HealthResponse } from '../lib/api'
import { useNavigate } from 'react-router-dom'

interface Props { onBackendStatus: (s: 'ok'|'degraded'|'offline') => void }

export default function Overview({ onBackendStatus }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [checking, setChecking] = useState(true)
  const navigate = useNavigate()

  const check = async () => {
    try {
      const h = await healthCheck()
      setHealth(h)
      onBackendStatus(h.status)
    } catch {
      setHealth(null)
      onBackendStatus('offline')
    } finally { setChecking(false) }
  }

  useEffect(() => { check(); const id = setInterval(check, 30000); return () => clearInterval(id) }, [])

  return (
    <div style={{ maxWidth: 800 }}>
      <PageHeader title="Overview" subtitle="System status and quick actions" />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>
            <Cpu size={16} strokeWidth={1.5} /> Backend status
          </div>
          {checking ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Checking…</div>
          ) : health ? (
            <KeyValueList items={[
              { label: 'Status', value: <StatusChip status={health.status} /> },
              { label: 'Model', value: health.model_name },
              { label: 'Model loaded', value: health.model_loaded ? 'Yes' : 'No' },
              { label: 'Version', value: health.version },
            ]} />
          ) : (
            <div>
              <StatusChip status="offline" />
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                Cannot reach the analysis server. Check the backend URL in Settings.
              </p>
              <Button size="sm" variant="secondary" onClick={check} style={{ marginTop: 8 }}>Retry</Button>
            </div>
          )}
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>
            <Activity size={16} strokeWidth={1.5} /> Quick actions
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Button onClick={() => navigate('/live')}>
              <Monitor size={14} /> Start demo session
            </Button>
            <Button variant="secondary" onClick={() => navigate('/devices')}>
              <Cpu size={14} /> Manage devices
            </Button>
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 16, marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Supported devices</div>
        <KeyValueList items={[
          { label: 'OpenBCI Cyton + Daisy', value: '16 ch · 125 Hz · USB dongle (Web Serial)' },
          { label: 'Simulator / EDF replay', value: 'Server-side · any recording' },
          { label: 'Local bridge (BrainFlow)', value: 'Phase 7 — coming soon' },
        ]} />
      </div>

      <div style={{ padding: 12, background: 'var(--surface-2)', borderRadius: 6, fontSize: 12, color: 'var(--text-muted)', borderLeft: '3px solid var(--border)' }}>
        Research prototype. Not a medical device. Data stays on your server and is not sent to any third party.
      </div>
    </div>
  )
}
