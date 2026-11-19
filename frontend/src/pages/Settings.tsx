import { useState, useEffect } from 'react'
import PageHeader from '../components/PageHeader'
import Button from '../components/Button'
import { getAccessKey, setAccessKey, healthCheck } from '../lib/api'

type Theme = 'light' | 'dark' | 'system'

interface Props { onThemeChange: (t: Theme) => void; currentTheme: Theme }

export default function SettingsPage({ onThemeChange, currentTheme }: Props) {
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem('sentry_api_url') ?? (import.meta.env.VITE_API_URL ?? ''))
  const [accessKey, setAccessKeyState] = useState(() => getAccessKey())
  const [preSeizureThreshold, setPreSeizureThreshold] = useState(() => Number(localStorage.getItem('thresh_pre') ?? '0.4'))
  const [seizureThreshold, setSeizureThreshold] = useState(() => Number(localStorage.getItem('thresh_seizure') ?? '0.6'))
  const [saved, setSaved] = useState(false)
  const [testResult, setTestResult] = useState<string | null>(null)

  const save = () => {
    localStorage.setItem('sentry_api_url', apiUrl)
    setAccessKey(accessKey)
    localStorage.setItem('thresh_pre', String(preSeizureThreshold))
    localStorage.setItem('thresh_seizure', String(seizureThreshold))
    setSaved(true); setTimeout(() => setSaved(false), 2000)
  }

  const testConnection = async () => {
    try {
      const h = await healthCheck()
      setTestResult(`Connected — model ${h.model_loaded ? 'loaded' : 'not loaded'}, status: ${h.status}`)
    } catch {
      setTestResult('Connection failed. Check the backend URL and ensure the server is running.')
    }
  }

  const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 20, marginBottom: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16, color: 'var(--text)' }}>{title}</div>
      {children}
    </div>
  )

  const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
      <label style={{ width: 180, fontSize: 13, color: 'var(--text-muted)', flexShrink: 0 }}>{label}</label>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  )

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)',
    background: 'var(--surface-2)', color: 'var(--text)', fontSize: 13, fontFamily: 'IBM Plex Mono',
    outline: 'none', boxSizing: 'border-box'
  }

  return (
    <div style={{ maxWidth: 700 }}>
      <PageHeader title="Settings" actions={
        <Button onClick={save}>{saved ? 'Saved!' : 'Save settings'}</Button>
      } />

      <Section title="Appearance">
        <Field label="Theme">
          <div style={{ display: 'flex', gap: 4 }}>
            {(['light','dark','system'] as Theme[]).map(t => (
              <button key={t} onClick={() => onThemeChange(t)} style={{
                padding: '5px 14px', borderRadius: 6, border: '1px solid var(--border)',
                background: currentTheme === t ? 'var(--surface-2)' : 'transparent',
                color: currentTheme === t ? 'var(--text)' : 'var(--text-muted)',
                cursor: 'pointer', fontSize: 12, textTransform: 'capitalize'
              }}>{t}</button>
            ))}
          </div>
        </Field>
      </Section>

      <Section title="Connection">
        <Field label="Backend URL">
          <input style={inputStyle} value={apiUrl} onChange={e => setApiUrl(e.target.value)} placeholder="https://seizure-sentry-api.onrender.com" />
        </Field>
        <Field label="Access key">
          <input style={inputStyle} type="password" value={accessKey} onChange={e => setAccessKeyState(e.target.value)} placeholder="Enter backend access key" />
        </Field>
        <Button variant="secondary" size="sm" onClick={testConnection}>Test connection</Button>
        {testResult && <p style={{ marginTop: 8, fontSize: 12, color: testResult.includes('failed') ? 'var(--crit)' : 'var(--ok)' }}>{testResult}</p>}
      </Section>

      <Section title="Alert thresholds">
        <Field label="Pre-seizure threshold">
          <input style={{ ...inputStyle, width: 100 }} type="number" min="0" max="1" step="0.05" value={preSeizureThreshold} onChange={e => setPreSeizureThreshold(Number(e.target.value))} />
        </Field>
        <Field label="Seizure threshold">
          <input style={{ ...inputStyle, width: 100 }} type="number" min="0" max="1" step="0.05" value={seizureThreshold} onChange={e => setSeizureThreshold(Number(e.target.value))} />
        </Field>
      </Section>
    </div>
  )
}
