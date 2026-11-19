import { useState, useEffect } from 'react'
import { FolderOpen, FileText, Download, Play, RefreshCw, X } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import EmptyState from '../components/EmptyState'
import Button from '../components/Button'
import {
  listSessions,
  getReplayRecordings,
  getClinicalReport,
  getReportEegPlotUrl,
  startReplay,
  type ReplayRecording,
} from '../lib/api'
import type { SessionRecord } from '../lib/types'

export default function Recordings() {
  const [sessions, setSessions] = useState<SessionRecord[]>([])
  const [recordings, setRecordings] = useState<ReplayRecording[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedReport, setSelectedReport] = useState<{ id: string; text: string } | null>(null)
  const [loadingReportId, setLoadingReportId] = useState<string | null>(null)

  const loadData = async () => {
    setIsLoading(true)
    try {
      const [sessList, recList] = await Promise.all([
        listSessions().catch(() => []),
        getReplayRecordings().catch(() => []),
      ])
      setSessions(sessList)
      setRecordings(recList)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const handleViewReport = async (sessionId: string) => {
    setLoadingReportId(sessionId)
    try {
      const text = await getClinicalReport(sessionId)
      setSelectedReport({ id: sessionId, text })
    } catch (err: any) {
      alert(`Could not retrieve report: ${err?.message || 'Session may still be active'}`)
    } finally {
      setLoadingReportId(null)
    }
  }

  return (
    <div style={{ maxWidth: 940 }}>
      <PageHeader
        title="Recordings & Replay"
        subtitle="Historical clinical recording sessions and bundled EDF replay datasets"
        actions={
          <Button variant="secondary" size="sm" onClick={loadData}>
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} /> Refresh
          </Button>
        }
      />

      {/* Bundled Replay Datasets */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Bundled Clinical Datasets
        </div>
        {recordings.length === 0 ? (
          <div style={{ padding: 12, background: 'var(--surface)', borderRadius: 6, border: '1px solid var(--border)', fontSize: 13, color: 'var(--text-muted)' }}>
            No bundled recordings found.
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {recordings.map(rec => (
              <div
                key={rec.id}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: 14,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ fontWeight: 500, fontSize: 14 }}>{rec.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'IBM Plex Mono' }}>
                  {rec.n_channels} channels · {rec.fs} Hz · {rec.duration_sec}s
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={async () => {
                    try {
                      await startReplay(rec.id, 1.0)
                      alert(`Started replay for ${rec.name}. Open Live Monitor to observe.`)
                    } catch (e: any) {
                      alert(`Replay start failed: ${e?.message}`)
                    }
                  }}
                  style={{ alignSelf: 'flex-start', marginTop: 4 }}
                >
                  <Play size={12} /> Play Stream
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Past Recorded Sessions */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Session History
        </div>

        {sessions.length === 0 ? (
          <EmptyState
            icon={<FolderOpen size={40} />}
            title="No recorded sessions yet"
            description="Start a live recording or demo session. Completed sessions will be archived here with clinical reports and waveform plots."
          />
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, background: 'var(--surface)', borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border)' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface-2)' }}>
                {['SESSION ID', 'CREATED', 'SOURCE', 'STATUS', 'ACTIONS'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map(s => (
                <tr key={s.session_id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 12px', fontFamily: 'IBM Plex Mono', fontSize: 12 }}>
                    {s.session_id.slice(0, 8)}…
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, color: 'var(--text-muted)' }}>
                    {new Date(s.created_at).toLocaleString()}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{s.source}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{ fontSize: 11, padding: '2px 6px', borderRadius: 4, background: 'var(--surface-2)', color: 'var(--text)' }}>
                      {s.status}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', display: 'flex', gap: 6 }}>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => handleViewReport(s.session_id)}
                      disabled={loadingReportId === s.session_id}
                    >
                      <FileText size={12} /> Report
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Report Modal */}
      {selectedReport && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              width: 720,
              maxHeight: '85vh',
              background: 'var(--surface)',
              borderRadius: 8,
              border: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: 600, fontSize: 15 }}>Clinical Session Report</span>
              <button onClick={() => setSelectedReport(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <X size={16} />
              </button>
            </div>
            <div style={{ padding: 18, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>EEG TRACE PLOT</div>
                <img
                  src={getReportEegPlotUrl(selectedReport.id)}
                  alt="EEG waveform"
                  style={{ width: '100%', height: 180, objectFit: 'cover', borderRadius: 4, border: '1px solid var(--border)' }}
                  onError={(e) => { (e.target as HTMLElement).style.display = 'none' }}
                />
              </div>
              <pre style={{
                background: 'var(--surface-2)',
                padding: 12,
                borderRadius: 4,
                fontSize: 12,
                fontFamily: 'IBM Plex Mono',
                whiteSpace: 'pre-wrap',
                margin: 0,
              }}>
                {selectedReport.text}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
