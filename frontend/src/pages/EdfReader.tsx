import { useState } from 'react'
import { FileText, ExternalLink, Maximize2, RefreshCw } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Button from '../components/Button'
import KeyValueList from '../components/KeyValueList'

const NEUROTRACE_URL = 'https://neurotrace-dashboard-eight.vercel.app/'

export default function EdfReader() {
  const [embedLoaded, setEmbedLoaded] = useState(false)
  const [embedKey, setEmbedKey] = useState(0)

  const handleOpenExternal = () => {
    window.open(NEUROTRACE_URL, '_blank', 'noopener,noreferrer')
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 48px - 36px - 48px)', gap: 16 }}>
      <PageHeader
        title="EDF File Reader"
        subtitle="NeuroTrace EEG Clinical Review Console — upload, inspect, and analyze .EDF recordings"
        actions={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Button
              onClick={handleOpenExternal}
              size="sm"
              style={{
                background: 'var(--accent)',
                color: 'var(--accent-contrast)',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <ExternalLink size={14} /> Open NeuroTrace in New Tab
            </Button>
          </div>
        }
      />

      {/* Hero launch card */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 6,
              background: 'color-mix(in srgb, var(--accent) 15%, transparent)',
              color: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <FileText size={24} strokeWidth={1.75} />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>
              NeuroTrace | EEG Clinical Review Console
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              Upload any standard European Data Format (.EDF / .EDF+) recording, review 16–22 channel raw traces, filter artifacts, and inspect offline seizure predictions.
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setEmbedKey(k => k + 1)}
            title="Reload embedded view"
          >
            <RefreshCw size={13} /> Reload
          </Button>
          <Button
            onClick={handleOpenExternal}
            size="sm"
            style={{
              background: 'var(--accent)',
              color: 'var(--accent-contrast)',
              fontWeight: 600,
            }}
          >
            <ExternalLink size={14} /> Open NeuroTrace Console
          </Button>
        </div>
      </div>

      {/* Embedded Console View (iframe with direct fallback) */}
      <div
        style={{
          flex: 1,
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          minHeight: 0,
        }}
      >
        <div
          style={{
            padding: '8px 14px',
            background: 'var(--surface-2)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: 12,
            color: 'var(--text-muted)',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--ok)' }} />
            {NEUROTRACE_URL}
          </div>
          <button
            onClick={handleOpenExternal}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            <Maximize2 size={12} /> Full screen
          </button>
        </div>

        <iframe
          key={embedKey}
          src={NEUROTRACE_URL}
          title="NeuroTrace EEG Clinical Review Console"
          onLoad={() => setEmbedLoaded(true)}
          style={{
            flex: 1,
            width: '100%',
            height: '100%',
            border: 'none',
            background: '#0B0E12',
          }}
          allow="clipboard-write; fullscreen"
        />
      </div>
    </div>
  )
}
