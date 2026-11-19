/**
 * DiagnosticsPanel component: 5-hop telemetry counters panel (FR-27 / Feature 6).
 * Displays end-to-end metrics from hardware device to client rendering.
 */

import { ArrowRight, CheckCircle2, AlertTriangle } from 'lucide-react'
import type { PipelineDiagnostics } from '../lib/types'

interface DiagnosticsPanelProps {
  diagnostics?: PipelineDiagnostics | null
  onRefresh?: () => void
}

export default function DiagnosticsPanel({ diagnostics }: DiagnosticsPanelProps) {
  if (!diagnostics) {
    return (
      <div
        style={{
          padding: '10px 14px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          fontSize: 12,
          color: 'var(--text-muted)',
        }}
      >
        Diagnostics telemetry initializing…
      </div>
    )
  }

  const hops = [
    { num: 1, label: 'Device Pkts', val: diagnostics.hop1_device_packets },
    { num: 2, label: 'Chunks Sent', val: diagnostics.hop2_chunks_sent },
    { num: 3, label: 'Frames Recv', val: diagnostics.hop3_frames_received },
    { num: 4, label: 'Inferred', val: diagnostics.hop4_windows_inferred },
    { num: 5, label: 'Displayed', val: diagnostics.hop5_predictions_displayed },
  ]

  const isHealthy = diagnostics.health === 'ok' && diagnostics.dropped_chunks === 0

  return (
    <div
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '10px 14px',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          5-Hop Pipeline Telemetry
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11, fontFamily: 'IBM Plex Mono' }}>
          <span style={{ color: isHealthy ? 'var(--ok)' : 'var(--warn)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            {isHealthy ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
            {diagnostics.health.toUpperCase()}
          </span>
          <span>Drops: {diagnostics.dropped_chunks}</span>
          <span>Latency: {diagnostics.last_latency_ms.toFixed(1)} ms</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflowX: 'auto' }}>
        {hops.map((hop, i) => (
          <div key={hop.num} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div
              style={{
                background: 'var(--surface-2)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '4px 8px',
                minWidth: 80,
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>
                Hop {hop.num}: {hop.label}
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, fontFamily: 'IBM Plex Mono', color: 'var(--text)' }}>
                {hop.val.toLocaleString()}
              </div>
            </div>
            {i < hops.length - 1 && (
              <ArrowRight size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
