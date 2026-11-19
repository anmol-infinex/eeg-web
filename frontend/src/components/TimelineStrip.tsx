/**
 * TimelineStrip component: Bounded 300-cell historical prediction timeline strip.
 * Supports color-coded state cells, alert pips, and clamped tooltips.
 */

import { useState, useRef } from 'react'
import type { EegPrediction } from '../lib/types'

export interface TimelineCell {
  t: number
  label: 'Normal' | 'Pre-Seizure' | 'Seizure'
  probs: { normal: number; pre_seizure: number; seizure: number }
  hasAlert?: boolean
  window_start?: number
  window_end?: number
}

interface TimelineStripProps {
  predictions: (TimelineCell | EegPrediction)[]
  onCellClick?: (index: number) => void
  maxCells?: number
}

const STATE_COLORS: Record<string, string> = {
  Normal: 'var(--ok)',
  'Pre-Seizure': 'var(--warn)',
  Seizure: 'var(--crit)',
}

export default function TimelineStrip({
  predictions,
  onCellClick,
  maxCells = 300,
}: TimelineStripProps) {
  const [hovered, setHovered] = useState<{
    idx: number
    x: number
    cell: TimelineCell
  } | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)

  // Bounded buffer of maxCells (default 300)
  const visible = predictions.slice(-maxCells)

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        height: 36,
        background: 'var(--surface-2)',
        borderRadius: 4,
        border: '1px solid var(--border)',
        padding: '3px 4px',
        display: 'flex',
        alignItems: 'center',
        userSelect: 'none',
      }}
    >
      <div style={{ display: 'flex', width: '100%', height: '100%', gap: 1 }}>
        {visible.map((item, i) => {
          const cell: TimelineCell = {
            t: (item as any).t ?? item.window_start ?? Date.now(),
            label: item.label,
            probs: item.probs,
            hasAlert: (item as any).hasAlert ?? (item.label === 'Seizure'),
            window_start: item.window_start,
            window_end: item.window_end,
          }
          const isSelected = hovered?.idx === i

          return (
            <div
              key={i}
              onClick={() => onCellClick?.(i)}
              onMouseEnter={(e) => {
                const rect = containerRef.current?.getBoundingClientRect()
                const cellRect = e.currentTarget.getBoundingClientRect()
                const relX = rect ? cellRect.left - rect.left + cellRect.width / 2 : 0
                setHovered({ idx: i, x: relX, cell })
              }}
              onMouseLeave={() => setHovered(null)}
              style={{
                flex: 1,
                minWidth: 2,
                height: '100%',
                borderRadius: 2,
                cursor: onCellClick ? 'pointer' : 'default',
                background: STATE_COLORS[cell.label] ?? 'var(--border)',
                opacity: isSelected ? 1 : 0.8,
                position: 'relative',
                transition: 'opacity 80ms',
              }}
            >
              {cell.hasAlert && (
                <div
                  style={{
                    position: 'absolute',
                    top: 1,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: 3,
                    height: 3,
                    borderRadius: '50%',
                    background: '#FFFFFF',
                  }}
                />
              )}
            </div>
          )
        })}

        {visible.length === 0 && (
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              color: 'var(--text-muted)',
            }}
          >
            Waiting for predictions…
          </div>
        )}
      </div>

      {/* Clamped Tooltip */}
      {hovered && (
        <div
          style={{
            position: 'absolute',
            bottom: 42,
            left: Math.max(120, Math.min(hovered.x, (containerRef.current?.clientWidth ?? 300) - 120)),
            transform: 'translateX(-50%)',
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '6px 10px',
            fontSize: 11,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            zIndex: 100,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          <div style={{ fontWeight: 600, color: STATE_COLORS[hovered.cell.label] ?? 'var(--text)' }}>
            {hovered.cell.label}
          </div>
          <div style={{ fontFamily: 'IBM Plex Mono', color: 'var(--text-muted)', fontSize: 10 }}>
            {hovered.cell.window_start && hovered.cell.window_end
              ? `${(hovered.cell.window_start / 1000).toFixed(1)}s – ${(hovered.cell.window_end / 1000).toFixed(1)}s`
              : new Date(hovered.cell.t).toLocaleTimeString()}
          </div>
          <div style={{ display: 'flex', gap: 8, fontFamily: 'IBM Plex Mono', fontSize: 10, marginTop: 2 }}>
            <span>N: {(hovered.cell.probs.normal * 100).toFixed(1)}%</span>
            <span>P: {(hovered.cell.probs.pre_seizure * 100).toFixed(1)}%</span>
            <span>S: {(hovered.cell.probs.seizure * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}
