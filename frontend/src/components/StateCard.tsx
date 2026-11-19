/**
 * Dedicated clinical state card displaying current ML classification,
 * state chip, 3-class probability bars, and model tag.
 */

import StateChip from './StateChip'
import ProbabilityRows from './ProbabilityRows'
import type { EegPrediction } from '../lib/types'

interface StateCardProps {
  currentPrediction: EegPrediction | null
  isWarmup?: boolean
}

export default function StateCard({ currentPrediction, isWarmup = false }: StateCardProps) {
  const rawState = currentPrediction?.label ?? (isWarmup ? 'WARMUP' : 'Unknown')
  const displayState = isWarmup ? 'Warmup' : rawState

  const stateColor =
    displayState === 'Normal'
      ? 'var(--ok)'
      : displayState === 'Pre-Seizure'
      ? 'var(--warn)'
      : displayState === 'Seizure'
      ? 'var(--crit)'
      : 'var(--text-muted)'

  const probs = currentPrediction?.probs ?? { normal: 0, pre_seizure: 0, seizure: 0 }

  return (
    <div
      style={{
        width: 210,
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        flexShrink: 0,
      }}
    >
      <div>
        <div
          style={{
            fontSize: 11,
            color: 'var(--text-muted)',
            marginBottom: 6,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            fontWeight: 500,
          }}
        >
          Current state
        </div>
        <div
          style={{
            fontSize: 26,
            fontWeight: 600,
            color: stateColor,
            marginBottom: 8,
            lineHeight: 1.2,
          }}
        >
          {displayState}
        </div>
        <StateChip state={(isWarmup ? 'Unknown' : rawState) as 'Normal' | 'Pre-Seizure' | 'Seizure' | 'Unknown'} />
      </div>

      <div style={{ flex: 1 }}>
        <ProbabilityRows
          normal={probs.normal}
          preSeizure={probs.pre_seizure}
          seizure={probs.seizure}
          currentState={displayState}
        />
      </div>

      <div
        style={{
          marginTop: 'auto',
          paddingTop: 8,
          borderTop: '1px solid var(--border)',
          fontSize: 11,
          fontFamily: 'IBM Plex Mono',
          color: 'var(--text-muted)',
        }}
      >
        TUSZ_EEG_DEEP_CNN v1.0.0
      </div>
    </div>
  )
}
