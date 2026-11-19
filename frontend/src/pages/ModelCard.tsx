import { useEffect, useState } from 'react'
import PageHeader from '../components/PageHeader'
import KeyValueList from '../components/KeyValueList'
import { healthCheck } from '../lib/api'

export default function ModelCard() {
  const [loaded, setLoaded] = useState<boolean | null>(null)

  useEffect(() => {
    healthCheck().then(h => setLoaded(h.model_loaded)).catch(() => setLoaded(false))
  }, [])

  return (
    <div style={{ maxWidth: 700 }}>
      <PageHeader title="Model" subtitle="Loaded ML model details and preprocessing specification" />
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 20, marginBottom: 16 }}>
        <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 16 }}>TUSZ_EEG_DEEP_CNN</div>
        <KeyValueList items={[
          { label: 'Model file', value: 'tusz_3class_eeg_model.h5' },
          { label: 'Version', value: '1.0.0' },
          { label: 'Framework', value: 'Keras 3 (PyTorch CPU backend)' },
          { label: 'Model loaded', value: loaded === null ? 'Checking…' : loaded ? 'Yes' : 'No' },
          { label: 'Input shape', value: '(None, 1280, 22) — float32' },
          { label: 'Sample rate', value: '256 Hz' },
          { label: 'Window duration', value: '5 seconds (1280 samples)' },
          { label: 'Channels', value: '22 (standard 10-20 EEG montage)' },
          { label: 'Output classes', value: 'Normal, Pre-Seizure, Seizure (softmax)' },
          { label: 'Preprocessing', value: 'Per-channel Z-score normalization (μ=0, σ=1)' },
          { label: 'Inference latency', value: '~4.85 ms (CPU)' },
        ]} />
      </div>
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Limitations</div>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
          <li>Not a medical device. Model output is for research purposes only.</li>
          <li>Trained on TUSZ dataset. Performance may differ on other recording setups.</li>
          <li>Requires 22-channel EEG at 256 Hz. 16-channel OpenBCI input is resampled and remapped.</li>
          <li>No real-time model retraining in this deployment.</li>
        </ul>
      </div>
    </div>
  )
}
