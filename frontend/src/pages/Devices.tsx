import { useState } from 'react'
import { Cpu, CheckCircle2, AlertTriangle, XCircle, Usb } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import KeyValueList from '../components/KeyValueList'
import Button from '../components/Button'
import DeviceModal from '../components/DeviceModal'
import { WebSerialSource } from '../lib/serial'

const DEVICES = [
  {
    id: 'openbci-cyton-daisy',
    name: 'OpenBCI Cyton + Daisy',
    status: 'Supported',
    fs: '125 Hz (Resampled to 256 Hz)',
    channels: 16,
    channelLabels: 'Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8',
    connection: 'FTDI USB Dongle (Web Serial API @ 115200 8N1)',
    compatibility: 'Compatible with adaptation',
    compatibilityStatus: 'adapted',
    adaptationNotes: 'Polyphase resampling 125→256 Hz and TCP montage remapping applied.',
  },
  {
    id: 'simulator',
    name: 'Simulator / EDF Replay',
    status: 'Available',
    fs: '256 Hz (Native)',
    channels: 22,
    channelLabels: 'Standard 10-20 montage (22 channels)',
    connection: 'Server-side streaming gateway',
    compatibility: 'Compatible (Native)',
    compatibilityStatus: 'compatible',
    adaptationNotes: 'Matches model tensor shape (1280, 22) exactly without interpolation.',
  },
  {
    id: 'consumer-headband',
    name: 'Generic 4-Channel Headband',
    status: 'Incompatible',
    fs: '200 Hz',
    channels: 4,
    channelLabels: 'TP9, AF7, AF8, TP10',
    connection: 'Bluetooth LE',
    compatibility: 'Incompatible (< 10 channels match)',
    compatibilityStatus: 'incompatible',
    adaptationNotes: 'Model requires at least 10 scalp channels for spatial derivation.',
  },
]

export default function Devices() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [activeDevice, setActiveDevice] = useState<string | null>(null)

  const isSerialSupported = typeof navigator !== 'undefined' && 'serial' in navigator

  const handleDeviceLinked = (_source: WebSerialSource) => {
    setActiveDevice('OpenBCI Cyton + Daisy')
  }

  return (
    <div style={{ maxWidth: 840 }}>
      <PageHeader
        title="Devices"
        subtitle="Hardware EEG sources, Web Serial link, and model compatibility matrix"
        actions={
          <Button
            onClick={() => setIsModalOpen(true)}
            size="sm"
            disabled={!isSerialSupported}
          >
            <Usb size={14} /> Link OpenBCI device
          </Button>
        }
      />

      {!isSerialSupported && (
        <div
          style={{
            padding: '10px 14px',
            background: 'color-mix(in srgb, var(--warn) 10%, var(--surface))',
            border: '1px solid var(--warn)',
            borderRadius: 6,
            color: 'var(--warn)',
            fontSize: 13,
            marginBottom: 16,
          }}
        >
          Web Serial is supported in Chromium-based browsers (Chrome, Edge). Replay and Demo modes remain fully functional.
        </div>
      )}

      {activeDevice && (
        <div
          style={{
            padding: '10px 14px',
            background: 'color-mix(in srgb, var(--ok) 10%, var(--surface))',
            border: '1px solid var(--ok)',
            borderRadius: 6,
            color: 'var(--ok)',
            fontSize: 13,
            marginBottom: 16,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <CheckCircle2 size={16} /> Active link: {activeDevice}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {DEVICES.map((d) => {
          const CompatIcon =
            d.compatibilityStatus === 'compatible'
              ? CheckCircle2
              : d.compatibilityStatus === 'adapted'
              ? AlertTriangle
              : XCircle
          const compatColor =
            d.compatibilityStatus === 'compatible'
              ? 'var(--ok)'
              : d.compatibilityStatus === 'adapted'
              ? 'var(--warn)'
              : 'var(--crit)'

          return (
            <div
              key={d.id}
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: 16,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: 12,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Cpu size={16} style={{ color: 'var(--accent)' }} />
                  <span style={{ fontSize: 15, fontWeight: 500, color: 'var(--text)' }}>
                    {d.name}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span
                    style={{
                      fontSize: 11,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      color: compatColor,
                      fontWeight: 500,
                    }}
                  >
                    <CompatIcon size={12} /> {d.compatibility}
                  </span>
                  <span
                    style={{
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 4,
                      background: 'var(--surface-2)',
                      color: 'var(--text-muted)',
                      border: '1px solid var(--border)',
                    }}
                  >
                    {d.status}
                  </span>
                </div>
              </div>

              <KeyValueList
                items={[
                  { label: 'Sample rate', value: d.fs },
                  { label: 'Channels', value: isNaN(d.channels) ? 'Variable' : String(d.channels) },
                  { label: 'Channel labels', value: d.channelLabels },
                  { label: 'Connection method', value: d.connection },
                  { label: 'Adaptation details', value: d.adaptationNotes },
                ]}
              />
            </div>
          )
        })}
      </div>

      <DeviceModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onDeviceLinked={handleDeviceLinked}
      />
    </div>
  )
}
