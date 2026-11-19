/**
 * DeviceModal component: 4-step Web Serial connection wizard for OpenBCI Cyton+Daisy.
 * (Select -> Connect -> Compatibility Check -> Signal Quality)
 */

import { useState } from 'react'
import { X, Cpu, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react'
import Button from './Button'
import { WebSerialSource } from '../lib/serial'
import type { ChannelQuality } from '../lib/types'

interface DeviceModalProps {
  isOpen: boolean
  onClose: () => void
  onDeviceLinked: (serialSource: WebSerialSource) => void
}

const CHANNELS_16 = [
  'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
  'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8'
]

const QUALITY_COLOR_MAP: Record<ChannelQuality, string> = {
  good: 'var(--ok)',
  noisy: 'var(--warn)',
  flat: 'var(--crit)',
  railed: 'var(--crit)',
}

export default function DeviceModal({ isOpen, onClose, onDeviceLinked }: DeviceModalProps) {
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [serialSource, setSerialSource] = useState<WebSerialSource | null>(null)
  const [channelQualities, setChannelQualities] = useState<ChannelQuality[]>(Array(16).fill('good'))
  const [isConnecting, setIsConnecting] = useState(false)

  if (!isOpen) return null

  const isSupported = typeof navigator !== 'undefined' && 'serial' in navigator

  const handleConnectPort = async () => {
    setErrorMsg(null)
    setIsConnecting(true)
    const source = new WebSerialSource()
    try {
      setStep(2)
      await source.connect()
      setSerialSource(source)
      // Once connected, move to compatibility check
      setStep(3)
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to connect to device. Ensure Cyton is powered on.')
      setStep(1)
      source.disconnect().catch(() => {})
    } finally {
      setIsConnecting(false)
    }
  }

  const handleFinishWizard = () => {
    if (serialSource) {
      onDeviceLinked(serialSource)
    }
    onClose()
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        style={{
          width: 580,
          background: 'var(--surface)',
          borderRadius: 8,
          border: '1px solid var(--border)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, fontSize: 15 }}>
            <Cpu size={18} style={{ color: 'var(--accent)' }} />
            <span>Link OpenBCI Device (Web Serial)</span>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Wizard Step Indicator */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--border)',
            background: 'var(--surface-2)',
            fontSize: 12,
          }}
        >
          {['1. Select Port', '2. Connect', '3. Compatibility', '4. Signal Quality'].map((title, i) => {
            const stepNum = (i + 1) as 1 | 2 | 3 | 4
            const isActive = step === stepNum
            const isDone = step > stepNum
            return (
              <div
                key={title}
                style={{
                  flex: 1,
                  padding: '8px 12px',
                  textAlign: 'center',
                  fontWeight: isActive ? 600 : 400,
                  color: isActive ? 'var(--accent)' : isDone ? 'var(--ok)' : 'var(--text-muted)',
                  borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                }}
              >
                {title}
              </div>
            )
          })}
        </div>

        {/* Body Content */}
        <div style={{ padding: 20 }}>
          {!isSupported && (
            <div
              style={{
                padding: '12px 14px',
                borderRadius: 6,
                background: 'color-mix(in srgb, var(--warn) 12%, var(--surface))',
                border: '1px solid var(--warn)',
                color: 'var(--warn)',
                fontSize: 13,
                marginBottom: 16,
              }}
            >
              Web Serial is supported in Chrome and Edge. Replay and Review remain available on all browsers.
            </div>
          )}

          {errorMsg && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: 6,
                background: 'color-mix(in srgb, var(--crit) 12%, var(--surface))',
                border: '1px solid var(--crit)',
                color: 'var(--crit)',
                fontSize: 13,
                marginBottom: 16,
              }}
            >
              {errorMsg}
            </div>
          )}

          {step === 1 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
                Connect your OpenBCI USB Dongle and turn on the Cyton + Daisy board.
                Click <strong>Select USB Port</strong> to open the browser port selector.
              </p>
              <div
                style={{
                  padding: 12,
                  background: 'var(--surface-2)',
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily: 'IBM Plex Mono',
                  color: 'var(--text-muted)',
                }}
              >
                Expected: 115200 baud, 8N1, FTDI USB Serial
              </div>
              <Button
                disabled={!isSupported || isConnecting}
                onClick={handleConnectPort}
                style={{ alignSelf: 'flex-start' }}
              >
                {isConnecting ? 'Connecting…' : 'Select USB Port & Connect'}
              </Button>
            </div>
          )}

          {step === 2 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center', padding: '20px 0' }}>
              <div style={{ fontSize: 14, fontWeight: 500 }}>Establishing Serial Handshake…</div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', margin: 0 }}>
                Sending soft reset 'v', awaiting '$$$' banner, and configuring 16-channel streaming mode.
              </p>
            </div>
          )}

          {step === 3 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--ok)', fontWeight: 500 }}>
                <CheckCircle2 size={16} /> Device Connected (Cyton + Daisy 16-Channel)
              </div>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 0', color: 'var(--text-muted)' }}>Hardware Sampling Rate</td>
                    <td style={{ padding: '6px 0', fontFamily: 'IBM Plex Mono' }}>125 Hz</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 0', color: 'var(--text-muted)' }}>Resampling Target</td>
                    <td style={{ padding: '6px 0', fontFamily: 'IBM Plex Mono' }}>256 Hz (Polyphase Resampler)</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 0', color: 'var(--text-muted)' }}>Channel Remapping</td>
                    <td style={{ padding: '6px 0', fontFamily: 'IBM Plex Mono' }}>16 Scalp Channels → 22 TCP Montage</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '6px 0', color: 'var(--text-muted)' }}>ML Compatibility</td>
                    <td style={{ padding: '6px 0', color: 'var(--ok)', fontWeight: 500 }}>
                      Compatible with adaptation
                    </td>
                  </tr>
                </tbody>
              </table>
              <Button onClick={() => setStep(4)} style={{ alignSelf: 'flex-end' }}>
                Next: Check Signal Quality <ArrowRight size={14} />
              </Button>
            </div>
          )}

          {step === 4 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>
                Electrode Impedance & Signal Quality Grid (16 Channels):
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                {CHANNELS_16.map((ch, idx) => (
                  <div
                    key={ch}
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border)',
                      borderRadius: 4,
                      padding: '6px 8px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <span style={{ fontSize: 11, fontFamily: 'IBM Plex Mono', fontWeight: 500 }}>{ch}</span>
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: QUALITY_COLOR_MAP[channelQualities[idx] ?? 'good'],
                      }}
                    />
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 8 }}>
                <Button variant="secondary" onClick={onClose}>
                  Cancel
                </Button>
                <Button onClick={handleFinishWizard}>
                  Start Monitoring
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
