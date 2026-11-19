/**
 * WebSocket client for Seizure-Sentry live session gateway (/ws/session/{session_id}).
 * Supports binary streaming (producer) and telemetry/prediction monitoring (viewer).
 */

import type { ConnectionStatus, EegPrediction, EegAlert, WsLinkStatus } from './types'

export type { ConnectionStatus, EegPrediction as Prediction, EegAlert as Alert, WsLinkStatus as WsStatus }

export class EegWsClient {
  private ws: WebSocket | null = null
  private backoffMs = 1000
  private maxBackoffMs = 30000
  private sessionId: string
  private role: 'producer' | 'viewer'
  private source: string
  private shareToken?: string
  private reconnectTimer?: ReturnType<typeof setTimeout>
  private intentionalClose = false

  public onStatusChange: (status: ConnectionStatus) => void = () => {}
  public onPrediction: (p: EegPrediction) => void = () => {}
  public onAlert: (a: EegAlert) => void = () => {}
  public onStatusUpdate: (s: WsLinkStatus) => void = () => {}
  public onEegChunk: (chunk: Float32Array) => void = () => {}

  constructor(
    sessionId = 'default',
    role: 'producer' | 'viewer' = 'viewer',
    source = 'replay',
    shareToken?: string
  ) {
    this.sessionId = sessionId
    this.role = role
    this.source = source
    this.shareToken = shareToken
  }

  public connect(key = ''): void {
    this.intentionalClose = false
    let baseUrl = (import.meta.env.VITE_WS_URL ?? '').replace(/\/$/, '')
    if (!baseUrl) {
      baseUrl = (import.meta.env.VITE_API_URL ?? '').replace(/^http/, 'ws').replace(/\/$/, '')
    }
    if (!baseUrl) {
      if (typeof window !== 'undefined' && window.location) {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        baseUrl = `${proto}//${window.location.host}`
      } else {
        baseUrl = 'ws://localhost:10000'
      }
    }

    const tokenQuery = this.shareToken ? `?token=${encodeURIComponent(this.shareToken)}` : ''
    const url = `${baseUrl}/ws/session/${encodeURIComponent(this.sessionId)}${tokenQuery}`

    this.onStatusChange('Connecting')

    try {
      this.ws = new WebSocket(url)
      this.ws.binaryType = 'arraybuffer'

      this.ws.onopen = () => {
        this.backoffMs = 1000
        const hello = {
          type: 'hello',
          key: key || this.shareToken || '',
          role: this.role,
          source: this.source,
          profile_id: this.source === 'cyton-serial' ? 'openbci-cyton-daisy' : 'simulator',
          fs: this.source === 'cyton-serial' ? 125 : 256,
          channels: [],
        }
        this.ws?.send(JSON.stringify(hello))
      }

      this.ws.onmessage = (ev) => this.handleMessage(ev)

      this.ws.onerror = () => {
        if (!this.intentionalClose) {
          this.scheduleReconnect(key)
        }
      }

      this.ws.onclose = () => {
        this.onStatusChange('Disconnected')
        if (!this.intentionalClose) {
          this.scheduleReconnect(key)
        }
      }
    } catch {
      if (!this.intentionalClose) {
        this.scheduleReconnect(key)
      }
    }
  }

  private handleMessage(ev: MessageEvent): void {
    if (typeof ev.data === 'string') {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'ready') {
          this.onStatusChange('Streaming')
        } else if (msg.type === 'prediction') {
          this.onPrediction(msg as EegPrediction)
        } else if (msg.type === 'alert') {
          this.onAlert(msg as EegAlert)
        } else if (msg.type === 'status') {
          this.onStatusUpdate(msg as WsLinkStatus)
        }
      } catch {
        // Ignore unparseable frames
      }
    } else if (ev.data instanceof ArrayBuffer) {
      // Inbound binary EEG payload (e.g., broadcast to viewer)
      if (ev.data.byteLength > 16) {
        const samples = new Float32Array(ev.data, 16)
        this.onEegChunk(samples)
      }
    }
  }

  private scheduleReconnect(key: string): void {
    if (this.reconnectTimer || this.intentionalClose) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined
      this.backoffMs = Math.min(this.backoffMs * 2, this.maxBackoffMs)
      this.connect(key)
    }, this.backoffMs)
  }

  /**
   * Pack and send 250ms binary frame (<IdHBB header + float32 payload).
   */
  public sendBinaryChunk(
    seq: number,
    t0_ms: number,
    samples: Float32Array,
    nChannels: number,
    flags = 0
  ): void {
    if (this.role === 'viewer') {
      // Viewers are prohibited from transmitting binary frames
      return
    }
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return

    const nSamples = Math.floor(samples.length / nChannels)
    const headerSize = 16 // uint32(4) + float64(8) + uint16(2) + uint8(1) + uint8(1)
    const buf = new ArrayBuffer(headerSize + samples.byteLength)
    const view = new DataView(buf)

    view.setUint32(0, seq, true)
    view.setFloat64(4, t0_ms, true)
    view.setUint16(12, nSamples, true)
    view.setUint8(14, nChannels)
    view.setUint8(15, flags & 0xFF)

    new Float32Array(buf, headerSize).set(samples)
    this.ws.send(buf)
  }

  public disconnect(): void {
    this.intentionalClose = true
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.onStatusChange('Disconnected')
  }
}

export default EegWsClient
