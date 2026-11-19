/**
 * Web Serial API manager and OpenBCI 33-byte packet parser for Cyton + Daisy.
 * Conforms to EegSource interface and matches architectural specifications.
 */

import type { EegSource, SampleChunk, ConnectionStatus } from './types'

export const ADS1299_SCALE_UV = 0.02235174 // 4.5 / (24 * (2^23 - 1)) * 1e6 uV/count
export const OPENBCI_PACKET_LEN = 33
export const HEADER_BYTE = 0xA0
export const SAMPLES_PER_FRAME_125HZ = 31 // 248ms ~ 250ms

export interface SerialPortOptions {
  baudRate: number
  dataBits?: 7 | 8
  stopBits?: 1 | 2
  parity?: 'none' | 'even' | 'odd'
  bufferSize?: number
}

export class OpenBciPacketParser {
  private buffer: Uint8Array = new Uint8Array(0)
  private lastSampleCounter: number | null = null
  private droppedPackets = 0
  private hadGap = false

  // Daisy pairing state: channels 0-7 from board (odd), channels 8-15 from daisy (even)
  private pendingBoardChannels: Float32Array | null = null

  public parseChunk(
    newData: Uint8Array,
    onSample16ch: (sample16: Float32Array, hadDrop: boolean) => void
  ): void {
    // Append new data to internal buffer
    const combined = new Uint8Array(this.buffer.length + newData.length)
    combined.set(this.buffer)
    combined.set(newData, this.buffer.length)
    this.buffer = combined

    let offset = 0
    while (offset + OPENBCI_PACKET_LEN <= this.buffer.length) {
      // Look for header byte 0xA0
      if (this.buffer[offset] !== HEADER_BYTE) {
        offset++
        continue
      }

      // Check stop byte 0xC0..0xC6
      const stopByte = this.buffer[offset + 32]
      if (stopByte < 0xC0 || stopByte > 0xC6) {
        // Corrupt framing, advance by 1 to re-sync
        offset++
        continue
      }

      // Valid 33-byte packet found
      const pkt = this.buffer.subarray(offset, offset + OPENBCI_PACKET_LEN)
      const counter = pkt[1]

      // Check continuity
      let isGap = false
      if (this.lastSampleCounter !== null) {
        const expected = (this.lastSampleCounter + 1) % 256
        if (counter !== expected) {
          const missed = (counter - expected + 256) % 256
          this.droppedPackets += missed
          this.hadGap = true
          isGap = true
        }
      }
      this.lastSampleCounter = counter

      // Extract 8 24-bit channels
      const channels8 = new Float32Array(8)
      for (let ch = 0; ch < 8; ch++) {
        const bOffset = 2 + ch * 3
        const b0 = pkt[bOffset]
        const b1 = pkt[bOffset + 1]
        const b2 = pkt[bOffset + 2]
        let raw24 = (b0 << 16) | (b1 << 8) | b2
        if (raw24 & 0x800000) {
          raw24 -= 0x1000000
        }
        channels8[ch] = raw24 * ADS1299_SCALE_UV
      }

      // Daisy pairing: odd counter = board (ch 1-8), even counter = daisy (ch 9-16)
      if (counter % 2 === 1) {
        this.pendingBoardChannels = channels8
      } else {
        if (this.pendingBoardChannels) {
          const sample16 = new Float32Array(16)
          sample16.set(this.pendingBoardChannels, 0)
          sample16.set(channels8, 8)
          this.pendingBoardChannels = null

          const dropFlag = this.hadGap || isGap
          this.hadGap = false
          onSample16ch(sample16, dropFlag)
        }
      }

      offset += OPENBCI_PACKET_LEN
    }

    // Keep remainder in buffer
    if (offset < this.buffer.length) {
      this.buffer = this.buffer.slice(offset)
    } else {
      this.buffer = new Uint8Array(0)
    }
  }

  public getDroppedCount(): number {
    return this.droppedPackets
  }

  public reset(): void {
    this.buffer = new Uint8Array(0)
    this.lastSampleCounter = null
    this.pendingBoardChannels = null
    this.hadGap = false
    this.droppedPackets = 0
  }
}

export class WebSerialSource implements EegSource {
  public readonly id = 'cyton-serial'
  public readonly profileId = 'openbci-cyton-daisy'

  private port: any = null
  private reader: any = null
  private parser = new OpenBciPacketParser()
  private status: ConnectionStatus = 'Disconnected'
  private chunkListeners: ((chunk: SampleChunk) => void)[] = []
  private statusListeners: ((status: ConnectionStatus) => void)[] = []

  // 250ms batching buffer (31 samples @ 125Hz)
  private batchSamples: Float32Array[] = []
  private seq = 0
  private frameHasDrop = false
  private isReading = false

  public isSupported(): boolean {
    return typeof navigator !== 'undefined' && 'serial' in navigator
  }

  public onChunk(cb: (chunk: SampleChunk) => void): void {
    this.chunkListeners.push(cb)
  }

  public onStatus(cb: (status: ConnectionStatus) => void): void {
    this.statusListeners.push(cb)
    cb(this.status)
  }

  private setStatus(status: ConnectionStatus): void {
    this.status = status
    for (const cb of this.statusListeners) cb(status)
  }

  public async connect(): Promise<void> {
    if (!this.isSupported()) {
      throw new Error('Web Serial is not supported in this browser. Please use Chrome or Edge.')
    }

    this.setStatus('Connecting')
    try {
      // Request serial port from user
      this.port = await (navigator as any).serial.requestPort()
      await this.port.open({
        baudRate: 115200,
        dataBits: 8,
        stopBits: 1,
        parity: 'none',
        bufferSize: 16384,
      })

      // Send soft reset 'v' and await '$$$' banner
      await this.writeString('v')
      await this.waitForBanner('$$$', 3000)

      // Set 16-channel mode 'C'
      await this.writeString('C')
      await new Promise(r => setTimeout(r, 200))

      // Start streaming 'b'
      await this.writeString('b')

      this.setStatus('Streaming')
      this.startReading()
    } catch (err: any) {
      await this.disconnect()
      throw err
    }
  }

  private async writeString(str: string): Promise<void> {
    if (!this.port || !this.port.writable) return
    const writer = this.port.writable.getWriter()
    const encoder = new TextEncoder()
    try {
      await writer.write(encoder.encode(str))
    } finally {
      writer.releaseLock()
    }
  }

  private async waitForBanner(banner: string, timeoutMs: number): Promise<void> {
    if (!this.port || !this.port.readable) return
    const reader = this.port.readable.getReader()
    const decoder = new TextDecoder()
    let accum = ''
    const start = Date.now()

    try {
      while (Date.now() - start < timeoutMs) {
        const { value, done } = await Promise.race([
          reader.read(),
          new Promise<{ value: undefined; done: boolean }>((_, rej) =>
            setTimeout(() => rej(new Error('Handshake timeout')), timeoutMs)
          ),
        ])
        if (done) break
        if (value) {
          accum += decoder.decode(value, { stream: true })
          if (accum.includes(banner)) {
            return
          }
        }
      }
    } catch {
      // If banner wait times out, continue gracefully
    } finally {
      reader.releaseLock()
    }
  }

  private async startReading(): Promise<void> {
    if (!this.port || !this.port.readable) return
    this.isReading = true
    this.parser.reset()
    this.batchSamples = []
    this.frameHasDrop = false

    try {
      while (this.port && this.port.readable && this.isReading) {
        this.reader = this.port.readable.getReader()
        try {
          while (this.isReading) {
            const { value, done } = await this.reader.read()
            if (done) break
            if (value) {
              this.parser.parseChunk(value, (sample16, hadDrop) => {
                if (hadDrop) this.frameHasDrop = true
                this.batchSamples.push(sample16)

                if (this.batchSamples.length >= SAMPLES_PER_FRAME_125HZ) {
                  this.emitBatch()
                }
              })
            }
          }
        } finally {
          this.reader.releaseLock()
          this.reader = null
        }
      }
    } catch (err) {
      if (this.isReading) {
        this.setStatus('Disconnected')
      }
    }
  }

  private emitBatch(): void {
    const nSamples = this.batchSamples.length
    const flatSamples = new Float32Array(nSamples * 16)
    for (let s = 0; s < nSamples; s++) {
      flatSamples.set(this.batchSamples[s], s * 16)
    }

    const chunk: SampleChunk = {
      seq: this.seq++,
      t0_ms: Date.now(),
      samples: flatSamples,
      channels: ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8'],
      fs: 125,
      flags: this.frameHasDrop ? 0x01 : 0x00,
    }

    this.batchSamples = []
    this.frameHasDrop = false

    for (const cb of this.chunkListeners) {
      cb(chunk)
    }
  }

  public async disconnect(): Promise<void> {
    this.isReading = false
    try {
      if (this.port && this.port.writable) {
        try {
          // Send stop streaming 's'
          await this.writeString('s')
        } catch { /* ignore */ }
      }
      if (this.reader) {
        await this.reader.cancel().catch(() => {})
        this.reader.releaseLock()
        this.reader = null
      }
      if (this.port) {
        await this.port.close().catch(() => {})
        this.port = null
      }
    } finally {
      this.setStatus('Disconnected')
    }
  }
}
