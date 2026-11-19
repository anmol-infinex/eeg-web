/**
 * Web Audio API synthesizer for clinical EEG alerts.
 * Generates warning chimes and critical alarm pulses without external audio assets.
 */

class AlertAudioSynthesizer {
  private ctx: AudioContext | null = null
  private criticalInterval: ReturnType<typeof setInterval> | null = null
  private muted = false
  private unlocked = false

  constructor() {
    // Lazy unlock on first user pointer/touch interaction
    if (typeof window !== 'undefined') {
      const unlock = () => {
        this.ensureContext()
        if (this.ctx && this.ctx.state === 'suspended') {
          this.ctx.resume().catch(() => {})
        }
        this.unlocked = true
        window.removeEventListener('pointerdown', unlock)
        window.removeEventListener('keydown', unlock)
      }
      window.addEventListener('pointerdown', unlock, { once: true })
      window.addEventListener('keydown', unlock, { once: true })
    }
  }

  private ensureContext(): AudioContext | null {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      if (AudioCtx) {
        this.ctx = new AudioCtx()
      }
    }
    return this.ctx
  }

  public unlock(): void {
    const ctx = this.ensureContext()
    if (ctx && ctx.state === 'suspended') {
      ctx.resume().catch(() => {})
    }
    this.unlocked = true
  }

  public isMuted(): boolean {
    return this.muted
  }

  public setMuted(muted: boolean): void {
    this.muted = muted
    if (muted) {
      this.stopCritical()
    }
  }

  /**
   * Play polite 2-tone warning chime (440Hz then 660Hz, 200ms total).
   */
  public playWarning(): void {
    if (this.muted) return
    const ctx = this.ensureContext()
    if (!ctx) return
    if (ctx.state === 'suspended') ctx.resume().catch(() => {})

    try {
      const now = ctx.currentTime
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sine'
      // Two-tone chime: 440Hz for 100ms, then 660Hz for 100ms
      osc.frequency.setValueAtTime(440, now)
      osc.frequency.setValueAtTime(660, now + 0.1)

      gain.gain.setValueAtTime(0.08, now)
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now)
      osc.stop(now + 0.25)
    } catch {
      // Audio playback failure ignored gracefully
    }
  }

  /**
   * Start pulsating 880Hz critical seizure alarm (150ms on / 150ms off pulses).
   */
  public startCritical(): void {
    if (this.muted || this.criticalInterval) return
    const ctx = this.ensureContext()
    if (!ctx) return
    if (ctx.state === 'suspended') ctx.resume().catch(() => {})

    const pulse = () => {
      if (this.muted) return
      try {
        const now = ctx.currentTime
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()

        osc.type = 'triangle'
        osc.frequency.setValueAtTime(880, now)

        // 150ms beep with smooth envelope
        gain.gain.setValueAtTime(0.12, now)
        gain.gain.setValueAtTime(0.12, now + 0.12)
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15)

        osc.connect(gain)
        gain.connect(ctx.destination)

        osc.start(now)
        osc.stop(now + 0.16)
      } catch {
        // Ignored
      }
    }

    pulse()
    // Repeating 300ms cycle (150ms tone + 150ms silence)
    this.criticalInterval = setInterval(pulse, 300)
  }

  public stopCritical(): void {
    if (this.criticalInterval) {
      clearInterval(this.criticalInterval)
      this.criticalInterval = null
    }
  }

  public stopAll(): void {
    this.stopCritical()
  }
}

export const alertAudio = new AlertAudioSynthesizer()
export default alertAudio
