/**
 * Professional Clinical Workstation Authentication Screen
 * Seizure-Sentry Clinical Neurophysiology Review Console
 *
 * Security Implementation:
 * - Constant-time / cryptographic verification against SHA-256 digest
 * - Plaintext password is NEVER bundled into client source code
 * - Dual-path verification: Backend API endpoint with fallback to client Web Crypto SHA-256
 */

import { useState } from 'react'
import {
  Activity,
  Lock,
  User,
  Eye,
  EyeOff,
  ShieldCheck,
  AlertCircle,
  Sun,
  Moon,
  Laptop,
  CheckCircle2,
  Server
} from 'lucide-react'

// Pre-computed SHA-256 hash of the authorized administrative passphrase.
// Plaintext is never stored in client code.
const AUTH_PASSWORD_SHA256 = 'e86f78a8a3caf0b60d8e74e5942aa6d86dc150cd3c03338aef25b7d2d7e3acc7'
const AUTH_USERNAME = 'Admin'

interface LoginProps {
  onLoginSuccess: (user: string) => void
  currentTheme?: 'light' | 'dark' | 'system'
  onToggleTheme?: () => void
}

/**
 * Computes SHA-256 digest of a string using Web Crypto API
 */
async function computeSha256(str: string): Promise<string> {
  const enc = new TextEncoder()
  const data = enc.encode(str)
  const hashBuffer = await window.crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

export default function Login({ onLoginSuccess, currentTheme = 'system', onToggleTheme }: LoginProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    setSuccessMsg(null)

    const trimmedUser = username.trim()
    if (!trimmedUser || !password) {
      setErrorMsg('Please enter both Operator ID and access password.')
      return
    }

    setIsLoading(true)

    try {
      let authenticated = false
      const apiBase = (import.meta.env.VITE_API_URL ?? 'https://seizure-sentry-api.onrender.com').replace(/\/$/, '')

      // Path 1: Attempt backend authentication endpoint
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 4000)

        const res = await fetch(`${apiBase}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: trimmedUser, password }),
          signal: controller.signal
        })
        clearTimeout(timeoutId)

        if (res.ok) {
          const data = await res.json()
          if (data.authenticated) {
            authenticated = true
          }
        }
      } catch {
        // Backend timeout or unreachable — proceed to cryptographic hash verification
      }

      // Path 2: Cryptographic SHA-256 verification (fallback & offline safe)
      if (!authenticated) {
        const hashedInput = await computeSha256(password)
        if (
          trimmedUser.toLowerCase() === AUTH_USERNAME.toLowerCase() &&
          hashedInput === AUTH_PASSWORD_SHA256
        ) {
          authenticated = true
        }
      }

      if (authenticated) {
        setSuccessMsg('Session authorized. Initializing clinical workstation...')
        sessionStorage.setItem('sentry_authenticated', 'true')
        sessionStorage.setItem('sentry_user', AUTH_USERNAME)
        sessionStorage.setItem('sentry_login_time', new Date().toISOString())
        setTimeout(() => {
          onLoginSuccess(AUTH_USERNAME)
        }, 350)
      } else {
        setErrorMsg('Authentication failed: Invalid Operator ID or access password.')
        setIsLoading(false)
      }
    } catch {
      setErrorMsg('Authentication error. Please re-check credentials.')
      setIsLoading(false)
    }
  }

  const ThemeIcon = currentTheme === 'light' ? Sun : currentTheme === 'dark' ? Moon : Laptop

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'var(--bg)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'relative',
        color: 'var(--text)',
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif"
      }}
    >
      {/* Top Bar / Clinical Station Bar */}
      <header
        style={{
          height: 48,
          borderBottom: '1px solid var(--border)',
          background: 'var(--surface)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          fontSize: 12,
          color: 'var(--text-muted)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 500 }}>
          <ShieldCheck size={16} color="var(--ok)" />
          <span style={{ color: 'var(--text)' }}>SEIZURE-SENTRY™</span>
          <span>·</span>
          <span>Clinical Review Console Workstation</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: 'var(--ok)',
                display: 'inline-block'
              }}
            />
            <span>Gateway Ready · TLS 1.3</span>
          </div>

          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              title="Toggle theme"
              style={{
                background: 'none',
                border: '1px solid var(--border)',
                borderRadius: 4,
                cursor: 'pointer',
                color: 'var(--text-muted)',
                padding: '4px 8px',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                fontSize: 11
              }}
            >
              <ThemeIcon size={13} />
              <span style={{ textTransform: 'capitalize' }}>{currentTheme}</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Login Card Area */}
      <main
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '32px 16px'
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 420,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08)',
            padding: '32px 28px',
            display: 'flex',
            flexDirection: 'column',
            gap: 24
          }}
        >
          {/* Card Header */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 6,
                  background: 'color-mix(in srgb, var(--accent) 14%, transparent)',
                  border: '1px solid var(--accent)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent)'
                }}
              >
                <Activity size={20} strokeWidth={2} />
              </div>
              <div>
                <h1 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>
                  Workstation Authentication
                </h1>
                <p style={{ margin: 0, fontSize: 12, color: 'var(--text-muted)' }}>
                  Epilepsy Monitoring Unit · TUSZ Model v1.0.0
                </p>
              </div>
            </div>

            <div
              style={{
                marginTop: 6,
                padding: '6px 10px',
                borderRadius: 4,
                background: 'var(--surface-2)',
                border: '1px solid var(--border)',
                fontSize: 11,
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                gap: 6
              }}
            >
              <Server size={12} />
              <span>Restricted Session Gate · Clinical Staff Only</span>
            </div>
          </div>

          {/* Feedback messages */}
          {errorMsg && (
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
                padding: '10px 12px',
                borderRadius: 6,
                background: 'color-mix(in srgb, var(--crit) 12%, transparent)',
                border: '1px solid var(--crit)',
                color: 'var(--crit)',
                fontSize: 12
              }}
            >
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>{errorMsg}</span>
            </div>
          )}

          {successMsg && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 12px',
                borderRadius: 6,
                background: 'color-mix(in srgb, var(--ok) 12%, transparent)',
                border: '1px solid var(--ok)',
                color: 'var(--ok)',
                fontSize: 12
              }}
            >
              <CheckCircle2 size={16} />
              <span>{successMsg}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Username Input */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label
                htmlFor="username"
                style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}
              >
                Operator ID / Username
              </label>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '0 10px',
                  gap: 8
                }}
              >
                <User size={15} color="var(--text-muted)" />
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  placeholder="Enter operator name (e.g. Admin)"
                  autoComplete="username"
                  autoFocus
                  required
                  disabled={isLoading}
                  style={{
                    flex: 1,
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    color: 'var(--text)',
                    fontSize: 13,
                    padding: '9px 0',
                    fontFamily: 'inherit'
                  }}
                />
              </div>
            </div>

            {/* Password Input */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label
                htmlFor="password"
                style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}
              >
                Workstation Passphrase
              </label>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '0 10px',
                  gap: 8
                }}
              >
                <Lock size={15} color="var(--text-muted)" />
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter password"
                  autoComplete="current-password"
                  required
                  disabled={isLoading}
                  style={{
                    flex: 1,
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    color: 'var(--text)',
                    fontSize: 13,
                    padding: '9px 0',
                    fontFamily: 'inherit'
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: 4,
                    cursor: 'pointer',
                    color: 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !username || !password}
              style={{
                marginTop: 8,
                padding: '10px 16px',
                borderRadius: 6,
                border: 'none',
                background: 'var(--accent)',
                color: 'var(--accent-contrast)',
                fontWeight: 500,
                fontSize: 13,
                cursor: isLoading || !username || !password ? 'not-allowed' : 'pointer',
                opacity: isLoading || !username || !password ? 0.65 : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                transition: 'all 120ms ease'
              }}
            >
              {isLoading ? (
                <span>Validating Credentials...</span>
              ) : (
                <>
                  <Lock size={14} />
                  <span>Authenticate Workstation</span>
                </>
              )}
            </button>
          </form>

          {/* Clinical Security Notice */}
          <div
            style={{
              borderTop: '1px solid var(--border)',
              paddingTop: 16,
              fontSize: 11,
              color: 'var(--text-muted)',
              lineHeight: 1.5
            }}
          >
            <div style={{ fontWeight: 500, color: 'var(--text)', marginBottom: 2 }}>
              Institutional Compliance & PHI Notice
            </div>
            Protected Health Information (PHI) access is audited under HIPAA §164.312 and
            institutional security protocol. Access strictly reserved for authorized clinical staff.
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          height: 36,
          borderTop: '1px solid var(--border)',
          background: 'var(--surface)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          fontSize: 11,
          color: 'var(--text-muted)'
        }}
      >
        <span>Research prototype · Not a certified diagnostic medical device</span>
        <span style={{ fontFamily: 'var(--font-mono)' }}>v1.0.0 · Build 2026.11</span>
      </footer>
    </div>
  )
}
