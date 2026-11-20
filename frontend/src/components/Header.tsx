/**
 * Header: Precision clinical workstation top bar.
 * Engineered for expert human-built ergonomics:
 * - Uniform 28px control height across all interactive elements
 * - Crisp 4px rectangular corner geometry (no bubbly AI pills)
 * - Cohesive telemetry chip & icon-only theme toggle
 * - Integrated operator profile badge & lock workstation control
 */

import { Sun, Moon, Laptop, LogOut } from 'lucide-react'
import StatusChip from './StatusChip'
import type { ConnectionStatus } from '../lib/types'

interface HeaderProps {
  title?: string
  connectionStatus?: ConnectionStatus
  backendStatus?: 'ok' | 'degraded' | 'offline'
  theme?: 'light' | 'dark' | 'system'
  onToggleTheme?: () => void
  sidebarOpen?: boolean
  currentUser?: string
  onLogout?: () => void
}

const CONN_STYLES: Record<ConnectionStatus, { bg: string; color: string; border: string }> = {
  Streaming:    { bg: 'color-mix(in srgb, var(--ok) 10%, transparent)',   color: 'var(--ok)',   border: 'color-mix(in srgb, var(--ok) 30%, transparent)' },
  Simulated:    { bg: 'color-mix(in srgb, var(--sim) 10%, transparent)',  color: 'var(--sim)',  border: 'color-mix(in srgb, var(--sim) 30%, transparent)' },
  Connecting:   { bg: 'color-mix(in srgb, var(--warn) 10%, transparent)', color: 'var(--warn)', border: 'color-mix(in srgb, var(--warn) 30%, transparent)' },
  Disconnected: { bg: 'var(--surface-2)',                                  color: 'var(--text-muted)', border: 'var(--border)' },
}

export default function Header({
  title = 'Overview',
  connectionStatus,
  backendStatus = 'offline',
  theme = 'system',
  onToggleTheme,
  currentUser = 'Admin',
  onLogout,
}: HeaderProps) {
  const ThemeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Laptop
  const connStyle = connectionStatus ? CONN_STYLES[connectionStatus] : null

  return (
    <header
      style={{
        height: 48,
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: 10,
        flexShrink: 0,
        userSelect: 'none'
      }}
    >
      {/* Title & Section Breadcrumb */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Console
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>/</span>
        <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 600 }}>
          {title}
        </span>
      </div>

      {/* Right-aligned Telemetry & Controls cluster */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Connection Telemetry Badge (if active) */}
        {connectionStatus && connStyle && (
          <span
            style={{
              height: 28,
              padding: '0 9px',
              borderRadius: 4,
              border: `1px solid ${connStyle.border}`,
              background: connStyle.bg,
              color: connStyle.color,
              fontSize: 11,
              fontWeight: 500,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: connStyle.color,
              }}
            />
            {connectionStatus}
          </span>
        )}

        {/* System / API Health Status */}
        <StatusChip status={backendStatus} />

        {/* Square Theme Switcher Button */}
        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            title={`Active theme: ${theme}. Click to switch.`}
            aria-label="Toggle theme"
            style={{
              width: 28,
              height: 28,
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              cursor: 'pointer',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 0,
              transition: 'all 120ms ease'
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = 'var(--text)'
              e.currentTarget.style.borderColor = 'var(--text-muted)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = 'var(--text-muted)'
              e.currentTarget.style.borderColor = 'var(--border)'
            }}
          >
            <ThemeIcon size={14} strokeWidth={1.75} />
          </button>
        )}

        {/* Subtle Separator */}
        <div
          style={{
            width: 1,
            height: 18,
            background: 'var(--border)',
            margin: '0 2px'
          }}
        />

        {/* Operator Profile Capsule */}
        {currentUser && (
          <div
            style={{
              height: 28,
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '0 8px',
              display: 'flex',
              alignItems: 'center',
              gap: 6
            }}
          >
            {/* Monogram avatar */}
            <div
              style={{
                width: 18,
                height: 18,
                borderRadius: 3,
                background: 'var(--accent)',
                color: 'var(--accent-contrast)',
                fontSize: 10,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                lineHeight: 1
              }}
            >
              {currentUser.charAt(0).toUpperCase()}
            </div>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>
              {currentUser}
            </span>
            <span
              style={{
                fontSize: 10,
                color: 'var(--text-muted)',
                paddingLeft: 4,
                borderLeft: '1px solid var(--border)',
                lineHeight: 1
              }}
            >
              EMU
            </span>
          </div>
        )}

        {/* Lock Workstation Button */}
        {onLogout && (
          <button
            onClick={onLogout}
            title="Lock Clinical Workstation"
            aria-label="Lock Workstation"
            style={{
              height: 28,
              padding: '0 9px',
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              cursor: 'pointer',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 11,
              fontWeight: 500,
              transition: 'all 120ms ease'
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = 'var(--crit)'
              e.currentTarget.style.borderColor = 'var(--crit)'
              e.currentTarget.style.background = 'color-mix(in srgb, var(--crit) 10%, transparent)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = 'var(--text-muted)'
              e.currentTarget.style.borderColor = 'var(--border)'
              e.currentTarget.style.background = 'var(--surface-2)'
            }}
          >
            <LogOut size={12} strokeWidth={1.75} />
            <span>Lock</span>
          </button>
        )}
      </div>
    </header>
  )
}
