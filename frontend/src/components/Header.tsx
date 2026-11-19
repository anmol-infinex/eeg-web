/**
 * Header topbar component displaying page title, connection status chip,
 * backend health status chip, and theme toggle.
 */

import { Sun, Moon, Laptop } from 'lucide-react'
import StatusChip from './StatusChip'
import type { ConnectionStatus } from '../lib/types'

interface HeaderProps {
  title?: string
  connectionStatus?: ConnectionStatus
  backendStatus?: 'ok' | 'degraded' | 'offline'
  theme?: 'light' | 'dark' | 'system'
  onToggleTheme?: () => void
  sidebarOpen?: boolean
}

const CONN_COLORS: Record<ConnectionStatus, { bg: string; color: string; border: string }> = {
  Streaming:    { bg: 'color-mix(in srgb, var(--ok) 12%, transparent)',   color: 'var(--ok)',   border: 'var(--ok)' },
  Simulated:    { bg: 'color-mix(in srgb, var(--sim) 12%, transparent)',  color: 'var(--sim)',  border: 'var(--sim)' },
  Connecting:   { bg: 'color-mix(in srgb, var(--warn) 12%, transparent)', color: 'var(--warn)', border: 'var(--warn)' },
  Disconnected: { bg: 'var(--surface-2)',                                  color: 'var(--text-muted)', border: 'var(--border)' },
}

export default function Header({
  title = 'Overview',
  connectionStatus,
  backendStatus = 'offline',
  theme = 'system',
  onToggleTheme,
  sidebarOpen = true,
}: HeaderProps) {
  const ThemeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Laptop
  const connStyle = connectionStatus ? CONN_COLORS[connectionStatus] : null

  return (
    <header
      style={{
        height: 48,
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: 12,
        flexShrink: 0,
      }}
    >
      <span style={{ flex: 1, fontSize: 13, color: 'var(--text-muted)', fontWeight: 500 }}>
        {title}
      </span>

      {connectionStatus && connStyle && (
        <span
          style={{
            fontSize: 11,
            fontWeight: 500,
            padding: '2px 8px',
            borderRadius: 4,
            border: `1px solid ${connStyle.border}`,
            background: connStyle.bg,
            color: connStyle.color,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 5,
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

      <StatusChip status={backendStatus} />

      {onToggleTheme && (
        <button
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          style={{
            background: 'none',
            border: '1px solid var(--border)',
            borderRadius: 6,
            cursor: 'pointer',
            color: 'var(--text-muted)',
            padding: '4px 8px',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 12,
          }}
        >
          <ThemeIcon size={14} />
          {sidebarOpen && <span style={{ textTransform: 'capitalize' }}>{theme}</span>}
        </button>
      )}
    </header>
  )
}
