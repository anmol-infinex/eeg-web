import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useState, useEffect, useCallback } from 'react'
import {
  Activity, Monitor, Cpu, FolderOpen, Bell, Settings, BookOpen,
  ChevronLeft, ChevronRight
} from 'lucide-react'
import Overview from './pages/Overview'
import LiveMonitor from './pages/LiveMonitor'
import Devices from './pages/Devices'
import Recordings from './pages/Recordings'
import AlertsPage from './pages/Alerts'
import ModelCard from './pages/ModelCard'
import SettingsPage from './pages/Settings'
import Header from './components/Header'

type Theme = 'light' | 'dark' | 'system'

function applyTheme(t: Theme) {
  const root = document.documentElement
  if (t === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', t)
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('theme') as Theme) ?? 'system'
  })
  const [backendStatus, setBackendStatus] = useState<'ok' | 'degraded' | 'offline'>('offline')
  const location = useLocation()

  useEffect(() => {
    applyTheme(theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const cycleTheme = useCallback(() => {
    setTheme(t => t === 'light' ? 'dark' : t === 'dark' ? 'system' : 'light')
  }, [])

  const navItems = [
    { to: '/', label: 'Overview', icon: Activity },
    { to: '/live', label: 'Live monitor', icon: Monitor },
    { to: '/devices', label: 'Devices', icon: Cpu },
    { to: '/recordings', label: 'Recordings', icon: FolderOpen },
    { to: '/alerts', label: 'Alerts', icon: Bell },
    { to: '/model', label: 'Model', icon: BookOpen },
    { to: '/settings', label: 'Settings', icon: Settings },
  ]

  const pageLabel = navItems.find(n => n.to === location.pathname)?.label ?? 'Overview'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg)' }}>
      {/* Sidebar */}
      <aside style={{
        width: sidebarOpen ? 220 : 48, transition: 'width 150ms ease',
        background: 'var(--surface)', borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column', flexShrink: 0
      }}>
        <div style={{
          padding: '16px 12px', borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 8
        }}>
          {sidebarOpen && (
            <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text)', flex: 1 }}>
              Seizure-Sentry
            </span>
          )}
          <button
            onClick={() => setSidebarOpen(o => !o)}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-muted)', padding: 4, borderRadius: 4,
              display: 'flex', alignItems: 'center'
            }}
          >
            {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>
        </div>
        <nav style={{ flex: 1, padding: '8px 0' }}>
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              style={({ isActive }) => ({
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 12px', textDecoration: 'none',
                color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                background: isActive ? 'color-mix(in srgb, var(--accent) 10%, transparent)' : 'transparent',
                borderRadius: 6, margin: '1px 8px', fontSize: 13, fontWeight: isActive ? 500 : 400,
                transition: 'all 120ms'
              })}
            >
              <Icon size={16} strokeWidth={1.5} />
              {sidebarOpen && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Topbar */}
        <Header
          title={pageLabel}
          backendStatus={backendStatus}
          theme={theme}
          onToggleTheme={cycleTheme}
          sidebarOpen={sidebarOpen}
        />

        {/* Content */}
        <main style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          <Routes>
            <Route path="/" element={<Overview onBackendStatus={setBackendStatus} />} />
            <Route path="/live" element={<LiveMonitor />} />
            <Route path="/devices" element={<Devices />} />
            <Route path="/recordings" element={<Recordings />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/model" element={<ModelCard />} />
            <Route path="/settings" element={<SettingsPage onThemeChange={setTheme} currentTheme={theme} />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer style={{
          height: 36, background: 'var(--surface)', borderTop: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', padding: '0 20px',
          fontSize: 11, color: 'var(--text-muted)', flexShrink: 0
        }}>
          Research prototype. Not a medical device.
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)' }}>
            v{import.meta.env.VITE_APP_VERSION ?? '1.0.0'}
          </span>
        </footer>
      </div>
    </div>
  )
}
