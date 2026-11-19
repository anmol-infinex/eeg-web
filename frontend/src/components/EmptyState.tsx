interface Props {
  icon: React.ReactNode
  title: string
  description: string
  action?: { label: string; onClick: () => void }
}

export default function EmptyState({ icon, title, description, action }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '64px 24px', gap: 12, textAlign: 'center' }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{icon}</div>
      <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>{title}</div>
      <div style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 320 }}>{description}</div>
      {action && (
        <button onClick={action.onClick} style={{ marginTop: 8, padding: '6px 16px', borderRadius: 6, background: 'var(--accent)', color: 'var(--accent-contrast)', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500 }}>
          {action.label}
        </button>
      )}
    </div>
  )
}
