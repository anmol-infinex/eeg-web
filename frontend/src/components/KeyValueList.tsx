interface Props {
  items: { label: string; value: string | React.ReactNode }[]
}

export default function KeyValueList({ items }: Props) {
  return (
    <dl style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
      {items.map(({ label, value }) => (
        <div key={label} style={{ display: 'flex', gap: 16, padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
          <dt style={{ width: 180, fontSize: 12, color: 'var(--text-muted)', flexShrink: 0 }}>{label}</dt>
          <dd style={{ margin: 0, fontSize: 13, color: 'var(--text)', fontFamily: typeof value === 'string' && /\d/.test(value) ? 'IBM Plex Mono' : 'inherit' }}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}
