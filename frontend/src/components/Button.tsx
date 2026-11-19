import { clsx } from 'clsx'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'destructive'
  size?: 'sm' | 'md'
}

const VARIANTS = {
  primary:     'bg-[var(--accent)] text-[var(--accent-contrast)] border-[var(--accent)]',
  secondary:   'bg-[var(--surface-2)] text-[var(--text)] border-[var(--border)]',
  ghost:       'bg-transparent text-[var(--text)] border-transparent',
  destructive: 'bg-transparent text-[var(--crit)] border-[var(--crit)]',
}
const SIZES = { sm: 'px-3 py-1 text-xs', md: 'px-4 py-2 text-sm' }

export default function Button({
  variant = 'primary', size = 'md', className, children, disabled, ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled}
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-[6px] border font-medium transition-all duration-120',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        VARIANTS[variant], SIZES[size], className
      )}
    >
      {children}
    </button>
  )
}
