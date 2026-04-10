import type { PropsWithChildren, ReactNode } from 'react'

export interface PanelProps extends PropsWithChildren {
  title: string
  description?: string
  eyebrow?: string
  actions?: ReactNode
  className?: string
}

export interface StatItem {
  label: string
  value: string
}

export interface DescriptionItem {
  label: string
  value: string
}

export interface ChipItem {
  label: string
  tone?: 'neutral' | 'info' | 'warning' | 'success'
}

export interface NoticeProps {
  title: string
  message: string
  tone?: 'info' | 'error' | 'success'
}

export interface EmptyStateProps {
  title: string
  description: string
  action?: ReactNode
}

export function Panel({
  title,
  description,
  eyebrow,
  actions,
  className,
  children,
}: PanelProps) {
  return (
    <section className={joinClassNames('panel', className)}>
      <header className="panel__header">
        <div className="panel__copy">
          {eyebrow ? <p className="panel__eyebrow">{eyebrow}</p> : null}
          <h2 className="panel__title">{title}</h2>
          {description ? <p className="panel__description">{description}</p> : null}
        </div>
        {actions ? <div className="panel__actions">{actions}</div> : null}
      </header>
      <div className="panel__body">{children}</div>
    </section>
  )
}

export function StatGrid({ items }: { items: StatItem[] }) {
  return (
    <div className="stat-grid">
      {items.map((item) => (
        <article className="stat-card" key={`${item.label}:${item.value}`}>
          <span className="stat-card__label">{item.label}</span>
          <strong className="stat-card__value">{item.value}</strong>
        </article>
      ))}
    </div>
  )
}

export function DescriptionList({ items }: { items: DescriptionItem[] }) {
  const filteredItems = items.filter((item) => item.value)
  if (filteredItems.length === 0) {
    return null
  }
  return (
    <dl className="detail-list">
      {filteredItems.map((item) => (
        <div className="detail-list__row" key={`${item.label}:${item.value}`}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function ChipRow({ items }: { items: ChipItem[] }) {
  if (items.length === 0) {
    return null
  }
  return (
    <div className="chip-row">
      {items.map((item) => (
        <span
          className={joinClassNames('chip', `chip--${item.tone ?? 'neutral'}`)}
          key={item.label}
        >
          {item.label}
        </span>
      ))}
    </div>
  )
}

export function Notice({ title, message, tone = 'info' }: NoticeProps) {
  return (
    <section className={joinClassNames('notice', `notice--${tone}`)} role="alert">
      <div className="notice__title">{title}</div>
      <p className="notice__message">{message}</p>
    </section>
  )
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <section className="empty-state">
      <div className="empty-state__icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" className="empty-state__glyph">
          <path
            d="M4 7.5A2.5 2.5 0 0 1 6.5 5h11A2.5 2.5 0 0 1 20 7.5v9A2.5 2.5 0 0 1 17.5 19h-11A2.5 2.5 0 0 1 4 16.5v-9Zm2.5-.75a.75.75 0 0 0-.75.75v9c0 .414.336.75.75.75h11a.75.75 0 0 0 .75-.75v-9a.75.75 0 0 0-.75-.75h-11Zm2.25 2.5h6.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1 0-1.5Zm0 3.5h3.5a.75.75 0 0 1 0 1.5h-3.5a.75.75 0 0 1 0-1.5Z"
            fill="currentColor"
          />
        </svg>
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action ? <div className="empty-state__action">{action}</div> : null}
    </section>
  )
}

export function LoadingBlock({
  title = 'Loading',
  description = 'Fetching the latest workspace state.',
}: {
  title?: string
  description?: string
}) {
  return (
    <section className="loading-block" aria-live="polite">
      <div className="loading-block__spinner" aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </section>
  )
}

export function FigurePanel({
  title,
  description,
  src,
}: {
  title: string
  description: string
  src: string
}) {
  return (
    <article className="figure-panel">
      <header className="figure-panel__header">
        <h3>{title}</h3>
        <p>{description}</p>
      </header>
      <img alt={title} className="figure-panel__image" loading="lazy" src={src} />
    </article>
  )
}

function joinClassNames(...classNames: Array<string | undefined>) {
  return classNames.filter(Boolean).join(' ')
}
