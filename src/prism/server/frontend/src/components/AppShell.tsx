import { startTransition, useEffect, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const [quickGene, setQuickGene] = useState('')

  useEffect(() => {
    if (!location.pathname.startsWith('/gene/')) {
      return
    }
    setQuickGene(decodeURIComponent(location.pathname.replace('/gene/', '')))
  }, [location.pathname])

  function handleQuickOpen(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const resolved = quickGene.trim()
    if (!resolved) {
      return
    }
    startTransition(() => {
      navigate(`/gene/${encodeURIComponent(resolved)}`)
    })
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__brand">
          <Link className="brand" to="/">
            <span className="brand__mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" className="brand__glyph">
                <path
                  d="M12 3 4 7v10l8 4 8-4V7l-8-4Zm0 2.045 5.75 2.875L12 10.795 6.25 7.92 12 5.045Zm-6.25 4.48 5.5 2.75v6.18l-5.5-2.75v-6.18Zm7 8.93v-6.18l5.5-2.75v6.18l-5.5 2.75Z"
                  fill="currentColor"
                />
              </svg>
            </span>
            <span>
              <strong>PRISM Server</strong>
              <small>Decoupled analysis workspace</small>
            </span>
          </Link>
        </div>
        <nav className="topbar__nav" aria-label="Primary">
          <Link
            className={location.pathname === '/' ? 'nav-link nav-link--active' : 'nav-link'}
            to="/"
          >
            Dashboard
          </Link>
        </nav>
        <form className="quick-open" onSubmit={handleQuickOpen}>
          <label className="quick-open__field">
            <span>Gene Lookup</span>
            <input
              onChange={(event) => setQuickGene(event.target.value)}
              placeholder="Enter gene name or index"
              type="text"
              value={quickGene}
            />
          </label>
          <button className="button button--primary" type="submit">
            Open
          </button>
        </form>
      </header>
      <main className="page-frame">{children}</main>
    </div>
  )
}
