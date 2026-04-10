import { startTransition, useDeferredValue, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { apiClient } from '../api/client'
import type { BrowseScope, BrowseSort, LoadContextPayload } from '../api/types'
import {
  ChipRow,
  DescriptionList,
  EmptyState,
  LoadingBlock,
  Notice,
  Panel,
  StatGrid,
} from '../components/ui'
import { useAsyncResource } from '../hooks/useAsyncResource'
import { useContextSnapshot } from '../hooks/useContextSnapshot'
import {
  formatDecimal,
  formatInteger,
  formatOptionalNumber,
  formatPercent,
} from '../lib/format'

const BROWSE_SCOPE_OPTIONS: Array<{ value: BrowseScope; label: string }> = [
  { value: 'auto', label: 'Auto' },
  { value: 'fitted', label: 'Checkpoint genes' },
  { value: 'all', label: 'All genes' },
]

const BROWSE_SORT_OPTIONS: Array<{ value: BrowseSort; label: string }> = [
  { value: 'total_count', label: 'Total count' },
  { value: 'detected_cells', label: 'Detected cells' },
  { value: 'detected_fraction', label: 'Detected fraction' },
  { value: 'gene_name', label: 'Gene name' },
  { value: 'gene_index', label: 'Gene index' },
]

const EMPTY_FORM: LoadContextPayload = {
  h5adPath: '',
  ckptPath: '',
  layer: '',
}

export default function DashboardPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const searchKey = searchParams.toString()
  const {
    snapshot,
    error: contextError,
    isLoading: isContextLoading,
    isSubmitting,
    loadContext,
  } = useContextSnapshot()
  const [loadForm, setLoadForm] = useState<LoadContextPayload>(EMPTY_FORM)

  const browseQuery = searchParams.get('q') ?? ''
  const browseScope = readBrowseScope(searchParams.get('scope'))
  const browseSort = readBrowseSort(searchParams.get('sort'))
  const browseDirection = searchParams.get('dir') === 'asc' ? 'asc' : 'desc'
  const browsePage = readPositiveInteger(searchParams.get('page'), 1)
  const [browseQueryInput, setBrowseQueryInput] = useState(browseQuery)
  const deferredBrowseQuery = useDeferredValue(browseQueryInput)

  useEffect(() => {
    setBrowseQueryInput(browseQuery)
  }, [browseQuery])

  useEffect(() => {
    if (!snapshot?.loaded || snapshot.dataset == null) {
      return
    }
    setLoadForm({
      h5adPath: snapshot.dataset.h5adPath,
      ckptPath: snapshot.checkpoint?.ckptPath ?? '',
      layer: snapshot.dataset.layer === '(X)' ? '' : snapshot.dataset.layer,
    })
  }, [snapshot?.contextKey, snapshot?.loaded, snapshot?.dataset, snapshot?.checkpoint])

  useEffect(() => {
    if (deferredBrowseQuery === browseQuery) {
      return
    }
    const nextSearchParams = new URLSearchParams(searchKey)
    if (deferredBrowseQuery) {
      nextSearchParams.set('q', deferredBrowseQuery)
    } else {
      nextSearchParams.delete('q')
    }
    nextSearchParams.set('page', '1')
    startTransition(() => {
      setSearchParams(nextSearchParams, { replace: true })
    })
  }, [browseQuery, deferredBrowseQuery, searchKey, setSearchParams])

  const browseResult = useAsyncResource(
    (signal) =>
      apiClient.browseGenes(
        {
          query: browseQuery,
          scope: browseScope,
          sortBy: browseSort,
          direction: browseDirection,
          page: browsePage,
        },
        { signal },
      ),
    [
      snapshot?.contextKey ?? 'unloaded',
      browseQuery,
      browseScope,
      browseSort,
      browseDirection,
      String(browsePage),
    ].join(':'),
    { enabled: Boolean(snapshot?.loaded) },
  )

  async function handleLoadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await loadContext(loadForm)
  }

  function updateBrowseParam(key: string, value: string) {
    const nextSearchParams = new URLSearchParams(searchParams)
    if (value) {
      nextSearchParams.set(key, value)
    } else {
      nextSearchParams.delete(key)
    }
    if (key !== 'page') {
      nextSearchParams.set('page', '1')
    }
    startTransition(() => {
      setSearchParams(nextSearchParams)
    })
  }

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div className="hero-panel__copy">
          <p className="hero-panel__eyebrow">FastAPI + React Workspace</p>
          <h1>Single-context gene analysis without server-rendered UI coupling.</h1>
          <p className="hero-panel__description">
            Load one dataset, inspect checkpoint readiness, browse genes, and jump into
            deep posterior or kBulk workflows through a stable API contract.
          </p>
          <ChipRow
            items={[
              { label: 'Data-dense dashboard', tone: 'info' },
              { label: 'URL-driven browse state', tone: 'neutral' },
              { label: 'Checkpoint-aware analysis', tone: 'warning' },
            ]}
          />
        </div>
        <div className="hero-panel__metrics">
          <div className="hero-metric">
            <span>Context</span>
            <strong>{snapshot?.loaded ? 'Loaded' : 'Idle'}</strong>
          </div>
          <div className="hero-metric">
            <span>API</span>
            <strong>/api/context · /api/gene-analysis</strong>
          </div>
          <div className="hero-metric">
            <span>Frontend</span>
            <strong>React + Vite + TypeScript</strong>
          </div>
        </div>
      </section>

      {contextError ? (
        <Notice
          title="Context request failed"
          message={contextError}
          tone="error"
        />
      ) : null}

      <Panel
        title="Load Workspace"
        description="Attach a local dataset and an optional checkpoint. The active context is shared across dashboard and gene workspaces."
        eyebrow="Workspace"
      >
        <form className="form-grid" onSubmit={handleLoadSubmit}>
          <label className="field field--wide">
            <span>Dataset (.h5ad)</span>
            <input
              onChange={(event) =>
                setLoadForm((current) => ({ ...current, h5adPath: event.target.value }))
              }
              placeholder="/path/to/data.h5ad"
              type="text"
              value={loadForm.h5adPath}
            />
          </label>
          <label className="field field--wide">
            <span>Checkpoint (optional)</span>
            <input
              onChange={(event) =>
                setLoadForm((current) => ({ ...current, ckptPath: event.target.value }))
              }
              placeholder="/path/to/model.ckpt"
              type="text"
              value={loadForm.ckptPath}
            />
          </label>
          <label className="field">
            <span>Layer (optional)</span>
            <input
              onChange={(event) =>
                setLoadForm((current) => ({ ...current, layer: event.target.value }))
              }
              placeholder="counts"
              type="text"
              value={loadForm.layer}
            />
          </label>
          <div className="button-row">
            <button className="button button--primary" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Loading…' : 'Load Context'}
            </button>
          </div>
        </form>
      </Panel>

      {isContextLoading ? (
        <LoadingBlock
          title="Loading context"
          description="Reading the current backend context snapshot."
        />
      ) : null}

      {!snapshot?.loaded || snapshot.dataset == null ? (
        <EmptyState
          title="No dataset is loaded yet"
          description="Once you load a dataset, this dashboard will expose dataset summary cards, checkpoint readiness, and a filterable gene browser."
        />
      ) : (
        <>
          <div className="panel-grid panel-grid--two">
            <Panel
              title="Dataset Snapshot"
              description="Dataset size, label coverage, and expression depth summary from the active analysis context."
              eyebrow="Dataset"
            >
              <ChipRow
                items={
                  snapshot.dataset.labelKeys.length > 0
                    ? snapshot.dataset.labelKeys.map((key) => ({
                        label: key,
                        tone: 'info' as const,
                      }))
                    : [{ label: 'No label columns detected', tone: 'neutral' as const }]
                }
              />
              <StatGrid
                items={[
                  { label: 'Cells', value: formatInteger(snapshot.dataset.nCells) },
                  { label: 'Genes', value: formatInteger(snapshot.dataset.nGenes) },
                  { label: 'Layer', value: snapshot.dataset.layer },
                  {
                    label: 'Mean total count',
                    value: formatDecimal(snapshot.dataset.totalCountMean, 2),
                  },
                  {
                    label: 'Median total count',
                    value: formatDecimal(snapshot.dataset.totalCountMedian, 2),
                  },
                  {
                    label: 'P99 total count',
                    value: formatDecimal(snapshot.dataset.totalCountP99, 2),
                  },
                ]}
              />
              <DescriptionList
                items={[
                  { label: 'Dataset path', value: snapshot.dataset.h5adPath },
                  { label: 'Context key', value: snapshot.contextKey ?? '-' },
                ]}
              />
            </Panel>

            <Panel
              title="Checkpoint Readiness"
              description="Checkpoint overlap, prior availability, and reference summary for checkpoint-backed analysis."
              eyebrow="Model"
            >
              {snapshot.checkpoint == null ? (
                <EmptyState
                  title="No checkpoint loaded"
                  description="Raw browsing and on-demand fit are still available. Load a checkpoint to unlock checkpoint posterior and kBulk comparisons."
                />
              ) : (
                <>
                  <ChipRow
                    items={[
                      {
                        label: snapshot.checkpoint.hasGlobalPrior
                          ? 'Global prior ready'
                          : 'Global prior missing',
                        tone: snapshot.checkpoint.hasGlobalPrior ? 'success' : 'warning',
                      },
                      {
                        label: `${formatInteger(snapshot.checkpoint.nLabelPriors)} label priors`,
                        tone: 'info',
                      },
                      {
                        label: snapshot.checkpoint.distribution,
                        tone: 'neutral',
                      },
                    ]}
                  />
                  <StatGrid
                    items={[
                      {
                        label: 'Checkpoint genes',
                        value: formatInteger(snapshot.checkpoint.geneCount),
                      },
                      {
                        label: 'Reference overlap',
                        value: `${formatInteger(
                          snapshot.checkpoint.nOverlapReferenceGenes,
                        )} / ${formatInteger(snapshot.checkpoint.nReferenceGenes)}`,
                      },
                      {
                        label: 'Scale',
                        value: formatOptionalNumber(snapshot.checkpoint.scale, 3),
                      },
                      {
                        label: 'Mean reference',
                        value: formatOptionalNumber(
                          snapshot.checkpoint.meanReferenceCount,
                          3,
                        ),
                      },
                    ]}
                  />
                  <DescriptionList
                    items={[
                      { label: 'Checkpoint path', value: snapshot.checkpoint.ckptPath },
                      {
                        label: 'Suggested label key',
                        value: snapshot.checkpoint.suggestedLabelKey ?? '-',
                      },
                      {
                        label: 'Label preview',
                        value:
                          snapshot.checkpoint.labelPreview.length > 0
                            ? snapshot.checkpoint.labelPreview.join(', ')
                            : '-',
                      },
                    ]}
                  />
                </>
              )}
            </Panel>
          </div>

          <Panel
            title="Gene Browser"
            description="Filter and rank genes against the active dataset. Browse state is preserved in the URL for direct links."
            eyebrow="Browse"
            actions={
              <button
                className="button button--secondary"
                onClick={browseResult.reload}
                type="button"
              >
                Refresh table
              </button>
            }
          >
            <div className="toolbar">
              <label className="field field--wide">
                <span>Substring search</span>
                <input
                  onChange={(event) => setBrowseQueryInput(event.target.value)}
                  placeholder="Filter by gene name"
                  type="search"
                  value={browseQueryInput}
                />
              </label>
              <label className="field">
                <span>Scope</span>
                <select
                  onChange={(event) => updateBrowseParam('scope', event.target.value)}
                  value={browseScope}
                >
                  {BROWSE_SCOPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Sort by</span>
                <select
                  onChange={(event) => updateBrowseParam('sort', event.target.value)}
                  value={browseSort}
                >
                  {BROWSE_SORT_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Direction</span>
                <select
                  onChange={(event) => updateBrowseParam('dir', event.target.value)}
                  value={browseDirection}
                >
                  <option value="desc">Descending</option>
                  <option value="asc">Ascending</option>
                </select>
              </label>
            </div>

            {browseResult.error ? (
              <Notice title="Browse request failed" message={browseResult.error} tone="error" />
            ) : null}

            {browseResult.isLoading || browseResult.data == null ? (
              <LoadingBlock
                title="Refreshing gene browser"
                description="Fetching the current browse page from the API."
              />
            ) : (
              (() => {
                const browsePageData = browseResult.data
                return (
                  <>
                <div className="table-meta">
                  <span>
                    Page {formatInteger(browsePageData.page)} /{' '}
                    {formatInteger(browsePageData.totalPages)}
                  </span>
                  <span>{formatInteger(browsePageData.totalItems)} genes matched</span>
                </div>
                <div className="table-shell">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Gene</th>
                        <th>Index</th>
                        <th>Total count</th>
                        <th>Detected cells</th>
                        <th>Detected fraction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {browsePageData.items.length === 0 ? (
                        <tr>
                          <td className="table-empty" colSpan={5}>
                            No genes matched the current filter.
                          </td>
                        </tr>
                      ) : (
                        browsePageData.items.map((item) => (
                          <tr key={item.geneName}>
                            <td>
                              <Link
                                className="table-link"
                                to={`/gene/${encodeURIComponent(item.geneName)}`}
                              >
                                {item.geneName}
                              </Link>
                            </td>
                            <td>{formatInteger(item.geneIndex)}</td>
                            <td>{formatInteger(item.totalCount)}</td>
                            <td>{formatInteger(item.detectedCells)}</td>
                            <td>{formatPercent(item.detectedFraction, 2)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="pagination-row">
                  <button
                    className="button button--secondary"
                    disabled={browsePageData.page <= 1}
                    onClick={() =>
                      updateBrowseParam('page', String(Math.max(1, browsePageData.page - 1)))
                    }
                    type="button"
                  >
                    Previous
                  </button>
                  <button
                    className="button button--secondary"
                    disabled={browsePageData.page >= browsePageData.totalPages}
                    onClick={() =>
                      updateBrowseParam(
                        'page',
                        String(
                          Math.min(
                            browsePageData.totalPages,
                            browsePageData.page + 1,
                          ),
                        ),
                      )
                    }
                    type="button"
                  >
                    Next
                  </button>
                </div>
                  </>
                )
              })()
            )}
          </Panel>
        </>
      )}
    </div>
  )
}

function readBrowseScope(value: string | null): BrowseScope {
  if (value === 'fitted' || value === 'all') {
    return value
  }
  return 'auto'
}

function readBrowseSort(value: string | null): BrowseSort {
  if (
    value === 'detected_cells' ||
    value === 'detected_fraction' ||
    value === 'gene_name' ||
    value === 'gene_index'
  ) {
    return value
  }
  return 'total_count'
}

function readPositiveInteger(value: string | null, fallback: number) {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) && parsed >= 1 ? parsed : fallback
}
