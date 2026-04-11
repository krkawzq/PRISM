import { startTransition, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'

import {
  ChipRow,
  DescriptionList,
  EmptyState,
  FigurePanel,
  LoadingBlock,
  Notice,
  Panel,
  StatGrid,
} from '../components/ui'
import { useGeneWorkspace } from '../hooks/useGeneWorkspace'
import type {
  AnalysisMode,
  GeneAnalysisData,
  PriorSource,
} from '../api/types'
import {
  readGeneWorkspaceUrlState,
  toGeneAnalysisRequest,
  toGeneSearchParams,
  toKbulkAnalysisRequest,
  type FitFormState,
  type GeneWorkspaceUrlState,
  type KBulkFormState,
} from '../lib/geneUrlState'
import {
  formatDecimal,
  formatInteger,
  formatOptionalNumber,
  formatPercent,
} from '../lib/format'

const MODE_OPTIONS: Array<{ value: AnalysisMode; label: string }> = [
  { value: 'raw', label: 'Raw only' },
  { value: 'checkpoint', label: 'Checkpoint posterior' },
  { value: 'fit', label: 'On-demand fit' },
]

const PRIOR_OPTIONS: Array<{ value: PriorSource; label: string }> = [
  { value: 'global', label: 'Global prior' },
  { value: 'label', label: 'Label prior' },
]

export default function GenePage() {
  const params = useParams<{ geneQuery: string }>()
  const geneQuery = decodeURIComponent(params.geneQuery ?? '')
  const [searchParams, setSearchParams] = useSearchParams()
  const searchKey = searchParams.toString()
  const urlState = readGeneWorkspaceUrlState(searchParams)
  const [draft, setDraft] = useState<GeneWorkspaceUrlState>(urlState)

  useEffect(() => {
    setDraft(urlState)
  }, [geneQuery, searchKey])

  const workspace = useGeneWorkspace({
    geneQuery,
    analysisRequest: toGeneAnalysisRequest(geneQuery, urlState),
    kbulkEnabled: urlState.kbulkEnabled,
    kbulkRequest: toKbulkAnalysisRequest(geneQuery, urlState),
  })

  const analysis = workspace.analysis
  const availableLabelKeys = analysis?.availableLabelKeys ?? []
  const availableLabels = analysis?.availableLabels ?? []
  const availableClassKeys = workspace.kbulk?.availableClassKeys ?? availableLabelKeys

  function commit(nextState: GeneWorkspaceUrlState, replace = false) {
    startTransition(() => {
      setSearchParams(toGeneSearchParams(nextState), { replace })
    })
  }

  function handleControlsSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    commit({ ...draft, kbulkEnabled: false })
  }

  function handleFitSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    commit({ ...draft, mode: 'fit', kbulkEnabled: false })
  }

  function handleKbulkSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    commit({ ...draft, kbulkEnabled: true })
  }

  function handleControlSelect(
    key: 'mode' | 'priorSource' | 'labelKey' | 'label',
    value: string,
  ) {
    setDraft((current) => ({ ...current, [key]: value }))
  }

  function handleFitChange(
    key: keyof FitFormState,
    value: string | boolean,
  ) {
    setDraft((current) => ({
      ...current,
      fit: {
        ...current.fit,
        [key]: value,
      },
    }))
  }

  function handleKbulkChange(
    key: keyof KBulkFormState,
    value: string | boolean,
  ) {
    setDraft((current) => ({
      ...current,
      kbulk: {
        ...current.kbulk,
        [key]: value,
      },
    }))
  }

  if (!geneQuery) {
    return (
      <EmptyState
        title="Missing gene query"
        description="Open a gene from the dashboard or type a gene name into the quick-open field."
        action={
          <Link className="button button--primary" to="/">
            Back to dashboard
          </Link>
        }
      />
    )
  }

  return (
    <div className="page-grid">
      <div className="page-intro">
        <div>
          <p className="page-intro__eyebrow">Gene Workspace</p>
          <h1>{geneQuery}</h1>
          <p className="page-intro__description">
            URL state controls the analysis mode, fit parameters, and optional kBulk
            comparison.
          </p>
        </div>
        <div className="page-intro__actions">
          <Link className="button button--secondary" to="/">
            Back to dashboard
          </Link>
        </div>
      </div>

      {workspace.analysisError ? (
        <Notice
          title="Analysis request failed"
          message={workspace.analysisError}
          tone="error"
        />
      ) : null}
      {workspace.kbulkError ? (
        <Notice
          title="kBulk request failed"
          message={workspace.kbulkError}
          tone="error"
        />
      ) : null}

      <div className="panel-grid panel-grid--two">
        <Panel
          title="Resolved Analysis"
          description="Current gene summary, resolved labels, and the active analysis source returned by the backend."
          eyebrow="Summary"
        >
          {workspace.isAnalysisLoading && analysis == null ? (
            <LoadingBlock
              title="Loading gene analysis"
              description="Fetching the latest analysis payload from the API."
            />
          ) : analysis == null ? (
            <EmptyState
              title="No analysis payload available"
              description="If the backend does not have a dataset loaded, return to the dashboard and load a context first."
            />
          ) : (
            <>
              <ChipRow
                items={[
                  { label: `Mode: ${analysis.mode}`, tone: 'info' },
                  { label: `Source: ${analysis.source}`, tone: 'neutral' },
                  { label: `Prior: ${analysis.priorSource}`, tone: 'warning' },
                ]}
              />
              <StatGrid items={buildAnalysisStats(analysis)} />
              <DescriptionList
                items={[
                  { label: 'Gene index', value: formatInteger(analysis.geneIndex) },
                  { label: 'Label key', value: analysis.labelKey ?? '-' },
                  { label: 'Label', value: analysis.label ?? '-' },
                  { label: 'Reference source', value: analysis.referenceSource },
                  {
                    label: 'Reference gene count',
                    value: formatInteger(analysis.referenceGeneCount),
                  },
                ]}
              />
            </>
          )}
        </Panel>

        <Panel
          title="Checkpoint Context"
          description="Checkpoint metadata for the active gene workspace. This is empty when the context only has raw data."
          eyebrow="Model"
        >
          {analysis?.checkpointSummary == null ? (
            <EmptyState
              title="Checkpoint summary unavailable"
              description="Load a checkpoint to compare checkpoint prior, posterior diagnostics, and label-aware kBulk behavior."
            />
          ) : (
            <>
              <ChipRow
                items={[
                  {
                    label: analysis.checkpointSummary.hasGlobalPrior
                      ? 'Global prior ready'
                      : 'Global prior missing',
                    tone: analysis.checkpointSummary.hasGlobalPrior ? 'success' : 'warning',
                  },
                  {
                    label: `${formatInteger(analysis.checkpointSummary.nLabelPriors)} label priors`,
                    tone: 'info',
                  },
                  {
                    label: analysis.checkpointSummary.distribution,
                    tone: 'neutral',
                  },
                ]}
              />
              <StatGrid
                items={[
                  {
                    label: 'Checkpoint genes',
                    value: formatInteger(analysis.checkpointSummary.geneCount),
                  },
                  {
                    label: 'Scale',
                    value: formatOptionalNumber(analysis.checkpointSummary.scale, 3),
                  },
                  {
                    label: 'Reference overlap',
                    value: `${formatInteger(
                      analysis.checkpointSummary.nOverlapReferenceGenes,
                    )} / ${formatInteger(analysis.checkpointSummary.nReferenceGenes)}`,
                  },
                  {
                    label: 'Mean reference',
                    value: formatOptionalNumber(
                      analysis.checkpointSummary.meanReferenceCount,
                      3,
                    ),
                  },
                ]}
              />
              <DescriptionList
                items={[
                  {
                    label: 'Checkpoint path',
                    value: analysis.checkpointSummary.ckptPath,
                  },
                  {
                    label: 'Suggested label key',
                    value: analysis.checkpointSummary.suggestedLabelKey ?? '-',
                  },
                ]}
              />
            </>
          )}
        </Panel>
      </div>

      <Panel
        title="Analysis Controls"
        description="Switch between raw, checkpoint, and fit-backed views. These controls update the URL and the backend request contract."
        eyebrow="Controls"
      >
        <form className="form-grid" onSubmit={handleControlsSubmit}>
          <label className="field">
            <span>Mode</span>
            <select
              onChange={(event) => handleControlSelect('mode', event.target.value)}
              value={draft.mode}
            >
              {MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Prior source</span>
            <select
              onChange={(event) => handleControlSelect('priorSource', event.target.value)}
              value={draft.priorSource}
            >
              {PRIOR_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Label key</span>
            <select
              disabled={availableLabelKeys.length === 0}
              onChange={(event) => handleControlSelect('labelKey', event.target.value)}
              value={draft.labelKey}
            >
              <option value="">Auto</option>
              {availableLabelKeys.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Label</span>
            <select
              disabled={availableLabels.length === 0}
              onChange={(event) => handleControlSelect('label', event.target.value)}
              value={draft.label}
            >
              <option value="">Auto</option>
              {availableLabels.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <div className="button-row">
            <button className="button button--primary" type="submit">
              Apply controls
            </button>
          </div>
        </form>
      </Panel>

      <Panel
        title="Fit Controls"
        description="Adjust the on-demand fit configuration. Submitting this form switches the workspace into fit mode."
        eyebrow="Fit"
      >
        <form className="form-grid form-grid--dense" onSubmit={handleFitSubmit}>
          <label className="field">
            <span>Scale</span>
            <input
              onChange={(event) => handleFitChange('scale', event.target.value)}
              type="text"
              value={draft.fit.scale}
            />
          </label>
          <label className="field">
            <span>Reference source</span>
            <select
              onChange={(event) => handleFitChange('referenceSource', event.target.value)}
              value={draft.fit.referenceSource}
            >
              <option value="checkpoint">checkpoint</option>
              <option value="dataset">dataset</option>
            </select>
          </label>
          <label className="field">
            <span>Support points</span>
            <input
              onChange={(event) => handleFitChange('nSupportPoints', event.target.value)}
              type="number"
              value={draft.fit.nSupportPoints}
            />
          </label>
          <label className="field">
            <span>Max EM iterations</span>
            <input
              onChange={(event) => handleFitChange('maxEmIterations', event.target.value)}
              type="number"
              value={draft.fit.maxEmIterations}
            />
          </label>
          <label className="field">
            <span>Convergence tolerance</span>
            <input
              onChange={(event) =>
                handleFitChange('convergenceTolerance', event.target.value)
              }
              type="text"
              value={draft.fit.convergenceTolerance}
            />
          </label>
          <label className="field">
            <span>Cell chunk size</span>
            <input
              onChange={(event) => handleFitChange('cellChunkSize', event.target.value)}
              type="number"
              value={draft.fit.cellChunkSize}
            />
          </label>
          <label className="field">
            <span>Support max from</span>
            <select
              onChange={(event) => handleFitChange('supportMaxFrom', event.target.value)}
              value={draft.fit.supportMaxFrom}
            >
              <option value="observed_max">observed_max</option>
              <option value="quantile">quantile</option>
            </select>
          </label>
          <label className="field">
            <span>Support spacing</span>
            <select
              onChange={(event) => handleFitChange('supportSpacing', event.target.value)}
              value={draft.fit.supportSpacing}
            >
              <option value="linear">linear</option>
              <option value="sqrt">sqrt</option>
            </select>
          </label>
          <label className="field">
            <span>Support scale</span>
            <input
              onChange={(event) => handleFitChange('supportScale', event.target.value)}
              type="text"
              value={draft.fit.supportScale}
            />
          </label>
          <label className="field">
            <span>Adaptive support</span>
            <select
              onChange={(event) =>
                handleFitChange('useAdaptiveSupport', event.target.value === '1')
              }
              value={draft.fit.useAdaptiveSupport ? '1' : '0'}
            >
              <option value="0">off</option>
              <option value="1">on</option>
            </select>
          </label>
          <label className="field">
            <span>Adaptive support scale</span>
            <input
              onChange={(event) =>
                handleFitChange('adaptiveSupportScale', event.target.value)
              }
              type="text"
              value={draft.fit.adaptiveSupportScale}
            />
          </label>
          <label className="field">
            <span>Adaptive quantile</span>
            <input
              onChange={(event) =>
                handleFitChange('adaptiveSupportQuantile', event.target.value)
              }
              type="text"
              value={draft.fit.adaptiveSupportQuantile}
            />
          </label>
          <label className="field">
            <span>Likelihood</span>
            <select
              onChange={(event) => handleFitChange('likelihood', event.target.value)}
              value={draft.fit.likelihood}
            >
              <option value="binomial">binomial</option>
              <option value="negative_binomial">negative_binomial</option>
              <option value="poisson">poisson</option>
            </select>
          </label>
          <label className="field">
            <span>NB overdispersion</span>
            <input
              onChange={(event) => handleFitChange('nbOverdispersion', event.target.value)}
              type="text"
              value={draft.fit.nbOverdispersion}
            />
          </label>
          <label className="field">
            <span>Torch dtype</span>
            <select
              onChange={(event) => handleFitChange('torchDtype', event.target.value)}
              value={draft.fit.torchDtype}
            >
              <option value="float64">float64</option>
              <option value="float32">float32</option>
            </select>
          </label>
          <label className="field">
            <span>Compile model</span>
            <select
              onChange={(event) =>
                handleFitChange('compileModel', event.target.value === '1')
              }
              value={draft.fit.compileModel ? '1' : '0'}
            >
              <option value="1">on</option>
              <option value="0">off</option>
            </select>
          </label>
          <label className="field">
            <span>Device</span>
            <input
              onChange={(event) => handleFitChange('device', event.target.value)}
              type="text"
              value={draft.fit.device}
            />
          </label>
          <div className="button-row">
            <button className="button button--primary" type="submit">
              Run fit analysis
            </button>
          </div>
        </form>
      </Panel>

      <Panel
        title="Figures"
        description="Rendered plots returned from the decoupled backend. The frontend only composes and displays these assets."
        eyebrow="Visuals"
      >
        {workspace.isAnalysisLoading && analysis == null ? (
          <LoadingBlock
            title="Preparing figures"
            description="Waiting for the analysis payload before rendering plot panels."
          />
        ) : analysis == null ? (
          <EmptyState
            title="No figures to display"
            description="Run a gene analysis first to populate raw, prior, posterior, and fit diagnostics."
          />
        ) : (
          <div className="figure-grid">
            {analysis.figures.rawOverview ? (
              <FigurePanel
                description="Raw count histogram, count-vs-depth scatter, and proxy view."
                src={analysis.figures.rawOverview}
                title="Raw overview"
              />
            ) : null}
            {analysis.figures.priorOverlay ? (
              <FigurePanel
                description="Selected prior profile, with checkpoint comparison when fit mode is active."
                src={analysis.figures.priorOverlay}
                title="Prior overlay"
              />
            ) : null}
            {analysis.figures.signalInterface ? (
              <FigurePanel
                description="Posterior signal against raw proxy, plus entropy vs information."
                src={analysis.figures.signalInterface}
                title="Signal interface"
              />
            ) : null}
            {analysis.figures.posteriorGallery ? (
              <FigurePanel
                description="Posterior traces for representative cells in the current gene context."
                src={analysis.figures.posteriorGallery}
                title="Posterior gallery"
              />
            ) : null}
            {analysis.figures.objectiveTrace ? (
              <FigurePanel
                description="Objective history across EM iterations during on-demand fit."
                src={analysis.figures.objectiveTrace}
                title="Objective trace"
              />
            ) : null}
          </div>
        )}
      </Panel>

      <div className="panel-grid panel-grid--two">
        <Panel
          title="kBulk Controls"
          description="Configure and run group-level sampling. kBulk is kept as a separate request flow from the main analysis."
          eyebrow="kBulk"
        >
          <form className="form-grid form-grid--dense" onSubmit={handleKbulkSubmit}>
            <label className="field">
              <span>Class key</span>
              <select
                disabled={availableClassKeys.length === 0}
                onChange={(event) => handleKbulkChange('classKey', event.target.value)}
                value={draft.kbulk.classKey}
              >
                <option value="">Auto</option>
                {availableClassKeys.map((key) => (
                  <option key={key} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>k</span>
              <input
                onChange={(event) => handleKbulkChange('k', event.target.value)}
                type="number"
                value={draft.kbulk.k}
              />
            </label>
            <label className="field">
              <span>Samples</span>
              <input
                onChange={(event) => handleKbulkChange('nSamples', event.target.value)}
                type="number"
                value={draft.kbulk.nSamples}
              />
            </label>
            <label className="field">
              <span>Seed</span>
              <input
                onChange={(event) => handleKbulkChange('sampleSeed', event.target.value)}
                type="number"
                value={draft.kbulk.sampleSeed}
              />
            </label>
            <label className="field">
              <span>Max classes</span>
              <input
                onChange={(event) => handleKbulkChange('maxClasses', event.target.value)}
                placeholder="auto"
                type="number"
                value={draft.kbulk.maxClasses}
              />
            </label>
            <label className="field">
              <span>Batch size</span>
              <input
                onChange={(event) => handleKbulkChange('sampleBatchSize', event.target.value)}
                type="number"
                value={draft.kbulk.sampleBatchSize}
              />
            </label>
            <label className="field">
              <span>Prior source</span>
              <select
                onChange={(event) =>
                  handleKbulkChange('kbulkPriorSource', event.target.value)
                }
                value={draft.kbulk.kbulkPriorSource}
              >
                <option value="global">global</option>
                <option value="label">label</option>
              </select>
            </label>
            <div className="button-row">
              <button className="button button--primary" type="submit">
                Run kBulk
              </button>
              {urlState.kbulkEnabled ? (
                <button
                  className="button button--secondary"
                  onClick={() => commit({ ...draft, kbulkEnabled: false })}
                  type="button"
                >
                  Clear kBulk
                </button>
              ) : null}
            </div>
          </form>
        </Panel>

        <Panel
          title="kBulk Result"
          description="Group summary statistics and rendered comparison figure from the dedicated kBulk request."
          eyebrow="Result"
        >
          {workspace.isKbulkLoading ? (
            <LoadingBlock
              title="Running kBulk"
              description="Sampling group combinations and computing signal/entropy summaries."
            />
          ) : workspace.kbulk == null ? (
            <EmptyState
              title="No kBulk result yet"
              description="Configure a class key and submit the kBulk form to populate group comparisons."
            />
          ) : (
            <>
              <ChipRow
                items={[
                  { label: `Class key: ${workspace.kbulk.classKey}`, tone: 'info' },
                  { label: `Prior: ${workspace.kbulk.priorSource}`, tone: 'neutral' },
                  { label: `k=${workspace.kbulk.k}`, tone: 'warning' },
                ]}
              />
              <StatGrid
                items={[
                  {
                    label: 'Groups',
                    value: formatInteger(workspace.kbulk.groups.length),
                  },
                  {
                    label: 'Samples',
                    value: formatInteger(workspace.kbulk.nSamples),
                  },
                  {
                    label: 'Seed',
                    value: formatInteger(workspace.kbulk.sampleSeed),
                  },
                ]}
              />
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Label</th>
                      <th>Cells</th>
                      <th>Samples</th>
                      <th>Mean signal</th>
                      <th>Std signal</th>
                      <th>Mean entropy</th>
                      <th>Std entropy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {workspace.kbulk.groups.map((group) => (
                      <tr key={group.label}>
                        <td>{group.label}</td>
                        <td>{formatInteger(group.nCells)}</td>
                        <td>{formatInteger(group.realizedSamples)}</td>
                        <td>{formatDecimal(group.meanSignal, 4)}</td>
                        <td>{formatDecimal(group.stdSignal, 4)}</td>
                        <td>{formatDecimal(group.meanEntropy, 4)}</td>
                        <td>{formatDecimal(group.stdEntropy, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {workspace.kbulk.figure ? (
                <FigurePanel
                  description="Distribution comparison of group-level MAP signal and posterior entropy."
                  src={workspace.kbulk.figure}
                  title="kBulk comparison"
                />
              ) : null}
            </>
          )}
        </Panel>
      </div>
    </div>
  )
}

function buildAnalysisStats(analysis: GeneAnalysisData) {
  const items = [
    { label: 'Cells', value: formatInteger(analysis.nCells) },
    { label: 'Mean raw count', value: formatDecimal(analysis.rawSummary.meanCount, 4) },
    {
      label: 'Detected fraction',
      value: formatPercent(analysis.rawSummary.detectedFraction, 2),
    },
    { label: 'Zero fraction', value: formatPercent(analysis.rawSummary.zeroFraction, 2) },
    {
      label: 'Depth correlation',
      value: formatDecimal(analysis.rawSummary.countTotalCorrelation, 4),
    },
  ]

  if (analysis.posterior != null) {
    items.push(
      {
        label: 'Mean signal',
        value: formatDecimal(analysis.posterior.summary.mapSignal.mean, 4),
      },
      {
        label: 'Mean entropy',
        value: formatDecimal(analysis.posterior.summary.posteriorEntropy.mean, 4),
      },
      {
        label: 'Mean MI',
        value: formatDecimal(analysis.posterior.summary.mutualInformation.mean, 4),
      },
    )
  }

  if (analysis.fit != null) {
    items.push({
      label: 'Final objective',
      value: formatOptionalNumber(analysis.fit.finalObjective, 4),
    })
  }

  return items
}
