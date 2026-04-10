import type {
  AnalysisMode,
  GeneAnalysisRequest,
  KBulkAnalysisRequest,
  PriorSource,
} from '../api/types'

export interface FitFormState {
  scale: string
  referenceSource: 'checkpoint' | 'dataset'
  nSupportPoints: string
  maxEmIterations: string
  convergenceTolerance: string
  cellChunkSize: string
  supportMaxFrom: 'observed_max' | 'quantile'
  supportSpacing: 'linear' | 'sqrt'
  supportScale: string
  useAdaptiveSupport: boolean
  adaptiveSupportScale: string
  adaptiveSupportQuantileHi: string
  likelihood: 'binomial' | 'negative_binomial' | 'poisson'
  nbOverdispersion: string
  torchDtype: 'float32' | 'float64'
  compileModel: boolean
  device: string
}

export interface KBulkFormState {
  classKey: string
  k: string
  nSamples: string
  sampleSeed: string
  maxClasses: string
  sampleBatchSize: string
  kbulkPriorSource: PriorSource
  torchDtype: 'float32' | 'float64'
  compileModel: boolean
  device: string
}

export interface GeneWorkspaceUrlState {
  mode: AnalysisMode
  priorSource: PriorSource
  labelKey: string
  label: string
  fit: FitFormState
  kbulkEnabled: boolean
  kbulk: KBulkFormState
}

const DEFAULT_FIT_FORM: FitFormState = {
  scale: '',
  referenceSource: 'checkpoint',
  nSupportPoints: '512',
  maxEmIterations: '200',
  convergenceTolerance: '0.000001',
  cellChunkSize: '512',
  supportMaxFrom: 'observed_max',
  supportSpacing: 'linear',
  supportScale: '1.5',
  useAdaptiveSupport: false,
  adaptiveSupportScale: '1.5',
  adaptiveSupportQuantileHi: '0.99',
  likelihood: 'binomial',
  nbOverdispersion: '0.01',
  torchDtype: 'float64',
  compileModel: true,
  device: 'cpu',
}

const DEFAULT_KBULK_FORM: KBulkFormState = {
  classKey: '',
  k: '8',
  nSamples: '24',
  sampleSeed: '0',
  maxClasses: '',
  sampleBatchSize: '32',
  kbulkPriorSource: 'global',
  torchDtype: 'float64',
  compileModel: true,
  device: 'cpu',
}

export function readGeneWorkspaceUrlState(
  searchParams: URLSearchParams,
): GeneWorkspaceUrlState {
  return {
    mode: readAnalysisMode(searchParams.get('mode')),
    priorSource: readPriorSource(searchParams.get('prior_source')),
    labelKey: searchParams.get('label_key') ?? '',
    label: searchParams.get('label') ?? '',
    fit: {
      scale: searchParams.get('scale') ?? DEFAULT_FIT_FORM.scale,
      referenceSource:
        searchParams.get('reference_source') === 'dataset'
          ? 'dataset'
          : DEFAULT_FIT_FORM.referenceSource,
      nSupportPoints: searchParams.get('n_support_points') ?? DEFAULT_FIT_FORM.nSupportPoints,
      maxEmIterations:
        searchParams.get('max_em_iterations') ?? DEFAULT_FIT_FORM.maxEmIterations,
      convergenceTolerance:
        searchParams.get('convergence_tolerance') ?? DEFAULT_FIT_FORM.convergenceTolerance,
      cellChunkSize: searchParams.get('cell_chunk_size') ?? DEFAULT_FIT_FORM.cellChunkSize,
      supportMaxFrom:
        searchParams.get('support_max_from') === 'quantile'
          ? 'quantile'
          : DEFAULT_FIT_FORM.supportMaxFrom,
      supportSpacing:
        searchParams.get('support_spacing') === 'sqrt'
          ? 'sqrt'
          : DEFAULT_FIT_FORM.supportSpacing,
      supportScale: searchParams.get('support_scale') ?? DEFAULT_FIT_FORM.supportScale,
      useAdaptiveSupport: readBoolean(
        searchParams.get('use_adaptive_support'),
        DEFAULT_FIT_FORM.useAdaptiveSupport,
      ),
      adaptiveSupportScale:
        searchParams.get('adaptive_support_scale') ?? DEFAULT_FIT_FORM.adaptiveSupportScale,
      adaptiveSupportQuantileHi:
        searchParams.get('adaptive_support_quantile_hi') ??
        DEFAULT_FIT_FORM.adaptiveSupportQuantileHi,
      likelihood: readLikelihood(searchParams.get('likelihood')),
      nbOverdispersion:
        searchParams.get('nb_overdispersion') ?? DEFAULT_FIT_FORM.nbOverdispersion,
      torchDtype:
        searchParams.get('torch_dtype') === 'float32' ? 'float32' : DEFAULT_FIT_FORM.torchDtype,
      compileModel: readBoolean(searchParams.get('compile_model'), DEFAULT_FIT_FORM.compileModel),
      device: searchParams.get('device') ?? DEFAULT_FIT_FORM.device,
    },
    kbulkEnabled: readBoolean(searchParams.get('kbulk'), false),
    kbulk: {
      classKey: searchParams.get('class_key') ?? DEFAULT_KBULK_FORM.classKey,
      k: searchParams.get('k') ?? DEFAULT_KBULK_FORM.k,
      nSamples: searchParams.get('n_samples') ?? DEFAULT_KBULK_FORM.nSamples,
      sampleSeed: searchParams.get('sample_seed') ?? DEFAULT_KBULK_FORM.sampleSeed,
      maxClasses: searchParams.get('max_classes') ?? DEFAULT_KBULK_FORM.maxClasses,
      sampleBatchSize:
        searchParams.get('sample_batch_size') ?? DEFAULT_KBULK_FORM.sampleBatchSize,
      kbulkPriorSource: readPriorSource(searchParams.get('kbulk_prior_source')),
      torchDtype:
        searchParams.get('kbulk_torch_dtype') === 'float32'
          ? 'float32'
          : DEFAULT_KBULK_FORM.torchDtype,
      compileModel: readBoolean(
        searchParams.get('kbulk_compile_model'),
        DEFAULT_KBULK_FORM.compileModel,
      ),
      device: searchParams.get('kbulk_device') ?? DEFAULT_KBULK_FORM.device,
    },
  }
}

export function toGeneAnalysisRequest(
  geneQuery: string,
  state: GeneWorkspaceUrlState,
): GeneAnalysisRequest {
  return {
    q: geneQuery,
    mode: state.mode,
    priorSource: state.priorSource,
    labelKey: state.labelKey || undefined,
    label: state.label || undefined,
    scale: readOptionalNumber(state.fit.scale),
    referenceSource: state.fit.referenceSource,
    nSupportPoints: readInteger(state.fit.nSupportPoints, 512),
    maxEmIterations: readOptionalInteger(state.fit.maxEmIterations),
    convergenceTolerance: readNumber(state.fit.convergenceTolerance, 0.000001),
    cellChunkSize: readInteger(state.fit.cellChunkSize, 512),
    supportMaxFrom: state.fit.supportMaxFrom,
    supportSpacing: state.fit.supportSpacing,
    supportScale: readNumber(state.fit.supportScale, 1.5),
    useAdaptiveSupport: state.fit.useAdaptiveSupport,
    adaptiveSupportScale: readNumber(state.fit.adaptiveSupportScale, 1.5),
    adaptiveSupportQuantileHi: readNumber(state.fit.adaptiveSupportQuantileHi, 0.99),
    likelihood: state.fit.likelihood,
    nbOverdispersion: readNumber(state.fit.nbOverdispersion, 0.01),
    torchDtype: state.fit.torchDtype,
    compileModel: state.fit.compileModel,
    device: state.fit.device || 'cpu',
  }
}

export function toKbulkAnalysisRequest(
  geneQuery: string,
  state: GeneWorkspaceUrlState,
): KBulkAnalysisRequest {
  return {
    q: geneQuery,
    classKey: state.kbulk.classKey || undefined,
    labelKey: state.labelKey || undefined,
    label: state.label || undefined,
    k: readInteger(state.kbulk.k, 8),
    nSamples: readInteger(state.kbulk.nSamples, 24),
    sampleSeed: readInteger(state.kbulk.sampleSeed, 0),
    maxClasses: state.kbulk.maxClasses ? readInteger(state.kbulk.maxClasses, 6) : undefined,
    sampleBatchSize: readInteger(state.kbulk.sampleBatchSize, 32),
    kbulkPriorSource: state.kbulk.kbulkPriorSource,
    torchDtype: state.kbulk.torchDtype,
    compileModel: state.kbulk.compileModel,
    device: state.kbulk.device || 'cpu',
  }
}

export function toGeneSearchParams(state: GeneWorkspaceUrlState): URLSearchParams {
  const searchParams = new URLSearchParams()
  searchParams.set('mode', state.mode)
  searchParams.set('prior_source', state.priorSource)
  setIfPresent(searchParams, 'label_key', state.labelKey)
  setIfPresent(searchParams, 'label', state.label)
  setIfPresent(searchParams, 'scale', state.fit.scale)
  searchParams.set('reference_source', state.fit.referenceSource)
  searchParams.set('n_support_points', state.fit.nSupportPoints)
  setIfPresent(searchParams, 'max_em_iterations', state.fit.maxEmIterations)
  searchParams.set('convergence_tolerance', state.fit.convergenceTolerance)
  searchParams.set('cell_chunk_size', state.fit.cellChunkSize)
  searchParams.set('support_max_from', state.fit.supportMaxFrom)
  searchParams.set('support_spacing', state.fit.supportSpacing)
  searchParams.set('support_scale', state.fit.supportScale)
  searchParams.set('use_adaptive_support', state.fit.useAdaptiveSupport ? '1' : '0')
  searchParams.set('adaptive_support_scale', state.fit.adaptiveSupportScale)
  searchParams.set('adaptive_support_quantile_hi', state.fit.adaptiveSupportQuantileHi)
  searchParams.set('likelihood', state.fit.likelihood)
  searchParams.set('nb_overdispersion', state.fit.nbOverdispersion)
  searchParams.set('torch_dtype', state.fit.torchDtype)
  searchParams.set('compile_model', state.fit.compileModel ? '1' : '0')
  searchParams.set('device', state.fit.device)

  if (state.kbulkEnabled) {
    searchParams.set('kbulk', '1')
    setIfPresent(searchParams, 'class_key', state.kbulk.classKey)
    searchParams.set('k', state.kbulk.k)
    searchParams.set('n_samples', state.kbulk.nSamples)
    searchParams.set('sample_seed', state.kbulk.sampleSeed)
    setIfPresent(searchParams, 'max_classes', state.kbulk.maxClasses)
    searchParams.set('sample_batch_size', state.kbulk.sampleBatchSize)
    searchParams.set('kbulk_prior_source', state.kbulk.kbulkPriorSource)
    searchParams.set('kbulk_torch_dtype', state.kbulk.torchDtype)
    searchParams.set('kbulk_compile_model', state.kbulk.compileModel ? '1' : '0')
    searchParams.set('kbulk_device', state.kbulk.device)
  }

  return searchParams
}

function setIfPresent(searchParams: URLSearchParams, key: string, value: string) {
  if (value) {
    searchParams.set(key, value)
  }
}

function readBoolean(value: string | null, fallback: boolean) {
  if (value == null) {
    return fallback
  }
  return value === '1' || value === 'true' || value === 'yes' || value === 'on'
}

function readAnalysisMode(value: string | null): AnalysisMode {
  if (value === 'raw' || value === 'fit') {
    return value
  }
  return 'checkpoint'
}

function readPriorSource(value: string | null): PriorSource {
  return value === 'label' ? 'label' : 'global'
}

function readLikelihood(
  value: string | null,
): 'binomial' | 'negative_binomial' | 'poisson' {
  if (value === 'negative_binomial' || value === 'poisson') {
    return value
  }
  return 'binomial'
}

function readNumber(value: string, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function readInteger(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function readOptionalNumber(value: string) {
  return value.trim() ? readNumber(value, 0) : undefined
}

function readOptionalInteger(value: string) {
  return value.trim() ? readInteger(value, 0) : undefined
}
