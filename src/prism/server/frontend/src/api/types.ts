export type AnalysisMode = 'raw' | 'checkpoint' | 'fit'
export type PriorSource = 'global' | 'label'
export type BrowseScope = 'auto' | 'fitted' | 'all'
export type BrowseSort =
  | 'total_count'
  | 'detected_cells'
  | 'detected_fraction'
  | 'gene_name'
  | 'gene_index'

export interface ApiErrorPayload {
  code: string
  message: string
  details: Record<string, unknown>
}

export interface ApiEnvelope<T> {
  ok: boolean
  data: T
  error: ApiErrorPayload | null
  meta: Record<string, unknown>
}

export interface DatasetSummary {
  nCells: number
  nGenes: number
  layer: string
  h5adPath: string
  labelKeys: string[]
  totalCountMean: number
  totalCountMedian: number
  totalCountP99: number
}

export interface CheckpointSummary {
  ckptPath: string
  geneCount: number
  hasGlobalPrior: boolean
  nLabelPriors: number
  labelPreview: string[]
  distribution: string
  supportDomain: string | null
  scale: number | null
  meanReferenceCount: number | null
  nReferenceGenes: number
  nOverlapReferenceGenes: number
  suggestedLabelKey: string | null
}

export interface ContextSnapshot {
  loaded: boolean
  contextKey: string | null
  dataset: DatasetSummary | null
  checkpoint: CheckpointSummary | null
}

export interface GeneCandidate {
  geneName: string
  geneIndex: number
  totalCount: number
  detectedCells: number
  detectedFraction: number
}

export interface GeneBrowsePage {
  query: string
  scope: BrowseScope
  sortBy: BrowseSort
  descending: boolean
  page: number
  pageSize: number
  totalItems: number
  totalPages: number
  items: GeneCandidate[]
}

export interface GeneSummary {
  totalCount: number
  meanCount: number
  medianCount: number
  p90Count: number
  p99Count: number
  maxCount: number
  detectedCells: number
  detectedFraction: number
  zeroFraction: number
  countTotalCorrelation: number
}

export interface PriorCurve {
  supportDomain: string
  scale: number
  support: number[]
  probabilities: number[]
}

export interface PosteriorMetricSummary {
  mean: number
  median: number
  p90: number
  max: number
}

export interface PosteriorSummary {
  supportDomain: string
  summary: {
    mapSignal: PosteriorMetricSummary
    mapProbability: PosteriorMetricSummary
    posteriorEntropy: PosteriorMetricSummary
    priorEntropy: PosteriorMetricSummary
    mutualInformation: PosteriorMetricSummary
  }
}

export interface FitSummary {
  objectiveHistory: number[]
  finalObjective: number | null
}

export interface AnalysisFigures {
  rawOverview: string | null
  priorOverlay: string | null
  signalInterface: string | null
  posteriorGallery: string | null
  objectiveTrace: string | null
}

export interface GeneAnalysisData {
  geneName: string
  geneIndex: number
  source: string
  mode: AnalysisMode
  priorSource: PriorSource
  labelKey: string | null
  label: string | null
  nCells: number
  referenceGeneCount: number
  referenceSource: string
  availableLabelKeys: string[]
  availableLabels: string[]
  rawSummary: GeneSummary
  checkpointSummary: CheckpointSummary | null
  prior: PriorCurve | null
  checkpointPrior: PriorCurve | null
  posterior: PosteriorSummary | null
  fit: FitSummary | null
  figures: AnalysisFigures
}

export interface KBulkGroupSummary {
  label: string
  nCells: number
  realizedSamples: number
  meanSignal: number
  stdSignal: number
  meanEntropy: number
  stdEntropy: number
}

export interface KBulkAnalysisData {
  geneName: string
  geneIndex: number
  classKey: string
  priorSource: PriorSource
  k: number
  nSamples: number
  sampleSeed: number
  maxClasses: number
  sampleBatchSize: number
  availableClassKeys: string[]
  groups: KBulkGroupSummary[]
  figure: string | null
}

export interface LoadContextPayload {
  h5adPath: string
  ckptPath: string
  layer: string
}

export interface GeneBrowseRequest {
  query: string
  scope: BrowseScope
  sortBy: BrowseSort
  direction: 'asc' | 'desc'
  page: number
}

export interface GeneAnalysisRequest {
  q: string
  mode: AnalysisMode
  priorSource: PriorSource
  labelKey?: string
  label?: string
  scale?: number
  referenceSource: 'checkpoint' | 'dataset'
  nSupportPoints: number
  maxEmIterations?: number
  convergenceTolerance: number
  cellChunkSize: number
  supportMaxFrom: 'observed_max' | 'quantile'
  supportSpacing: 'linear' | 'sqrt'
  supportScale: number
  useAdaptiveSupport: boolean
  adaptiveSupportScale: number
  adaptiveSupportQuantile: number
  likelihood: 'binomial' | 'negative_binomial' | 'poisson'
  nbOverdispersion: number
  torchDtype: 'float32' | 'float64'
  compileModel: boolean
  device: string
}

export interface KBulkAnalysisRequest {
  q: string
  classKey?: string
  labelKey?: string
  label?: string
  k: number
  nSamples: number
  sampleSeed: number
  maxClasses?: number
  sampleBatchSize: number
  kbulkPriorSource: PriorSource
  torchDtype: 'float32' | 'float64'
  compileModel: boolean
  device: string
}
