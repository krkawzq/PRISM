import type {
  ApiEnvelope,
  ContextSnapshot,
  GeneAnalysisData,
  GeneAnalysisRequest,
  GeneBrowsePage,
  GeneBrowseRequest,
  GeneCandidate,
  KBulkAnalysisData,
  KBulkAnalysisRequest,
  LoadContextPayload,
} from './types'

export class ApiClientError extends Error {
  readonly code: string
  readonly status: number
  readonly details: Record<string, unknown>

  constructor(message: string, options: { code: string; status: number; details?: Record<string, unknown> }) {
    super(message)
    this.code = options.code
    this.status = options.status
    this.details = options.details ?? {}
  }
}

interface RequestOptions extends RequestInit {
  signal?: AbortSignal
}

export const apiClient = {
  getContext: (options?: RequestOptions) => request<ContextSnapshot>('/api/context', options),
  loadContext: (payload: LoadContextPayload, options?: RequestOptions) =>
    request<ContextSnapshot>('/api/context/load', {
      ...options,
      method: 'POST',
      body: JSON.stringify({
        h5ad_path: payload.h5adPath,
        ckpt_path: payload.ckptPath || null,
        layer: payload.layer || null,
      }),
    }),
  browseGenes: (payload: GeneBrowseRequest, options?: RequestOptions) =>
    request<GeneBrowsePage>(`/api/genes?${toQueryString({
      query: payload.query,
      scope: payload.scope,
      sort_by: payload.sortBy,
      direction: payload.direction,
      page: payload.page,
    })}`, options),
  searchGenes: (query: string, limit?: number, options?: RequestOptions) =>
    request<{ items: GeneCandidate[] }>(
      `/api/genes/search?${toQueryString({ q: query, limit })}`,
      options,
    ),
  getGeneAnalysis: (payload: GeneAnalysisRequest, options?: RequestOptions) =>
    request<GeneAnalysisData>(`/api/gene-analysis?${toQueryString({
      q: payload.q,
      mode: payload.mode,
      prior_source: payload.priorSource,
      label_key: payload.labelKey,
      label: payload.label,
      scale: payload.scale,
      reference_source: payload.referenceSource,
      n_support_points: payload.nSupportPoints,
      max_em_iterations: payload.maxEmIterations,
      convergence_tolerance: payload.convergenceTolerance,
      cell_chunk_size: payload.cellChunkSize,
      support_max_from: payload.supportMaxFrom,
      support_spacing: payload.supportSpacing,
      support_scale: payload.supportScale,
      use_adaptive_support: payload.useAdaptiveSupport,
      adaptive_support_scale: payload.adaptiveSupportScale,
      adaptive_support_quantile_hi: payload.adaptiveSupportQuantileHi,
      likelihood: payload.likelihood,
      nb_overdispersion: payload.nbOverdispersion,
      torch_dtype: payload.torchDtype,
      compile_model: payload.compileModel,
      device: payload.device,
    })}`, options),
  getKbulkAnalysis: (payload: KBulkAnalysisRequest, options?: RequestOptions) =>
    request<KBulkAnalysisData>(`/api/kbulk-analysis?${toQueryString({
      q: payload.q,
      class_key: payload.classKey,
      label_key: payload.labelKey,
      label: payload.label,
      k: payload.k,
      n_samples: payload.nSamples,
      sample_seed: payload.sampleSeed,
      max_classes: payload.maxClasses,
      sample_batch_size: payload.sampleBatchSize,
      kbulk_prior_source: payload.kbulkPriorSource,
      torch_dtype: payload.torchDtype,
      compile_model: payload.compileModel,
      device: payload.device,
    })}`, options),
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown request error'
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  })

  const payload = (await response.json()) as ApiEnvelope<T>
  if (!response.ok || !payload.ok || payload.error) {
    const message = payload.error?.message ?? `Request failed with status ${response.status}`
    throw new ApiClientError(message, {
      code: payload.error?.code ?? 'request_failed',
      status: response.status,
      details: payload.error?.details,
    })
  }
  return payload.data
}

function toQueryString(values: Record<string, string | number | boolean | null | undefined>): string {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') {
      return
    }
    params.set(key, String(value))
  })
  return params.toString()
}
