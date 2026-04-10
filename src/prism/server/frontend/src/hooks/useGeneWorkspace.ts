import { useEffect, useState } from 'react'

import { apiClient, getErrorMessage, isAbortError } from '../api/client'
import type {
  GeneAnalysisData,
  GeneAnalysisRequest,
  KBulkAnalysisData,
  KBulkAnalysisRequest,
} from '../api/types'

interface GeneWorkspaceOptions {
  geneQuery: string
  analysisRequest: GeneAnalysisRequest
  kbulkEnabled: boolean
  kbulkRequest: KBulkAnalysisRequest
}

interface GeneWorkspaceState {
  analysis: GeneAnalysisData | null
  analysisError: string | null
  isAnalysisLoading: boolean
  kbulk: KBulkAnalysisData | null
  kbulkError: string | null
  isKbulkLoading: boolean
  reload: () => void
}

const EMPTY_STATE: Omit<GeneWorkspaceState, 'reload'> = {
  analysis: null,
  analysisError: null,
  isAnalysisLoading: true,
  kbulk: null,
  kbulkError: null,
  isKbulkLoading: false,
}

export function useGeneWorkspace(options: GeneWorkspaceOptions): GeneWorkspaceState {
  const [state, setState] = useState<Omit<GeneWorkspaceState, 'reload'>>(EMPTY_STATE)
  const [reloadToken, setReloadToken] = useState(0)

  const requestKey = JSON.stringify({
    geneQuery: options.geneQuery,
    analysisRequest: options.analysisRequest,
    kbulkEnabled: options.kbulkEnabled,
    kbulkRequest: options.kbulkRequest,
    reloadToken,
  })

  useEffect(() => {
    if (!options.geneQuery) {
      setState({ ...EMPTY_STATE, isAnalysisLoading: false })
      return
    }

    const controller = new AbortController()
    let disposed = false

    setState((current) => ({
      ...current,
      isAnalysisLoading: true,
      isKbulkLoading: options.kbulkEnabled,
      analysisError: null,
      kbulkError: null,
    }))

    async function loadWorkspace() {
      let analysis: GeneAnalysisData | null = null
      let analysisError: string | null = null

      try {
        analysis = await apiClient.getGeneAnalysis(options.analysisRequest, {
          signal: controller.signal,
        })
      } catch (caughtError) {
        if (isAbortError(caughtError)) {
          return
        }
        analysisError = getErrorMessage(caughtError)
        if (options.analysisRequest.mode !== 'raw') {
          try {
            analysis = await apiClient.getGeneAnalysis(
              {
                ...options.analysisRequest,
                mode: 'raw',
                priorSource: 'global',
              },
              { signal: controller.signal },
            )
          } catch (fallbackError) {
            if (isAbortError(fallbackError)) {
              return
            }
            analysisError = getErrorMessage(fallbackError)
          }
        }
      }

      let kbulk: KBulkAnalysisData | null = null
      let kbulkError: string | null = null

      if (options.kbulkEnabled) {
        try {
          kbulk = await apiClient.getKbulkAnalysis(options.kbulkRequest, {
            signal: controller.signal,
          })
        } catch (caughtError) {
          if (!isAbortError(caughtError)) {
            kbulkError = getErrorMessage(caughtError)
          }
        }
      }

      if (disposed) {
        return
      }

      setState({
        analysis,
        analysisError,
        isAnalysisLoading: false,
        kbulk,
        kbulkError,
        isKbulkLoading: false,
      })
    }

    void loadWorkspace()

    return () => {
      disposed = true
      controller.abort()
    }
  }, [requestKey])

  return {
    ...state,
    reload: () => setReloadToken((current) => current + 1),
  }
}
