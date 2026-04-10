import { useEffect, useState } from 'react'

import { apiClient, getErrorMessage, isAbortError } from '../api/client'
import type { ContextSnapshot, LoadContextPayload } from '../api/types'

interface ContextSnapshotState {
  snapshot: ContextSnapshot | null
  error: string | null
  isLoading: boolean
  isSubmitting: boolean
  reload: () => void
  loadContext: (payload: LoadContextPayload) => Promise<ContextSnapshot | null>
}

export function useContextSnapshot(): ContextSnapshotState {
  const [snapshot, setSnapshot] = useState<ContextSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let disposed = false

    setIsLoading(true)
    setError(null)

    void apiClient
      .getContext({ signal: controller.signal })
      .then((result) => {
        if (disposed) {
          return
        }
        setSnapshot(result)
        setIsLoading(false)
      })
      .catch((caughtError) => {
        if (disposed || isAbortError(caughtError)) {
          return
        }
        setError(getErrorMessage(caughtError))
        setIsLoading(false)
      })

    return () => {
      disposed = true
      controller.abort()
    }
  }, [reloadToken])

  async function loadContext(payload: LoadContextPayload) {
    setIsSubmitting(true)
    setError(null)
    try {
      const nextSnapshot = await apiClient.loadContext(payload)
      setSnapshot(nextSnapshot)
      return nextSnapshot
    } catch (caughtError) {
      if (!isAbortError(caughtError)) {
        setError(getErrorMessage(caughtError))
      }
      return null
    } finally {
      setIsSubmitting(false)
    }
  }

  return {
    snapshot,
    error,
    isLoading,
    isSubmitting,
    reload: () => setReloadToken((current) => current + 1),
    loadContext,
  }
}
