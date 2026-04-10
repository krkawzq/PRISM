import { useCallback, useEffect, useRef, useState } from 'react'

import { getErrorMessage, isAbortError } from '../api/client'

interface UseAsyncResourceOptions {
  enabled?: boolean
}

export interface AsyncResourceState<T> {
  data: T | null
  isLoading: boolean
  error: string | null
  reload: () => void
}

export function useAsyncResource<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  dependencyKey: string,
  options: UseAsyncResourceOptions = {},
): AsyncResourceState<T> {
  const enabled = options.enabled ?? true
  const [data, setData] = useState<T | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(enabled)
  const [error, setError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)
  const loaderRef = useRef(loader)

  loaderRef.current = loader

  const reload = useCallback(() => {
    setReloadToken((current) => current + 1)
  }, [])

  useEffect(() => {
    if (!enabled) {
      setData(null)
      setIsLoading(false)
      setError(null)
      return
    }

    const controller = new AbortController()
    let disposed = false
    setIsLoading(true)
    setError(null)

    void loaderRef.current(controller.signal)
      .then((result) => {
        if (disposed) {
          return
        }
        setData(result)
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
  }, [dependencyKey, enabled, reloadToken])

  return { data, isLoading, error, reload }
}
