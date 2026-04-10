import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useAsyncResource } from './useAsyncResource'

afterEach(() => {
  cleanup()
})

function TestHarness({
  dependencyKey,
  loader,
}: {
  dependencyKey: string
  loader: (signal: AbortSignal) => Promise<string>
}) {
  const resource = useAsyncResource(loader, dependencyKey)
  return <div>{resource.isLoading ? 'Loading' : resource.data ?? resource.error ?? 'Empty'}</div>
}

describe('useAsyncResource', () => {
  it('does not refetch when rerenders create a new loader function identity', async () => {
    const loader = vi.fn(async (_signal: AbortSignal) => {
      await Promise.resolve()
      return 'Loaded'
    })

    function InlineLoaderHarness() {
      return (
        <TestHarness
          dependencyKey="stable-key"
          loader={(signal) => loader(signal)}
        />
      )
    }

    render(<InlineLoaderHarness />)

    await screen.findByText('Loaded')
    await new Promise((resolve) => setTimeout(resolve, 25))

    expect(loader).toHaveBeenCalledTimes(1)
  })

  it('refetches when the dependency key changes', async () => {
    const loader = vi.fn(async (_signal: AbortSignal) => {
      await Promise.resolve()
      return 'Loaded'
    })

    const { rerender } = render(
      <TestHarness dependencyKey="page:1" loader={loader} />,
    )

    await screen.findByText('Loaded')
    await waitFor(() => {
      expect(loader).toHaveBeenCalledTimes(1)
    })

    rerender(<TestHarness dependencyKey="page:2" loader={loader} />)

    await waitFor(() => {
      expect(loader).toHaveBeenCalledTimes(2)
    })
  })
})
