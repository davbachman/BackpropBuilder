import { Suspense } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import LearningStudio from './LearningStudio'
import { LESSONS } from './presets'

afterEach(cleanup)

function renderStudio() {
  return render(<Suspense fallback={<p role="status">Opening the local model…</p>}><LearningStudio /></Suspense>)
}

async function openPreset(user: ReturnType<typeof userEvent.setup>, title: string) {
  const gallery = screen.getByRole('region', { name: 'Learning presets' })
  const heading = within(gallery).getByRole('heading', { name: title, level: 3 })
  await user.click(heading.closest('button')!)
  return screen.findByRole('heading', { name: title, level: 1 }, { timeout: 3000 })
}

describe('preset gallery and studio navigation', () => {
  it('starts at a gallery with all ten learning questions and the blank builder', () => {
    renderStudio()
    const gallery = screen.getByRole('region', { name: 'Learning presets' })
    expect(within(gallery).getAllByRole('button')).toHaveLength(10)
    expect(within(gallery).getAllByRole('heading', { level: 3 })).toHaveLength(10)
    for (const preset of LESSONS) {
      expect(within(gallery).getByRole('heading', { name: preset.title })).toBeVisible()
      expect(within(gallery).getByText(preset.question)).toBeVisible()
    }
    expect(screen.getByRole('button', { name: 'Blank builder' })).toBeVisible()
    expect(screen.queryByRole('navigation', { name: 'Preset navigation' })).not.toBeInTheDocument()
  })

  it.each(LESSONS)('opens the actual $id lesson from the gallery', async preset => {
    const user = userEvent.setup()
    renderStudio()
    await openPreset(user, preset.title)
    expect(await screen.findByRole('navigation', { name: 'Preset navigation' })).toBeVisible()
    if (['linear', 'neuron', 'small-network', 'playground'].includes(preset.id)) {
      expect(await screen.findByRole('button', { name: '1 Forward' })).toBeVisible()
      expect(screen.getByTestId('network-prediction')).toHaveTextContent('—')
    } else {
      expect(await screen.findByRole('button', { name: 'Run forward' })).toBeVisible()
      expect(screen.getByLabelText('Model visual workspace')).toBeVisible()
    }
    expect(screen.queryByRole('region', { name: 'Learning presets' })).not.toBeInTheDocument()
  })

  it('opens an empty free builder and returns to the gallery', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await user.click(screen.getByRole('button', { name: 'Blank builder' }))
    expect(await screen.findByRole('heading', { name: 'Backprop Builder' }, { timeout: 3000 })).toBeVisible()
    expect(screen.getByText('0 nodes, 0 edges')).toBeVisible()
    expect(container.querySelectorAll('.builder-node')).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Preset gallery' }))
    expect(await screen.findByRole('region', { name: 'Learning presets' })).toBeVisible()
  })

  it('switches between real network and transformer lessons, hides navigation, and returns through Presets', async () => {
    const user = userEvent.setup()
    renderStudio()
    await openPreset(user, 'A line that learns')
    await screen.findByRole('button', { name: '1 Forward' })
    const navigation = screen.getByRole('navigation', { name: 'Preset navigation' })
    await user.click(within(navigation).getByRole('button', { name: /Stable softmax/ }))
    expect(await screen.findByRole('heading', { name: 'Scores become choices', level: 1 })).toBeVisible()
    expect(await screen.findByRole('button', { name: 'Run forward' })).toBeVisible()
    expect(screen.queryByTestId('network-prediction')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Hide preset navigation' }))
    expect(screen.queryByRole('navigation', { name: 'Preset navigation' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show preset navigation' }))
    await user.click(within(screen.getByRole('navigation', { name: 'Preset navigation' })).getByRole('button', { name: /Weights, bias & activation/ }))
    expect(await screen.findByRole('heading', { name: 'Inside one neuron', level: 1 })).toBeVisible()
    expect(await screen.findByRole('button', { name: '1 Forward' })).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Presets' }))
    expect(await screen.findByRole('region', { name: 'Learning presets' })).toBeVisible()
  })

  it('passes the current prepared network to the builder and starts a fresh blank graph afterward', async () => {
    const user = userEvent.setup()
    renderStudio()
    await openPreset(user, 'A line that learns')
    await user.click(await screen.findByRole('button', { name: 'Open in graph builder' }))
    expect(await screen.findByRole('heading', { name: 'Backprop Builder' }, { timeout: 3000 })).toBeVisible()
    expect(screen.getByText('8 nodes, 7 edges, 1 groups')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Open module Layer 1' })).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Preset gallery' }))
    await user.click(await screen.findByRole('button', { name: 'Blank builder' }))
    expect(await screen.findByText('0 nodes, 0 edges')).toBeVisible()
  })
})
