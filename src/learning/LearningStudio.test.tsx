import { Suspense } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
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
  await screen.findByRole('heading', { name: 'Backprop Builder', level: 1 }, { timeout: 3000 })
  expect(screen.getAllByText(title)[0]).toBeVisible()
}

describe('one canvas for every preset', () => {
  it('starts with every learning question and the blank builder', () => {
    renderStudio()
    const gallery = screen.getByRole('region', { name: 'Learning presets' })
    expect(within(gallery).getAllByRole('button')).toHaveLength(LESSONS.length)
    for (const preset of LESSONS) {
      expect(within(gallery).getByRole('heading', { name: preset.title })).toBeVisible()
      expect(within(gallery).getByText(preset.question)).toBeVisible()
    }
    expect(screen.getByRole('button', { name: 'Blank builder' })).toBeVisible()
  })

  it.each(LESSONS)('opens $id directly as an executable, editable graph', async preset => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, preset.title)
    expect(screen.getByRole('region', { name: 'Graph canvas' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Matrix product' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Run forward' })).toBeEnabled()
    expect(container.querySelectorAll('.react-flow__node').length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expect(screen.queryByText('Ready to evaluate')).not.toBeInTheDocument()
    expect(screen.queryByText('Complete the connections')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Learning presets' })).not.toBeInTheDocument()
    const train = screen.getByRole('button', { name: 'Run one full training step' })
    if (['embeddings', 'attention', 'causal', 'cnn'].includes(preset.id)) expect(train).toBeDisabled()
    else expect(train).toBeEnabled()
  })

  it('edits an actual scalar neuron weight and recomputes its output', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, 'Inside one neuron')
    const neuronOutput = () => container.querySelector('[data-id="layer-0/neuron-0/activation"] .semantic-operation-value')
    expect(neuronOutput()).toHaveTextContent('1.900')
    fireEvent.click(container.querySelector('[data-id="layer-0/weight-0-0"]')!)
    fireEvent.change(screen.getByLabelText('Tensor values'), { target: { value: '1' } })
    await user.click(screen.getByRole('button', { name: 'Apply values' }))
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expect(neuronOutput()).toHaveTextContent('1.300')
    expect(screen.getByText('Current loss 0.221')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run one full training step' }))
    expect(screen.getByText('Epoch 1')).toBeInTheDocument()
  })

  it('runs a full decoder forward and backward pass on the architecture canvas', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, 'A tiny model, end to end')
    expect(screen.getByRole('button', { name: 'Zoom into Transformer block 1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zoom into Transformer block 2' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    await user.click(screen.getByRole('button', { name: 'Step' }))
    expect(screen.getByRole('heading', { name: /Backpropagate through/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run one full training step' }))
    expect(screen.getByText('Epoch 1')).toBeInTheDocument()
    await user.click(container.querySelector('.react-flow__pane')!)
    expect(screen.getByRole('button', { name: 'Generate next token' })).toBeEnabled()
  })

  it('edits a neuron coordinate inside an MLP and sees the edit in its shared weight matrix', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, 'A tiny model, end to end')
    // Exercise explicit navigation independently of browser viewport geometry.
    await user.click(screen.getByRole('checkbox', { name: 'Zoom reveals detail' }))
    await user.click(screen.getByRole('button', { name: 'Zoom into Transformer block 1' }))
    await user.click(screen.getByRole('button', { name: 'Zoom into Feedforward MLP · 8 → 16 → 8' }))
    await user.click(screen.getByRole('button', { name: 'Inspect Hidden layer · 8 → 16 · ReLU neuron 1' }))
    fireEvent.click(container.querySelector('[data-id="inspect:blocks.0.ff1.layer:0:w0"]')!)
    expect(screen.getByText(/Coordinate 1 of Weights/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Edit tensor coordinate'), { target: { value: '0.75' } })
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expect(screen.getByLabelText('Edit tensor coordinate')).toHaveValue(.75)
    await user.click(screen.getByRole('button', { name: 'Zoom out of Hidden layer · 8 → 16 · ReLU' }))
    await user.click(screen.getByRole('button', { name: 'Zoom into Hidden layer · 8 → 16 · ReLU' }))
    fireEvent.click(container.querySelector('[data-id="blocks.0.ff1"]')!)
    expect((screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',')[0]).toBe('0.75')
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeEnabled()
  })

  it('returns from a populated model to a fresh blank graph', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, 'Connect the neurons')
    await user.click(screen.getByRole('button', { name: 'Preset gallery' }))
    expect(await screen.findByRole('region', { name: 'Learning presets' })).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Blank builder' }))
    expect(await screen.findByRole('heading', { name: 'Backprop Builder' })).toBeVisible()
    expect(screen.getByText('0 nodes, 0 edges')).toBeVisible()
    expect(container.querySelectorAll('.react-flow__node')).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Preset gallery' }))
    expect(await screen.findByRole('region', { name: 'Learning presets' })).toBeVisible()
  })

  it('protects a held-out example from point-wise training while allowing dataset training', async () => {
    const user = userEvent.setup()
    renderStudio()
    await openPreset(user, 'A line that learns')
    await user.click(screen.getByRole('button', { name: 'Use sample 1, test set' }))
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run 10 training steps' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run forward' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Train 1 epoch' })).toBeEnabled()
    fireEvent.change(screen.getByLabelText('Dataset sample'), { target: { value: '1' } })
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeEnabled()
  })

  it('opens CNN classifier arithmetic at its only image row and edits the shared matrix coordinate', async () => {
    const user = userEvent.setup()
    const { container } = renderStudio()
    await openPreset(user, 'Read a handwritten digit')
    expect(screen.getByRole('heading', { name: 'A tiny digit recognizer' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Inspect 36 features → 10 digit scores neuron 1' }))
    const output = container.querySelector('[data-id="inspect:digit-classifier:0:output"]')!
    expect(output).toBeInTheDocument()
    expect(output).not.toHaveTextContent('sigmoid')
    const row = screen.getByLabelText(/Inspected (token|sample) row/)
    expect(row).toHaveValue(1)
    expect(row).toHaveAttribute('max', '1')
    fireEvent.click(container.querySelector('[data-id="inspect:digit-classifier:0:w0"]')!)
    fireEvent.change(screen.getByLabelText('Edit tensor coordinate'), { target: { value: '.875' } })
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expect(screen.getByLabelText('Edit tensor coordinate')).toHaveValue(.875)
    await user.click(screen.getByRole('checkbox', { name: 'Zoom reveals detail' }))
    await user.click(screen.getByRole('button', { name: 'Zoom out of 36 features → 10 digit scores' }))
    await user.click(screen.getByRole('button', { name: 'Zoom into 36 features → 10 digit scores' }))
    fireEvent.click(container.querySelector('[data-id="classifier-weights"]')!)
    expect((screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',')[0]).toBe('0.875')
  })

  it('protects held-out digit images and trains after choosing a training example', async () => {
    const user = userEvent.setup()
    renderStudio()
    await openPreset(user, 'Read a handwritten digit')
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run forward' })).toBeEnabled()
    fireEvent.change(screen.getByLabelText('Digit dataset split'), { target: { value: 'train' } })
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Run one full training step' }))
    expect(screen.getByText('Epoch 1')).toBeInTheDocument()
  })
})
