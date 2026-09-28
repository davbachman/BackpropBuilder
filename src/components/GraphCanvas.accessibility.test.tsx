import { render, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { GraphCanvas } from './GraphCanvas'
import { createModelPreset } from '../domain/modelPresets'
import { projectDenseNeurons } from '../domain/neuronProjection'
import { tensorValue } from '../domain/tensor'

const callbacks = {
  onGraphChange: vi.fn(), onSelectionChange: vi.fn(), onCreateNode: vi.fn(),
  onCancelPendingPlacement: vi.fn(), onNodeValueChange: vi.fn(), onActivationChange: vi.fn(),
  onGroupCreate: vi.fn(), onGroupExplode: vi.fn(), onGroupMove: vi.fn(),
}

it('keeps covered node controls inert and exposes their full values only after zooming in', async () => {
  const graph = createModelPreset('linear')
  const weight = graph.nodes.find(node => node.id === 'layer-0/weight-0-0')!
  weight.value = tensorValue([3], [1, 2, 3])
  weight.params.value = weight.value
  graph.view = { ...graph.view!, viewport: { x: 0, y: 0, zoom: .1 } }
  const props = { ...callbacks, graph, showMath: true, showGradient: true, phase: 'edit' }
  const { container, rerender } = render(<GraphCanvas {...props} />)
  const node = () => container.querySelector('[data-id="layer-0/weight-0-0"] .builder-node')!
  expect(node()).toHaveAttribute('inert')
  expect(node()).toHaveAttribute('aria-hidden', 'true')
  expect(node().querySelector('input')!.closest('[inert]')).toBe(node())
  expect(node().querySelector('[data-tooltip]')).toBeNull()

  rerender(<GraphCanvas {...props} graph={{ ...graph, view: { ...graph.view!, viewport: { x: 0, y: 0, zoom: 20 } } }} />)
  await waitFor(() => expect(node()).not.toHaveAttribute('inert'))
  expect(node()).toHaveAttribute('aria-hidden', 'false')
  expect(node().querySelector('.node-metrics [data-tooltip]')).toHaveAttribute('data-tooltip', 'out [3] [1.000, 2.000, 3.000]')
})

it('offers coordinate values without unsupported projected topology or activation controls', () => {
  const graph = createModelPreset('decoder')
  const displayGraph = projectDenseNeurons(graph, { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 }).graph
  const { container } = render(<GraphCanvas {...callbacks} graph={graph} displayGraph={displayGraph} showMath showGradient phase="edit" />)
  const projected = (suffix: string) => container.querySelector(`[data-id="inspect:blocks.0.ff1.layer:0:${suffix}"]`)!
  expect(projected('w0').querySelector('input')).toBeInTheDocument()
  expect(projected('bias').querySelector('input')).toBeInTheDocument()
  expect(projected('sum').querySelector('.node-add-input-button')).toBeNull()
  expect(projected('output').querySelector('select')).toBeNull()
})
