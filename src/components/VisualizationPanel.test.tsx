import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { forwardPass } from '../domain/engine'
import { scalarValue, tensorValue } from '../domain/tensor'
import type { GraphModel } from '../domain/types'
import { VisualizationPanel } from './VisualizationPanel'

describe('VisualizationPanel', () => {
  it('plots target data points and current predictions for a single-input network', () => {
    const graph = forwardPass(singleInputGraph()).graph
    const { container } = render(<VisualizationPanel graph={graph} />)

    expect(screen.getByRole('img', { name: /Input-output visualization/i })).toBeInTheDocument()
    expect(screen.getByText('x-axis: x')).toBeInTheDocument()
    expect(screen.getByText('Target data')).toBeInTheDocument()
    expect(screen.getByText('Predictions')).toBeInTheDocument()
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(3)
    expect(container.querySelectorAll('.visualization-prediction-point')).toHaveLength(3)
  })

  it('shows a two-input heatmap with a switch between target and current predictions', async () => {
    const user = userEvent.setup()
    const graph = forwardPass(twoInputGraph()).graph
    const { container } = render(<VisualizationPanel graph={graph} />)

    expect(screen.getByRole('img', { name: /Target heatmap/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Target/i })).toHaveAttribute('aria-pressed', 'true')
    expect(container.querySelectorAll('.visualization-heatmap-cell')).toHaveLength(4)

    await user.click(screen.getByRole('button', { name: /Predictions/i }))

    expect(screen.getByRole('img', { name: /Prediction heatmap/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Predictions/i })).toHaveAttribute('aria-pressed', 'true')
    expect(container.querySelectorAll('.visualization-heatmap-cell')).toHaveLength(4)
  })
})

function singleInputGraph(): GraphModel {
  return {
    learningRate: 0.1,
    nodes: [
      {
        id: 'x',
        type: 'input',
        label: 'x',
        position: { x: 40, y: 80 },
        params: { value: tensorValue([3], [0, 1, 2]) },
      },
      { id: 'w', type: 'weight', label: 'w', position: { x: 40, y: 240 }, params: { value: scalarValue(2) } },
      { id: 'b', type: 'bias', label: 'b', position: { x: 320, y: 240 }, params: { value: scalarValue(1) } },
      { id: 'mul', type: 'multiply', label: 'x * w', position: { x: 320, y: 120 }, params: {} },
      { id: 'add', type: 'add', label: 'xw + b', position: { x: 600, y: 160 }, params: {} },
      {
        id: 'target',
        type: 'target',
        label: 'y',
        position: { x: 600, y: 340 },
        params: { value: tensorValue([3], [1, 3, 6]) },
      },
      { id: 'loss', type: 'loss', label: 'loss', position: { x: 880, y: 220 }, params: {} },
    ],
    edges: [
      { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
      { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
      { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
      { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
      { id: 'add-loss', source: 'add', target: 'loss', inputSlot: 0 },
      { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
    ],
  }
}

function twoInputGraph(): GraphModel {
  return {
    learningRate: 0.1,
    nodes: [
      {
        id: 'x1',
        type: 'input',
        label: 'x1',
        position: { x: 40, y: 80 },
        params: { value: tensorValue([2, 2], [0, 1, 0, 1]) },
      },
      {
        id: 'x2',
        type: 'input',
        label: 'x2',
        position: { x: 40, y: 240 },
        params: { value: tensorValue([2, 2], [0, 0, 1, 1]) },
      },
      { id: 'add', type: 'add', label: 'x1 + x2', position: { x: 340, y: 160 }, params: {} },
      {
        id: 'target',
        type: 'target',
        label: 'y',
        position: { x: 340, y: 340 },
        params: { value: tensorValue([2, 2], [0, 1, 1, 2]) },
      },
      { id: 'loss', type: 'loss', label: 'loss', position: { x: 620, y: 220 }, params: {} },
    ],
    edges: [
      { id: 'x1-add', source: 'x1', target: 'add', inputSlot: 0 },
      { id: 'x2-add', source: 'x2', target: 'add', inputSlot: 1 },
      { id: 'add-loss', source: 'add', target: 'loss', inputSlot: 0 },
      { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
    ],
  }
}
