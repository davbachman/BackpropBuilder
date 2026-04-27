import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import appCss from '../App.css?raw'
import { forwardPass } from '../domain/engine'
import { scalarValue, tensorValue } from '../domain/tensor'
import type { GraphModel } from '../domain/types'
import { VisualizationPanel } from './VisualizationPanel'

describe('VisualizationPanel', () => {
  it('does not force all target points to use one CSS fill color', () => {
    expect(appCss).not.toMatch(/\.visualization-target-point\s*{[^}]*\bfill\s*:/)
  })

  it('plots target data points and current predictions for a single-input network', () => {
    const graph = forwardPass(singleInputGraph()).graph
    const { container } = render(<VisualizationPanel graph={graph} />)

    expect(screen.getByRole('img', { name: /Input-output visualization/i })).toBeInTheDocument()
    expect(screen.getByText('x-axis: x')).toBeInTheDocument()
    expect(screen.getByText('Target data')).toBeInTheDocument()
    expect(screen.getByText('Predictions')).toBeInTheDocument()
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(3)
    expect(container.querySelector('.visualization-prediction-line')).toHaveAttribute('data-sample-count', '80')
  })

  it('keeps target point positions fixed when only predictions change', () => {
    const firstGraph = forwardPass(singleInputGraph()).graph
    const secondGraph = forwardPass(singleInputGraph({ weight: 10, bias: 10 })).graph
    const { container, rerender } = render(<VisualizationPanel graph={firstGraph} />)
    const firstTargetPositions = targetPointPositions(container)

    rerender(<VisualizationPanel graph={secondGraph} />)

    expect(targetPointPositions(container)).toEqual(firstTargetPositions)
  })

  it('overlays target data points on a sampled prediction heatmap for two-input networks', () => {
    const graph = forwardPass(twoInputGraph()).graph
    const { container } = render(<VisualizationPanel graph={graph} />)

    expect(screen.getByRole('img', { name: /Two-input prediction heatmap/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Target/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Predictions/i })).not.toBeInTheDocument()
    expect(screen.getByText('x-axis: x1')).toBeInTheDocument()
    expect(screen.getByText('y-axis: x2')).toBeInTheDocument()
    expect(container.querySelectorAll('.visualization-heatmap-cell')).toHaveLength(625)
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(4)
  })

  it('visualizes a network whose input and target values come from a dataset node', () => {
    const { container } = render(<VisualizationPanel graph={datasetBackedSingleInputGraph()} />)

    expect(screen.getByRole('img', { name: /Input-output visualization/i })).toBeInTheDocument()
    expect(screen.getByText('x-axis: x')).toBeInTheDocument()
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(20)
    expect(container.querySelector('.visualization-prediction-line')).toHaveAttribute('data-sample-count', '80')
  })
})

function targetPointPositions(container: HTMLElement): Array<{ cx: string | null; cy: string | null }> {
  return Array.from(container.querySelectorAll('.visualization-target-point')).map((point) => ({
    cx: point.getAttribute('cx'),
    cy: point.getAttribute('cy'),
  }))
}

function singleInputGraph(values: { weight?: number; bias?: number } = {}): GraphModel {
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
      { id: 'w', type: 'weight', label: 'w', position: { x: 40, y: 240 }, params: { value: scalarValue(values.weight ?? 2) } },
      { id: 'b', type: 'bias', label: 'b', position: { x: 320, y: 240 }, params: { value: scalarValue(values.bias ?? 1) } },
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
        params: { value: tensorValue([4], [0, 1, 0, 1]) },
      },
      {
        id: 'x2',
        type: 'input',
        label: 'x2',
        position: { x: 40, y: 240 },
        params: { value: tensorValue([4], [0, 0, 1, 1]) },
      },
      { id: 'add', type: 'add', label: 'x1 + x2', position: { x: 340, y: 160 }, params: {} },
      {
        id: 'target',
        type: 'target',
        label: 'y',
        position: { x: 340, y: 340 },
        params: { value: tensorValue([4], [0, 1, 1, 2]) },
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

function datasetBackedSingleInputGraph(): GraphModel {
  return {
    learningRate: 0.1,
    nodes: [
      {
        id: 'dataset',
        type: 'dataset',
        label: 'dataset',
        position: { x: 20, y: 160 },
        params: { dataset: 'line-1d' },
      },
      {
        id: 'x',
        type: 'input',
        label: 'x',
        position: { x: 260, y: 80 },
        params: { value: tensorValue([1], [0]) },
      },
      { id: 'w', type: 'weight', label: 'w', position: { x: 260, y: 240 }, params: { value: scalarValue(1) } },
      { id: 'b', type: 'bias', label: 'b', position: { x: 540, y: 240 }, params: { value: scalarValue(0) } },
      { id: 'mul', type: 'multiply', label: 'x * w', position: { x: 540, y: 120 }, params: {} },
      { id: 'add', type: 'add', label: 'xw + b', position: { x: 820, y: 160 }, params: {} },
      {
        id: 'target',
        type: 'target',
        label: 'y',
        position: { x: 820, y: 340 },
        params: { value: tensorValue([1], [0]) },
      },
      { id: 'loss', type: 'loss', label: 'loss', position: { x: 1100, y: 220 }, params: {} },
    ],
    edges: [
      { id: 'dataset-x', source: 'dataset', sourceSlot: 0, target: 'x', inputSlot: 0 },
      { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
      { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
      { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
      { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
      { id: 'add-loss', source: 'add', target: 'loss', inputSlot: 0 },
      { id: 'dataset-target', source: 'dataset', sourceSlot: 1, target: 'target', inputSlot: 0 },
      { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
    ],
  }
}
