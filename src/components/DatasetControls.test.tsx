import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { parameterValues } from '../domain/engine'
import { createModelPreset } from '../domain/modelPresets'
import type { GraphModel } from '../domain/types'
import { DatasetControls } from './DatasetControls'

function Harness({ initial, changed }: { initial: GraphModel; changed: (graph: GraphModel, epochs?: number, loss?: number) => void }) {
  const [graph, setGraph] = useState(initial)
  return <DatasetControls graph={graph} onGraphChange={(next, epochs, loss) => { setGraph(next); changed(next, epochs, loss) }} />
}

describe('dataset experiment controls', () => {
  it('selects a plotted sample into the actual graph inputs and target', () => {
    const changed = vi.fn()
    render(<Harness initial={createModelPreset('linear')} changed={changed} />)
    fireEvent.click(screen.getByRole('button', { name: 'Use sample 1, test set' }))
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'input-0')?.value?.data).toEqual([-2.4])
    expect(next.nodes.find(node => node.id === 'target')?.value?.data).toEqual([-3.62])
    expect(screen.getByLabelText('Dataset sample')).toHaveValue('0')
    expect(screen.getByText('15 training')).toBeInTheDocument()
    expect(screen.getByText('5 held out')).toBeInTheDocument()
  })

  it('switches to included cubic data and keeps that choice in the model', () => {
    const changed = vi.fn()
    render(<Harness initial={createModelPreset('linear')} changed={changed} />)
    fireEvent.change(screen.getByLabelText('Training dataset'), { target: { value: 'cubic-1d' } })
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'target')?.value?.data).toEqual([-8.97])
    expect(next.groups?.find(group => group.id === 'network')?.detail?.datasetKind).toBe('cubic-1d')
    expect(screen.getByLabelText('Training dataset')).toHaveValue('cubic-1d')
  })

  it('trains the canvas weights, updates the regression curve and reports held-out loss', async () => {
    const initial = createModelPreset('linear'), changed = vi.fn()
    const { container } = render(<Harness initial={initial} changed={changed} />)
    const beforeLoss = Number(screen.getByLabelText('Training loss').textContent)
    const beforeCurve = container.querySelector('.dataset-regression-curve')?.getAttribute('d')
    fireEvent.click(screen.getByRole('button', { name: 'Train 25 epochs' }))
    await waitFor(() => expect(changed).toHaveBeenCalled(), { timeout: 5000 })
    const [next, epochs, loss] = changed.mock.calls[0]
    expect(epochs).toBe(25)
    expect(loss).toBeLessThan(beforeLoss / 10)
    expect(parameterValues(next)).not.toEqual(parameterValues(initial))
    expect(container.querySelector('.dataset-regression-curve')?.getAttribute('d')).not.toEqual(beforeCurve)
    expect(Number(screen.getByLabelText('Test loss').textContent)).toBeLessThan(beforeLoss)
  })

  it('uses the included circle dataset with a real class-probability decision map', () => {
    const changed = vi.fn()
    const { container } = render(<Harness initial={createModelPreset('playground')} changed={changed} />)
    expect(screen.getByRole('group', { name: 'Decision boundary and dataset samples' })).toBeInTheDocument()
    expect(container.querySelectorAll('.dataset-boundary-cell')).toHaveLength(400)
    fireEvent.change(screen.getByLabelText('Training dataset'), { target: { value: 'circle-center' } })
    fireEvent.change(screen.getByLabelText('Dataset sample'), { target: { value: '10' } })
    const next: GraphModel = changed.mock.calls.at(-1)![0]
    expect(next.nodes.find(node => node.id === 'input-0')?.value?.data).toEqual([1.68])
    expect(next.nodes.find(node => node.id === 'input-1')?.value?.data).toEqual([.08])
    expect(next.nodes.find(node => node.id === 'target')?.value).toEqual({ shape: [1], data: [1] })
    expect(next.nodes.find(node => node.id === 'network-probabilities')?.value?.shape).toEqual([1, 2])
  })
})
