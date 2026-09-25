import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { tensorValue } from './domain/tensor'
import { createModelPreset } from './domain/modelPresets'
import type { GraphModel } from './domain/types'

vi.mock('./domain/examples', async (importOriginal) => {
  const original = await importOriginal<typeof import('./domain/examples')>()
  return {
    ...original,
    createStarterGraph: () => createTwoInputGraph(),
  }
})

import App from './App'

describe('Backprop Builder visualization data edits', () => {
  it('shows the complete linear dataset without the old experiment and instruction panels', () => {
    const { container } = render(<App initialGraph={createModelPreset('linear')} />)
    expect(screen.queryByText('Fit a function')).not.toBeInTheDocument()
    expect(screen.queryByText('One model. Every scale.')).not.toBeInTheDocument()
    expect(screen.queryByText('Build with the canvas')).not.toBeInTheDocument()
    expect(screen.queryByText('Build a CNN from scratch')).not.toBeInTheDocument()
    expect(screen.queryByText('Build a transformer from scratch')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Visualization' }))
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(20)
  })

  it('fits the first model to the training split and updates the loss beneath its line', async () => {
    const { container } = render(<App initialGraph={createModelPreset('linear')} />)
    fireEvent.click(container.querySelector('[data-id="model-data"]')!)
    fireEvent.click(screen.getByRole('tab', { name: 'Visualization' }))
    const trainingLoss = () => Number(screen.getByLabelText('Training loss').textContent)
    const heldOutLoss = () => Number(screen.getByLabelText('Held-out loss').textContent)
    const line = () => container.querySelector('.visualization-prediction-line')?.getAttribute('d')
    const initialLoss = trainingLoss(), initialHeldOutLoss = heldOutLoss(), initialLine = line()

    fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
    fireEvent.click(screen.getByRole('button', { name: 'Train 1 epoch' }))
    await waitFor(() => expect(screen.queryByRole('status')?.textContent).toMatch(/Completed 1 epoch/))
    expect(screen.getByText('Epoch 1')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Visualization' }))
    const afterOneEpoch = trainingLoss()
    expect(afterOneEpoch).toBeLessThan(initialLoss)
    expect(line()).not.toBe(initialLine)

    fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
    fireEvent.click(screen.getByRole('button', { name: 'Train 5 epochs' }))
    await waitFor(() => expect(screen.getByText('Epoch 6')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('tab', { name: 'Visualization' }))
    expect(trainingLoss()).toBeLessThan(afterOneEpoch)
    expect(heldOutLoss()).toBeLessThan(initialHeldOutLoss)
    expect(screen.getByText('Epoch 6')).toBeInTheDocument()
    expect(container.querySelectorAll('.visualization-target-point')).toHaveLength(20)
  })

  it('updates two-input target point colors immediately when target values change', () => {
    const { container } = render(<App />)

    fireFileMenuItem(/^Starter$/i)
    fireEvent.click(screen.getByRole('tab', { name: 'Visualization' }))
    const firstTargetPoint = () => container.querySelector('.visualization-target-point')
    const initialFill = firstTargetPoint()?.getAttribute('fill')

    fireEvent.change(screen.getByDisplayValue('[0,1,1,2]'), { target: { value: '[2,1,1,0]' } })

    expect(firstTargetPoint()?.getAttribute('fill')).not.toBe(initialFill)
  })
})

function fireFileMenuItem(name: RegExp): void {
  fireEvent.click(screen.getByRole('button', { name: /^File$/i }))
  fireEvent.click(screen.getByRole('menuitem', { name }))
}

function createTwoInputGraph(): GraphModel {
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
