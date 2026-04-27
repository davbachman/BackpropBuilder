import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { tensorValue } from './domain/tensor'
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
  it('updates two-input target point colors immediately when target values change', () => {
    const { container } = render(<App />)

    fireFileMenuItem(/^Starter$/i)
    fireEvent.click(screen.getByRole('button', { name: /Show visualization/i }))
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
