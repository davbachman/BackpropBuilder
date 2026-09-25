import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { tensorValue } from './domain/tensor'
import type { GraphModel } from './domain/types'

vi.mock('./components/GraphCanvas', () => ({
  GraphCanvas: ({ graph, onInspectEdge }: { graph: GraphModel; onInspectEdge?: (edgeId: string) => void }) => (
    <button type="button" onClick={() => onInspectEdge?.(graph.edges[0].id)}>Inspect wire</button>
  ),
}))

import App from './App'

describe('wire inspection in the right sidebar', () => {
  it('opens Details from another tab and shows every value in the wire', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x', type: 'input', label: 'x', params: { value: tensorValue([4], [0, 1, 2, 3]) }, position: { x: 0, y: 0 } },
        { id: 'relu', type: 'activation', label: 'ReLU', params: { activation: 'relu' }, position: { x: 250, y: 0 } },
      ],
      edges: [{ id: 'x-relu', source: 'x', target: 'relu', inputSlot: 0 }],
    }
    render(<App initialGraph={graph} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
    fireEvent.click(screen.getByRole('button', { name: 'Collapse right sidebar' }))
    fireEvent.click(screen.getByRole('button', { name: 'Inspect wire' }))

    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('button', { name: 'Collapse right sidebar' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'One connection' })).toBeInTheDocument()
    expect(screen.getByText('Forward value →').parentElement).toHaveTextContent('[4] [0.000, 1.000, 2.000, 3.000]')
  })
})
