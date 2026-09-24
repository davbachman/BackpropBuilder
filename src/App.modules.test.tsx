import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App'
import { createSingleNeuronGraph } from './domain/examples'
import { mergeNodesIntoVisualGroup } from './domain/grouping'

function nestedGraph() {
  let graph = createSingleNeuronGraph('identity', { x: 2, w: 3, b: 1, target: 9 })
  graph = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul']).graph
  graph = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  graph = mergeNodesIntoVisualGroup(graph, ['target', 'loss']).graph
  graph.view = { ...graph.view!, semanticZoom: false }
  return graph
}

describe('module navigation in the builder', () => {
  it('opens nested arithmetic and closes back while preserving selection and a collapsed neighbor', async () => {
    const user = userEvent.setup()
    const { container } = render(<App initialGraph={nestedGraph()} />)
    expect(container.querySelectorAll('.builder-node')).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Zoom into Group 2' }))
    expect(screen.getByRole('button', { name: 'Zoom into Group 3' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Zoom into Group 1' }))
    const multiplication = container.querySelector('[data-id="mul"]')!
    fireEvent.click(multiplication)
    expect(multiplication).toHaveClass('selected')
    expect(multiplication).toHaveTextContent('6.000')
    await user.click(screen.getByRole('button', { name: 'Zoom out of Group 2' }))
    expect(container.querySelector('[data-id="mul"]')).not.toBeInTheDocument()
    expect(screen.getByText('8 nodes, 7 edges, 3 groups')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Zoom into Group 2' }))
    expect(container.querySelector('[data-id="mul"]')).toHaveClass('selected')
    expect(container.querySelector('[data-id="mul"]')).toHaveTextContent('6.000')
    expect(screen.getByRole('button', { name: 'Zoom into Group 3' })).toBeInTheDocument()
  })

  it('undoes module navigation without erasing the module', async () => {
    const user = userEvent.setup()
    render(<App initialGraph={nestedGraph()} />)
    await user.click(screen.getByRole('button', { name: 'Zoom into Group 2' }))
    fireEvent.keyDown(document, { key: 'z', ctrlKey: true })
    expect(screen.getByRole('button', { name: 'Zoom into Group 2' })).toBeInTheDocument()
    expect(screen.getByText('8 nodes, 7 edges, 3 groups')).toBeInTheDocument()
  })

  it('steps through inference without a target and disables training', async () => {
    const user = userEvent.setup()
    const graph = createSingleNeuronGraph('identity', { x: 2, w: 3, b: 1, target: 9 })
    graph.nodes = graph.nodes.filter((node) => !['loss', 'target'].includes(node.type))
    graph.edges = graph.edges.filter((edge) => edge.target !== 'loss')
    const { container } = render(<App initialGraph={graph} />)
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    for (let i = 0; i < 5; i += 1) await user.click(screen.getByRole('button', { name: /^Step$/ }))
    expect(container.querySelector('[data-id="pred"]')).toHaveTextContent('out 7.000')
    expect(screen.getByText('Epoch 0')).toBeInTheDocument()
  })
})
