import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { GraphCanvas } from './GraphCanvas'
import { createNode } from '../domain/examples'
import type { GraphModel } from '../domain/types'

it('displays connected variable names, saves edited slots, and follows renames and rewiring', async () => {
  const input = createNode('input', 0), standard = createNode('standardize', 0)
  standard.params = { outputName: 'u', standardization: { mean: [0], scale: [1], count: 2 } }
  const weight = { ...createNode('weight', 0), label: 'w' }, arithmetic = createNode('arithmetic', 0)
  const graph: GraphModel = { learningRate: .1, nodes: [input, standard, weight, arithmetic], edges: [
    { id: 'raw', source: input.id, target: standard.id, inputSlot: 0 },
    { id: 'left', source: standard.id, target: arithmetic.id, inputSlot: 0 },
    { id: 'right', source: weight.id, target: arithmetic.id, inputSlot: 1 },
  ] }
  const onExpressionChange = vi.fn()
  const canvas = (model: GraphModel) => <GraphCanvas graph={model} showMath showGradient phase="edit"
    onGraphChange={vi.fn()} onSelectionChange={vi.fn()} onCreateNode={vi.fn()} onCancelPendingPlacement={vi.fn()}
    onNodeValueChange={vi.fn()} onActivationChange={vi.fn()} onGroupCreate={vi.fn()} onGroupExplode={vi.fn()} onGroupMove={vi.fn()}
    onExpressionChange={onExpressionChange}/>
  const { rerender } = render(canvas(graph))
  const field = await screen.findByRole('textbox', { name: 'Arithmetic expression' })
  expect(field).toHaveValue('u * w')
  fireEvent.change(field, { target: { value: 'u + w^2' } })
  fireEvent.blur(field)
  expect(onExpressionChange).toHaveBeenLastCalledWith(arithmetic.id, 'x1 + x2^2')
  arithmetic.params.expression = 'x1 + x2^2'
  standard.params.outputName = 'scaled'
  rerender(canvas({ ...graph, nodes: [...graph.nodes] }))
  await waitFor(() => expect(field).toHaveValue('scaled + w^2'))
  rerender(canvas({ ...graph, edges: graph.edges.map(edge => edge.id === 'left' ? { ...edge, source: weight.id } : edge.id === 'right' ? { ...edge, source: standard.id } : edge) }))
  await waitFor(() => expect(field).toHaveValue('w + scaled^2'))
  fireEvent.change(field, { target: { value: 'w +' } })
  fireEvent.blur(field)
  expect(field).toHaveClass('is-invalid')
  expect(onExpressionChange).toHaveBeenCalledTimes(1)
  rerender(canvas({ ...graph, edges: [] }))
  await waitFor(() => expect(field).toHaveValue('x1 + x2^2'))
})
