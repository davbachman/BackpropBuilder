import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { GraphCanvas } from './GraphCanvas'
import { createStarterGraph } from '../domain/examples'
import { parseCustomCsv } from '../domain/customCsv'
import { forwardPass } from '../domain/engine'
import { mergePreservingLayout as mergeNodesIntoVisualGroup, ungroupPreservingLayout as explodeVisualGroup } from '../domain/mergeLayout'

it.each([false, true])('merges and ungroups CSV models with explicit splits (dataset selected: %s)', async (includeDataset) => {
  const user = userEvent.setup()
  const csv = parseCustomCsv('weight,year,mpg,split\n3000,1970,20,train\n2500,1975,25,train\n2000,1980,30,test\n3500,1972,18,test', 'cars.csv')
  const initial = createStarterGraph(true)
  initial.nodes[0].params = { dataset: 'custom-csv', customCsv: csv, datasetMode: 'sample' }
  const ids = includeDataset ? ['starter-data', 'x', 'mul'] : ['mul', 'add']
  const original = JSON.stringify(initial)
  const expectedLoss = forwardPass(initial).graph.nodes.find(node => node.id === 'loss')?.value
  let current = initial
  function Harness() {
    const [graph, setGraph] = useState(initial)
    const [groupId, setGroupId] = useState<string>()
    current = graph
    return <GraphCanvas graph={graph} selectedNodeIds={groupId ? [] : ids} selectedGroupId={groupId}
      showMath showGradient phase="edit" onGraphChange={setGraph} onSelectionChange={vi.fn()}
      onCreateNode={vi.fn()} onCancelPendingPlacement={vi.fn()} onNodeValueChange={vi.fn()}
      onActivationChange={vi.fn()} onGroupMove={vi.fn()}
      onGroupCreate={() => { const merged = mergeNodesIntoVisualGroup(graph, ids); setGraph(merged.graph); setGroupId(merged.group?.id) }}
      onGroupExplode={id => { setGraph(explodeVisualGroup(graph, id)); setGroupId(undefined) }} />
  }
  const { container } = render(<Harness />)
  await user.click(screen.getByRole('button', { name: /Merge selection/i }))
  expect(container.querySelector('[data-id="visual-group:group-1"]')).toBeInTheDocument()
  expect(current.nodes[0].params.customCsv).toEqual(csv)
  expect(current.nodes[0].params.customCsv?.splits).toEqual(['train', 'train', 'test', 'test'])
  expect(forwardPass(current).graph.nodes.find(node => node.id === 'loss')?.value).toEqual(expectedLoss)
  await user.click(screen.getByRole('button', { name: /Ungroup module/i }))
  expect(container.querySelector('[data-id="visual-group:group-1"]')).not.toBeInTheDocument()
  expect(container.querySelector('[data-id="mul"]')).toBeInTheDocument()
  expect(JSON.stringify(initial)).toBe(original)
})

it('releases the selected group when zoom reveals its selectable contents', async () => {
  const merged = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add'])
  const onSelectionChange = vi.fn()
  const props = {
    selectedNodeIds: [], selectedGroupId: merged.group!.id, showMath: true, showGradient: true,
    phase: 'edit' as const, onGraphChange: vi.fn(), onSelectionChange,
    onCreateNode: vi.fn(), onCancelPendingPlacement: vi.fn(), onNodeValueChange: vi.fn(),
    onActivationChange: vi.fn(), onGroupMove: vi.fn(), onGroupCreate: vi.fn(), onGroupExplode: vi.fn(),
  }
  const graph = { ...merged.graph, view: { ...merged.graph.view!, viewport: { x: 0, y: 0, zoom: .5 } } }
  const { container, rerender } = render(<GraphCanvas {...props} graph={graph} />)
  const group = () => container.querySelector('[data-id="visual-group:group-1"]')!
  expect(group()).toHaveClass('selected')
  rerender(<GraphCanvas {...props} graph={{ ...graph, view: { ...graph.view, viewport: { x: 0, y: 0, zoom: 20 } } }} />)
  await waitFor(() => expect(group()).not.toHaveClass('selected'))
  const inside = container.querySelector('[data-id="mul"]')!
  expect(inside).toHaveClass('selectable')
  fireEvent.click(inside)
  expect(onSelectionChange).toHaveBeenCalledWith({ nodeIds: ['mul'] })
})
