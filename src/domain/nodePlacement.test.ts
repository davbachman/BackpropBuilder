import { describe, expect, it } from 'vitest'
import { LESSONS } from '../learning/presets'
import { compactVisualHierarchy, layoutContinuousScene } from './continuousScene'
import { createNode, createEmptyGraph } from './examples'
import { createModelPreset } from './modelPresets'
import { projectDenseNeurons } from './neuronProjection'
import { placeCanvasNode } from './nodePlacement'
import { layoutSemanticGraph, type SemanticLayout } from './semanticLayout'
import type { GraphModel } from './types'

const geometry = (graph: GraphModel) => graph.view?.semanticZoom === false
  ? layoutSemanticGraph(graph) : layoutContinuousScene(compactVisualHierarchy(graph))

function expectUnmoved(before: SemanticLayout, after: SemanticLayout) {
  for (const kind of ['nodes', 'groups'] as const) for (const [id, rect] of before[kind]) {
    const actual = after[kind].get(id)!
    expect(actual.x, `${kind}:${id} x`).toBeCloseTo(rect.x, 8)
    expect(actual.y, `${kind}:${id} y`).toBeCloseTo(rect.y, 8)
    expect(actual.width).toBeCloseTo(rect.width, 8)
    expect(actual.height).toBeCloseTo(rect.height, 8)
  }
}

describe.each([true, false])('canvas insertion with continuous zoom %s', semanticZoom => {
  it.each(LESSONS)('preserves every existing block in $id through successive placements', ({ id }) => {
    let graph = createModelPreset(id)
    graph.view = { ...graph.view!, semanticZoom, viewport: { x: 125, y: -40, zoom: 2.5 },
      layoutOffsets: { [graph.nodes[0].id]: { x: -15, y: 12 }, 'visual-group:network': { x: 25, y: -30 } } }
    for (const [index, type] of (['weight', 'add', 'target'] as const).entries()) {
      const original = structuredClone(graph)
      const before = geometry(graph)
      const position = { x: -190.25 + index * 340, y: 318.125 - index * 80 }
      const next = placeCanvasNode(graph, { ...createNode(type, 99 + index), position })
      const after = geometry(next)
      expectUnmoved(before, after)
      const added = after.nodes.get(next.nodes.at(-1)!.id)!
      expect(added.x).toBeCloseTo(position.x, 8)
      expect(added.y).toBeCloseTo(position.y, 8)
      expect(next.nodes.slice(0, -1)).toEqual(original.nodes)
      expect(next.edges).toBe(graph.edges)
      expect(next.groups).toBe(graph.groups)
      expect(next.view?.viewport).toEqual(original.view?.viewport)
      expect(graph).toEqual(original)
      graph = next
    }
  })
})

it('preserves projected neuron geometry without copying virtual calculations into the model', () => {
  const graph = createModelPreset('decoder')
  const inspectedNeuron = { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 }
  graph.view = { ...graph.view!, inspectedNeuron }
  const display = projectDenseNeurons(graph, inspectedNeuron).graph
  const before = geometry(display)
  const next = placeCanvasNode(graph, { ...createNode('weight', 99), position: { x: 200, y: 100 } }, display)
  expectUnmoved(before, geometry(projectDenseNeurons(next, inspectedNeuron).graph))
  expect(next.nodes).toHaveLength(graph.nodes.length + 1)
  expect(next.nodes.some(node => node.id.startsWith('inspect:'))).toBe(false)
})

it('keeps free-form builder placement in native canvas coordinates', () => {
  const graph = createEmptyGraph()
  const node = { ...createNode('input', 1), position: { x: -23.5, y: 174 } }
  const next = placeCanvasNode(graph, node)
  expect(next.nodes).toEqual([node])
  expect(next.view).toEqual(graph.view)
})
