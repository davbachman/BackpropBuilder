import { describe, expect, it } from 'vitest'
import { compactVisualHierarchy, layoutContinuousScene } from './continuousScene'
import { preserveLayoutAfterDeletion } from './deletionLayout'
import { removeNodesFromVisualGroups } from './grouping'
import { createModelPreset } from './modelPresets'
import { createNode } from './examples'
import { placeCanvasNode } from './nodePlacement'
import { builderCardHeight, builderCardWidth } from './builderGeometry'
import { LESSONS } from '../learning/presets'
import type { GraphModel } from './types'

const geometry = (graph: GraphModel) => graph.groups?.length || graph.view?.preservedLayouts
  ? layoutContinuousScene(compactVisualHierarchy(graph))
  : { nodes: new Map(graph.nodes.map(node => [node.id, { ...node.position, width: builderCardWidth(node), height: builderCardHeight(node) }])), groups: new Map() }
const remove = (graph: GraphModel, id: string) => preserveLayoutAfterDeletion(graph, {
  ...graph,
  nodes: graph.nodes.filter(node => node.id !== id),
  edges: graph.edges.filter(edge => edge.source !== id && edge.target !== id),
  groups: removeNodesFromVisualGroups(graph, new Set([id])),
})
function unchanged(before: ReturnType<typeof geometry>, after: ReturnType<typeof geometry>) {
  for (const kind of ['nodes', 'groups'] as const) for (const [id, rect] of after[kind]) {
    const previous = before[kind].get(id)!
    for (const key of ['x', 'y', 'width', 'height'] as const) expect(rect[key]).toBeCloseTo(previous[key], 8)
  }
}

describe('deletion layout preservation', () => {
  it.each(LESSONS)('retains moved and manually placed blocks across repeated deletions in $id', ({ id }) => {
    let graph = createModelPreset(id)
    graph.view = { ...graph.view!, layoutOffsets: { [graph.nodes[0].id]: { x: -20, y: 10 } } }
    graph = placeCanvasNode(graph, { ...createNode('weight', 99), position: { x: -300, y: 180 } })
    const before = geometry(graph)
    const original = structuredClone(graph)
    const first = remove(graph, graph.nodes[0].id)
    unchanged(before, geometry(first))
    unchanged(before, geometry(remove(first, 'weight-99')))
    expect(graph).toEqual(original)
  })

  it('keeps the last member in place when deleting a node dissolves its group', () => {
    const first = createNode('input', 1), second = createNode('activation', 1)
    const graph: GraphModel = {
      nodes: [first, second], edges: [{ id: 'wire', source: first.id, target: second.id, inputSlot: 0 }], learningRate: .1,
      groups: [{ id: 'pair', label: 'Pair', nodeIds: [first.id, second.id], position: { x: 100, y: 100 }, dimensions: { width: 238, height: 136 } }],
    }
    const before = geometry(graph)
    const after = remove(graph, first.id)
    expect(after.groups).toEqual([])
    unchanged(before, geometry(after))
  })
})
