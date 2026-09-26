import { describe, expect, it } from 'vitest'
import { LESSONS } from '../learning/presets'
import { compactVisualHierarchy, layoutContinuousScene, routeContinuousScene } from './continuousScene'
import { forwardPass } from './engine'
import { createNode } from './examples'
import { connectGraphNodes } from './graphEditing'
import { preserveLayoutForWiring } from './layoutState'
import { createModelPreset } from './modelPresets'
import { projectDenseNeurons } from './neuronProjection'
import { placeCanvasNode } from './nodePlacement'
import { builderCardHeight, builderCardWidth } from './builderGeometry'
import type { SemanticLayout } from './semanticLayout'
import type { GraphModel } from './types'

const geometry = (graph: GraphModel): SemanticLayout => graph.groups?.length
  ? layoutContinuousScene(compactVisualHierarchy(graph))
  : { nodes: new Map(graph.nodes.map(node => [node.id, { ...node.position, width: builderCardWidth(node), height: builderCardHeight(node) }])), groups: new Map() }

describe.each([true, false])('wiring without rearrangement, continuous zoom %s', semanticZoom => {
  it.each(LESSONS)('keeps every level of $id fixed when connecting, replacing and removing wires', ({ id }) => {
    let graph = createModelPreset(id)
    graph.view = { ...graph.view!, semanticZoom, expandedGroupIds: graph.groups?.map(group => group.id) ?? [],
      viewport: { x: -180, y: 200, zoom: 1.7 }, layoutOffsets: { [graph.nodes[0].id]: { x: -20, y: 10 } } }
    graph = placeCanvasNode(graph, { ...createNode('weight', 99), position: { x: -300, y: 180 } })
    const original = structuredClone(graph)
    const before = geometry(graph)
    const oldEdge = graph.edges[0]
    const connected = connectGraphNodes(graph, { source: 'weight-99', target: oldEdge.target, targetHandle: `in-${oldEdge.inputSlot ?? 0}` }, () => 'replacement')!
    expect(connected.edges.find(edge => edge.id === 'replacement')?.source).toBe('weight-99')
    expect(connected.edges.some(edge => edge.id === oldEdge.id)).toBe(false)
    expect(geometry(connected)).toEqual(before)
    expect(connected.view?.viewport).toEqual(graph.view?.viewport)
    expect(connected.nodes).toEqual(graph.nodes)
    expect(graph).toEqual(original)

    const removed = { ...preserveLayoutForWiring(connected), edges: connected.edges.filter(edge => edge.id !== 'replacement') }
    expect(geometry(removed)).toEqual(before)
    const restored = connectGraphNodes(removed, { source: oldEdge.source, sourceHandle: `out-${oldEdge.sourceSlot ?? 0}`, target: oldEdge.target, targetHandle: `in-${oldEdge.inputSlot ?? 0}` })!
    expect(geometry(restored)).toEqual(before)
    const added = placeCanvasNode(connected, { ...createNode('bias', 99), position: { x: 215, y: -100 } })
    const addedLayout = geometry(added)
    for (const kind of ['nodes', 'groups'] as const) for (const [nodeId, rect] of before[kind]) {
      const actual = addedLayout[kind].get(nodeId)!
      expect(actual.x).toBeCloseTo(rect.x, 8)
      expect(actual.y).toBeCloseTo(rect.y, 8)
      expect(actual.width).toBe(rect.width)
      expect(actual.height).toBe(rect.height)
    }
  })
})

it('updates the real calculation and routes the edited wires while keeping nested neuron geometry fixed', () => {
  const graph = createModelPreset('linear')
  const sum = graph.nodes.find(node => node.type === 'add')!, weight = graph.nodes.find(node => node.type === 'weight')!
  const edited = connectGraphNodes(graph, { source: weight.id, target: sum.id, targetHandle: 'in-1' }, () => 'weight-to-sum')!
  expect(geometry(edited)).toEqual(geometry(graph))
  expect(forwardPass(graph).graph.nodes.find(node => node.type === 'activation')!.value!.data[0]).toBeCloseTo(1.9)
  expect(forwardPass(edited).graph.nodes.find(node => node.type === 'activation')!.value!.data[0]).toBeCloseTo(3.9)
  const display = compactVisualHierarchy(edited)
  const routes = routeContinuousScene(display, layoutContinuousScene(display))
  expect(routes.some(wire => wire.edgeId === 'weight-to-sum')).toBe(true)
  expect(routes.every(wire => edited.edges.some(edge => edge.id === wire.edgeId))).toBe(true)
})

it('keeps projected neurons stable through wiring changes and switching the inspected neuron', () => {
  const graph = createModelPreset('decoder')
  const edge = graph.edges[0]
  const edited = connectGraphNodes(graph, { source: graph.nodes.find(node => node.type === 'weight')!.id, target: edge.target, targetHandle: `in-${edge.inputSlot ?? 0}` })!
  for (const unitIndex of [0, 3]) {
    const focus = { groupId: 'blocks.0.ff1.layer', unitIndex, row: 0 }
    expect(geometry(projectDenseNeurons(edited, focus).graph)).toEqual(geometry(projectDenseNeurons(graph, focus).graph))
  }
  expect(edited.view!.layoutEdges!.some(edge => edge.id.startsWith('inspect:'))).toBe(false)
})
