import type { GraphModel } from './types'
import { describe, expect, it } from 'vitest'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { createEmptyGraph, createNode, createStarterGraph } from './examples'
import { mergeNodesIntoVisualGroup, visualGroupInterface } from './grouping'

describe('graph clipboard', () => {
  it('copies selected nodes and pastes them with remapped ids and internal edges only', () => {
    const graph = createStarterGraph()

    const fragment = copyGraphSelection(graph, { nodeIds: ['x', 'w', 'mul'] })
    const result = pasteGraphClipboard(graph, fragment!, { x: 40, y: 50 })

    expect(fragment).toBeDefined()
    expect(result.selection.groupId).toBeUndefined()
    expect(result.selection.nodeIds).toHaveLength(3)
    expect(result.graph.nodes).toHaveLength(graph.nodes.length + 3)
    expect(result.graph.edges).toHaveLength(graph.edges.length + 2)

    const pastedNodes = result.selection.nodeIds.map((nodeId) => result.graph.nodes.find((node) => node.id === nodeId))
    expect(pastedNodes.map((node) => node?.label)).toEqual(['x1', 'w1', 'x * w'])
    expect(pastedNodes.map((node) => node?.position)).toEqual([
      { x: 80, y: 110 },
      { x: 80, y: 310 },
      { x: 360, y: 210 },
    ])

    const pastedNodeIds = new Set(result.selection.nodeIds)
    const pastedEdges = result.graph.edges.filter((edge) => pastedNodeIds.has(edge.source) && pastedNodeIds.has(edge.target))
    expect(pastedEdges).toHaveLength(2)
    expect(pastedEdges.map((edge) => edge.inputSlot)).toEqual([0, 1])
    expect(result.graph.edges.some((edge) => pastedNodeIds.has(edge.source) && edge.target === 'add')).toBe(false)
  })

  it('copies a selected visual group and pastes it with the next group label and boundary handles', () => {
    const groupedGraph = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    const fragment = copyGraphSelection(groupedGraph, { nodeIds: [], groupId: 'group-1' })
    const result = pasteGraphClipboard(groupedGraph, fragment!, { x: 25, y: 35 })

    expect(fragment).toBeDefined()
    expect(result.selection.nodeIds).toEqual([])
    expect(result.selection.groupId).toBe('group-2')
    expect(result.graph.nodes).toHaveLength(groupedGraph.nodes.length + 2)
    expect(result.graph.edges).toHaveLength(groupedGraph.edges.length + 4)
    expect(result.graph.groups).toHaveLength(2)

    const pastedGroup = result.graph.groups?.find((group) => group.id === result.selection.groupId)
    expect(pastedGroup).toBeDefined()
    expect(pastedGroup?.label).toBe('Group 2')
    expect(pastedGroup?.nodeIds).toHaveLength(2)
    expect(pastedGroup?.position).toEqual({
      x: groupedGraph.groups![0].position.x + 25,
      y: groupedGraph.groups![0].position.y + 35,
    })

    const pastedGroupNodeIds = new Set(pastedGroup?.nodeIds)
    const pastedInternalEdges = result.graph.edges.filter(
      (edge) => pastedGroupNodeIds.has(edge.source) && pastedGroupNodeIds.has(edge.target),
    )
    expect(pastedInternalEdges).toHaveLength(1)

    const pastedInterface = visualGroupInterface(result.graph, pastedGroup!)
    expect(pastedInterface.inputs).toHaveLength(3)
    expect(pastedInterface.outputs).toHaveLength(1)
    expect(result.graph.edges.filter(edge => edge.target === 'pred')).toHaveLength(1)
  })

  it('pastes copied source nodes with the next generated label for that node type', () => {
    const input = createNode('input', 1)
    const graph = { ...createEmptyGraph(), nodes: [input] }

    const fragment = copyGraphSelection(graph, { nodeIds: [input.id] })
    const result = pasteGraphClipboard(graph, fragment!, { x: 20, y: 20 })

    expect(result.selection.nodeIds).toEqual(['input-2'])
    expect(result.graph.nodes.find((node) => node.id === 'input-2')).toEqual(
      expect.objectContaining({
        label: 'x2',
        position: { x: input.position.x + 20, y: input.position.y + 20 },
      }),
    )
  })
})


it('does not reconnect a missing external input to an unrelated newly generated node ID', () => {
  const original: GraphModel = {
    learningRate: .01,
    nodes: [
      { id: 'weight-1', label: 'External parameter', type: 'weight', params: { value: 3 }, position: { x: 0, y: 0 } },
      { id: 'weight-2', label: 'Internal parameter', type: 'weight', params: { value: 4 }, position: { x: 200, y: 0 } },
      { id: 'add', label: 'Sum', type: 'add', params: {}, position: { x: 400, y: 0 } },
    ],
    edges: [{ id: 'external', source: 'weight-1', target: 'add', inputSlot: 0 }, { id: 'internal', source: 'weight-2', target: 'add', inputSlot: 1 }],
    groups: [{ id: 'g', label: 'Building block', nodeIds: ['weight-2', 'add'], position: { x: 200, y: 0 }, dimensions: {} }],
  }
  const fragment = copyGraphSelection(original, { nodeIds: [], groupId: 'g' })!
  const pasted = pasteGraphClipboard({ nodes: [], edges: [], learningRate: .01 }, fragment, { x: 0, y: 0 }).graph
  expect(pasted.nodes.some(node => node.id === 'weight-1')).toBe(true)
  expect(pasted.edges).toHaveLength(1)
  expect(pasted.edges[0].inputSlot).toBe(1)
})
