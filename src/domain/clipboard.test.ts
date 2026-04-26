import { describe, expect, it } from 'vitest'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { createStarterGraph } from './examples'
import { mergeNodesIntoVisualGroup } from './grouping'

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
    expect(pastedNodes.map((node) => node?.label)).toEqual(['x', 'w', 'x * w'])
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

  it('copies a selected visual group and pastes it as a new collapsed group', () => {
    const groupedGraph = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    const fragment = copyGraphSelection(groupedGraph, { nodeIds: [], groupId: 'group-1' })
    const result = pasteGraphClipboard(groupedGraph, fragment!, { x: 25, y: 35 })

    expect(fragment).toBeDefined()
    expect(result.selection.nodeIds).toEqual([])
    expect(result.selection.groupId).toBeDefined()
    expect(result.graph.nodes).toHaveLength(groupedGraph.nodes.length + 2)
    expect(result.graph.edges).toHaveLength(groupedGraph.edges.length + 1)
    expect(result.graph.groups).toHaveLength(2)

    const pastedGroup = result.graph.groups?.find((group) => group.id === result.selection.groupId)
    expect(pastedGroup).toBeDefined()
    expect(pastedGroup?.label).toBe('Group 1')
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
  })
})
