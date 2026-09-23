import { describe, expect, it } from 'vitest'
import { backwardPass, cloneGraph, forwardPass, runTrainingStep } from './engine'
import { createSingleNeuronGraph } from './examples'
import { collapsedGroupForNode, explodeVisualGroup, groupAncestors, mergeNodesIntoVisualGroup, moveVisualGroup, setVisualGroupExpanded, visibleGroups } from './grouping'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { createProjectStateFile, parseProjectStateFile } from './session'

function nestedGraph() {
  let graph = backwardPass(forwardPass(createSingleNeuronGraph('identity', { x: 2, w: 3, b: 1, target: 9 })).graph).graph
  graph = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul']).graph
  graph = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  graph = mergeNodesIntoVisualGroup(graph, ['target', 'loss']).graph
  return graph
}

describe('persistent nested modules', () => {
  it('opens one branch while preserving computations, gradients, layouts and neighbor summaries', () => {
    const original = nestedGraph()
    expect(groupAncestors(original, 'group-1').map((g) => g.id)).toEqual(['group-2', 'group-1'])
    const opened = setVisualGroupExpanded(original, 'group-1', true)
    expect(opened.nodes).toBe(original.nodes)
    expect(opened.edges).toBe(original.edges)
    expect(opened.groups).toBe(original.groups)
    expect(collapsedGroupForNode(opened, 'mul')).toBeUndefined()
    expect(collapsedGroupForNode(opened, 'loss')?.id).toBe('group-3')
    const closed = setVisualGroupExpanded(opened, 'group-2', false)
    expect(visibleGroups(closed).map((g) => g.id)).toEqual(['group-2', 'group-3'])
    expect(closed.groups).toHaveLength(3)
    const reopened = setVisualGroupExpanded(closed, 'group-2', true)
    expect(reopened.nodes).toEqual(original.nodes)
    expect(reopened.edges).toEqual(original.edges)
    expect(forwardPass(reopened).loss).toBe(2) // Existing builder's half-square convention.
    expect(runTrainingStep(reopened).loss).toBeCloseTo(runTrainingStep(original).loss!)
  })

  it('ungroups explicitly and reparents children without changing calculations', () => {
    const graph = nestedGraph()
    const next = explodeVisualGroup(graph, 'group-2')
    expect(next.groups).toHaveLength(2)
    expect(next.groups?.find((g) => g.id === 'group-1')?.parentId).toBeUndefined()
    expect(next.nodes).toBe(graph.nodes)
  })

  it('moves descendants together and copies the complete nested hierarchy', () => {
    const graph = nestedGraph()
    const parent = graph.groups!.find((g) => g.id === 'group-2')!
    const moved = moveVisualGroup(graph, parent.id, { x: parent.position.x + 30, y: parent.position.y + 15 })
    expect(moved.groups![0].position.x).toBe(graph.groups![0].position.x + 30)
    const copied = copyGraphSelection(graph, { nodeIds: [], groupId: parent.id })!
    const pasted = pasteGraphClipboard(graph, copied, { x: 30, y: 15 })
    const copiedParent = pasted.graph.groups!.find((g) => g.id === pasted.selection.groupId)!
    const copiedChild = pasted.graph.groups!.find((g) => g.parentId === copiedParent.id)!
    expect(copiedChild.nodeIds).toHaveLength(3)
    expect(copiedChild.nodeIds.every((id) => copiedParent.nodeIds.includes(id))).toBe(true)
    expect(copiedChild.nodeIds.some((id) => graph.nodes.some((node) => node.id === id))).toBe(false)
  })

  it('restores hierarchy, numeric snapshots, selected internals, and viewport from a save', () => {
    const graph = setVisualGroupExpanded(nestedGraph(), 'group-1', true)
    graph.view!.viewport = { x: 12, y: 40, zoom: 0.8 }
    const snapshot = createProjectStateFile({ graph, visualizationGraph: graph, initialParameterValues: {}, selectedNodeIds: ['mul'], phase: 'backward', traceSteps: [], traceIndex: 0, epoch: 0, currentLoss: 2, display: { showMath: true, showGradient: true, showCode: false, showVisualization: false } })
    const parsed = parseProjectStateFile(JSON.stringify(snapshot))
    expect(parsed.ok).toBe(true)
    if (!parsed.ok) return
    expect(parsed.file.state.graph).toEqual(cloneGraph(graph))
    expect(parsed.file.state.selectedNodeIds).toEqual(['mul'])
    graph.view!.expandedGroupIds.length = 0
    expect(parsed.file.state.graph.view!.expandedGroupIds).toEqual(['group-2', 'group-1'])
  })
})
