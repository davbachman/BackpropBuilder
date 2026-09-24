import { describe, expect, it } from 'vitest'
import { backwardPass, forwardPass } from './engine'
import { visibleGraphForTrace, visibleStepEdgeIds } from './traceVisibility'
import type { GraphModel, GraphNode } from './types'

const source = (id: string, value: number): GraphNode => ({ id, type: 'weight', label: id, params: { value }, position: { x: 0, y: 0 } })
const op = (id: string, type: GraphNode['type']): GraphNode => ({ id, type, label: id, params: {}, position: { x: 0, y: 0 } })
function branches(): GraphModel {
  return {
    learningRate: .01,
    nodes: [source('x', 2), source('c1', 3), source('c2', 5), op('a', 'multiply'), op('b', 'multiply'), op('sum', 'add'), { ...source('target', 0), type: 'target' }, op('loss', 'loss')],
    edges: [['x', 'a', 0], ['c1', 'a', 1], ['x', 'b', 0], ['c2', 'b', 1], ['a', 'sum', 0], ['b', 'sum', 1], ['sum', 'loss', 0], ['target', 'loss', 1]].map(([from, to, slot]) => ({ id: `${from}-${to}`, source: String(from), target: String(to), inputSlot: Number(slot) })),
  }
}

describe('trace-visible numerical information', () => {
  it('keeps gradients unknown before backpropagation without changing computed values', () => {
    const forward = forwardPass(branches())
    const visible = visibleGraphForTrace(forward.graph, [], 0, 'edit')
    expect(visible.nodes.find(node => node.id === 'loss')!.value).toEqual(forward.graph.nodes.find(node => node.id === 'loss')!.value)
    expect(visible.edges.find(edge => edge.id === 'x-a')!.value).toEqual(forward.graph.edges.find(edge => edge.id === 'x-a')!.value)
    expect(visible.nodes.every(node => node.grad === undefined)).toBe(true)
    expect(visible.edges.every(edge => edge.grad === undefined)).toBe(true)
    expect(forward.graph.nodes.find(node => node.id === 'x')!.grad).toBeDefined()
  })

  it('shows only reached forward values, leaving future wires unknown', () => {
    const forward = forwardPass(branches())
    const stepIndex = forward.steps.findIndex(step => step.nodeId === 'a')
    const visible = visibleGraphForTrace(forward.graph, forward.steps, stepIndex, 'forward')
    expect(visible.edges.find(edge => edge.id === 'x-a')!.value!.data).toEqual([2])
    expect(visible.edges.find(edge => edge.id === 'a-sum')!.value!.data).toEqual([6])
    expect(visible.edges.find(edge => edge.id === 'sum-loss')!.value).toBeUndefined()
    expect(visible.edges.every(edge => edge.grad === undefined)).toBe(true)
  })

  it('accumulates a shared node gradient only as its individual branch contributions arrive', () => {
    const backward = backwardPass(forwardPass(branches()).graph)
    const firstBranch = backward.steps.find(step => step.nodeId === 'a' || step.nodeId === 'b')!
    const index = backward.steps.indexOf(firstBranch)
    const visible = visibleGraphForTrace(backward.graph, backward.steps, index, 'backward')
    const reachedEdge = `x-${firstBranch.nodeId}`
    const laterEdge = `x-${firstBranch.nodeId === 'a' ? 'b' : 'a'}`
    const contribution = backward.graph.edges.find(edge => edge.id === reachedEdge)!.grad!.data[0]
    expect(visible.nodes.find(node => node.id === 'x')!.grad!.data).toEqual([contribution])
    expect(visible.nodes.find(node => node.id === 'x')!.grad!.data[0]).not.toBe(128)
    expect(visible.edges.find(edge => edge.id === laterEdge)!.grad).toBeUndefined()
    const complete = visibleGraphForTrace(backward.graph, backward.steps, backward.steps.length - 1, 'backward')
    expect(complete.nodes.find(node => node.id === 'x')!.grad!.data).toEqual([128])
  })

  it('does not leak values through pass-through inputs and traces their backward wire', () => {
    const graph = branches()
    graph.nodes.push(op('bridge', 'input'))
    graph.edges.find(edge => edge.id === 'sum-loss')!.source = 'bridge'
    graph.edges.push({ id: 'sum-bridge', source: 'sum', target: 'bridge', inputSlot: 0 })
    const forward = forwardPass(graph)
    const early = visibleGraphForTrace(forward.graph, forward.steps, 0, 'forward')
    expect(early.nodes.find(node => node.id === 'bridge')!.value).toBeUndefined()
    expect(early.edges.find(edge => edge.id === 'sum-loss')!.value).toBeUndefined()
    const backward = backwardPass(forward.graph)
    const bridgeIndex = backward.steps.findIndex(step => step.nodeId === 'bridge')
    expect(bridgeIndex).toBeGreaterThan(0)
    const shown = visibleGraphForTrace(backward.graph, backward.steps, bridgeIndex, 'backward')
    expect(shown.edges.find(edge => edge.id === 'sum-bridge')!.grad!.data).toEqual([16])
  })

  it('animates collapsed-region boundary wires for all summarized steps without future output', () => {
    const graph = branches()
    graph.groups = [{ id: 'module', kind: 'module', label: 'Branch sums', nodeIds: ['a', 'b', 'sum'], position: { x: 0, y: 0 }, dimensions: {} }]
    graph.view = { expandedGroupIds: [] }
    const forward = forwardPass(graph), sumIndex = forward.steps.findIndex(step => step.nodeId === 'sum')
    const edges = visibleStepEdgeIds(forward.graph, forward.steps, sumIndex)
    expect(edges).toEqual(expect.arrayContaining(['x-a', 'c1-a', 'x-b', 'c2-b', 'sum-loss']))
    const first = visibleStepEdgeIds(forward.graph, forward.steps, 0)
    expect(first).not.toContain('sum-loss')
    const backward = backwardPass(forward.graph), firstBranch = backward.steps.findIndex(step => step.nodeId === 'a' || step.nodeId === 'b')
    const backwardEdges = visibleStepEdgeIds(backward.graph, backward.steps, firstBranch)
    expect(backwardEdges).toContain(`x-${backward.steps[firstBranch].nodeId}`)
    const expanded = { ...forward.graph, view: { expandedGroupIds: ['module'] } }
    expect(visibleStepEdgeIds(expanded, forward.steps, sumIndex)).toEqual(forward.steps[sumIndex].edgeIds)
  })

  it('seeds cross-entropy loss and does not expose future parameter gradients', () => {
    const graph: GraphModel = { learningRate: .01, nodes: [
      { ...source('logits', 0), params: { value: { shape: [1, 2], data: [.2, .8] } } },
      { ...source('targets', 0), type: 'target', params: { value: { shape: [1], data: [1] } } },
      op('loss', 'cross-entropy'),
    ], edges: [{ id: 'prediction', source: 'logits', target: 'loss', inputSlot: 0 }, { id: 'target', source: 'targets', target: 'loss', inputSlot: 1 }] }
    const backward = backwardPass(forwardPass(graph).graph)
    const visible = visibleGraphForTrace(backward.graph, backward.steps, 0, 'backward')
    expect(visible.nodes.find(node => node.id === 'loss')!.grad!.data).toEqual([1])
    expect(visible.edges.find(edge => edge.id === 'prediction')!.grad).toEqual(backward.graph.edges.find(edge => edge.id === 'prediction')!.grad)
  })
})
