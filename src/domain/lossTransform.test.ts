import { describe, expect, it } from 'vitest'
import { blockPalette } from './blockPalette'
import { backwardPass, forwardPass, runTrainingStep, validateGraph } from './engine'
import { createNode } from './examples'
import { tensorValue } from './tensor'
import type { GraphModel, GraphNode, NodeParams, TensorTransformKind } from './types'

function graphWithOperation(input: GraphNode, operation: GraphNode): GraphModel {
  return { nodes: [input, operation], edges: [{ id: 'input-operation', source: input.id, target: operation.id, inputSlot: 0 }], learningRate: 0.05 }
}

describe('unified loss and tensor transform blocks', () => {
  it('offers the unified entries in the palette', () => {
    const paletteTypes = blockPalette.map(item => item.type)
    expect(paletteTypes).toContain('loss')
    expect(paletteTypes).toContain('tensor-transform')
    expect(paletteTypes).not.toContain('cross-entropy')
    for (const type of ['reshape', 'transpose', 'slice', 'mean']) expect(paletteTypes).not.toContain(type)
  })

  it('computes multiclass cross-entropy and its gradient through the Loss block', () => {
    const logits = createNode('weight', 1)
    logits.params.value = tensorValue([2, 3], [2, 1, 0, 0, 1, 2])
    const target = createNode('target', 1)
    target.params.value = tensorValue([2], [0, 2])
    const loss = createNode('loss', 1)
    loss.params.loss = 'cross-entropy'
    const graph: GraphModel = { nodes: [logits, target, loss], learningRate: 0.1, edges: [
      { id: 'logits-loss', source: logits.id, target: loss.id, inputSlot: 0 },
      { id: 'target-loss', source: target.id, target: loss.id, inputSlot: 1 },
    ] }
    expect(validateGraph(graph)).toEqual([])
    const forward = forwardPass(graph)
    expect(forward.loss).toBeCloseTo(0.407605964)
    const backward = backwardPass(forward.graph)
    const gradient = backward.graph.nodes.find(node => node.id === logits.id)!.grad!
    expect(gradient.shape).toEqual([2, 3])
    expect(gradient.data[0]).toBeCloseTo(-0.1673795)
    expect(gradient.data[5]).toBeCloseTo(-0.1673795)
    expect(backward.graph.nodes.find(node => node.id === target.id)?.grad?.data).toEqual([0, 0])
    expect(runTrainingStep(graph).loss).toBeLessThan(forward.loss!)
  })

  it('accepts one vector of logits with one scalar class ID', () => {
    const logits = createNode('input', 1)
    logits.params.value = tensorValue([3], [1, 2, 3])
    const target = createNode('target', 1)
    target.params.value = tensorValue([], [2])
    const loss = createNode('loss', 1)
    loss.params.loss = 'cross-entropy'
    const graph: GraphModel = { nodes: [logits, target, loss], learningRate: 0.1, edges: [
      { id: 'logits', source: logits.id, target: loss.id, inputSlot: 0 },
      { id: 'class', source: target.id, target: loss.id, inputSlot: 1 },
    ] }
    expect(validateGraph(graph)).toEqual([])
    expect(forwardPass(graph).loss).toBeCloseTo(0.407605964)
    const backward = backwardPass(forwardPass(graph).graph)
    expect(backward.graph.nodes.find(node => node.id === logits.id)?.grad?.shape).toEqual([3])
  })

  it.each([
    ['reshape', { shape: [3, 2] }, [3, 2], [1, 2, 3, 4, 5, 6]],
    ['transpose', { axes: [1, 0] }, [3, 2], [1, 4, 2, 5, 3, 6]],
    ['slice', { axis: 1, start: 1, end: 3 }, [2, 2], [2, 3, 5, 6]],
    ['mean', { axis: 1 }, [2], [2, 5]],
  ] as const)('runs %s through one Tensor transform block', (kind, params, shape, data) => {
    const input = createNode('input', 1)
    input.params.value = tensorValue([2, 3], [1, 2, 3, 4, 5, 6])
    const transform = createNode('tensor-transform', 1)
    transform.params = { ...transform.params, transform: kind as TensorTransformKind, ...(params as unknown as NodeParams) }
    const result = forwardPass(graphWithOperation(input, transform))
    expect(result.graph.nodes.find(node => node.id === transform.id)?.value).toEqual(tensorValue([...shape], [...data]))
  })

  it('backpropagates through a selected Mean transform', () => {
    const input = createNode('weight', 1)
    input.params.value = tensorValue([2, 3], [1, 2, 3, 4, 5, 6])
    const transform = createNode('tensor-transform', 1)
    transform.params = { transform: 'mean', axis: 1 }
    const target = createNode('target', 1)
    target.params.value = tensorValue([2], [0, 0])
    const loss = createNode('loss', 1)
    loss.params.loss = 'mse'
    const graph: GraphModel = { nodes: [input, transform, target, loss], learningRate: 0.01, edges: [
      { id: 'transform', source: input.id, target: transform.id, inputSlot: 0 },
      { id: 'prediction', source: transform.id, target: loss.id, inputSlot: 0 },
      { id: 'target', source: target.id, target: loss.id, inputSlot: 1 },
    ] }
    const backward = backwardPass(forwardPass(graph).graph)
    const gradient = backward.graph.nodes.find(node => node.id === input.id)!.grad!.data
    gradient.slice(0, 3).forEach(value => expect(value).toBeCloseTo(2 / 3))
    gradient.slice(3).forEach(value => expect(value).toBeCloseTo(5 / 3))
  })
})
