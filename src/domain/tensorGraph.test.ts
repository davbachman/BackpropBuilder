import { describe, expect, it } from 'vitest'
import { backwardPass, cloneGraph, forwardPass, formulaForNode, runTrainingStep, validateGraph } from './engine'
import { tensorValue } from './tensor'
import type { GraphModel, GraphNode, NodeParams, NodeType, TensorValue } from './types'

const t = tensorValue
function operationGraph(type: NodeType, values: TensorValue[], params: NodeParams = {}, constants: number[] = []): GraphModel {
  const nodes: GraphNode[] = values.map((value, index) => ({ id: `x${index}`, label: `x${index}`, type: constants.includes(index) ? 'input' : 'weight', params: { value }, position: { x: 0, y: 200 * index } }))
  nodes.push({ id: 'op', type, label: type, params, position: { x: 300, y: 0 } })
  const graph: GraphModel = { nodes, edges: values.map((_, index) => ({ id: `e${index}`, source: `x${index}`, target: 'op', inputSlot: index })), learningRate: 0.01 }
  if (type !== 'cross-entropy') {
    const output = forwardPass(graph).graph.nodes.find((node) => node.id === 'op')!.value!
    nodes.push({ id: 'target', label: 'target', type: 'target', params: { value: t(output.shape, output.data.map((_, index) => 0.13 * (index + 1))) }, position: { x: 300, y: 300 } })
    nodes.push({ id: 'loss', label: 'loss', type: 'loss', params: { loss: 'mse' }, position: { x: 600, y: 0 } })
    graph.edges.push({ id: 'pred', source: 'op', target: 'loss', inputSlot: 0 }, { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 })
  }
  return graph
}

function expectNumericalGradients(graph: GraphModel, tolerance = 5) {
  expect(validateGraph(graph, { requireLoss: true })).toEqual([])
  const actual = backwardPass(forwardPass(graph).graph).graph
  for (const source of graph.nodes.filter((node) => node.type === 'weight')) {
    const value = source.params.value as TensorValue
    for (let index = 0; index < value.data.length; index++) {
      const plus = cloneGraph(graph), minus = cloneGraph(graph), epsilon = 1e-5
      ;(plus.nodes.find((node) => node.id === source.id)!.params.value as TensorValue).data[index] += epsilon
      ;(minus.nodes.find((node) => node.id === source.id)!.params.value as TensorValue).data[index] -= epsilon
      const numerical = (forwardPass(plus).loss! - forwardPass(minus).loss!) / (2 * epsilon)
      expect(actual.nodes.find((node) => node.id === source.id)!.grad!.data[index]).toBeCloseTo(numerical, tolerance)
    }
  }
}

describe('tensor primitives in the editable graph', () => {
  it('reduces broadcast add and multiply adjoints over all expanded axes', () => {
    const values = [t([2, 2], [0.3, 0.5, 0.7, 1.2]), t([2], [0.6, -0.4])]
    expectNumericalGradients(operationGraph('add', values))
    expectNumericalGradients(operationGraph('multiply', values))
    expectNumericalGradients(operationGraph('add', [t([2, 1, 2], [1, 2, 3, 4]), t([1, 3, 1], [0.2, 0.4, 0.6])]))
  })

  it('gathers embeddings and sums repeated-token parameter gradients', () => {
    expectNumericalGradients(operationGraph('embedding', [t([3, 2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), t([3], [2, 0, 2])], {}, [1]))
  })

  it('differentiates transpose, slice, reshape, concat and mean', () => {
    const matrix = t([2, 3], [0.2, 0.4, 0.6, 0.8, 1, 1.2])
    expectNumericalGradients(operationGraph('transpose', [matrix]))
    expectNumericalGradients(operationGraph('slice', [matrix], { axis: 1, start: 1, end: 3 }))
    expectNumericalGradients(operationGraph('reshape', [matrix], { shape: [3, 2] }))
    expectNumericalGradients(operationGraph('concat', [matrix, t([2, 1], [0.5, 0.7])], { axis: 1 }))
    expectNumericalGradients(operationGraph('concat', [t([2], [0.5, 0.7]), t([2], [0.2, 0.4])], { axis: 1 }))
    expectNumericalGradients(operationGraph('concat', [matrix, t([2], [0.5, 0.7])], { axis: 1 }))
    expectNumericalGradients(operationGraph('mean', [matrix], { axis: 1, keepDims: true }))
    expectNumericalGradients(operationGraph('mean', [matrix]))
  })

  it('stacks equal-length vectors as columns on axis 1 without changing axis 0 concatenation', () => {
    const vectors = Array.from({ length: 4 }, (_, index) => t([112], Array.from({ length: 112 }, (_, row) => row + index)))
    const graph: GraphModel = {
      learningRate: 0.01,
      nodes: [
        ...vectors.map((value, index): GraphNode => ({ id: `x${index}`, label: `x${index}`, type: 'input', params: { value }, position: { x: 0, y: index * 100 } })),
        { id: 'concat', label: 'Concatenate', type: 'concat', params: { axis: 1, inputCount: 4 }, position: { x: 300, y: 0 } },
      ],
      edges: vectors.map((_, index) => ({ id: `e${index}`, source: `x${index}`, target: 'concat', inputSlot: index })),
    }
    expect(validateGraph(graph)).toEqual([])
    expect(formulaForNode(graph.nodes.at(-1)!, graph)).toContain('column_stack(')
    const stacked = forwardPass(graph).graph.nodes.find(node => node.id === 'concat')!.value!
    expect(stacked.shape).toEqual([112, 4])
    expect(stacked.data.slice(0, 8)).toEqual([0, 1, 2, 3, 1, 2, 3, 4])

    graph.nodes.at(-1)!.params.axis = 0
    const joined = forwardPass(graph).graph.nodes.find(node => node.id === 'concat')!.value!
    expect(joined.shape).toEqual([448])
    expect(joined.data.slice(0, 4)).toEqual([0, 1, 2, 3])
  })

  it('rejects mismatched column lengths instead of silently broadcasting them', () => {
    const graph = operationGraph('concat', [t([3], [1, 2, 3]), t([2], [4, 5])], { axis: 0 })
    graph.nodes.find(node => node.id === 'op')!.params.axis = 1
    expect(validateGraph(graph)).toContainEqual(expect.objectContaining({ code: 'shape-mismatch', nodeId: 'op', message: expect.stringContaining('same row count') }))
  })

  it('differentiates softmax, layer norm and cross entropy', () => {
    const matrix = t([2, 3], [0.2, 0.4, 0.6, 0.8, 1.4, 1.2])
    expectNumericalGradients(operationGraph('softmax', [matrix]))
    expectNumericalGradients(operationGraph('layer-norm', [matrix, t([3], [1.1, 0.8, 0.7]), t([3], [0.1, 0.2, 0.3])]))
    expectNumericalGradients(operationGraph('cross-entropy', [matrix, t([2], [0, 2])], {}, [1]))
  })

  it('preserves exact causal exclusions across graph edges and round-trip cloning', () => {
    const graph = operationGraph('causal-mask', [t([2, 2], [-2e9, 10, 1, 2])])
    graph.nodes = graph.nodes.filter((node) => node.id !== 'target' && node.id !== 'loss')
    graph.edges = graph.edges.filter((edge) => edge.target === 'op')
    graph.nodes.push({ id: 'prob', label: 'probabilities', type: 'softmax', params: {}, position: { x: 600, y: 0 } })
    graph.edges.push({ id: 'mask-prob', source: 'op', target: 'prob', inputSlot: 0 })
    const evaluated = cloneGraph(forwardPass(graph).graph)
    expect(evaluated.nodes.find((node) => node.id === 'op')!.value!.excluded).toEqual([false, true, false, false])
    const probabilities = evaluated.nodes.find((node) => node.id === 'prob')!.value!.data
    expect(probabilities.slice(0, 2)).toEqual([1, 0])
    expect(probabilities[2] + probabilities[3]).toBeCloseTo(1)
  })

  it('backpropagates through causal attention to shared Q/K inputs', () => {
    const graph: GraphModel = {
      learningRate: 0.02,
      nodes: [
        { id: 'x', label: 'x', type: 'weight', params: { value: t([2, 2], [0.5, 0.7, -0.2, 0.9]) }, position: { x: 0, y: 0 } },
        { id: 'transpose', label: 'transpose', type: 'transpose', params: {}, position: { x: 200, y: 0 } },
        { id: 'scores', label: 'scores', type: 'matmul', params: {}, position: { x: 400, y: 0 } },
        { id: 'mask', label: 'mask', type: 'causal-mask', params: {}, position: { x: 600, y: 0 } },
        { id: 'prob', label: 'prob', type: 'softmax', params: {}, position: { x: 800, y: 0 } },
        { id: 'target', label: 'target', type: 'target', params: { value: t([2, 2], [1, 0, 0.8, 0.2]) }, position: { x: 800, y: 200 } },
        { id: 'loss', label: 'loss', type: 'loss', params: { loss: 'mse' }, position: { x: 1000, y: 0 } },
      ],
      edges: [['x', 'transpose', 0], ['x', 'scores', 0], ['transpose', 'scores', 1], ['scores', 'mask', 0], ['mask', 'prob', 0], ['prob', 'loss', 0], ['target', 'loss', 1]].map(([source, target, slot], index) => ({ id: `e${index}`, source: String(source), target: String(target), inputSlot: Number(slot) })),
    }
    expectNumericalGradients(graph)
    const initial = forwardPass(graph).loss!
    expect(runTrainingStep(graph).loss!).toBeLessThan(initial)
  })

  it('differentiates convolution pixels, filters and biases with spatially shared weights', () => {
    expectNumericalGradients(operationGraph('conv2d', [
      t([3, 3, 1], [0.1, .2, .3, .4, .5, .6, .7, .8, .9]),
      t([2, 2, 2, 1], [.1, .2, -.1, .4, .2, -.3, .4, .1]),
      t([2], [.2, -.1]),
    ]))
  })

  it('differentiates independent channel pooling, including unused odd border pixels', () => {
    expectNumericalGradients(operationGraph('avgpool2d', [t([3, 3, 2], Array.from({ length: 18 }, (_, i) => i * .1))]))
  })

  it('re-evaluates edited source values and shapes without reusing stale caches', () => {
    const graph = operationGraph('embedding', [t([3, 2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), t([1], [0])], {}, [1])
    graph.nodes = graph.nodes.filter(node => node.id !== 'target' && node.id !== 'loss')
    graph.edges = graph.edges.filter(edge => edge.target === 'op')
    const evaluated = forwardPass(graph).graph
    evaluated.nodes.find(node => node.id === 'x1')!.params.value = t([2], [1, 2])
    const edited = forwardPass(evaluated).graph.nodes.find(node => node.id === 'op')!.value!
    expect(edited).toEqual(t([2, 2], [0.4, 0.5, 0.6, 0.7]))
  })

  it('blocks invalid token IDs before evaluating a user-edited graph', () => {
    const graph = operationGraph('embedding', [t([2, 2], [1, 2, 3, 4]), t([1], [0])], {}, [1])
    graph.nodes.find(node => node.id === 'x1')!.params.value = t([1], [2])
    expect(validateGraph(graph)).toContainEqual(expect.objectContaining({ code: 'invalid-value', nodeId: 'op' }))
    graph.nodes.find(node => node.id === 'x1')!.params.value = t([1], [0.5])
    expect(validateGraph(graph)).toContainEqual(expect.objectContaining({ code: 'invalid-value', nodeId: 'op' }))
  })

  it('validates transform shape contracts and gives meaningful formulas', () => {
    const graph = operationGraph('transpose', [t([2, 2], [1, 2, 3, 4])])
    graph.nodes.find((node) => node.id === 'op')!.params.axes = [0, 0]
    expect(validateGraph(graph).some((issue) => issue.code === 'shape-mismatch')).toBe(true)
    expect(formulaForNode({ id: 'e', type: 'cross-entropy', label: 'loss', params: {}, position: { x: 0, y: 0 } })).toContain('softmax')
  })
})
