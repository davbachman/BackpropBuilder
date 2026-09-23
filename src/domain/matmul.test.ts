import { describe, expect, it } from 'vitest'
import { backwardPass, cloneGraph, forwardPass, matrixProduct, runTrainingStep } from './engine'
import { createSingleNeuronGraph } from './examples'
import { tensorValue, toTensor } from './tensor'
import type { GraphModel } from './types'

function matrixGraph(shared = false): GraphModel {
  return { learningRate: 0.01, nodes: [
    { id: 'a', type: 'weight', label: 'A', position: { x: 0, y: 0 }, params: { value: tensorValue([2, 2], [1, 2, -1, 3]) } },
    ...(!shared ? [{ id: 'b', type: 'weight' as const, label: 'B', position: { x: 0, y: 100 }, params: { value: tensorValue([2, 2], [2, -1, 0.5, 1]) } }] : []),
    { id: 'out', type: 'matmul', label: 'A @ B', position: { x: 200, y: 0 }, params: {} },
    { id: 'target', type: 'target', label: 'target', position: { x: 200, y: 100 }, params: { value: tensorValue([2, 2], [0, 1, 1, 0]) } },
    { id: 'loss', type: 'loss', label: 'loss', position: { x: 400, y: 0 }, params: { loss: 'mse' } },
  ], edges: [
    { id: 'a-out', source: 'a', target: 'out', inputSlot: 0 },
    { id: 'b-out', source: shared ? 'a' : 'b', target: 'out', inputSlot: 1 },
    { id: 'out-loss', source: 'out', target: 'loss', inputSlot: 0 },
    { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
  ] }
}

describe('matrix products and inference', () => {
  it('computes non-square row by column sums and rejects incompatible shapes', () => {
    expect(matrixProduct(tensorValue([2, 3], [1, 2, 3, 4, 5, 6]), tensorValue([3, 1], [2, -1, 3]))).toEqual(tensorValue([2, 1], [9, 21]))
    expect(() => matrixProduct(tensorValue([2], [1, 2]), tensorValue([2], [1, 2]))).toThrow(/shapes/)
  })
  for (const shared of [false, true]) {
    it(`matches central differences for matrix gradients (shared=${shared})`, () => {
      const graph = matrixGraph(shared)
      const analytic = backwardPass(forwardPass(graph).graph).graph
      for (const parameter of graph.nodes.filter((node) => node.type === 'weight')) {
        const value = toTensor(parameter.params.value)
        for (let index = 0; index < value.data.length; index += 1) {
          const losses = [-1, 1].map((sign) => {
            const perturbed = cloneGraph(graph)
            const target = perturbed.nodes.find((node) => node.id === parameter.id)!
            const changed = toTensor(target.params.value)
            changed.data[index] += sign * 1e-5
            target.params.value = changed
            return forwardPass(perturbed).loss!
          })
          expect(analytic.nodes.find((node) => node.id === parameter.id)!.grad!.data[index]).toBeCloseTo((losses[1] - losses[0]) / 2e-5, 6)
        }
      }
    })
  }
  it('performs inference without target or loss, preserving parameters and refusing training', () => {
    const graph = createSingleNeuronGraph('identity', { x: 2, w: 3, b: 1, target: 9 })
    graph.nodes = graph.nodes.filter((node) => !['target', 'loss'].includes(node.type))
    graph.edges = graph.edges.filter((edge) => edge.target !== 'loss')
    expect(forwardPass(graph).graph.nodes.find((node) => node.id === 'pred')?.value?.data).toEqual([7])
    expect(forwardPass(graph).loss).toBeUndefined()
    expect(graph.nodes.find((node) => node.id === 'w')!.params.value).toEqual(tensorValue([], [3]))
    expect(() => backwardPass(graph)).toThrow(/loss/)
    expect(() => runTrainingStep(graph)).toThrow(/loss/)
  })
})
