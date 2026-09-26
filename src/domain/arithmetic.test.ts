import { describe, expect, it } from 'vitest'
import { appendArithmeticInput, arithmeticInputCount, evaluateArithmetic, parseArithmetic } from './arithmetic'
import { backwardPass, forwardPass, runTrainingStep } from './engine'
import { blockPalette } from './blockPalette'
import { createNode } from './examples'
import { denseGroupDetail } from './authoring'
import { scalarValue, tensorValue } from './tensor'
import type { GraphGroup, GraphModel } from './types'

describe('editable arithmetic block', () => {
  it('replaces separate Add and Multiply entries with Arithmetic and Param', () => {
    expect(blockPalette.filter(item => ['weight', 'bias', 'add', 'multiply', 'arithmetic'].includes(item.type)))
      .toEqual([{ type: 'weight', label: 'Param' }, { type: 'arithmetic', label: 'Arithmetic' }])
    expect(createNode('weight', 1).label).toBe('Param 1')
    expect(createNode('arithmetic', 1).params.expression).toBe('x1 * x2')
  })

  it('parses normal arithmetic precedence and adjusts port count', () => {
    expect(evaluateArithmetic('-x1^2 + (x2 + 1) * 3', [scalarValue(2), scalarValue(4)]).value.data).toEqual([11])
    expect(arithmeticInputCount('x1^2')).toBe(1)
    expect(arithmeticInputCount('x1 + x2 + x3')).toBe(3)
    expect(() => parseArithmetic('x2 + 1')).toThrow(/consecutive/)
    expect(() => parseArithmetic('x1^x2')).toThrow(/exponent must be a number/)
    expect(() => parseArithmetic('alert(x1)')).toThrow()
  })

  it('extends an additive or multiplicative expression with a working input', () => {
    const sum = appendArithmeticInput('x1 + x2')
    const product = appendArithmeticInput('x1 * x2')
    expect(sum).toBe('x1 + x2 + x3')
    expect(product).toBe('x1 * x2 * x3')
    expect(evaluateArithmetic(sum, [scalarValue(2), scalarValue(3), scalarValue(4)]).value.data).toEqual([9])
    expect(evaluateArithmetic(product, [scalarValue(2), scalarValue(3), scalarValue(4)]).value.data).toEqual([24])
  })

  it('broadcasts tensors and sums broadcast gradients back to each input shape', () => {
    const x1 = tensorValue([2, 2], [1, 2, 3, 4])
    const x2 = tensorValue([2], [10, 20])
    const result = evaluateArithmetic('x1 / x2 + x1^2', [x1, x2], tensorValue([2, 2], [1, 1, 1, 1]))
    expect(result.value.shape).toEqual([2, 2])
    expect(result.value.data).toEqual([1.1, 4.1, 9.3, 16.2])
    expect(result.gradients[0].data).toEqual([2.1, 4.05, 6.1, 8.05])
    expect(result.gradients[1].data[0]).toBeCloseTo(-0.04)
    expect(result.gradients[1].data[1]).toBeCloseTo(-0.015)
  })

  it('trains a graph with a repeated parameter inside one expression', () => {
    const input = createNode('input', 1)
    const param = createNode('weight', 1)
    const arithmetic = createNode('arithmetic', 1)
    const target = createNode('target', 1)
    const loss = createNode('loss', 1)
    input.params.value = scalarValue(2)
    param.params.value = scalarValue(3)
    target.params.value = scalarValue(10)
    arithmetic.params.expression = 'x1*x2 + x2^2'
    const graph: GraphModel = {
      nodes: [input, param, arithmetic, target, loss], learningRate: 0.01,
      edges: [
        { id: 'x', source: input.id, target: arithmetic.id, inputSlot: 0 },
        { id: 'p', source: param.id, target: arithmetic.id, inputSlot: 1 },
        { id: 'z', source: arithmetic.id, target: loss.id, inputSlot: 0 },
        { id: 'y', source: target.id, target: loss.id, inputSlot: 1 },
      ],
    }
    const forward = forwardPass(graph)
    expect(forward.graph.nodes.find(node => node.id === arithmetic.id)?.value?.data).toEqual([15])
    const backward = backwardPass(forward.graph)
    expect(backward.graph.nodes.find(node => node.id === input.id)?.grad?.data).toEqual([15])
    expect(backward.graph.nodes.find(node => node.id === param.id)?.grad?.data).toEqual([40])
    const trained = runTrainingStep(graph)
    expect(trained.graph.nodes.find(node => node.id === param.id)?.params.value).toEqual(scalarValue(2.6))
    expect(trained.loss).toBeLessThan(forward.loss!)
  })

  it('recognizes a layer built with Param for both matrix and bias', () => {
    const input = createNode('input', 1)
    const matrix = createNode('weight', 1)
    const bias = createNode('weight', 2)
    const product = createNode('matmul', 1)
    const sum = createNode('arithmetic', 1)
    const output = createNode('activation', 1)
    sum.params.expression = 'x1 + x2'
    const nodes = [input, matrix, bias, product, sum, output]
    const group: GraphGroup = { id: 'layer', label: 'Dense layer', nodeIds: nodes.map(node => node.id), position: { x: 0, y: 0 }, dimensions: { width: 300, height: 200 } }
    const graph: GraphModel = { nodes, groups: [group], learningRate: 0.01, edges: [
      { id: 'input-product', source: input.id, target: product.id, inputSlot: 0 },
      { id: 'matrix-product', source: matrix.id, target: product.id, inputSlot: 1 },
      { id: 'product-sum', source: product.id, target: sum.id, inputSlot: 0 },
      { id: 'bias-sum', source: bias.id, target: sum.id, inputSlot: 1 },
      { id: 'sum-output', source: sum.id, target: output.id, inputSlot: 0 },
    ] }
    expect(denseGroupDetail(graph, group)).toMatchObject({ weightNodeId: matrix.id, biasNodeId: bias.id, preNodeId: sum.id })
  })
})
