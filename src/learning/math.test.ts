import { describe, expect, it } from 'vitest'
import { Tensor, tensor, scalar, add, mul, sub, scale, matmul, sum, mean, reshape, transpose, slice, concat, embedding, relu, softmax, causalMask, layerNorm, crossEntropy } from './math'

function checkGradients(inputs: Tensor[], objective: () => Tensor, tolerance = 2e-5) {
  inputs.forEach(t => t.zeroGrad())
  objective().backward()
  const analytic = inputs.map(t => [...t.grad]), epsilon = 1e-5
  inputs.forEach((input, n) => input.data.forEach((value, index) => {
    input.data[index] = value + epsilon; const plus = objective().data[0]
    input.data[index] = value - epsilon; const minus = objective().data[0]
    input.data[index] = value
    expect(Math.abs(analytic[n][index] - (plus - minus) / (2 * epsilon))).toBeLessThan(tolerance)
  }))
}
const weighted = (a: Tensor) => sum(mul(a, tensor(a.shape, a.data.map((_, i) => 0.1 + (i % 7) * 0.17))))

describe('bounded tensor differentiation', () => {
  it('keeps elementwise multiplication distinct from matrix multiplication', () => {
    const a = tensor([2, 2], [1, 2, 3, 4]), b = tensor([2, 2], [5, 6, 7, 8])
    expect(mul(a, b).data).toEqual([5, 12, 21, 32])
    expect(matmul(a, b).data).toEqual([19, 22, 43, 50])
    expect(() => matmul(a, tensor([3], [1, 2, 3]))).toThrow(/Matmul/)
  })
  it('reduces broadcasting adjoints over every repeated dimension', () => {
    const a = tensor([2, 1, 3], [0.2, 0.5, -0.8, 1.4, 1.7, -1.9], true), b = tensor([1, 2, 1], [1.3, -2], true), c = scalar(0.7, true)
    checkGradients([a, b, c], () => weighted(sub(mul(add(a, b), c), b)))
    expect(add(a, b).shape).toEqual([2, 2, 3])
    expect(() => add(a, tensor([2], [1, 2]))).toThrow(/broadcast/)
  })
  it('differentiates matrix products and shared parameter paths', () => {
    const a = tensor([2, 3], [0.2, -0.7, 1.1, 0.8, -0.5, 1.3], true), b = tensor([3, 2], [0.4, 0.6, -0.3, 0.9, 0.7, -1.2], true)
    checkGradients([a, b], () => weighted(add(matmul(a, b), matmul(a, b))))
    const w = scalar(3, true), x = scalar(2), bias = scalar(1, true)
    const error = sub(add(mul(w, x), bias), scalar(9)), loss = mul(error, error)
    loss.backward(); expect(loss.data[0]).toBe(4); expect(w.grad).toEqual([-8]); expect(bias.grad).toEqual([-4])
    w.data[0] -= 0.05 * w.grad[0]; bias.data[0] -= 0.05 * bias.grad[0]
    expect(w.data[0]).toBe(3.4); expect(bias.data[0]).toBe(1.2)
    expect(add(mul(w, x), bias).data[0]).toBe(8)
    const shared = scalar(2, true)
    const total = add(mul(shared, shared), scale(shared, 3))
    total.backward(); expect(shared.grad).toEqual([7])
    total.backward(); expect(shared.grad).toEqual([14])
  })
  it('differentiates reduction, reshape, transpose, head slicing and concatenation', () => {
    const a = tensor([2, 3, 2], [0.4, -1.2, 0.6, 0.7, -0.9, 1.8, 1.1, -0.8, 0.3, 2.1, 0.2, 0.9], true)
    checkGradients([a], () => weighted(mean(a, 1, true)))
    checkGradients([a], () => weighted(sum(a, 2)))
    checkGradients([a], () => weighted(transpose(a, [1, 2, 0])))
    checkGradients([a], () => {
      const matrix = reshape(a, [3, 4])
      return weighted(concat([slice(matrix, 1, 2, 4), slice(matrix, 1, 0, 2), slice(matrix, 1, 0, 2)], 1))
    })
    expect(sum(a).shape).toEqual([])
    expect(sum(a, undefined, true).shape).toEqual([1, 1, 1])
  })
  it('scatter-adds repeated embedding token gradients', () => {
    const table = tensor([3, 2], [1, 2, 3, 4, 5, 6], true)
    checkGradients([table], () => weighted(embedding(table, [1, 0, 1])))
    table.zeroGrad(); sum(embedding(table, [1, 0, 1])).backward()
    expect(table.grad).toEqual([1, 1, 2, 2, 0, 0])
  })
  it('differentiates stable softmax, masking, relu and cross entropy', () => {
    const a = tensor([3, 3], [1.2, -0.7, 0.5, 2.3, -1.1, 0.4, 0.8, 1.5, -0.9], true)
    checkGradients([a], () => weighted(softmax(causalMask(a))))
    checkGradients([a], () => weighted(relu(a)))
    checkGradients([a], () => crossEntropy(a, [0, 2, 1]))
    const stable = softmax(tensor([1, 3], [10000, 10001, 9999]))
    expect(stable.data.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    expect(crossEntropy(tensor([1, 2], [-10000, 10000]), [0]).data).toEqual([20000])
    a.zeroGrad(); sum(causalMask(a)).backward()
    expect(a.grad).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1])
    const extreme = tensor([2, 2], [-1e12, 1e12, -1e12, -1e12], true)
    const probabilities = softmax(causalMask(extreme))
    expect(probabilities.data).toEqual([1, 0, 0.5, 0.5])
    weighted(probabilities).backward()
    expect(extreme.grad[1]).toBe(0)
  })
  it('differentiates layer normalization including learned scale and bias', () => {
    const a = tensor([2, 3], [0.5, -0.7, 1.2, 0.3, 0.8, -1.4], true), gamma = tensor([3], [0.8, 1.2, -0.3], true), beta = tensor([3], [0.2, -0.1, 0.4], true)
    checkGradients([a, gamma, beta], () => weighted(layerNorm(a, gamma, beta)))
    expect(layerNorm(tensor([1, 3], [2, 2, 2])).data).toEqual([0, 0, 0])
  })
  it('rejects invalid shapes, values and ambiguous backward seeds', () => {
    expect(() => tensor([2], [1])).toThrow()
    expect(() => tensor([1], [Infinity])).toThrow()
    expect(() => tensor([0], [])).toThrow()
    expect(() => tensor([200001], Array(200001).fill(0))).toThrow(/oversized/)
    expect(() => matmul(tensor([500, 1], Array(500).fill(1)), tensor([1, 500], Array(500).fill(1)))).toThrow(/oversized/)
    expect(() => add(tensor([500, 1], Array(500).fill(1)), tensor([1, 500], Array(500).fill(1)))).toThrow(/oversized/)
    expect(() => tensor([2], [1, 2], true).backward()).toThrow(/seed/)
  })
})
