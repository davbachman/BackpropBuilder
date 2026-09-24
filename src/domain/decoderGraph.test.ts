import { describe, expect, it } from 'vitest'
import { decoderForward, getCheckpoint, loadCheckpoint } from '../learning/decoder'
import { crossEntropy } from '../learning/math'
import { backwardPass, forwardPass, runTrainingStep } from './engine'
import { createModelPreset } from './modelPresets'

// This exercises the complete shared-parameter DAG through both transformer
// blocks, every head, residual branch, normalization and feedforward layer.
describe('full transformer graph backpropagation', () => {
  it.each(['trained', 'untrained'] as const)('matches all %s checkpoint parameter gradients', checkpoint => {
    const graph = createModelPreset('decoder', checkpoint)
    const model = loadCheckpoint(getCheckpoint(checkpoint))
    const expected = crossEntropy(decoderForward(model, [0, 1, 2]).logitsTensor, [1, 2, 3])
    expected.backward()
    const actual = backwardPass(forwardPass(graph).graph)
    expect(actual.loss).toBeCloseTo(expected.data[0], 12)
    Object.entries(model.parameters).forEach(([id, parameter]) => {
      const gradient = actual.graph.nodes.find(node => node.id === id)!.grad!
      expect(gradient.shape).toEqual(parameter.shape)
      parameter.grad.forEach((value, index) => expect(gradient.data[index]).toBeCloseTo(value, 10))
    })
    expect(runTrainingStep(graph).loss!).toBeLessThan(actual.loss!)
  })
})
