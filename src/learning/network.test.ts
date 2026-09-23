import { describe, expect, it } from 'vitest'
import { applyNetworkGradients, createNetwork, createNetworkData, networkLoss, networkToGraph, neuronCalculation, runNetwork, trainNetwork } from './network'
import { backwardPass, forwardPass, validateGraph } from '../domain/engine'

describe('shared tensor network execution', () => {
  it('matches the identity fixture through separate backward and simultaneous update', () => {
    const model = createNetwork('linear')
    const example = { id: 'fixture', x: [2], y: 9, split: 'train' as const }
    expect(runNetwork(model, example.x).prediction).toBe(7)
    const { loss, gradients } = networkLoss(model, [example], true)
    expect(loss).toBe(4)
    expect(gradients[0].weights).toEqual([-8])
    expect(gradients[0].biases).toEqual([-4])
    expect(model.layers[0].weights).toEqual([3])
    const updated = applyNetworkGradients(model, gradients)
    expect(updated.layers[0].weights[0]).toBeCloseTo(3.4, 12)
    expect(updated.layers[0].biases[0]).toBeCloseTo(1.2, 12)
    expect(runNetwork(updated, example.x).prediction).toBe(8)
    expect(networkLoss(updated, [example]).loss).toBe(1)
  })

  it('uses genuine ReLU and exactly zero derivatives for a closed gate', () => {
    const model = createNetwork('neuron')
    model.layers[0].activation = 'relu'
    const example = { id: 'negative', x: [-2], y: 1, split: 'train' as const }
    expect(runNetwork(model, example.x).prediction).toBe(0)
    expect(networkLoss(model, [example], true).gradients[0]).toEqual({ id: 'layer-0', weights: [0], biases: [0] })
  })

  it('every neuron scalar trace reconstructs the very same batched matrix operation', () => {
    const model = createNetwork('small-network')
    const trace = runNetwork(model, [1, 2])
    expect(trace.layers[0].output).toEqual([2.5, 1.5])
    expect(trace.prediction).toBe(1)
    model.layers.forEach((layer, l) => layer.biases.forEach((_, n) => {
      const scalar = neuronCalculation(model, trace, l, n)
      expect(scalar.products.reduce((sum, value) => sum + value, scalar.bias)).toBeCloseTo(scalar.weightedSum, 12)
      expect(scalar.output).toBe(trace.layers[l].output[n])
    }))
  })

  it('checks all tiny-network weight and bias derivatives with finite differences', () => {
    const model = createNetwork('small-network')
    const example = { id: 'fixture', x: [1, 2], y: 1.5, split: 'train' as const }
    const analytic = networkLoss(model, [example], true).gradients
    const epsilon = 1e-5
    model.layers.forEach((layer, l) => {
      ;(['weights', 'biases'] as const).forEach(key => layer[key].forEach((_, index) => {
        const high = structuredClone(model), low = structuredClone(model)
        high.layers[l][key][index] += epsilon
        low.layers[l][key][index] -= epsilon
        const numeric = (networkLoss(high, [example]).loss - networkLoss(low, [example]).loss) / (2 * epsilon)
        expect(analytic[l][key][index]).toBeCloseTo(numeric, 7)
      }))
    })
  })

  it('checks classifier softmax-cross-entropy gradients through multiple dense layers', () => {
    const model = createNetwork('playground', 42, [3])
    const data = createNetworkData('playground').slice(0, 4)
    const analytic = networkLoss(model, data, true).gradients
    const epsilon = 1e-5
    model.layers.forEach((layer, l) => {
      ;(['weights', 'biases'] as const).forEach(key => layer[key].forEach((_, index) => {
        const high = structuredClone(model), low = structuredClone(model)
        high.layers[l][key][index] += epsilon
        low.layers[l][key][index] -= epsilon
        expect(analytic[l][key][index]).toBeCloseTo((networkLoss(high, data).loss - networkLoss(low, data).loss) / (2 * epsilon), 6)
      }))
    })
  })

  it('produces independently verified deterministic two-dimensional labels and disjoint splits', () => {
    const data = createNetworkData('playground', 137)
    expect(data).toEqual(createNetworkData('playground', 137))
    expect(data).not.toEqual(createNetworkData('playground', 138))
    expect(data.filter(point => point.split === 'train')).toHaveLength(72)
    expect(data.filter(point => point.split === 'validation')).toHaveLength(24)
    expect(new Set(data.map(point => point.id)).size).toBe(96)
    data.forEach(point => expect(point.y).toBe(Number(point.x[0] * point.x[1] > 0)))
  })

  it('learns reproducibly while holding validation points out of gradient updates', () => {
    const model = createNetwork('playground')
    const data = createNetworkData('playground')
    const before = networkLoss(model, data.filter(point => point.split === 'train')).loss
    const trained = trainNetwork(model, data, 20)
    expect(trained.losses.at(-1)!.train).toBeLessThan(before)
    expect(trained).toEqual(trainNetwork(model, data, 20))
    const changedValidation = data.map(point => point.split === 'validation' ? { ...point, y: 1 - point.y } : point)
    expect(trainNetwork(model, changedValidation, 20).model).toEqual(trained.model)
    expect(trained.model).not.toEqual(model)
    const parameters = JSON.stringify(trained.model)
    runNetwork(trained.model, [.4, .6])
    networkLoss(trained.model, data)
    expect(JSON.stringify(trained.model)).toBe(parameters)
  })

  it('exports a numerical-equivalent builder network preserving parameter IDs', () => {
    const model = createNetwork('small-network')
    const example = { id: 'fixture', x: [1, 2], y: 1.5, split: 'train' as const }
    const graph = networkToGraph(model, example)
    const forward = forwardPass(graph)
    expect(forward.loss).toBeCloseTo(networkLoss(model, [example]).loss, 10)
    const backward = backwardPass(forward.graph)
    const gradients = networkLoss(model, [example], true).gradients
    expect(backward.graph.nodes.find(node => node.id === 'layer-0/weight-0-0')?.grad?.data[0]).toBeCloseTo(gradients[0].weights[0], 10)
  })

  it('exports eight-wide layers without exceeding the builder add-port limit', () => {
    const model = createNetwork('playground', 42, [8, 8])
    const example = createNetworkData('playground')[1]
    const graph = networkToGraph(model, example)
    expect(validateGraph(graph, { requireLoss: false })).toEqual([])
    const forward = forwardPass(graph)
    runNetwork(model, example.x).output.forEach((value, index) => {
      expect(forward.graph.nodes.find(node => node.id === `layer-2/neuron-${index}/activation`)?.value?.data[0]).toBeCloseTo(value, 10)
    })
  })
})
