import { describe, expect, it } from 'vitest'
import { backwardPass, forwardPass, parameterValues } from './engine'
import { createModelPreset } from './modelPresets'
import { graphWithDatasetSample, isHeldOutSample, modelDataset, modelDatasetEvaluation, trainModelDataset } from './modelDatasets'
import { CNN_DIGITS, createCnnPreset, selectDigit } from '../learning/cnn'
import { datasetOutputValueForSlot } from './datasets'

describe('datasets use the editable graph', () => {
  it.each(['linear', 'small-network', 'playground'] as const)('matches separate per-example execution when evaluating a %s batch', kind => {
    const graph = createModelPreset(kind), dataset = modelDataset(kind === 'playground' ? 'xor' : kind === 'small-network' ? 'plane-2d' : 'line-1d')
    const actual = modelDatasetEvaluation(graph, dataset.samples)
    const examples = dataset.samples.map((_, index) => {
      const evaluated = forwardPass(graphWithDatasetSample(graph, dataset, index))
      const probability = evaluated.graph.nodes.find(node => node.id === 'network-probabilities')?.value
      const predictionEdge = evaluated.graph.edges.find(edge => edge.target === 'loss' && edge.inputSlot === 0)!
      return { loss: evaluated.loss!, prediction: probability ? probability.data[1] : evaluated.graph.nodes.find(node => node.id === predictionEdge.source)!.value!.data[0] }
    })
    expect(actual.loss).toBeCloseTo(examples.reduce((sum, example) => sum + example.loss, 0) / examples.length, 10)
    actual.predictions.forEach((prediction, index) => expect(prediction).toBeCloseTo(examples[index].prediction, 10))
  })

  it('fits the regression points and returns the same scalar canvas structure', () => {
    const graph = createModelPreset('linear'), dataset = modelDataset('line-1d')
    const initialLoss = modelDatasetEvaluation(graph, dataset.samples).loss
    const trained = trainModelDataset(graph, dataset, 25)
    expect(modelDatasetEvaluation(trained, dataset.samples).loss).toBeLessThan(initialLoss / 10)
    expect(parameterValues(trained)).not.toEqual(parameterValues(graph))
    expect(trained.nodes.map(node => node.id)).toEqual(graph.nodes.map(node => node.id))
    expect(trained.edges).toEqual(forwardPass(graph).graph.edges.map(edge => expect.objectContaining({ id: edge.id, source: edge.source, target: edge.target })))
    expect(trained.nodes.find(node => node.id === 'input-0')!.value!.shape).toEqual([])
  })

  it('keeps held-out labels completely out of parameter updates', () => {
    const graph = createModelPreset('linear'), data = modelDataset('line-1d')
    const altered = { ...data, samples: data.samples.map(sample => sample.split === 'test' ? { ...sample, y: sample.y + 1000 } : sample) }
    expect(parameterValues(trainModelDataset(graph, data, 3))).toEqual(parameterValues(trainModelDataset(graph, altered, 3)))
  })

  it('trains a two-class graph without retaining temporary batch reshape sizes', () => {
    const graph = createModelPreset('playground'), dataset = modelDataset('xor')
    const before = modelDatasetEvaluation(graph, dataset.samples).loss
    const trained = trainModelDataset(graph, dataset, 25)
    expect(modelDatasetEvaluation(trained, dataset.samples).loss).toBeLessThan(before)
    expect(trained.nodes.find(node => node.id === 'network-logits')?.params.shape).toEqual([1, 2])
    expect(trained.nodes.find(node => node.id === 'network-probabilities')?.value?.shape).toEqual([1, 2])
    expect(modelDataset('xor')).toEqual(dataset)
  })

  it('matches the mean of per-example classifier gradients for every canonical parameter', () => {
    const graph = createModelPreset('playground'), dataset = modelDataset('xor')
    const initial = parameterValues(graph), expectedGradients = Object.fromEntries(Object.entries(initial).map(([id]) => [id, 0]))
    const trainCount = dataset.samples.filter(sample => sample.split === 'train').length
    dataset.samples.forEach((sample, index) => {
      if (sample.split !== 'train') return
      const result = backwardPass(graphWithDatasetSample(graph, dataset, index)).graph
      for (const node of result.nodes) if (node.type === 'weight' || node.type === 'bias') expectedGradients[node.id] += node.grad!.data[0] / trainCount
    })
    const actual = parameterValues(trainModelDataset(graph, dataset, 1))
    Object.entries(actual).forEach(([id, parameter]) => expect(parameter.data[0]).toBeCloseTo(initial[id].data[0] - graph.learningRate * expectedGradients[id], 10))
  })

  it('identifies a selected held-out point until its inputs change', () => {
    const graph = createModelPreset('linear'), dataset = modelDataset('line-1d')
    const heldOut = graphWithDatasetSample(graph, dataset, 0)
    expect(isHeldOutSample(heldOut)).toBe(true)
    expect(isHeldOutSample(graphWithDatasetSample(graph, dataset, 1))).toBe(false)
    const data = heldOut.nodes.find(node=>node.type === 'dataset')!
    data.params.datasetValues = [{shape:[],data:[99]},datasetOutputValueForSlot(data,1)]
    expect(isHeldOutSample(heldOut)).toBe(false)
  })

  it('protects an actual held-out digit image and permits training images or custom pixels', () => {
    const graph = createCnnPreset()
    expect(isHeldOutSample(graph)).toBe(true)
    const train = CNN_DIGITS.find(sample => sample.split === 'train')!
    expect(isHeldOutSample(selectDigit(graph, train))).toBe(false)
    const data = graph.nodes.find(node=>node.type === 'dataset')!
    const value = datasetOutputValueForSlot(data,0)
    data.params.datasetValues = [{...value,data:value.data.map((pixel,index)=>index === 0 ? 1-pixel : pixel)},datasetOutputValueForSlot(data,1)]
    expect(isHeldOutSample(graph)).toBe(false)
  })
})
