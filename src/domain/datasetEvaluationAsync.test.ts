import { describe, expect, it } from 'vitest'
import { evaluateDataset, evaluateDatasetAsync } from './datasetTraining'
import { createModelPreset } from './modelPresets'

describe('asynchronous dataset evaluation', () => {
  it.each(['linear', 'playground', 'decoder', 'cnn'] as const)('matches synchronous metrics for %s', async kind => {
    const graph = createModelPreset(kind)
    const dataset = graph.nodes.find(node => node.type === 'dataset')!
    const expected = evaluateDataset(graph, dataset.id, 'test')
    expect(await evaluateDatasetAsync(graph, dataset.id, 'test')).toEqual(expected)
    const losses = await evaluateDatasetAsync(graph, dataset.id, 'test', { includePredictions: false })
    expect(losses.loss).toBe(expected.loss)
    expect(losses.examples).toBe(expected.examples)
    expect(losses.rows).toEqual([])
  })

  it('yields to the event loop and respects cancellation before publishing metrics', async () => {
    const graph = createModelPreset('linear')
    const dataset = graph.nodes.find(node => node.type === 'dataset')!
    const controller = new AbortController()
    const pending = evaluateDatasetAsync(graph, dataset.id, 'test', { signal: controller.signal })
    controller.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })
})
