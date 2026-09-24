import { datasetOutputValueForSlot } from './datasets'
import { describe, expect, it } from 'vitest'
import { decoderForward, getCheckpoint, loadCheckpoint } from '../learning/decoder'
import { createNetwork, networkLoss, runNetwork } from '../learning/network'
import { LESSONS } from '../learning/presets'
import { backwardPass, forwardPass, runTrainingStep, validateGraph } from './engine'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { createModelPreset } from './modelPresets'
import type { GraphModel, TensorValue } from './types'

const at = (graph: GraphModel, id: string): TensorValue => graph.nodes.find(node => node.id === id)!.value!
function expectTensor(actual: TensorValue, expected: TensorValue) {
  expect(actual.shape).toEqual(expected.shape)
  expect(actual.data).toHaveLength(expected.data.length)
  actual.data.forEach((number, index) => expect(number).toBeCloseTo(expected.data[index], 10))
}

describe('one executable model canvas', () => {
  it.each(LESSONS.map(lesson => lesson.id))('%s has valid wires, a true nested hierarchy and finite output', (kind) => {
    const graph = createModelPreset(kind)
    // Softmax is a readable auxiliary output of logits; cross-entropy consumes
    // logits directly for numerical stability, so that branch has no loss.
    expect(validateGraph(graph, { requireLoss: false }).filter(issue => issue.code !== 'disconnected')).toEqual([])
    const result = forwardPass(graph)
    expect(result.steps.length).toBeGreaterThan(0)
    expect(result.graph.nodes.every(node => node.value && node.value.data.every(Number.isFinite))).toBe(true)
    for (const group of graph.groups ?? []) {
      expect(group.nodeIds.length).toBeGreaterThan(0)
      expect(group.nodeIds.every(id => graph.nodes.some(node => node.id === id))).toBe(true)
      if (group.parentId) {
        const parent = graph.groups!.find(candidate => candidate.id === group.parentId)!
        expect(parent).toBeDefined()
        expect(group.nodeIds.every(id => parent.nodeIds.includes(id))).toBe(true)
      }
    }
  })

  it.each(['trained', 'untrained'] as const)('reconstructs every decoder stage from the %s checkpoint', checkpointName => {
    const checkpoint = getCheckpoint(checkpointName), graph = createModelPreset('decoder', checkpointName)
    const result = forwardPass(graph).graph, expected = decoderForward(loadCheckpoint(checkpoint), [0, 1, 2])
    expectTensor(at(result, 'token-vectors'), expected.embedding)
    expectTensor(at(result, 'position-vectors'), expected.position)
    expectTensor(at(result, 'embedding-output'), expected.input)
    expected.blocks.forEach((block, index) => {
      const prefix = `blocks.${index}`
      expectTensor(at(result, `${prefix}.norm1.output`), block.norm1)
      expectTensor(at(result, `${prefix}.q.output`), block.q)
      expectTensor(at(result, `${prefix}.k.output`), block.k)
      expectTensor(at(result, `${prefix}.v.output`), block.v)
      block.heads.forEach((head, headIndex) => {
        expectTensor(at(result, `${prefix}.head.${headIndex}.scores`), head.scores)
        expectTensor(at(result, `${prefix}.head.${headIndex}.weights`), head.weights)
        expectTensor(at(result, `${prefix}.head.${headIndex}.output`), head.output)
      })
      expectTensor(at(result, `${prefix}.attention.output`), block.attention)
      expectTensor(at(result, `${prefix}.attention-residual`), block.residual)
      expectTensor(at(result, `${prefix}.norm2.output`), block.norm2)
      expectTensor(at(result, `${prefix}.ff1.pre`), block.ffPre)
      expectTensor(at(result, `${prefix}.ff1.output`), block.ffHidden)
      expectTensor(at(result, `${prefix}.ff2.output`), block.ffOut)
      expectTensor(at(result, `${prefix}.output`), block.output)
    })
    expectTensor(at(result, 'finalNorm.output'), expected.norm)
    expectTensor(at(result, 'decoder-logits'), expected.logits)
    expectTensor(at(result, 'decoder-probabilities'), expected.probabilities)
    Object.entries(checkpoint.parameters).forEach(([id, parameter]) => expect(graph.nodes.find(node => node.id === id)?.params.value).toEqual(parameter))
  })

  it.each(['linear', 'neuron', 'small-network', 'playground'] as const)('computes %s using the concrete scalar neuron nodes', kind => {
    const model = createNetwork(kind), preset = createModelPreset(kind), dataset = preset.nodes.find(node => node.type === 'dataset')!
    const input = Array.from({length:kind === 'linear' || kind === 'neuron' ? 1 : 2},(_,slot)=>datasetOutputValueForSlot(dataset,slot).data[0])
    if (kind === 'neuron') model.layers[0].activation = 'relu'
    const graph = backwardPass(forwardPass(createModelPreset(kind)).graph).graph
    const calculation = runNetwork(model, input)
    const expected = networkLoss(model, [{ id: 'example', x: input, y: datasetOutputValueForSlot(dataset,input.length).data[0], split: 'train' }], true)
    expect(at(graph, 'loss').data[0]).toBeCloseTo(expected.loss, 10)
    model.layers.forEach((layer, layerIndex) => {
      for (let neuron = 0; neuron < layer.outputs; neuron++) {
        expect(at(graph, `${layer.id}/neuron-${neuron}/activation`).data[0]).toBeCloseTo(calculation.layers[layerIndex].output[neuron], 10)
        for (let input = 0; input < layer.inputs; input++) {
          expect(graph.nodes.find(node => node.id === `${layer.id}/weight-${input}-${neuron}`)!.grad!.data[0]).toBeCloseTo(expected.gradients[layerIndex].weights[input * layer.outputs + neuron], 10)
        }
      }
    })
  })

  it('lets a copied ReLU neuron become an independent, trainable part of a larger model', () => {
    const original = createModelPreset('neuron')
    const fragment = copyGraphSelection(original, { nodeIds: [], groupId: 'layer-0/neuron-0' })!
    const { graph, selection } = pasteGraphClipboard(original, fragment, { x: 0, y: 700 })
    const copied = graph.groups!.find(group => group.id === selection.groupId)!
    const copiedOutput = copied.detail!.outputNodeId as string
    const copiedWeights = copied.detail!.weightNodeIds as string[]
    expect(copiedWeights).not.toEqual(fragment.group!.detail!.weightNodeIds)
    expect(copied.detail!.inputNodeIds).toEqual(['input-0'])
    expect(copied.nodeIds).toContain(copiedOutput)
    expect(validateGraph(graph, { requireLoss: false }).filter(issue => issue.code !== 'disconnected')).toEqual([])
    graph.nodes.push({ id: 'combine-neurons', type: 'add', label: 'Combine neurons', position: { x: 2000, y: 100 }, params: {} })
    graph.edges = graph.edges.map(edge => edge.target === 'loss' && edge.inputSlot === 0 ? { ...edge, source: 'combine-neurons' } : edge)
    graph.edges.push({ id: 'first-neuron-combine', source: 'layer-0/neuron-0/activation', target: 'combine-neurons', inputSlot: 0 }, { id: 'second-neuron-combine', source: copiedOutput, target: 'combine-neurons', inputSlot: 1 })
    const forward = forwardPass(graph)
    expect(at(forward.graph, 'combine-neurons').data[0]).toBeCloseTo(3.8)
    const trained = runTrainingStep(graph).graph
    const trainedLoss = forwardPass(trained).loss!
    expect(trainedLoss).toBeLessThan(forward.loss!)
    expect(trained.nodes.find(node => node.id === copiedWeights[0])!.params.value).not.toEqual(original.nodes.find(node => node.id === 'layer-0/weight-0-0')!.params.value)
    expect(original.nodes.find(node => node.id === 'layer-0/weight-0-0')!.params.value).toBe(3)
  })

  it('keeps dense MLP inspection bindings attached to the same live parameters, including after copying', () => {
    const graph = createModelPreset('decoder')
    const layer = graph.groups!.find(group => group.id === 'blocks.0.ff1.layer')!
    expect(layer.detail).toMatchObject({ inputNodeId: 'blocks.0.norm2.output', weightNodeId: 'blocks.0.ff1', biasNodeId: 'blocks.0.ff1Bias', preNodeId: 'blocks.0.ff1.pre', outputNodeId: 'blocks.0.ff1.output', activation: 'relu' })
    const fragment = copyGraphSelection(graph, { nodeIds: [], groupId: layer.id })!
    const pasted = pasteGraphClipboard(graph, fragment, { x: 0, y: 500 })
    const copy = pasted.graph.groups!.find(group => group.id === pasted.selection.groupId)!
    expect(copy.detail!.inputNodeId).toBe(layer.detail!.inputNodeId)
    for (const key of ['weightNodeId', 'biasNodeId', 'preNodeId', 'outputNodeId']) {
      expect(copy.detail![key]).not.toBe(layer.detail![key])
      expect(copy.nodeIds).toContain(copy.detail![key])
    }
  })

  it('preserves the causal boundary when a later value is changed', () => {
    const graph = createModelPreset('causal')
    const before = forwardPass(graph).graph
    const dataset = graph.nodes.find(node => node.type === 'dataset')!
    dataset.params.datasetValues = [0,1,2,3].map(slot=>datasetOutputValueForSlot(dataset,slot))
    dataset.params.datasetValues[2] = { shape: [3, 2], data: [1, 0, 0, 2, 100, -200] }
    const after = forwardPass(graph).graph
    expect(at(before, 'attention-output').data.slice(0, 4)).toEqual(at(after, 'attention-output').data.slice(0, 4))
    expect(at(before, 'attention-output').data.slice(4)).not.toEqual(at(after, 'attention-output').data.slice(4))
  })
})
