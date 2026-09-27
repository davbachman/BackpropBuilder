import { describe, it, expect } from 'vitest'
import { prepareTextDocuments, importText, encodeText, isTextDatasetData } from './textData'
import { datasetExamplesForNode, datasetOutputValueForSlot } from './datasets'
import { buildTextModel, type TextModelKind } from '../test/textModels'
import { backwardPass, forwardPass, parameterValues, runTrainingStep, runTrainingStepFast, validateGraph } from './engine'
import { evaluateDataset, trainDataset } from './datasetTraining'
import { generateText, predictText } from './textGeneration'
import { generatePyTorchExport } from './pytorchExport'
import { createProjectStateFile, parseProjectStateFile } from './session'
import type { GraphModel } from './types'

const reviews = () => prepareTextDocuments([
  {text: 'good funny good film', label: 'positive', split: 'train'},
  {text: 'bad dull bad film', label: 'negative', split: 'train'},
  {text: 'wonderful good film', label: 'positive', split: 'test'},
  {text: 'bad film', label: 'negative', split: 'test'},
], 'reviews.csv', {task: 'sentiment', representation: 'tokens', maxLength: 8})
const corpus = () => prepareTextDocuments([
  {text: 'alice was beginning to get very tired of sitting by her sister on the bank.', split: 'train'},
  {text: 'and of having nothing to do once or twice she had peeped into the book.', split: 'test'},
], 'alice.txt', {task: 'language', maxLength: 8})
const output = (graph: GraphModel, id = 'positive-probability') => forwardPass(graph, false).graph.nodes.find(node => node.id === id)!.value!.data
const reverse = (graph: GraphModel) => ({...graph, nodes: graph.nodes.map(node => node.type === 'dataset' ? {...node, params: {...node.params, datasetValues: [{shape: datasetOutputValueForSlot(node, 0).shape, data: [...datasetOutputValueForSlot(node, 0).data].reverse()}, datasetOutputValueForSlot(node, 1), datasetOutputValueForSlot(node, 2)]}} : node)})

describe('text curriculum data and mathematical behavior', () => {
  it('fits the vocabulary on training text only, preserves supplied splits, and round-trips recipes', () => {
    const data = reviews()
    expect(data.vocabulary).not.toContain('wonderful')
    expect(encodeText('wonderful', data)).toEqual([0])
    expect(isTextDatasetData(JSON.parse(JSON.stringify(data)))).toBe(true)
    expect(isTextDatasetData({...data, maxLength: 0})).toBe(false)
    const graph = buildTextModel(data, 'mean')
    const node = graph.nodes[0]
    expect(datasetExamplesForNode({...node, params: {...node.params, trainPercent: 90}}).map(example => example.split)).toEqual(['train', 'train', 'test', 'test'])
    const file = createProjectStateFile({graph, visualizationGraph: graph, initialParameterValues: parameterValues(graph), selectedNodeIds: [], phase: 'edit', traceSteps: [], traceIndex: 0, epoch: 0, currentLoss: null, display: {showMath: true, showGradient: true, showCode: false, showVisualization: false}})
    expect(parseProjectStateFile(JSON.stringify(file)).ok).toBe(true)
  })
  it('parses quoted review CSV and prevents malformed split/label data', () => {
    const data = importText('text,label,split\n"good, funny",positive,train\n"bad\nfilm",negative,train\nokay,positive,test\n', 'reviews.csv', {task: 'sentiment'})
    expect(data.documents[1].text).toBe('bad\nfilm')
    expect(() => importText('text,label,split\na,positive,other\nb,negative,train', 'bad.csv', {task: 'sentiment'})).toThrow(/Split/)
  })
  it('creates aligned next-token targets without crossing document or split boundaries', () => {
    const data = corpus()
    const examples = datasetExamplesForNode(buildTextModel(data, 'alice-baseline').nodes[0])
    for (const example of examples) {
      const ids = encodeText(data.documents[example.documentIndex!].text, data)
      expect(example.features[0].data).toEqual(ids.slice(example.offset, example.offset! + 8))
      expect(example.target.data).toEqual(ids.slice(example.offset! + 1, example.offset! + 9))
      expect(example.split).toBe(data.documents[example.documentIndex!].split)
    }
  })
  it.each(['counts-linear', 'counts-mlp', 'mean', 'position-mean', 'attention', 'position-attention', 'transformer', 'alice-baseline', 'alice-transformer'] as TextModelKind[])('validates, differentiates, exports and matches fast SGD for %s', kind => {
    const graph = buildTextModel(kind.startsWith('alice') ? corpus() : reviews(), kind, 4)
    expect(validateGraph(graph).filter(issue => issue.code !== 'disconnected')).toEqual([])
    const traced = runTrainingStep(graph)
    expect(Number.isFinite(traced.loss)).toBe(true)
    const fast = parameterValues(runTrainingStepFast(graph))
    const slow = parameterValues(traced.graph)
    for (const id of Object.keys(slow)) slow[id].data.forEach((value, i) => expect(fast[id].data[i]).toBeCloseTo(value, 12))
    expect(generatePyTorchExport(graph).script).toContain('class BuilderModel')
  })
  it.each(['mean', 'position-mean', 'attention'] as TextModelKind[])('%s is permutation invariant after pooling', kind => {
    const graph = buildTextModel(reviews(), kind, 4)
    expect(output(graph)[0]).toBeCloseTo(output(reverse(graph))[0], 12)
  })
  it('positions before attention allow order dependence', () => {
    const graph = buildTextModel(reviews(), 'position-attention', 4)
    expect(Math.abs(output(graph)[0] - output(reverse(graph))[0])).toBeGreaterThan(1e-8)
  })
  it('one-hot matrix multiplication equals lookup, including repeated-token gradients', () => {
    const graph = buildTextModel(reviews(), 'mean', 4)
    const explicit: GraphModel = {...graph, nodes: [...graph.nodes.map(node => node.id === 'embedded-tokens' ? {...node, type: 'matmul' as const} : node), {id: 'one-hot', label: 'One-hot', type: 'one-hot', params: {numClasses: reviews().vocabulary.length}, position: {x: 0, y: 0}}], edges: [...graph.edges.filter(edge => edge.target !== 'embedded-tokens'), {id: 'ids-onehot', source: 'text-data', sourceSlot: 0, target: 'one-hot', inputSlot: 0}, {id: 'hot-product', source: 'one-hot', target: 'embedded-tokens', inputSlot: 0}, {id: 'table-product', source: 'word-embeddings', target: 'embedded-tokens', inputSlot: 1}]}
    expect(output(explicit)).toEqual(output(graph))
    const gradients = [graph, explicit].map(model => backwardPass(forwardPass(model, false).graph).graph.nodes.find(node => node.id === 'word-embeddings')!.grad!.data)
    expect(gradients[0]).toEqual(gradients[1])
    expect(generatePyTorchExport(explicit).script).toContain('F.one_hot')
  })
  it('causal masking prevents future-token influence and generation leaves parameters unchanged', () => {
    const graph = buildTextModel(corpus(), 'alice-transformer', 4)
    const source = graph.nodes[0]
    const values = [0, 1, 2].map(slot => datasetOutputValueForSlot(source, slot))
    const modified = {...graph, nodes: graph.nodes.map(node => node.id === source.id ? {...node, params: {...node.params, datasetValues: [{...values[0], data: values[0].data.map((id, i) => i > 3 ? (id + 1) % corpus().vocabulary.length : id)}, values[1], values[2]]}} : node)}
    const size = corpus().vocabulary.length
    expect(output(graph, 'vocabulary-bias-add').slice(0, 4 * size)).toEqual(output(modified, 'vocabulary-bias-add').slice(0, 4 * size))
    const before = JSON.stringify(graph)
    expect(predictText(graph, 'alice was beginning').ids).toHaveLength(8)
    expect(generateText(graph, 'alice', 12).length).toBeGreaterThan(5)
    expect(JSON.stringify(graph)).toBe(before)
  })
  it('actually learns review labels while preserving held-out examples', async () => {
    const graph = buildTextModel(reviews(), 'counts-linear', 4)
    graph.learningRate = .1
    const initial = evaluateDataset(graph, 'text-data', 'train').loss
    const trained = await trainDataset(graph, 'text-data', 20)
    expect(evaluateDataset(trained, 'text-data', 'train').loss).toBeLessThan(initial / 2)
    expect(trained.nodes[0].params.textData).toEqual(graph.nodes[0].params.textData)
  })
})

it('rejects target leakage during inference', () => {
  const graph = buildTextModel(corpus(), 'alice-baseline', 4)
  const leaked = {...graph, edges:graph.edges.map(edge => edge.target === 'embedded-tokens' && edge.inputSlot === 1 ? {...edge,sourceSlot:2} : edge)}
  expect(() => predictText(leaked, 'alice')).toThrow('must not depend on dataset targets')
})

it('round-trips a generated unknown token without expanding it into characters', () => {
  expect(encodeText('a<unk>a', corpus())).toEqual([...encodeText('a', corpus()),0,...encodeText('a', corpus())])
})
