import { DATASET_OPTIONS, datasetExamplesForNode, datasetExampleIndex, datasetForNode, datasetMode } from './datasets'
import { forwardPass, isLossNode, runTrainingStep } from './engine'
import { toTensor } from './tensor'
import type { DatasetKind, GraphModel } from './types'
import { CNN_DIGITS } from '../learning/cnn'

export type ModelDatasetKind = DatasetKind | 'xor'
export interface ModelSample { x: number[]; y: number; split: 'train' | 'test' }
export interface ModelDataset { id: ModelDatasetKind; label: string; classification: boolean; samples: ModelSample[] }

export function modelDataset(kind: ModelDatasetKind): ModelDataset {
  const data = DATASET_OPTIONS.find(dataset => dataset.kind === kind)!
  return { id: kind, label: data.label, classification: data.task === 'binary-classification', samples: data.targetValue.data.map((y, index) => ({ x: data.featureValues.map(values => values.data[index]), y, split: index % 4 === 0 ? 'test' : 'train' })) }
}

export function datasetsForModel(graph: GraphModel): ModelDataset[] {
  const twoInputs = graph.nodes.some(node => node.id === 'input-1')
  const classification = graph.nodes.some(node => node.id === 'loss' && (node.type === 'cross-entropy' || (node.type === 'loss' && node.params.loss === 'cross-entropy')))
  const kinds: ModelDatasetKind[] = !twoInputs ? ['line-1d', 'cubic-1d'] : classification ? ['xor', 'circle-center', 'parabola-boundary'] : ['plane-2d']
  const source = graph.nodes.find(node => node.type === 'dataset')
  const actual = source && datasetForNode(source)
  if (actual && !actual.examples && !kinds.includes(actual.kind)) kinds.unshift(actual.kind)
  return kinds.map(modelDataset)
}

/** A batch is the same scalar neuron graph broadcast over examples. Only the
 * known output packing nodes need row-aware shapes; every learned parameter
 * remains the original scalar and receives its complete batch gradient. */
function batchGraph(graph: GraphModel, samples: ModelSample[]): GraphModel {
  if (!samples.length) throw new Error('Choose at least one example.')
  const count = samples.length, classification = graph.nodes.some(node => node.id === 'loss' && (node.type === 'cross-entropy' || (node.type === 'loss' && node.params.loss === 'cross-entropy')))
  return { ...graph, nodes: graph.nodes.map(node => {
    const params = { ...node.params }
    if (node.type === 'dataset') params.datasetValues = [...samples[0].x.map((_, axis) => ({shape:[count],data:samples.map(sample=>sample.x[axis])})), {shape:[count],data:samples.map(sample=>sample.y)}]
    const input = /^input-(\d+)$/.exec(node.id)
    if (input) params.value = { shape: [count], data: samples.map(sample => sample.x[Number(input[1])]) }
    if (node.id === 'target') params.value = { shape: [count], data: samples.map(sample => sample.y) }
    if (classification && /^output-\d+$/.test(node.id) && node.type === 'reshape') params.shape = [count, 1]
    if (classification && node.id === 'output-vector') params.axis = 1
    if (classification && node.id === 'network-logits') params.shape = [count, 2]
    return { ...node, params, value: undefined, grad: undefined, cache: undefined }
  }) }
}

export function modelDatasetEvaluation(graph: GraphModel, samples: ModelSample[]): { predictions: number[]; loss: number } {
  const result = forwardPass(batchGraph(graph, samples))
  const probability = result.graph.nodes.find(node => node.id === 'network-probabilities')?.value
  const lossNode = result.graph.nodes.find(isLossNode)
  const outputId = graph.edges.find(edge => edge.target === lossNode?.id && (edge.inputSlot ?? 0) === 0)?.source
  const output = result.graph.nodes.find(node => node.id === outputId)?.value
  if (!output || result.loss === undefined) throw new Error('Connect the model output and target to a loss.')
  return { predictions: probability ? samples.map((_, index) => probability.data[index * 2 + 1]) : output.data, loss: result.loss }
}

export function graphWithDatasetSample(graph: GraphModel, dataset: ModelDataset, index?: number): GraphModel {
  const sample = index === undefined ? undefined : dataset.samples[index]
  const next = { ...graph, groups: graph.groups?.map(group => group.id === 'network' ? { ...group, detail: { ...group.detail, datasetKind: dataset.id, sampleIndex: index, selectedSampleSplit: sample?.split } } : group), nodes: graph.nodes.map(node => {
    if (node.type === 'dataset') return {...node,params:{...node.params,dataset:dataset.id,datasetMode:'sample' as const,datasetIndex:index ?? 0,datasetValues:undefined}}
    if (!sample) return node
    const input = /^input-(\d+)$/.exec(node.id)
    if (!input && node.id !== 'target') return node
    const scalar = input ? sample.x[Number(input[1])] : sample.y
    const value = { shape: !input && dataset.classification ? [1] : [], data: [scalar] }
    return { ...node, params: { ...node.params, value }, value, grad: undefined, cache: undefined }
  }) }
  return forwardPass(next).graph
}

/** A selected held-out example may be inspected and differentiated, but a
 * point-wise optimizer must not use it as a training example. */
export function isHeldOutSample(graph: GraphModel): boolean {
  const sources = graph.nodes.filter(node => node.type === 'dataset')
  if (sources.length) return sources.some(node => !node.params.datasetValues && (datasetMode(node) === 'sample'
    ? datasetExamplesForNode(node)[datasetExampleIndex(node)].split === 'test'
    : node.params.datasetSplit !== 'train'))
  const digitId = graph.groups?.find(group => group.id === 'features')?.detail?.sampleId
  const digit = typeof digitId === 'string' ? CNN_DIGITS.find(sample => sample.id === digitId) : undefined
  if (digit?.split === 'test') {
    const image = toTensor(graph.nodes.find(node => node.id === 'image-input')?.params.value)
    return image.shape.join(',') === '8,8,1' && image.data.every((pixel, index) => pixel === digit.pixels[index] / 16)
  }
  const detail = graph.groups?.find(group => group.id === 'network')?.detail
  if (detail?.selectedSampleSplit !== 'test' || typeof detail.sampleIndex !== 'number') return false
  const dataset = datasetsForModel(graph).find(candidate => candidate.id === detail.datasetKind)
  const sample = dataset?.samples[detail.sampleIndex]
  return sample?.split === 'test' && sample.x.every((number, index) => {
    const input = graph.nodes.find(node => node.id === `input-${index}`)
    const current = toTensor(input?.params.value)
    return current.data.length === 1 && current.data[0] === number
  })
}

export function trainModelDataset(graph: GraphModel, dataset: ModelDataset, epochs: number): GraphModel {
  let batch = batchGraph(graph, dataset.samples.filter(sample => sample.split === 'train'))
  for (let epoch = 0; epoch < epochs; epoch++) batch = runTrainingStep(batch).graph
  const updated = new Map(batch.nodes.filter(node => node.type === 'weight' || node.type === 'bias').map(node => [node.id, toTensor(node.params.value)]))
  const canonical = { ...graph, nodes: graph.nodes.map(node => {
    const parameter = updated.get(node.id)
    return parameter ? { ...node, params: { ...node.params, value: parameter }, value: parameter, grad: undefined, cache: undefined } : node
  }) }
  return forwardPass(canonical).graph
}
