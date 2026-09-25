import { datasetExamplesForNode, datasetForNode, datasetMode, datasetOutputValueForSlot } from './datasets'
import { forwardPass, isLossNode, runTrainingStep } from './engine'
import type { GraphModel, GraphNode } from './types'

export function withDatasetExample(graph: GraphModel, id: string, index: number): GraphModel {
  return { ...graph, nodes: graph.nodes.map(node => node.id === id ? {
    ...node, params: { ...node.params, datasetMode: 'sample', datasetIndex: index, datasetValues: undefined },
  } : node) }
}

function withDatasetBatch(graph: GraphModel, id: string, split: 'train' | 'test'): GraphModel {
  return {...graph,nodes:graph.nodes.map(node=>node.id === id ? {...node,params:{...node.params,datasetSplit:split,datasetValues:undefined}} : node)}
}

/** Discover the scores through the loss connection, independent of node names. */
export function predictionNode(graph: GraphModel): GraphNode | undefined {
  const loss = graph.nodes.find(isLossNode)
  const edge = graph.edges.find(edge => edge.target === loss?.id && (edge.inputSlot ?? 0) === 0)
  return graph.nodes.find(node => node.id === edge?.source)
}

export interface DatasetPrediction { example: string; actual: string; predicted: string; correct?: boolean }
export interface DatasetMetrics { loss: number; accuracy?: number; examples: number; predictions: number; rows: DatasetPrediction[] }

export function evaluateDataset(graph: GraphModel, id: string, split: 'train' | 'test'): DatasetMetrics {
  const source = graph.nodes.find(node => node.id === id && node.type === 'dataset')
  if (!source) throw new Error('Choose a dataset block.')
  const dataset = datasetForNode(source)
  const examples = datasetExamplesForNode(source)
  const batch = datasetMode(source) === 'batch'
  const indices = examples.flatMap((example,index)=>example.split === split ? [index] : [])
  let loss = 0, count = 0, correct = 0, predictions = 0
  const rows: DatasetPrediction[] = []
  for (const index of batch ? indices.slice(0,1) : indices) {
    const result = forwardPass(batch ? withDatasetBatch(graph,id,split) : withDatasetExample(graph, id, index))
    if (result.loss === undefined || !Number.isFinite(result.loss)) throw new Error('Connect predictions and dataset targets to a loss before evaluating.')
    loss += result.loss
    count++
    const lossNode = result.graph.nodes.find(isLossNode)!
    const output = predictionNode(result.graph)?.value
    const targetEdge = result.graph.edges.find(edge => edge.target === lossNode.id && edge.inputSlot === 1)
    const targetNode = result.graph.nodes.find(node => node.id === targetEdge?.source)
    const target = targetNode?.type === 'dataset' ? datasetOutputValueForSlot(targetNode, targetEdge?.sourceSlot ?? 0) : targetNode?.value
    const categorical = dataset.task.includes('classification') || dataset.task === 'sequence' || lossNode.type === 'cross-entropy' || lossNode.params.loss === 'cross-entropy'
    const width = output && target ? output.data.length / target.data.length : 0
    if (output && target && Number.isInteger(width) && width >= 1 && (width === 1 || output.shape.at(-1) === width)) {
      target.data.forEach((actual, row) => {
        const scores = output.data.slice(row * width, (row + 1) * width)
        const predicted = categorical && width > 1 ? scores.indexOf(Math.max(...scores))
          : dataset.task === 'binary-classification' ? Number(scores[0] >= .5) : scores[0]
        const scored = categorical && (width > 1 || dataset.task === 'binary-classification')
        const matched = scored ? predicted === actual : undefined
        if (matched !== undefined) { correct += Number(matched); predictions++ }
        const exampleIndex = batch ? indices[row] : index
        rows.push({
          example: `${examples[exampleIndex]?.label ?? `Example ${exampleIndex + 1}`}${target.data.length > 1 && !batch ? ` · output ${row + 1}` : ''}`,
          actual: displayPrediction(actual, scored, dataset.classLabels, dataset.vocabulary),
          predicted: displayPrediction(predicted, scored, dataset.classLabels, dataset.vocabulary),
          ...(matched === undefined ? {} : { correct: matched }),
        })
      })
    }
  }
  if (!count) throw new Error(`This dataset has no ${split} examples.`)
  return { loss: loss / count, examples: indices.length, predictions, rows, accuracy: predictions ? correct / predictions : undefined }
}

function displayPrediction(value: number, categorical: boolean, classLabels?: string[], vocabulary?: string[]): string {
  if (categorical) return classLabels?.[value] ?? vocabulary?.[value] ?? String(value)
  return Number(value.toPrecision(5)).toString()
}

/** SGD traverses training examples only and restores the inspected example.
 * Yield between chunks so the canvas remains responsive and training can stop. */
export async function trainDataset(graph: GraphModel, id: string, epochs: number, options: { signal?: AbortSignal; progress?: (done: number, total: number) => void } = {}): Promise<GraphModel> {
  const source = graph.nodes.find(node => node.id === id && node.type === 'dataset')
  if (!source) throw new Error('Choose a dataset block.')
  if (graph.nodes.filter(node => node.type === 'dataset').length !== 1) throw new Error('Use one dataset block to keep features and targets synchronized during training.')
  const indices = datasetExamplesForNode(source).flatMap((example, index) => example.split === 'train' ? [index] : [])
  if (!indices.length || !Number.isInteger(epochs) || epochs < 1) throw new Error('Choose training examples and a positive epoch count.')
  const batch = datasetMode(source) === 'batch'
  let next = graph, done = 0
  for (let epoch = 0; epoch < epochs; epoch++) {
    // Included image files are sorted by class. Shuffle each epoch so SGD
    // doesn't forget earlier classes while processing a long run of one digit.
    const order = [...indices]
    let seed = 42 + epoch
    for (let i = order.length - 1; i > 0; i--) {
      seed = (1664525 * seed + 1013904223) >>> 0
      const j = Math.floor(seed / 4294967296 * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    for (const index of batch ? order.slice(0,1) : order) {
      if (options.signal?.aborted) throw new Error('Training stopped; the previous parameters are unchanged.')
      next = runTrainingStep(batch ? withDatasetBatch(next,id,'train') : withDatasetExample(next, id, index)).graph
      if (next.nodes.some(node => node.value?.data.some(value => !Number.isFinite(value)))) throw new Error('Training diverged. Lower the learning rate and try again.')
      done += batch ? indices.length : 1
      options.progress?.(done, epochs * indices.length)
      if (batch || done % 4 === 0) await new Promise(resolve => setTimeout(resolve, 0))
    }
  }
  if (options.signal?.aborted) throw new Error('Training stopped; the previous parameters are unchanged.')
  return forwardPass({ ...next, nodes: next.nodes.map(node => node.id === id ? { ...node, params: { ...source.params } } : node) }).graph
}
