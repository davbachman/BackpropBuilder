import type { GraphGroup, GraphModel, NodeType, TensorValue } from './types'

export type Initializer = 'zeros' | 'ones' | 'xavier' | 'he' | 'uniform'
export function initializeTensor(shape: number[], mode: Initializer, seed = 42): TensorValue {
  const size = shape.reduce((a, b) => a * b, 1)
  if (shape.some(n => !Number.isInteger(n) || n < 1) || size > 65536) throw new Error('Use positive dimensions and at most 65,536 values.')
  let state = seed >>> 0
  const random = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296 }
  const fanIn = shape.length === 4 ? shape.slice(1).reduce((a,b) => a*b,1) : shape[0] ?? 1
  const fanOut = shape.length === 4 ? shape[0] * shape[1] * shape[2] : shape[1] ?? fanIn
  const limit = mode === 'xavier' ? Math.sqrt(6 / (fanIn + fanOut)) : mode === 'he' ? Math.sqrt(6 / fanIn) : .1
  return { shape: [...shape], data: Array.from({length:size}, () => mode === 'zeros' ? 0 : mode === 'ones' ? 1 : (random()*2-1)*limit) }
}

export const operationHelp: Partial<Record<NodeType, string>> = {
  matmul: 'Left [rows, inner] × right [inner, columns] → [rows, columns]. Use a Weight for the right matrix.',
  embedding: 'Port 1: table [vocabulary, width]. Port 2: integer IDs [tokens]. Output: [tokens, width].',
  transpose: 'Empty axes reverses the axes. For a matrix, 1, 0 swaps rows and columns.',
  slice: 'Choose an axis and a start-inclusive, end-exclusive range. Slice Q, K and V on axis 1 to build attention heads.',
  concat: 'Join tensors on one axis; all other dimensions must match. Join attention heads on axis 1.',
  softmax: 'Normalizes the last axis into probabilities. For classification training, wire logits directly to Cross-entropy.',
  'causal-mask': 'Masks positions above the diagonal of a square [tokens, tokens] score matrix before softmax.',
  'layer-norm': 'Ports: input [tokens, width], learned scale γ [width], learned bias β [width]. Initialize γ to ones and β to zeros.',
  reshape: 'Changes shape without changing the number of values. Use one -1 to infer a dimension (e.g. -1, 1 for numeric batches). Empty shape makes a scalar (one value only).',
  mean: 'Empty axis averages every value. Keep dimensions retains a size-1 axis for broadcasting.',
  'cross-entropy': 'Port 1: raw logits [examples, classes]. Port 2: integer target IDs [examples]. For sequences, examples are token positions.',
  conv2d: 'Ports: image [height, width, channels], filters [count, height, width, channels], biases [count]. Valid convolution, stride 1. Output: [H−KH+1, W−KW+1, count].',
  avgpool2d: 'Averages each 2 × 2 patch with stride 2, independently per channel. Output: [floor(H/2), floor(W/2), channels].',
  multiply: 'Elementwise multiplication with broadcasting. For attention scaling, multiply by an Input containing 1/√head width.',
  add: 'Adds its inputs with broadcasting. Use for biases or residual connections. Input count controls the number of ports.',
  input: 'An editable input or fixed constant. An incoming wire overrides its stored value. Connect a Dataset output for training data.',
  target: 'Connect the dataset target output here, or wire it directly to the loss.',
}

/** Recognize a user-built dense layer by connections, never by preset IDs. */
export function denseGroupDetail(graph: GraphModel, group: GraphGroup): GraphGroup['detail'] {
  const nodes = graph.nodes.filter(node => group.nodeIds.includes(node.id))
  for (const activation of nodes.filter(node => node.type === 'activation')) {
    const pre = graph.nodes.find(node => node.id === graph.edges.find(edge => edge.target === activation.id)?.source)
    if (pre?.type !== 'add') continue
    const operands = graph.edges.filter(edge => edge.target === pre.id).map(edge => graph.nodes.find(node => node.id === edge.source))
    const product = operands.find(node => node?.type === 'matmul'), bias = operands.find(node => node?.type === 'bias')
    if (!product || !bias) continue
    const input = graph.edges.find(edge => edge.target === product.id && (edge.inputSlot ?? 0) === 0)?.source
    const weight = graph.nodes.find(node => node.id === graph.edges.find(edge => edge.target === product.id && edge.inputSlot === 1)?.source)
    if (input && weight?.type === 'weight') return { ...group.detail, inputNodeId: input, weightNodeId: weight.id, biasNodeId: bias.id, preNodeId: pre.id, outputNodeId: activation.id, activation: activation.params.activation ?? 'identity' }
  }
  return group.detail
}
