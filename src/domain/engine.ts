import { resolveReshape } from './reshape'
import { arithmeticInputCount, evaluateArithmetic, parseArithmetic } from './arithmetic'
import { conv2d, conv2dBackward, avgpool2d, avgpool2dBackward } from '../learning/cnnMath'
import * as autograd from '../learning/math'
import type {
  ActivationKind,
  EvaluationResult,
  EvaluationTraceStep,
  GraphEdge,
  GraphModel,
  GraphNode,
  LossKind,
  NodeType,
  ParameterUpdate,
  TensorValue,
  UpdateResult,
  ValidationIssue,
} from './types'
import {
  datasetForNode,
  datasetOutputCountForNode,
  datasetOutputLabelForSlot,
  datasetOutputValueForSlot,
} from './datasets'
import {
  addTensorsExact,
  broadcastShapeForShapes,
  broadcastShapeForTensors,
  cloneTensor,
  elementwiseTensors,
  fillLike,
  formatShape,
  formatTensor,
  isScalarTensor,
  multiplyTensors,
  oneLike,
  reduceToShape,
  scalarFromTensor,
  scalarValue,
  scaleTensor,
  subtractTensors,
  sumTensor,
  tensorSize,
  tensorValue,
  toTensor,
  zeroLike,
} from './tensor'

export {
  DATASET_OPTIONS,
  datasetForNode,
  datasetOutputCountForNode,
  datasetOutputLabelForSlot,
  datasetOutputValueForSlot,
  isDatasetKind,
  remapDatasetOutputSlot,
} from './datasets'

const SOURCE_TYPES = new Set<NodeType>(['dataset', 'input', 'weight', 'bias', 'target'])
const OPTIONAL_PASSTHROUGH_TYPES = new Set<NodeType>(['input', 'target'])
const FLEXIBLE_INPUT_TYPES = new Set<NodeType>(['multiply', 'add', 'concat'])
const TENSOR_OPERATION_TYPES = new Set<NodeType>(['conv2d', 'avgpool2d', 'embedding', 'transpose', 'slice', 'concat', 'softmax', 'causal-mask', 'layer-norm', 'reshape', 'mean', 'cross-entropy'])

export function isLossNode(node: GraphNode): boolean { return node.type === 'loss' || node.type === 'cross-entropy' }

export const NODE_WIDTH = 176
export const MIN_NODE_HEIGHT = 150
export const FLEX_INPUT_HEIGHT_STEP = 32
export const MIN_FLEX_INPUT_COUNT = 2
export const BASE_FLEX_INPUT_CAPACITY = 3
export const MAX_FLEX_INPUT_COUNT = 64
const BCE_EPSILON = 1e-7

export const LOSS_OPTIONS: Array<{ kind: LossKind; label: string }> = [
  { kind: 'squared-error', label: 'Squared error' },
  { kind: 'mse', label: 'Mean squared error' },
  { kind: 'mae', label: 'Mean absolute error' },
  { kind: 'binary-cross-entropy', label: 'Binary cross entropy' },
]
const TENSOR_LOSS_OPTIONS = LOSS_OPTIONS.filter((option) => option.kind !== 'squared-error')

export const inputArityByType: Record<NodeType, number> = {
  dataset: 0,
  input: 1,
  weight: 0,
  bias: 0,
  multiply: 2,
  matmul: 2,
  add: 2,
  arithmetic: 2,
  activation: 1,
  target: 1,
  loss: 2,
  embedding: 2,
  transpose: 1,
  slice: 1,
  concat: 2,
  softmax: 1,
  'causal-mask': 1,
  'layer-norm': 3,
  reshape: 1,
  mean: 1,
  'cross-entropy': 2,
  conv2d: 3,
  avgpool2d: 1,
}

export function outputArityForNode(node: GraphNode): number {
  if (isLossNode(node)) return 0
  if (node.type === 'dataset') return datasetOutputCountForNode(node)
  return 1
}

export function outputLabelForNodeSlot(node: GraphNode, slot: number): string {
  if (node.type === 'dataset') return datasetOutputLabelForSlot(node, slot)
  return node.label
}

function isOptionalPassThroughNode(node: GraphNode): boolean {
  return OPTIONAL_PASSTHROUGH_TYPES.has(node.type)
}

export function isFlexibleInputNodeType(type: NodeType): boolean {
  return FLEXIBLE_INPUT_TYPES.has(type)
}

export function inputArityForNode(node: GraphNode): number {
  if (node.type === 'arithmetic') {
    try { return arithmeticInputCount(node.params.expression ?? 'x1 * x2') }
    catch { return 2 }
  }
  if (!isFlexibleInputNodeType(node.type)) return inputArityByType[node.type]
  return normalizeFlexibleInputCount(node.params.inputCount)
}

export function inputCountForNodeHeight(height: number | undefined): number {
  if (height === undefined || !Number.isFinite(height)) return BASE_FLEX_INPUT_CAPACITY
  const extraInputs = Math.floor((Math.max(height, MIN_NODE_HEIGHT) - MIN_NODE_HEIGHT) / FLEX_INPUT_HEIGHT_STEP)
  return normalizeFlexibleInputCount(BASE_FLEX_INPUT_CAPACITY + extraInputs)
}

export function heightForInputCount(inputCount: number): number {
  return MIN_NODE_HEIGHT + Math.max(0, normalizeFlexibleInputCount(inputCount) - BASE_FLEX_INPUT_CAPACITY) * FLEX_INPUT_HEIGHT_STEP
}

function normalizeFlexibleInputCount(inputCount: number | undefined): number {
  if (inputCount === undefined || !Number.isFinite(inputCount)) return MIN_FLEX_INPUT_COUNT
  return Math.min(MAX_FLEX_INPUT_COUNT, Math.max(MIN_FLEX_INPUT_COUNT, Math.round(inputCount)))
}

export function formatNumber(value: TensorValue | number | undefined, digits = 3): string {
  return formatTensor(value, digits)
}

type TensorFormatter = (value: TensorValue | number | undefined, digits?: number) => string

export function formulaForNode(node: GraphNode, graph?: GraphModel, valueFormatter: TensorFormatter = formatNumber): string {
  const inputLabels = inputLabelsForFormula(node, graph)
  const outputLabel = outputLabelForFormula(node, graph)
  switch (node.type) {
    case 'dataset': {
      const dataset = datasetForNode(node)
      return `${node.label} = ${dataset.label}`
    }
    case 'input':
      return `${node.label} = ${inputLabels[0] ?? valueFormatter(node.params.value)}`
    case 'weight':
      return `${node.label} = ${valueFormatter(node.params.value)}`
    case 'bias':
      return `${node.label} = ${valueFormatter(node.params.value)}`
    case 'target':
      return `${node.label} = ${inputLabels[0] ?? valueFormatter(node.params.value)}`
    case 'matmul':
      return `${outputLabel} = ${inputLabels.join(' @ ')}`
    case 'multiply':
      return `${outputLabel} = ${inputLabels.join(' * ')}`
    case 'add':
      return `${outputLabel} = ${inputLabels.join(' + ')}`
    case 'arithmetic':
      return `${outputLabel} = ${(node.params.expression ?? 'x1 * x2').replace(/x([1-9]\d*)\b/g, (_, index: string) => inputLabels[Number(index) - 1] ?? `x${index}`)}`
    case 'activation':
      return `${outputLabel} = ${node.params.activation ?? 'identity'}(${inputLabels[0]})`
    case 'conv2d': return `${outputLabel}[r,c,o] = Σ(kernel[o] · ${inputLabels[0]}[window r,c]) + bias[o]`
    case 'avgpool2d': return `${outputLabel}[r,c,k] = mean(${inputLabels[0]}[2×2 window r,c,k])`
    case 'embedding': return `${outputLabel} = ${inputLabels[0]}[${inputLabels[1]}]`
    case 'transpose': return `${outputLabel} = transpose(${inputLabels[0]})`
    case 'slice': return `${outputLabel} = slice(${inputLabels[0]}, axis=${node.params.axis ?? 0}, ${node.params.start ?? 0}:${node.params.end ?? '?'})`
    case 'concat': return `${outputLabel} = concat(${inputLabels.join(', ')}, axis=${node.params.axis ?? 1})`
    case 'softmax': return `${outputLabel} = softmax(${inputLabels[0]})`
    case 'causal-mask': return `${outputLabel}[i,j] = ${inputLabels[0]}[i,j] if j ≤ i; otherwise −∞`
    case 'layer-norm': return `${outputLabel} = γ · (${inputLabels[0]} − mean) / √(variance + ε) + β`
    case 'reshape': return `${outputLabel} = reshape(${inputLabels[0]}, [${node.params.shape ?? []}])`
    case 'mean': return `${outputLabel} = mean(${inputLabels[0]}${node.params.axis === undefined ? '' : `, axis=${node.params.axis}`})`
    case 'cross-entropy': return `L = mean(−log softmax(${inputLabels[0]})[${inputLabels[1]}])`
    case 'loss':
      return lossFormula(lossKindForNode(node, graph), inputLabels, Boolean(graph && hasNonScalarIncomingValue(node, graph)))
  }
}

export function lossOptionsForNode(node: GraphNode, graph?: GraphModel): Array<{ kind: LossKind; label: string }> {
  if (node.type === 'loss' && graph && hasNonScalarIncomingValue(node, graph)) return TENSOR_LOSS_OPTIONS
  return LOSS_OPTIONS
}

export function lossKindForNode(node: GraphNode, graph?: GraphModel): LossKind {
  return lossKindForOptions(node, lossOptionsForNode(node, graph))
}

function lossKindForInputs(node: GraphNode, inputs: TensorValue[]): LossKind {
  return lossKindForOptions(node, inputs.some((input) => !isScalarTensor(input)) ? TENSOR_LOSS_OPTIONS : LOSS_OPTIONS)
}

function lossKindForOptions(node: GraphNode, options: Array<{ kind: LossKind; label: string }>): LossKind {
  const selected = node.params.loss
  if (selected && options.some((option) => option.kind === selected)) return selected
  return options[0]?.kind ?? 'squared-error'
}

function lossLabel(kind: LossKind): string {
  return LOSS_OPTIONS.find((option) => option.kind === kind)?.label ?? 'Squared error'
}

function lossFormula(kind: LossKind, inputLabels: string[], isTensor: boolean): string {
  const prediction = inputLabels[0] ?? 'prediction'
  const target = inputLabels[1] ?? 'target'

  if (!isTensor) {
    if (kind === 'mse') return `L = (${prediction} - ${target})^2`
    if (kind === 'mae') return `L = |${prediction} - ${target}|`
    if (kind === 'binary-cross-entropy') {
      return `L = -(${target} * log(${prediction}) + (1 - ${target}) * log(1 - ${prediction}))`
    }
    return `L = 0.5 * (${prediction} - ${target})^2`
  }

  const predictionEntry = `${prediction}_i`
  const targetEntry = `${target}_i`
  if (kind === 'mse') return `L = (1/n) * Σ_i (${predictionEntry} - ${targetEntry})^2`
  if (kind === 'mae') return `L = (1/n) * Σ_i |${predictionEntry} - ${targetEntry}|`
  if (kind === 'binary-cross-entropy') {
    return `L = -(1/n) * Σ_i [${targetEntry} * log(${predictionEntry}) + (1 - ${targetEntry}) * log(1 - ${predictionEntry})]`
  }
  return `L = 0.5 * Σ_i (${predictionEntry} - ${targetEntry})^2`
}

function hasNonScalarIncomingValue(node: GraphNode, graph: GraphModel): boolean {
  const inferredShapes = inferredOutputShapes(graph)
  return incomingEdges(graph, node.id).some((edge) => {
    const source = graph.nodes.find((candidate) => candidate.id === edge.source)
    if (!source) return false
    const inferredShape = outputShapeForSourceEdge(source, edge, inferredShapes)
    if (inferredShape) return inferredShape.length > 0
    return !isScalarTensor(sourceValueForEdgeSource(source, edge))
  })
}

function inferredOutputShapes(graph: GraphModel): Map<string, number[]> {
  const shapeByNode = new Map<string, number[]>()

  for (const nodeId of topologicalSort(graph)) {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) continue

    const incoming = incomingEdges(graph, node.id)
    const existingValueShape = node.value !== undefined ? toTensor(node.value).shape : undefined
    if (SOURCE_TYPES.has(node.type)) {
      shapeByNode.set(node.id, sourceShapeForNode(graph, node, incoming, shapeByNode))
      continue
    }

    const expected = inputArityForNode(node)
    if (incoming.length !== expected) {
      if (existingValueShape) shapeByNode.set(node.id, existingValueShape)
      continue
    }

    const inputShapes = incoming.map((edge) => {
      const source = graph.nodes.find((candidate) => candidate.id === edge.source)
      return source ? outputShapeForSourceEdge(source, edge, shapeByNode) : undefined
    })
    if (inputShapes.some((shape) => !shape)) {
      if (existingValueShape) shapeByNode.set(node.id, existingValueShape)
      continue
    }

    const outputShape = outputShapeForNode(node, inputShapes as number[][])
    if (outputShape) {
      shapeByNode.set(node.id, outputShape)
    } else if (existingValueShape) {
      shapeByNode.set(node.id, existingValueShape)
    }
  }

  return shapeByNode
}

function inputLabelsForFormula(node: GraphNode, graph?: GraphModel): string[] {
  const fallback = fallbackInputLabels(node)
  if (!graph) return fallback

  const labels = [...fallback]
  for (const edge of incomingEdges(graph, node.id)) {
    const slot = edge.inputSlot ?? 0
    labels[slot] =
      outputLabelForFormula(graph.nodes.find((candidate) => candidate.id === edge.source), graph, edge.sourceSlot ?? 0) ??
      fallback[slot] ??
      '?'
  }

  return labels
}

function fallbackInputLabels(node: GraphNode): string[] {
  if (SOURCE_TYPES.has(node.type)) return []
  if (node.type === 'activation') return ['u']
  if (isLossNode(node)) return ['prediction', 'target']
  if (node.type === 'embedding') return ['table', 'token_ids']
  if (node.type === 'conv2d') return ['image', 'kernel', 'bias']
  if (node.type === 'layer-norm') return ['input', 'γ', 'β']
  return Array.from({ length: inputArityForNode(node) }, (_, index) => inputLabelForIndex(index))
}

function inputLabelForIndex(index: number): string {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz'
  return alphabet[index] ?? `input${index + 1}`
}

function outputLabelForFormula(node: GraphNode | undefined, graph?: GraphModel, sourceSlot = 0): string | undefined {
  if (!node) return undefined
  if (node.type === 'dataset') return datasetOutputLabelForSlot(node, sourceSlot)
  if (SOURCE_TYPES.has(node.type)) return node.label
  if (isLossNode(node)) return 'L'
  if (!graph) return 'z'
  return computedOutputLabels(graph).get(node.id) ?? 'z'
}

function computedOutputLabels(graph: GraphModel): Map<string, string> {
  const order = topologicalSort(graph)
  const orderedIds = new Set(order)
  const fallbackIds = graph.nodes
    .filter((node) => !orderedIds.has(node.id))
    .sort(
      (first, second) =>
        first.position.x - second.position.x ||
        first.position.y - second.position.y ||
        first.id.localeCompare(second.id),
    )
    .map((node) => node.id)

  let index = 1
  const labels = new Map<string, string>()
  for (const nodeId of [...order, ...fallbackIds]) {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node || SOURCE_TYPES.has(node.type) || isLossNode(node)) continue
    labels.set(node.id, `z${index}`)
    index += 1
  }
  return labels
}

export function cloneGraph(graph: GraphModel): GraphModel {
  return {
    learningRate: graph.learningRate,
    view: graph.view ? {
      ...graph.view,
      inspectedNeuron: graph.view.inspectedNeuron ? { ...graph.view.inspectedNeuron } : undefined,
      layoutOffsets: graph.view.layoutOffsets ? Object.fromEntries(Object.entries(graph.view.layoutOffsets).map(([id, offset]) => [id, { ...offset }])) : undefined,
      manualNodePlacements: graph.view.manualNodePlacements ? Object.fromEntries(Object.entries(graph.view.manualNodePlacements).map(([id, placement]) => [id, { ...placement, offset: { ...placement.offset } }])) : undefined,
      layoutEdges: graph.view.layoutEdges?.map(edge => ({ ...edge })),
      expandedGroupIds: [...graph.view.expandedGroupIds],
      viewport: graph.view.viewport ? { ...graph.view.viewport } : undefined,
    } : undefined,
    groups: graph.groups?.map((group) => ({
      ...group,
      nodeIds: [...group.nodeIds],
      position: { ...group.position },
      dimensions: { ...group.dimensions },
      detail: group.detail ? structuredClone(group.detail) : undefined,
    })),
    nodes: graph.nodes.map((node) => ({
      ...node,
      params: {
        ...node.params,
        datasetValues: node.params.datasetValues?.map(cloneTensor),
        axes: node.params.axes ? [...node.params.axes] : undefined,
        shape: node.params.shape ? [...node.params.shape] : undefined,
        value: node.params.value !== undefined ? toTensor(node.params.value) : undefined,
      },
      position: { ...node.position },
      cache: node.cache
        ? {
            ...node.cache,
            inputValues: node.cache.inputValues.map(cloneTensor),
            outputValue: cloneTensor(node.cache.outputValue),
            localDerivatives: node.cache.localDerivatives.map(cloneTensor),
            error: node.cache.error ? cloneTensor(node.cache.error) : undefined,
          }
        : undefined,
      dimensions: node.dimensions ? { ...node.dimensions } : undefined,
      value: node.value !== undefined ? cloneTensor(node.value) : undefined,
      grad: node.grad !== undefined ? cloneTensor(node.grad) : undefined,
      localDerivative: node.localDerivative !== undefined ? cloneTensor(node.localDerivative) : undefined,
    })),
    edges: graph.edges.map((edge) => ({
      ...edge,
      value: edge.value ? cloneTensor(edge.value) : undefined,
      grad: edge.grad ? cloneTensor(edge.grad) : undefined,
    })),
  }
}

export function parameterValues(graph: GraphModel): Record<string, TensorValue> {
  return Object.fromEntries(
    graph.nodes
      .filter((node) => node.type === 'weight' || node.type === 'bias')
      .map((node) => [node.label, toTensor(node.params.value)]),
  )
}

export function validateGraph(graph: GraphModel, options: { requireLoss?: boolean } = {}): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const nodeIds = new Set(graph.nodes.map((node) => node.id))
  const lossNodes = graph.nodes.filter(isLossNode)

  if (lossNodes.length === 0 && options.requireLoss) {
    issues.push({
      code: 'missing-loss',
      message: 'Add exactly one loss node so the app knows what to optimize.',
    })
  }

  if (lossNodes.length > 1) {
    issues.push({
      code: 'multiple-losses',
      message: 'Use one loss node at a time in this teaching graph.',
    })
  }

  for (const edge of graph.edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      issues.push({
        code: 'unknown-node',
        edgeId: edge.id,
        message: 'An edge points to a node that no longer exists.',
      })
      continue
    }

    const source = graph.nodes.find((node) => node.id === edge.source)
    const outputSlot = edge.sourceSlot ?? 0
    if (source && (outputSlot < 0 || outputSlot >= outputArityForNode(source))) {
      issues.push({
        code: 'invalid-arity',
        nodeId: source.id,
        edgeId: edge.id,
        message: `${source.label} has an edge connected from unavailable output ${outputSlot + 1}.`,
      })
    }
  }

  for (const node of graph.nodes) {
    if (node.type === 'arithmetic') {
      try { parseArithmetic(node.params.expression ?? 'x1 * x2') }
      catch (error) { issues.push({ code: 'invalid-value', nodeId: node.id, message: `${node.label}: ${error instanceof Error ? error.message : 'Invalid expression.'}` }) }
    }
    const incoming = incomingEdges(graph, node.id)
    const expected = inputArityForNode(node)
    const hasValidInputCount = isOptionalPassThroughNode(node)
      ? incoming.length <= expected
      : incoming.length === expected
    if (!hasValidInputCount) {
      issues.push({
        code: 'invalid-arity',
        nodeId: node.id,
        message: `${node.label} expects ${expected} input${expected === 1 ? '' : 's'} but has ${incoming.length}.`,
      })
    }
    if (expected > 0 && (!isOptionalPassThroughNode(node) || incoming.length > 0)) {
      for (let slot = 0; slot < expected; slot += 1) {
        if (!incoming.some((edge) => (edge.inputSlot ?? 0) === slot)) {
          issues.push({
            code: 'missing-input',
            nodeId: node.id,
            message: `${node.label} is missing input ${slot + 1}.`,
          })
        }
      }
      for (const edge of incoming) {
        const slot = edge.inputSlot ?? 0
        if (slot < 0 || slot >= expected) {
          issues.push({
            code: 'invalid-arity',
            nodeId: node.id,
            edgeId: edge.id,
            message: `${node.label} has an edge connected to unavailable input ${slot + 1}.`,
          })
        }
      }
    }
  }

  const sorted = topologicalSort(graph)
  if (sorted.length !== graph.nodes.length) {
    issues.push({
      code: 'cycle',
      message: 'The graph contains a cycle. Backprop Builder supports directed acyclic graphs only.',
    })
  }

  if (lossNodes.length === 1) {
    const connectedToLoss = ancestorsOf(graph, lossNodes[0].id)
    const disconnected = graph.nodes.filter((node) => !connectedToLoss.has(node.id))
    for (const node of disconnected) {
      issues.push({
        code: 'disconnected',
        nodeId: node.id,
        message: `${node.label} is not connected to the loss node.`,
      })
    }
  }

  issues.push(...validateTensorShapes(graph))
  issues.push(...validateDiscreteInputs(graph))

  return issues
}

function validateDiscreteInputs(graph: GraphModel): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const shapes = inferredOutputShapes(graph)
  for (const node of graph.nodes) {
    if (node.type !== 'embedding' && node.type !== 'cross-entropy') continue
    const incoming = incomingEdges(graph, node.id)
    const ids = graph.nodes.find(candidate => candidate.id === incoming[1]?.source)
    if (!ids || !['input', 'target', 'weight', 'bias'].includes(ids.type) || incomingEdges(graph, ids.id).length) continue
    const firstShape = shapes.get(incoming[0]?.source)
    const limit = firstShape?.[node.type === 'embedding' ? 0 : 1]
    if (toTensor(ids.params.value).data.some(id => !Number.isInteger(id) || id < 0 || (limit !== undefined && id >= limit))) {
      issues.push({ code: 'invalid-value', nodeId: node.id, message: `${node.label} needs integer token IDs ${limit === undefined ? 'greater than or equal to 0' : `from 0 to ${limit - 1}`}. Edit ${ids.label} to use valid IDs.` })
    }
  }
  return issues
}

function validateTensorShapes(graph: GraphModel): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const shapeByNode = new Map<string, number[]>()

  for (const nodeId of topologicalSort(graph)) {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) continue

    const incoming = incomingEdges(graph, node.id)
    if (SOURCE_TYPES.has(node.type)) {
      shapeByNode.set(node.id, sourceShapeForNode(graph, node, incoming, shapeByNode))
      continue
    }

    const expected = inputArityForNode(node)
    if (incoming.length !== expected) continue
    if (incoming.some((edge) => (edge.inputSlot ?? 0) < 0 || (edge.inputSlot ?? 0) >= expected)) continue

    const inputShapes = incoming.map((edge) => {
      const source = graph.nodes.find((candidate) => candidate.id === edge.source)
      return source ? outputShapeForSourceEdge(source, edge, shapeByNode) : undefined
    })
    if (inputShapes.some((shape) => !shape)) continue
    const outputShape = outputShapeForNode(node, inputShapes as number[][])
    if (outputShape) {
      shapeByNode.set(node.id, outputShape)
      continue
    }

    issues.push({
      code: 'shape-mismatch',
      nodeId: node.id,
      message: `${node.label} received incompatible tensor shapes ${inputShapes.map((shape) => formatShape(shape ?? [])).join(', ')}. Use matching shapes or scalars.`,
    })
  }

  return issues
}

function outputShapeForNode(node: GraphNode, inputShapes: number[][]): number[] | undefined {
  if (node.type === 'matmul') {
    const [a, b] = inputShapes
    return a.length === 2 && b.length === 2 && a[1] === b[0] ? [a[0], b[1]] : undefined
  }
  const [first, second, third] = inputShapes
  const axis = node.params.axis ?? (node.type === 'concat' ? 1 : 0)
  if (node.type === 'conv2d') return first.length === 3 && second.length === 4 && third.length === 1 && first[2] === second[3] && second[0] === third[0] && first[0] >= second[1] && first[1] >= second[2] ? [first[0] - second[1] + 1, first[1] - second[2] + 1, second[0]] : undefined
  if (node.type === 'avgpool2d') return first.length === 3 && first[0] >= 2 && first[1] >= 2 ? [Math.floor(first[0] / 2), Math.floor(first[1] / 2), first[2]] : undefined
  if (node.type === 'embedding') return first.length === 2 && second.length === 1 ? [second[0], first[1]] : undefined
  if (node.type === 'transpose') {
    const axes = node.params.axes ?? first.map((_, index) => first.length - index - 1)
    return axes.length === first.length && new Set(axes).size === axes.length && axes.every((a) => Number.isInteger(a) && a >= 0 && a < first.length) ? axes.map((a) => first[a]) : undefined
  }
  if (node.type === 'slice') {
    const start = node.params.start ?? 0, end = node.params.end ?? first[axis]
    if (![axis, start, end].every(Number.isInteger) || axis < 0 || axis >= first.length || start < 0 || end > first[axis] || end <= start) return undefined
    return first.map((dimension, index) => index === axis ? end - start : dimension)
  }
  if (node.type === 'concat') {
    if (!Number.isInteger(axis) || axis < 0 || axis >= first.length || inputShapes.some((shape) => shape.length !== first.length || shape.some((d, index) => index !== axis && d !== first[index]))) return undefined
    return first.map((d, index) => index === axis ? inputShapes.reduce((sum, shape) => sum + shape[axis], 0) : d)
  }
  if (node.type === 'softmax') return first.length && first.at(-1)! > 0 ? [...first] : undefined
  if (node.type === 'causal-mask') return first.length === 2 && first[0] === first[1] ? [...first] : undefined
  if (node.type === 'layer-norm') return first.length > 0 && second.length === 1 && third.length === 1 && second[0] === first.at(-1) && third[0] === first.at(-1) && (node.params.epsilon === undefined || (Number.isFinite(node.params.epsilon) && node.params.epsilon > 0)) ? [...first] : undefined
  if (node.type === 'reshape') {
    const shape = node.params.shape ?? first
    try { return resolveReshape(shape,tensorSize(first)) } catch { return undefined }
  }
  if (node.type === 'mean') {
    if (node.params.axis === undefined) return node.params.keepDims ? first.map(() => 1) : []
    if (!Number.isInteger(axis) || axis < 0 || axis >= first.length) return undefined
    return first.flatMap((d, index) => index === axis ? (node.params.keepDims ? [1] : []) : [d])
  }
  if (node.type === 'cross-entropy') return first.length === 2 && second.length === 1 && second[0] === first[0] ? [] : undefined
  if (node.type === 'activation') return [...inputShapes[0]]
  if (node.type === 'add' || node.type === 'multiply' || node.type === 'arithmetic') return broadcastShapeForShapes(inputShapes)
  if (node.type === 'loss') return broadcastShapeForShapes(inputShapes) ? [] : undefined
  return []
}

export function topologicalSort(graph: GraphModel): string[] {
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map<string, string[]>()
  for (const edge of graph.edges) {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) continue
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target])
  }

  const queue = graph.nodes
    .filter((node) => indegree.get(node.id) === 0)
    .sort((a, b) => a.position.x - b.position.x || a.position.y - b.position.y)
    .map((node) => node.id)
  const order: string[] = []

  while (queue.length > 0) {
    const id = queue.shift()!
    order.push(id)
    for (const target of outgoing.get(id) ?? []) {
      const next = (indegree.get(target) ?? 0) - 1
      indegree.set(target, next)
      if (next === 0) queue.push(target)
    }
  }

  return order
}

export function forwardPass(graph: GraphModel): EvaluationResult {
  assertValid(graph)
  const next = cloneGraph(graph)
  const order = topologicalSort(next)
  const steps: EvaluationTraceStep[] = []

  for (const nodeId of order) {
    const node = mustNode(next, nodeId)
    const incoming = incomingEdges(next, nodeId)
    const inputValues = incoming.map((edge) => valueForSourceEdge(next, edge))
    const computed = computeForward(node, inputValues)

    node.value = computed.value
    node.grad = zeroLike(computed.value)
    node.localDerivative = computed.localDerivative
    node.cache = {
      inputValues: inputValues.map(cloneTensor),
      outputValue: cloneTensor(computed.value),
      localDerivatives: computed.localDerivatives.map(cloneTensor),
      error: computed.error ? cloneTensor(computed.error) : undefined,
    }

    for (const edge of next.edges.filter((candidate) => candidate.source === node.id)) {
      const outputValue = sourceValueForEdgeSource(node, edge)
      edge.value = cloneTensor(outputValue)
      edge.grad = zeroLike(outputValue)
    }

    if (!SOURCE_TYPES.has(node.type) && inputArityForNode(node) > 0) {
      steps.push(forwardStep(node, incoming, inputValues, next))
    }
  }

  const lossValue = next.nodes.find(isLossNode)?.value
  const loss = lossValue ? scalarFromTensor(lossValue) : undefined
  return { graph: next, steps, loss }
}

export function backwardPass(graph: GraphModel): EvaluationResult {
  assertValid(graph, true)
  const next = cloneGraph(graph)
  const order = topologicalSort(next).reverse()
  const steps: EvaluationTraceStep[] = []
  for (const node of next.nodes) {
    node.grad = zeroLike(sourceDefaultValue(node))
  }
  for (const edge of next.edges) {
    edge.grad = edge.value ? zeroLike(edge.value) : undefined
  }

  const lossNode = next.nodes.find(isLossNode)
  if (!lossNode) throw new Error('Cannot run backward pass without a loss node.')
  lossNode.grad = scalarValue(1)

  for (const nodeId of order) {
    const node = mustNode(next, nodeId)
    const incoming = incomingEdges(next, nodeId)
    const downstreamGrad = toTensor(node.grad)
    const contributions = computeBackward(node, incoming, next, downstreamGrad)

    for (const contribution of contributions) {
      const source = mustNode(next, contribution.sourceId)
      if (source.type !== 'dataset') {
        source.grad = addTensorsExact(source.grad ?? zeroLike(contribution.gradient), contribution.gradient)
      }
      const edge = next.edges.find((candidate) => candidate.id === contribution.edgeId)
      if (edge) edge.grad = cloneTensor(contribution.gradient)
    }

    if (!SOURCE_TYPES.has(node.type) || contributions.length > 0) {
      steps.push(backwardStep(node, incoming, downstreamGrad, contributions, next))
    }
  }

  return { graph: next, steps, loss: lossNode.value ? scalarFromTensor(lossNode.value) : undefined }
}

export function updateParameters(graph: GraphModel, learningRate = graph.learningRate): UpdateResult {
  const next = cloneGraph(graph)
  const updates: ParameterUpdate[] = []

  for (const node of next.nodes) {
    if (node.type !== 'weight' && node.type !== 'bias') continue
    const oldValue = toTensor(node.params.value)
    const gradient = node.grad ?? zeroLike(oldValue)
    const newValue = addTensorsExact(oldValue, scaleTensor(gradient, -learningRate))
    node.params.value = newValue
    node.value = newValue
    updates.push({
      nodeId: node.id,
      label: node.label,
      oldValue: cloneTensor(oldValue),
      gradient: cloneTensor(gradient),
      learningRate,
      newValue: cloneTensor(newValue),
    })
  }

  for (const node of next.nodes) {
    node.grad = zeroLike(sourceDefaultValue(node))
  }
  for (const edge of next.edges) {
    edge.grad = edge.value ? zeroLike(edge.value) : undefined
  }

  const steps = updates.map((update) => updateStep(update))
  return { graph: next, steps, updates }
}

export function runTrainingStep(graph: GraphModel, learningRate = graph.learningRate): EvaluationResult {
  const forward = forwardPass(graph)
  const backward = backwardPass(forward.graph)
  const updated = updateParameters(backward.graph, learningRate)
  const afterUpdate = forwardPass(updated.graph)
  return {
    graph: afterUpdate.graph,
    steps: [...forward.steps, ...backward.steps, ...updated.steps],
    loss: afterUpdate.loss,
  }
}

function computeForward(
  node: GraphNode,
  inputs: TensorValue[],
): { value: TensorValue; localDerivative?: TensorValue; localDerivatives: TensorValue[]; error?: TensorValue } {
  if (SOURCE_TYPES.has(node.type)) {
    return { value: sourceForwardValue(node, inputs), localDerivatives: [] }
  }

  if (TENSOR_OPERATION_TYPES.has(node.type)) {
    const { output } = tensorOperation(node, inputs)
    const value = output.toValue()
    if (output.excluded) value.excluded = [...output.excluded]
    return { value, localDerivatives: [] }
  }

  if (node.type === 'matmul') {
    return { value: matrixProduct(inputs[0], inputs[1]), localDerivatives: [] }
  }

  if (node.type === 'multiply') {
    const value = elementwiseTensors(inputs, (entries) => entries.reduce((product, input) => product * input, 1))
    return {
      value,
      localDerivatives: inputs.map((_, index) => productExceptIndex(inputs, index)),
    }
  }

  if (node.type === 'add') {
    const value = elementwiseTensors(inputs, (entries) => entries.reduce((sum, input) => sum + input, 0))
    return {
      value,
      localDerivative: oneLike(value),
      localDerivatives: inputs.map(() => oneLike(value)),
    }
  }

  if (node.type === 'arithmetic') {
    const result = evaluateArithmetic(node.params.expression ?? 'x1 * x2', inputs)
    const derivatives = evaluateArithmetic(node.params.expression ?? 'x1 * x2', inputs, oneLike(result.value)).gradients
    return { value: result.value, localDerivative: derivatives[0], localDerivatives: derivatives }
  }

  if (node.type === 'activation') {
    const activation = node.params.activation ?? 'identity'
    const value = mapActivation(activation, inputs[0])
    const derivative = activationDerivativeTensor(activation, inputs[0], value)
    return { value, localDerivative: derivative, localDerivatives: [derivative] }
  }

  return computeLossForward(lossKindForInputs(node, inputs), inputs[0], inputs[1])
}

function computeLossForward(
  kind: LossKind,
  prediction: TensorValue,
  target: TensorValue,
): { value: TensorValue; localDerivative?: TensorValue; localDerivatives: TensorValue[]; error?: TensorValue } {
  const error = subtractTensors(prediction, target)
  const predictionDerivative = lossGradient(kind, prediction, target)

  return {
    value: scalarValue(lossValue(kind, prediction, target, error)),
    localDerivative: predictionDerivative,
    localDerivatives: [predictionDerivative, zeroLike(predictionDerivative)],
    error,
  }
}

function lossValue(kind: LossKind, prediction: TensorValue, target: TensorValue, error: TensorValue): number {
  if (kind === 'mse') {
    return sumTensor(tensorValue(error.shape, error.data.map((entry) => entry ** 2))) / meanDenominator(error)
  }
  if (kind === 'mae') {
    return sumTensor(tensorValue(error.shape, error.data.map((entry) => Math.abs(entry)))) / meanDenominator(error)
  }
  if (kind === 'binary-cross-entropy') {
    const losses = elementwiseTensors([prediction, target], ([rawPrediction, targetEntry]) => {
      const clippedPrediction = clampProbability(rawPrediction)
      return -(
        targetEntry * Math.log(clippedPrediction) +
        (1 - targetEntry) * Math.log(1 - clippedPrediction)
      )
    })
    return sumTensor(losses) / meanDenominator(losses)
  }
  return 0.5 * error.data.reduce((sum, entry) => sum + entry ** 2, 0)
}

function lossGradient(kind: LossKind, prediction: TensorValue, target: TensorValue): TensorValue {
  const error = subtractTensors(prediction, target)
  if (kind === 'mse') return scaleTensor(error, 2 / meanDenominator(error))
  if (kind === 'mae') {
    const denominator = meanDenominator(error)
    return tensorValue(
      error.shape,
      error.data.map((entry) => {
        if (entry === 0) return 0
        return (entry > 0 ? 1 : -1) / denominator
      }),
    )
  }
  if (kind === 'binary-cross-entropy') {
    const entries = elementwiseTensors([prediction, target], ([rawPrediction, targetEntry]) => {
      const clippedPrediction = clampProbability(rawPrediction)
      return -targetEntry / clippedPrediction + (1 - targetEntry) / (1 - clippedPrediction)
    })
    return scaleTensor(entries, 1 / meanDenominator(entries))
  }
  return error
}

function meanDenominator(value: TensorValue): number {
  return Math.max(1, tensorSize(value.shape))
}

function clampProbability(value: number): number {
  return Math.min(1 - BCE_EPSILON, Math.max(BCE_EPSILON, value))
}

function computeBackward(
  node: GraphNode,
  incoming: GraphEdge[],
  graph: GraphModel,
  downstreamGrad: TensorValue,
): Array<{ edgeId: string; sourceId: string; gradient: TensorValue }> {
  if (SOURCE_TYPES.has(node.type)) {
    return isOptionalPassThroughNode(node) && incoming.length === 1
      ? [{ edgeId: incoming[0].id, sourceId: incoming[0].source, gradient: cloneTensor(downstreamGrad) }]
      : []
  }
  const values = incoming.map((edge) => valueForSourceEdge(graph, edge))

  if (TENSOR_OPERATION_TYPES.has(node.type)) {
    const { output, operands } = tensorOperation(node, values, true)
    output.backward(downstreamGrad.data)
    return incoming.map((edge, index) => ({ edgeId: edge.id, sourceId: edge.source, gradient: tensorValue(operands[index].shape, operands[index].grad) }))
  }

  if (node.type === 'matmul') {
    const gradients = [matrixProduct(downstreamGrad, transposeMatrix(values[1])), matrixProduct(transposeMatrix(values[0]), downstreamGrad)]
    return incoming.map((edge, index) => ({ edgeId: edge.id, sourceId: edge.source, gradient: gradients[index] }))
  }

  if (node.type === 'multiply') {
    return incoming.map((edge, index) => ({
      edgeId: edge.id,
      sourceId: edge.source,
      gradient: reduceToShape(multiplyTensors(downstreamGrad, productExceptIndex(values, index)), values[index].shape),
    }))
  }

  if (node.type === 'add') {
    return incoming.map((edge, index) => ({
      edgeId: edge.id,
      sourceId: edge.source,
      gradient: reduceToShape(downstreamGrad, values[index].shape),
    }))
  }

  if (node.type === 'arithmetic') {
    const gradients = evaluateArithmetic(node.params.expression ?? 'x1 * x2', values, downstreamGrad).gradients
    return incoming.map((edge, index) => ({ edgeId: edge.id, sourceId: edge.source, gradient: gradients[index] }))
  }

  if (node.type === 'activation') {
    const derivative = node.localDerivative ?? node.cache?.localDerivatives[0] ?? oneLike(values[0])
    return [
      {
        edgeId: incoming[0].id,
        sourceId: incoming[0].source,
        gradient: reduceToShape(multiplyTensors(downstreamGrad, derivative), values[0].shape),
      },
    ]
  }

  const prediction = values[0]
  const target = values[1]
  const predictionDerivative = lossGradient(lossKindForInputs(node, values), prediction, target)
  return [
    {
      edgeId: incoming[0].id,
      sourceId: incoming[0].source,
      gradient: reduceToShape(multiplyTensors(downstreamGrad, predictionDerivative), prediction.shape),
    },
    { edgeId: incoming[1].id, sourceId: incoming[1].source, gradient: zeroLike(target) },
  ]
}

/** Explicit rank-two matrix product; multiply remains elementwise broadcasting. */
export function matrixProduct(a: TensorValue, b: TensorValue): TensorValue {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) {
    throw new Error('Matrix multiplication needs shapes [rows, inner] and [inner, columns].')
  }
  const [rows, inner] = a.shape
  const columns = b.shape[1]
  const data = Array.from({ length: rows * columns }, (_, index) => {
    const row = Math.floor(index / columns)
    const column = index % columns
    let sum = 0
    for (let k = 0; k < inner; k += 1) sum += a.data[row * inner + k] * b.data[k * columns + column]
    return sum
  })
  return tensorValue([rows, columns], data)
}

function transposeMatrix(value: TensorValue): TensorValue {
  const [rows, columns] = value.shape
  return tensorValue([columns, rows], Array.from({ length: rows * columns }, (_, i) => value.data[(i % rows) * columns + Math.floor(i / rows)]))
}

function productExceptIndex(values: TensorValue[], excludedIndex: number): TensorValue {
  const outputShape = broadcastShapeForTensors(values)
  if (!outputShape) {
    throw new Error(`Cannot multiply incompatible tensor shapes ${values.map((value) => formatShape(value.shape)).join(', ')}.`)
  }
  const includedValues = values.filter((_, index) => index !== excludedIndex)
  if (includedValues.length === 0) {
    return fillLike({ shape: outputShape, data: Array.from({ length: outputShape.reduce((size, dimension) => size * dimension, 1) }, () => 1) }, 1)
  }
  return elementwiseTensors(includedValues, (entries) => entries.reduce((product, input) => product * input, 1))
}

function activate(kind: ActivationKind, input: number): number {
  switch (kind) {
    case 'identity':
      return input
    case 'relu':
      return Math.max(0, input)
    case 'sigmoid':
      return 1 / (1 + Math.exp(-input))
    case 'tanh':
      return Math.tanh(input)
  }
}

function mapActivation(kind: ActivationKind, input: TensorValue): TensorValue {
  return {
    shape: [...input.shape],
    data: input.data.map((entry) => activate(kind, entry)),
  }
}

function activationDerivative(kind: ActivationKind, input: number, output: number): number {
  switch (kind) {
    case 'identity':
      return 1
    case 'relu':
      return input > 0 ? 1 : 0
    case 'sigmoid':
      return output * (1 - output)
    case 'tanh':
      return 1 - output ** 2
  }
}

function activationDerivativeTensor(kind: ActivationKind, input: TensorValue, output: TensorValue): TensorValue {
  return {
    shape: [...input.shape],
    data: input.data.map((entry, index) => activationDerivative(kind, entry, output.data[index] ?? 0)),
  }
}

function incomingEdges(graph: GraphModel, nodeId: string): GraphEdge[] {
  return graph.edges
    .filter((edge) => edge.target === nodeId)
    .sort((a, b) => (a.inputSlot ?? 0) - (b.inputSlot ?? 0) || a.id.localeCompare(b.id))
}

function ancestorsOf(graph: GraphModel, nodeId: string): Set<string> {
  const ancestors = new Set<string>([nodeId])
  const stack = [nodeId]
  while (stack.length > 0) {
    const current = stack.pop()!
    for (const edge of graph.edges.filter((candidate) => candidate.target === current)) {
      if (!ancestors.has(edge.source)) {
        ancestors.add(edge.source)
        stack.push(edge.source)
      }
    }
  }
  return ancestors
}

function mustNode(graph: GraphModel, nodeId: string): GraphNode {
  const node = graph.nodes.find((candidate) => candidate.id === nodeId)
  if (!node) throw new Error(`Node ${nodeId} was not found.`)
  return node
}

function valueForSourceEdge(graph: GraphModel, edge: GraphEdge): TensorValue {
  return sourceValueForEdgeSource(mustNode(graph, edge.source), edge)
}

function sourceValueForEdgeSource(source: GraphNode, edge: Pick<GraphEdge, 'sourceSlot'>): TensorValue {
  if (source.type === 'dataset') return datasetOutputValueForSlot(source, edge.sourceSlot ?? 0)
  return sourceDefaultValue(source)
}

function sourceDefaultValue(node: GraphNode): TensorValue {
  if (node.type === 'dataset') return datasetOutputValueForSlot(node, 0)
  return toTensor(node.value ?? node.params.value)
}

function sourceForwardValue(node: GraphNode, inputs: TensorValue[]): TensorValue {
  if (isOptionalPassThroughNode(node) && inputs[0]) return cloneTensor(inputs[0])
  if (node.type === 'dataset') return datasetOutputValueForSlot(node, 0)
  return toTensor(node.params.value)
}

function sourceShapeForNode(
  graph: GraphModel,
  node: GraphNode,
  incoming: GraphEdge[],
  shapeByNode: Map<string, number[]>,
): number[] {
  if (isOptionalPassThroughNode(node) && incoming.length === 1) {
    const source = graph.nodes.find((candidate) => candidate.id === incoming[0].source)
    const incomingShape = source ? outputShapeForSourceEdge(source, incoming[0], shapeByNode) : undefined
    if (incomingShape) return incomingShape
  }
  if (node.type === 'dataset') return datasetOutputValueForSlot(node, 0).shape
  return toTensor(node.params.value).shape
}

function outputShapeForSourceEdge(
  source: GraphNode,
  edge: Pick<GraphEdge, 'sourceSlot'>,
  shapeByNode: Map<string, number[]>,
): number[] | undefined {
  if (source.type === 'dataset') return datasetOutputValueForSlot(source, edge.sourceSlot ?? 0).shape
  return shapeByNode.get(source.id)
}

function assertValid(graph: GraphModel, requireLoss = false): void {
  const blockingIssues = validateGraph(graph, { requireLoss }).filter(
    (issue) => issue.code !== 'disconnected',
  )
  if (blockingIssues.length > 0) {
    throw new Error(blockingIssues.map((issue) => issue.message).join(' '))
  }
}

function forwardStep(node: GraphNode, incoming: GraphEdge[], inputValues: TensorValue[], graph: GraphModel): EvaluationTraceStep {
  const phase = isLossNode(node) ? 'loss' : 'forward'
  const calculation = calculationForForward(node, inputValues)
  const isTensorStep = inputValues.some((value) => !isScalarTensor(value))
  return {
    id: `${phase}-${node.id}`,
    phase,
    nodeId: node.id,
    edgeIds: incoming.map((edge) => edge.id),
    title: phase === 'loss' ? `Compute ${node.type === 'cross-entropy' ? 'cross entropy' : lossLabel(lossKindForNode(node, graph)).toLowerCase()}` : `Evaluate ${node.label}`,
    explanation:
      phase === 'loss'
        ? 'The loss compares the prediction with the target and turns the error into a positive scalar.'
        : `${node.label} receives ${isTensorStep ? 'tensor' : 'scalar'} inputs, applies its formula, and stores one ${isTensorStep ? 'tensor' : 'scalar'} output for downstream nodes.`,
    formula: formulaForNode(node, graph),
    calculation,
    pseudocode: pseudocodeForNode(node, graph),
  }
}

function backwardStep(
  node: GraphNode,
  incoming: GraphEdge[],
  downstreamGrad: TensorValue,
  contributions: Array<{ edgeId: string; sourceId: string; gradient: TensorValue }>,
  graph: GraphModel,
): EvaluationTraceStep {
  const contributionText =
    contributions.length === 0
      ? 'No upstream inputs receive gradients from this source node.'
      : contributions
          .map((contribution) => {
            const source = contribution.sourceId
            return `to ${source}: ${formatNumber(contribution.gradient)}`
          })
          .join(', ')

  return {
    id: `backward-${node.id}`,
    phase: 'backward',
    nodeId: node.id,
    edgeIds: incoming.map((edge) => edge.id),
    title: `Backpropagate through ${node.label}`,
    explanation: 'The incoming gradient is multiplied by local derivatives and accumulated on upstream nodes.',
    formula: derivativeFormula(node, graph),
    calculation: `incoming gradient ${formatNumber(downstreamGrad)} -> ${contributionText}`,
    pseudocode: ['g = node.grad', ...pseudocodeForBackward(node, graph)],
  }
}

function updateStep(update: ParameterUpdate): EvaluationTraceStep {
  return {
    id: `update-${update.nodeId}`,
    phase: 'update',
    nodeId: update.nodeId,
    edgeIds: [],
    title: `Update ${update.label}`,
    explanation: 'Gradient descent nudges the trainable parameter in the direction that reduces loss.',
    formula: 'new value = old value - learning_rate * gradient',
    calculation: `${formatNumber(update.newValue)} = ${formatNumber(update.oldValue)} - ${formatNumber(update.learningRate)} * ${formatNumber(update.gradient)}`,
    pseudocode: [`${update.label} -= lr * ${update.label}.grad`, `${update.label}.grad = 0`],
  }
}

function calculationForForward(node: GraphNode, inputs: TensorValue[]): string {
  if (SOURCE_TYPES.has(node.type)) {
    return `${node.label} stores ${formatNumber(node.value)}.`
  }
  if (node.type === 'matmul') return `${formatNumber(inputs[0])} @ ${formatNumber(inputs[1])} = ${formatNumber(node.value)} (row × column sums)`
  if (node.type === 'multiply') {
    return `${inputs.map((input) => formatNumber(input)).join(' * ')} = ${formatNumber(node.value)}`
  }
  if (node.type === 'add') {
    return `${inputs.map((input) => formatNumber(input)).join(' + ')} = ${formatNumber(node.value)}`
  }
  if (node.type === 'arithmetic') {
    return `${(node.params.expression ?? 'x1 * x2').replace(/x([1-9]\d*)\b/g, (_, index: string) => formatNumber(inputs[Number(index) - 1]))} = ${formatNumber(node.value)}`
  }
  if (node.type === 'activation') {
    return `${node.params.activation ?? 'identity'}(${formatNumber(inputs[0])}) = ${formatNumber(node.value)}`
  }
  if (TENSOR_OPERATION_TYPES.has(node.type)) return `${node.type}(${inputs.map((input) => formatShape(input.shape)).join(', ')}) = ${formatNumber(node.value)}`
  return lossCalculationForForward(node, inputs)
}

function lossCalculationForForward(node: GraphNode, inputs: TensorValue[]): string {
  const kind = lossKindForInputs(node, inputs)
  const error = node.cache?.error ?? subtractTensors(inputs[0], inputs[1])
  const denominator = meanDenominator(error)
  if (kind === 'mse') {
    return `Σ(error^2) / ${denominator} = ${formatNumber(node.value)}; error = ${formatNumber(error)}`
  }
  if (kind === 'mae') {
    return `Σ(|error|) / ${denominator} = ${formatNumber(node.value)}; error = ${formatNumber(error)}`
  }
  if (kind === 'binary-cross-entropy') {
    return `mean binary cross entropy over ${denominator} value${denominator === 1 ? '' : 's'} = ${formatNumber(node.value)}; error = ${formatNumber(error)}`
  }
  return `0.5 * Σ(error^2) = ${formatNumber(node.value)}; error = ${formatNumber(error)}`
}

function derivativeFormula(node: GraphNode, graph?: GraphModel): string {
  const inputLabels = inputLabelsForFormula(node, graph)
  if (isOptionalPassThroughNode(node)) return 'dInput = gradient (identity connection)'
  if (TENSOR_OPERATION_TYPES.has(node.type)) return TENSOR_DERIVATIVE_FORMULAS[node.type] ?? 'Accumulate the vector–Jacobian product into each input.'
  if (node.type === 'matmul') return 'dA = gradient @ B.T; dB = A.T @ gradient'
  if (node.type === 'multiply') {
    return inputLabels
      .map((label, index) => `dz/d${label} = ${inputLabels.filter((_, otherIndex) => otherIndex !== index).join(' * ')}`)
      .join(', ')
  }
  if (node.type === 'add') return inputLabels.map((label) => `dz/d${label} = 1`).join(', ')
  if (node.type === 'arithmetic') return `Apply the chain rule to ${node.params.expression ?? 'x1 * x2'} for each input.`
  if (node.type === 'activation') {
    const activation = node.params.activation ?? 'identity'
    if (activation === 'sigmoid') return `dz/d${inputLabels[0]} = sigmoid'(${inputLabels[0]})`
    if (activation === 'relu') return `dz/d${inputLabels[0]} = 1 if ${inputLabels[0]} > 0, otherwise 0`
    if (activation === 'tanh') return `dz/d${inputLabels[0]} = 1 - tanh(${inputLabels[0]})^2`
    return `dz/d${inputLabels[0]} = 1`
  }
  if (node.type === 'loss') return lossDerivativeFormula(lossKindForNode(node, graph), inputLabels, Boolean(graph && hasNonScalarIncomingValue(node, graph)))
  return 'Gradient accumulates here.'
}

function lossDerivativeFormula(kind: LossKind, inputLabels: string[], isTensor: boolean): string {
  const prediction = inputLabels[0] ?? 'prediction'
  const target = inputLabels[1] ?? 'target'
  const suffix = isTensor ? '_i' : ''
  const denominator = isTensor ? ' / n' : ''

  if (kind === 'mse') return `dL/d${prediction}${suffix} = 2 * (${prediction}${suffix} - ${target}${suffix})${denominator}`
  if (kind === 'mae') return `dL/d${prediction}${suffix} = sign(${prediction}${suffix} - ${target}${suffix})${denominator}`
  if (kind === 'binary-cross-entropy') {
    return `dL/d${prediction}${suffix} = (-${target}${suffix} / ${prediction}${suffix} + (1 - ${target}${suffix}) / (1 - ${prediction}${suffix}))${denominator}`
  }
  return `dL/d${prediction}${suffix} = ${prediction}${suffix} - ${target}${suffix}`
}

function pseudocodeForNode(node: GraphNode, graph?: GraphModel): string[] {
  if (TENSOR_OPERATION_TYPES.has(node.type)) return [formulaForNode(node, graph)]
  if (node.type === 'input') return [`${node.label} = ${formatNumber(node.params.value)}`]
  if (node.type === 'weight' || node.type === 'bias') return [`${node.label} = Parameter(${formatNumber(node.params.value)})`]
  if (node.type === 'target') return [`${node.label} = ${formatNumber(node.params.value)}`]
  if (node.type === 'matmul') return ['z = a @ b']
  if (node.type === 'multiply') return ['z = product(inputs)']
  if (node.type === 'add') return ['z = sum(inputs)']
  if (node.type === 'arithmetic') return [formulaForNode(node, graph).replaceAll('^', '**')]
  if (node.type === 'activation') return [`z = ${node.params.activation ?? 'identity'}(u)`]
  const lossKind = lossKindForNode(node, graph)
  if (lossKind === 'mse') return ['loss = mean((prediction - target) ** 2)']
  if (lossKind === 'mae') return ['loss = mean(abs(prediction - target))']
  if (lossKind === 'binary-cross-entropy') {
    return ['loss = mean(-(target * log(prediction) + (1 - target) * log(1 - prediction)))']
  }
  return ['loss = 0.5 * sum((prediction - target) ** 2)']
}

function pseudocodeForBackward(node: GraphNode, graph?: GraphModel): string[] {
  if (isOptionalPassThroughNode(node)) return ['upstream.grad += g']
  if (TENSOR_OPERATION_TYPES.has(node.type)) return ['input_grads = vector_jacobian_product(operation, inputs, g)', 'for each input: input.grad += input_grads[input]']
  if (node.type === 'matmul') return ['a.grad += g @ b.T', 'b.grad += a.T @ g']
  if (node.type === 'multiply') return ['for each input i:', '  input_i.grad += g * product(other inputs)']
  if (node.type === 'add') return ['for each input:', '  input.grad += g']
  if (node.type === 'arithmetic') return ['for each input i:', '  input_i.grad += g * ∂expression/∂input_i']
  if (node.type === 'activation') return ['u.grad += g * local_derivative']
  if (node.type === 'loss') return lossBackwardPseudocode(lossKindForNode(node, graph))
  return ['accumulate gradient']
}

function lossBackwardPseudocode(kind: LossKind): string[] {
  if (kind === 'mse') return ['prediction.grad += 2 * (prediction - target) / n']
  if (kind === 'mae') return ['prediction.grad += sign(prediction - target) / n']
  if (kind === 'binary-cross-entropy') {
    return ['prediction.grad += (-target / prediction + (1 - target) / (1 - prediction)) / n']
  }
  return ['prediction.grad += prediction - target']
}


/** Evaluate one tensor primitive using the shared autodiff implementation.
 * Each input is a distinct leaf: the graph engine accumulates contributions
 * across repeated inputs and shared nodes, rather than duplicating parameters. */
function tensorOperation(node: GraphNode, values: TensorValue[], requiresGrad = false) {
  const operands = values.map((value) => {
    const operand = autograd.tensor(value.shape, value.data, requiresGrad)
    if (value.excluded) operand.excluded = [...value.excluded]
    return operand
  })
  const [first, second, third] = operands
  let output: autograd.Tensor
  switch (node.type) {
    case 'conv2d': {
      const value = conv2d(first, second, third)
      output = autograd.tensor(value.shape, value.data, requiresGrad)
      if (requiresGrad) {
        output.parents = operands
        output.backwardRule = () => {
          const gradients = conv2dBackward(first, second, third, tensorValue(output.shape, output.grad))
          gradients.forEach((gradient, index) => gradient.data.forEach((entry, coordinate) => { operands[index].grad[coordinate] += entry }))
        }
      }
      break
    }
    case 'avgpool2d': {
      const value = avgpool2d(first)
      output = autograd.tensor(value.shape, value.data, requiresGrad)
      if (requiresGrad) {
        output.parents = [first]
        output.backwardRule = () => {
          const gradient = avgpool2dBackward(first, tensorValue(output.shape, output.grad))
          gradient.data.forEach((entry, index) => { first.grad[index] += entry })
        }
      }
      break
    }
    case 'embedding': output = autograd.embedding(first, second.data); break
    case 'transpose': output = autograd.transpose(first, node.params.axes); break
    case 'slice': output = autograd.slice(first, node.params.axis ?? 0, node.params.start ?? 0, node.params.end ?? first.shape[node.params.axis ?? 0]); break
    case 'concat': output = autograd.concat(operands, node.params.axis ?? 1); break
    case 'softmax': output = autograd.softmax(first); break
    case 'causal-mask': output = autograd.causalMask(first); break
    case 'layer-norm': output = autograd.layerNorm(first, second, third, node.params.epsilon ?? 1e-5); break
    case 'reshape': output = autograd.reshape(first, resolveReshape(node.params.shape ?? first.shape, first.data.length)); break
    case 'mean': output = autograd.mean(first, node.params.axis, node.params.keepDims); break
    case 'cross-entropy': output = autograd.crossEntropy(first, second.data); break
    default: throw new Error(`Unknown tensor operation ${node.type}.`)
  }
  return { output, operands }
}

const TENSOR_DERIVATIVE_FORMULAS: Partial<Record<NodeType, string>> = {
  conv2d: 'dKernel += input window · output gradient; dInput += kernel · output gradient; dBias += output gradient',
  avgpool2d: 'Each of the four input cells receives one quarter of its pooled output gradient',
  embedding: 'dTable[token] += gradient[row]; token IDs are discrete constants',
  transpose: 'dInput = transpose(gradient, inverse permutation)',
  slice: 'dInput[slice] += gradient; all other entries receive zero',
  concat: 'dInput_i = slice(gradient, the region belonging to input i)',
  softmax: 'dInput = probabilities · (gradient − sum(gradient · probabilities))',
  'causal-mask': 'dInput[i,j] = gradient[i,j] if j ≤ i; otherwise 0',
  'layer-norm': 'dInput = inverse_std · (scaled_gradient − mean(scaled_gradient) − normalized · mean(scaled_gradient · normalized))',
  reshape: 'dInput = reshape(gradient, original shape)',
  mean: 'dInput = broadcast(gradient) / number of reduced entries',
  'cross-entropy': 'dLogits = (softmax(logits) − one_hot(target)) / number of tokens',
}
