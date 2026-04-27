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
const FLEXIBLE_INPUT_TYPES = new Set<NodeType>(['multiply', 'add'])

export const NODE_WIDTH = 176
export const MIN_NODE_HEIGHT = 150
export const FLEX_INPUT_HEIGHT_STEP = 32
export const MIN_FLEX_INPUT_COUNT = 2
export const BASE_FLEX_INPUT_CAPACITY = 3
export const MAX_FLEX_INPUT_COUNT = 8
const BCE_EPSILON = 1e-7

export const LOSS_OPTIONS: Array<{ kind: LossKind; label: string }> = [
  { kind: 'squared-error', label: 'Squared error' },
  { kind: 'mse', label: 'Mean squared error' },
  { kind: 'mae', label: 'Mean absolute error' },
  { kind: 'binary-cross-entropy', label: 'Binary cross entropy' },
]

export const inputArityByType: Record<NodeType, number> = {
  dataset: 0,
  input: 1,
  weight: 0,
  bias: 0,
  multiply: 2,
  add: 2,
  activation: 1,
  target: 1,
  loss: 2,
}

export function outputArityForNode(node: GraphNode): number {
  if (node.type === 'loss') return 0
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
    case 'multiply':
      return `${outputLabel} = ${inputLabels.join(' * ')}`
    case 'add':
      return `${outputLabel} = ${inputLabels.join(' + ')}`
    case 'activation':
      return `${outputLabel} = ${node.params.activation ?? 'identity'}(${inputLabels[0]})`
    case 'loss':
      return lossFormula(lossKindForNode(node), inputLabels, Boolean(graph && hasNonScalarIncomingValue(node, graph)))
  }
}

export function lossKindForNode(node: GraphNode): LossKind {
  const selected = node.params.loss
  if (selected && LOSS_OPTIONS.some((option) => option.kind === selected)) return selected
  return 'squared-error'
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
  if (node.type === 'loss') return ['prediction', 'target']
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
  if (node.type === 'loss') return 'L'
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
    if (!node || SOURCE_TYPES.has(node.type) || node.type === 'loss') continue
    labels.set(node.id, `z${index}`)
    index += 1
  }
  return labels
}

export function cloneGraph(graph: GraphModel): GraphModel {
  return {
    learningRate: graph.learningRate,
    groups: graph.groups?.map((group) => ({
      ...group,
      nodeIds: [...group.nodeIds],
      position: { ...group.position },
      dimensions: { ...group.dimensions },
    })),
    nodes: graph.nodes.map((node) => ({
      ...node,
      params: {
        ...node.params,
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

export function validateGraph(graph: GraphModel): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const nodeIds = new Set(graph.nodes.map((node) => node.id))
  const lossNodes = graph.nodes.filter((node) => node.type === 'loss')

  if (lossNodes.length === 0) {
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
  if (node.type === 'activation') return [...inputShapes[0]]
  if (node.type === 'add' || node.type === 'multiply') return broadcastShapeForShapes(inputShapes)
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

  const lossValue = next.nodes.find((node) => node.type === 'loss')?.value
  const loss = lossValue ? scalarFromTensor(lossValue) : undefined
  return { graph: next, steps, loss }
}

export function backwardPass(graph: GraphModel): EvaluationResult {
  assertValid(graph)
  const next = cloneGraph(graph)
  const order = topologicalSort(next).reverse()
  const steps: EvaluationTraceStep[] = []
  for (const node of next.nodes) {
    node.grad = zeroLike(sourceDefaultValue(node))
  }
  for (const edge of next.edges) {
    edge.grad = edge.value ? zeroLike(edge.value) : undefined
  }

  const lossNode = next.nodes.find((node) => node.type === 'loss')
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

    if (!SOURCE_TYPES.has(node.type)) {
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

  if (node.type === 'activation') {
    const activation = node.params.activation ?? 'identity'
    const value = mapActivation(activation, inputs[0])
    const derivative = activationDerivativeTensor(activation, inputs[0], value)
    return { value, localDerivative: derivative, localDerivatives: [derivative] }
  }

  return computeLossForward(lossKindForNode(node), inputs[0], inputs[1])
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
  if (SOURCE_TYPES.has(node.type)) return []
  const values = incoming.map((edge) => valueForSourceEdge(graph, edge))

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
  const predictionDerivative = lossGradient(lossKindForNode(node), prediction, target)
  return [
    {
      edgeId: incoming[0].id,
      sourceId: incoming[0].source,
      gradient: reduceToShape(multiplyTensors(downstreamGrad, predictionDerivative), prediction.shape),
    },
    { edgeId: incoming[1].id, sourceId: incoming[1].source, gradient: zeroLike(target) },
  ]
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
  return sourceDefaultValue(node)
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
  return sourceDefaultValue(node).shape
}

function outputShapeForSourceEdge(
  source: GraphNode,
  edge: Pick<GraphEdge, 'sourceSlot'>,
  shapeByNode: Map<string, number[]>,
): number[] | undefined {
  if (source.type === 'dataset') return datasetOutputValueForSlot(source, edge.sourceSlot ?? 0).shape
  return shapeByNode.get(source.id)
}

function assertValid(graph: GraphModel): void {
  const blockingIssues = validateGraph(graph).filter(
    (issue) => issue.code !== 'disconnected',
  )
  if (blockingIssues.length > 0) {
    throw new Error(blockingIssues.map((issue) => issue.message).join(' '))
  }
}

function forwardStep(node: GraphNode, incoming: GraphEdge[], inputValues: TensorValue[], graph: GraphModel): EvaluationTraceStep {
  const phase = node.type === 'loss' ? 'loss' : 'forward'
  const calculation = calculationForForward(node, inputValues)
  const isTensorStep = inputValues.some((value) => !isScalarTensor(value))
  return {
    id: `${phase}-${node.id}`,
    phase,
    nodeId: node.id,
    edgeIds: incoming.map((edge) => edge.id),
    title: phase === 'loss' ? `Compute ${lossLabel(lossKindForNode(node)).toLowerCase()}` : `Evaluate ${node.label}`,
    explanation:
      phase === 'loss'
        ? 'The loss compares the prediction with the target and turns the error into a positive scalar.'
        : `${node.label} receives ${isTensorStep ? 'tensor' : 'scalar'} inputs, applies its formula, and stores one ${isTensorStep ? 'tensor' : 'scalar'} output for downstream nodes.`,
    formula: formulaForNode(node, graph),
    calculation,
    pseudocode: pseudocodeForNode(node),
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
    pseudocode: ['g = node.grad', ...pseudocodeForBackward(node)],
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
  if (node.type === 'multiply') {
    return `${inputs.map((input) => formatNumber(input)).join(' * ')} = ${formatNumber(node.value)}`
  }
  if (node.type === 'add') {
    return `${inputs.map((input) => formatNumber(input)).join(' + ')} = ${formatNumber(node.value)}`
  }
  if (node.type === 'activation') {
    return `${node.params.activation ?? 'identity'}(${formatNumber(inputs[0])}) = ${formatNumber(node.value)}`
  }
  return lossCalculationForForward(node, inputs)
}

function lossCalculationForForward(node: GraphNode, inputs: TensorValue[]): string {
  const kind = lossKindForNode(node)
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
  if (node.type === 'multiply') {
    return inputLabels
      .map((label, index) => `dz/d${label} = ${inputLabels.filter((_, otherIndex) => otherIndex !== index).join(' * ')}`)
      .join(', ')
  }
  if (node.type === 'add') return inputLabels.map((label) => `dz/d${label} = 1`).join(', ')
  if (node.type === 'activation') {
    const activation = node.params.activation ?? 'identity'
    if (activation === 'sigmoid') return `dz/d${inputLabels[0]} = sigmoid'(${inputLabels[0]})`
    if (activation === 'relu') return `dz/d${inputLabels[0]} = 1 if ${inputLabels[0]} > 0, otherwise 0`
    if (activation === 'tanh') return `dz/d${inputLabels[0]} = 1 - tanh(${inputLabels[0]})^2`
    return `dz/d${inputLabels[0]} = 1`
  }
  if (node.type === 'loss') return lossDerivativeFormula(lossKindForNode(node), inputLabels, Boolean(graph && hasNonScalarIncomingValue(node, graph)))
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

function pseudocodeForNode(node: GraphNode): string[] {
  if (node.type === 'input') return [`${node.label} = ${formatNumber(node.params.value)}`]
  if (node.type === 'weight' || node.type === 'bias') return [`${node.label} = Parameter(${formatNumber(node.params.value)})`]
  if (node.type === 'target') return [`${node.label} = ${formatNumber(node.params.value)}`]
  if (node.type === 'multiply') return ['z = product(inputs)']
  if (node.type === 'add') return ['z = sum(inputs)']
  if (node.type === 'activation') return [`z = ${node.params.activation ?? 'identity'}(u)`]
  const lossKind = lossKindForNode(node)
  if (lossKind === 'mse') return ['loss = mean((prediction - target) ** 2)']
  if (lossKind === 'mae') return ['loss = mean(abs(prediction - target))']
  if (lossKind === 'binary-cross-entropy') {
    return ['loss = mean(-(target * log(prediction) + (1 - target) * log(1 - prediction)))']
  }
  return ['loss = 0.5 * sum((prediction - target) ** 2)']
}

function pseudocodeForBackward(node: GraphNode): string[] {
  if (node.type === 'multiply') return ['for each input i:', '  input_i.grad += g * product(other inputs)']
  if (node.type === 'add') return ['for each input:', '  input.grad += g']
  if (node.type === 'activation') return ['u.grad += g * local_derivative']
  if (node.type === 'loss') return lossBackwardPseudocode(lossKindForNode(node))
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
