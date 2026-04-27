import { cloneGraph } from './engine'
import { isDatasetKind } from './datasets'
import { cloneTensor, isTensorValue } from './tensor'
import type {
  ActivationKind,
  DatasetKind,
  EvaluationTraceStep,
  GraphEdge,
  GraphModel,
  GraphNode,
  GraphPhase,
  LossKind,
  NodeDimensions,
  NodeParams,
  Position,
  ProjectDisplayState,
  ProjectStateFile,
  ProjectStateParseResult,
  ProjectStateSnapshot,
  TensorValue,
} from './types'

const PROJECT_STATE_KIND = 'backprop-builder-state'
const PROJECT_STATE_VERSION = 1
const NODE_TYPES = new Set([
  'dataset',
  'input',
  'weight',
  'bias',
  'multiply',
  'add',
  'activation',
  'target',
  'loss',
])
const ACTIVATION_KINDS = new Set<ActivationKind>(['identity', 'relu', 'sigmoid', 'tanh'])
const LOSS_KINDS = new Set<LossKind>(['squared-error', 'mse', 'mae', 'binary-cross-entropy'])
const GRAPH_PHASES = new Set<GraphPhase>(['edit', 'forward', 'loss', 'backward', 'update'])

export function createProjectStateFile(state: ProjectStateSnapshot): ProjectStateFile {
  return {
    kind: PROJECT_STATE_KIND,
    version: PROJECT_STATE_VERSION,
    savedAt: new Date().toISOString(),
    state: cloneProjectStateSnapshot(state),
  }
}

export function downloadProjectStateFile(file: ProjectStateFile): void {
  const blob = new Blob([JSON.stringify(file, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `backprop-builder-state-${file.savedAt.slice(0, 10)}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

export function parseProjectStateFile(text: string): ProjectStateParseResult {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return { ok: false, error: 'Import failed: choose a valid Backprop Builder state JSON file.' }
  }

  const validationError = projectStateFileError(parsed)
  if (validationError) return { ok: false, error: validationError }

  return {
    ok: true,
    file: {
      ...(parsed as ProjectStateFile),
      state: cloneProjectStateSnapshot((parsed as ProjectStateFile).state),
    },
  }
}

function cloneProjectStateSnapshot(state: ProjectStateSnapshot): ProjectStateSnapshot {
  return {
    graph: cloneGraph(state.graph),
    visualizationGraph: cloneGraph(state.visualizationGraph),
    initialParameterValues: cloneParameterValueMap(state.initialParameterValues),
    selectedNodeIds: [...state.selectedNodeIds],
    selectedGroupId: state.selectedGroupId ?? undefined,
    phase: state.phase,
    traceSteps: cloneTraceSteps(state.traceSteps),
    traceIndex: state.traceIndex,
    epoch: state.epoch,
    currentLoss: state.currentLoss,
    display: { ...state.display },
  }
}

function cloneParameterValueMap(values: Record<string, TensorValue>): Record<string, TensorValue> {
  return Object.fromEntries(Object.entries(values).map(([label, value]) => [label, cloneTensor(value)]))
}

function cloneTraceSteps(steps: EvaluationTraceStep[]): EvaluationTraceStep[] {
  return steps.map((step) => ({
    ...step,
    edgeIds: [...step.edgeIds],
    pseudocode: [...step.pseudocode],
  }))
}

function projectStateFileError(value: unknown): string | undefined {
  if (!isRecord(value)) return 'Import failed: choose a valid Backprop Builder state JSON file.'
  if (value.kind !== PROJECT_STATE_KIND) return 'Import failed: this is not a Backprop Builder state file.'
  if (value.version !== PROJECT_STATE_VERSION) return 'Import failed: this state file version is not supported.'
  if (typeof value.savedAt !== 'string') return 'Import failed: the state file is missing its saved timestamp.'
  if (!isProjectStateSnapshot(value.state)) return 'Import failed: the state file is missing required graph data.'
  return undefined
}

function isProjectStateSnapshot(value: unknown): value is ProjectStateSnapshot {
  if (!isRecord(value)) return false
  const currentLoss = value.currentLoss
  return (
    isGraphModel(value.graph) &&
    isGraphModel(value.visualizationGraph) &&
    isTensorMap(value.initialParameterValues) &&
    Array.isArray(value.selectedNodeIds) &&
    value.selectedNodeIds.every((id) => typeof id === 'string') &&
    (value.selectedGroupId === undefined || value.selectedGroupId === null || typeof value.selectedGroupId === 'string') &&
    isGraphPhase(value.phase) &&
    Array.isArray(value.traceSteps) &&
    value.traceSteps.every(isTraceStep) &&
    isNonNegativeInteger(value.traceIndex) &&
    isNonNegativeInteger(value.epoch) &&
    (currentLoss === null || isFiniteNumber(currentLoss)) &&
    isDisplayState(value.display)
  )
}

function isGraphModel(value: unknown): value is GraphModel {
  if (!isRecord(value)) return false
  return (
    Array.isArray(value.nodes) &&
    value.nodes.every(isGraphNode) &&
    Array.isArray(value.edges) &&
    value.edges.every(isGraphEdge) &&
    (value.groups === undefined || (Array.isArray(value.groups) && value.groups.every(isGraphGroup))) &&
    isFiniteNumber(value.learningRate)
  )
}

function isGraphNode(value: unknown): value is GraphNode {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    isNodeType(value.type) &&
    typeof value.label === 'string' &&
    isPosition(value.position) &&
    isNodeParams(value.params) &&
    (value.dimensions === undefined || isNodeDimensions(value.dimensions)) &&
    (value.value === undefined || isTensorValue(value.value)) &&
    (value.grad === undefined || isTensorValue(value.grad)) &&
    (value.localDerivative === undefined || isTensorValue(value.localDerivative)) &&
    (value.cache === undefined || isDerivativeCache(value.cache))
  )
}

function isGraphEdge(value: unknown): value is GraphEdge {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    typeof value.source === 'string' &&
    typeof value.target === 'string' &&
    isOptionalNonNegativeInteger(value.sourceSlot) &&
    isOptionalNonNegativeInteger(value.inputSlot) &&
    (value.value === undefined || isTensorValue(value.value)) &&
    (value.grad === undefined || isTensorValue(value.grad))
  )
}

function isGraphGroup(value: unknown): boolean {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    typeof value.label === 'string' &&
    Array.isArray(value.nodeIds) &&
    value.nodeIds.every((id) => typeof id === 'string') &&
    isPosition(value.position) &&
    isNodeDimensions(value.dimensions)
  )
}

function isNodeParams(value: unknown): value is NodeParams {
  if (!isRecord(value)) return false
  return (
    (value.value === undefined || isFiniteNumber(value.value) || isTensorValue(value.value)) &&
    (value.activation === undefined || ACTIVATION_KINDS.has(value.activation as ActivationKind)) &&
    (value.loss === undefined || LOSS_KINDS.has(value.loss as LossKind)) &&
    (value.dataset === undefined || isDatasetKind(value.dataset as DatasetKind)) &&
    isOptionalNonNegativeInteger(value.inputCount)
  )
}

function isDerivativeCache(value: unknown): boolean {
  if (!isRecord(value)) return false
  return (
    Array.isArray(value.inputValues) &&
    value.inputValues.every(isTensorValue) &&
    isTensorValue(value.outputValue) &&
    Array.isArray(value.localDerivatives) &&
    value.localDerivatives.every(isTensorValue) &&
    (value.error === undefined || isTensorValue(value.error))
  )
}

function isTraceStep(value: unknown): value is EvaluationTraceStep {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    isGraphPhase(value.phase) &&
    (value.nodeId === undefined || typeof value.nodeId === 'string') &&
    Array.isArray(value.edgeIds) &&
    value.edgeIds.every((id) => typeof id === 'string') &&
    typeof value.title === 'string' &&
    typeof value.explanation === 'string' &&
    typeof value.formula === 'string' &&
    typeof value.calculation === 'string' &&
    Array.isArray(value.pseudocode) &&
    value.pseudocode.every((line) => typeof line === 'string')
  )
}

function isTensorMap(value: unknown): value is Record<string, TensorValue> {
  return isRecord(value) && Object.values(value).every(isTensorValue)
}

function isDisplayState(value: unknown): value is ProjectDisplayState {
  if (!isRecord(value)) return false
  return (
    typeof value.showMath === 'boolean' &&
    typeof value.showGradient === 'boolean' &&
    typeof value.showCode === 'boolean' &&
    typeof value.showVisualization === 'boolean'
  )
}

function isPosition(value: unknown): value is Position {
  return isRecord(value) && isFiniteNumber(value.x) && isFiniteNumber(value.y)
}

function isNodeDimensions(value: unknown): value is NodeDimensions {
  return (
    isRecord(value) &&
    (value.width === undefined || isFiniteNumber(value.width)) &&
    (value.height === undefined || isFiniteNumber(value.height))
  )
}

function isNodeType(value: unknown): value is GraphNode['type'] {
  return typeof value === 'string' && NODE_TYPES.has(value)
}

function isGraphPhase(value: unknown): value is GraphPhase {
  return typeof value === 'string' && GRAPH_PHASES.has(value as GraphPhase)
}

function isOptionalNonNegativeInteger(value: unknown): boolean {
  return value === undefined || isNonNegativeInteger(value)
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
