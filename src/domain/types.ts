export type NodeType =
  | 'dataset'
  | 'input'
  | 'weight'
  | 'bias'
  | 'multiply'
  | 'add'
  | 'activation'
  | 'target'
  | 'loss'

export type ActivationKind = 'identity' | 'relu' | 'sigmoid' | 'tanh'
export type LossKind = 'squared-error' | 'mse' | 'mae' | 'binary-cross-entropy'
export type DatasetKind =
  | 'line-1d'
  | 'cubic-1d'
  | 'plane-2d'
  | 'threshold-1d'
  | 'circle-center'
  | 'parabola-boundary'
export type DatasetTask = 'regression' | 'binary-classification'

export type GraphPhase = 'edit' | 'forward' | 'loss' | 'backward' | 'update'

export interface Position {
  x: number
  y: number
}

export interface NodeParams {
  value?: TensorValue | number
  activation?: ActivationKind
  loss?: LossKind
  dataset?: DatasetKind
  inputCount?: number
}

export interface NodeDimensions {
  width?: number
  height?: number
}

export interface DerivativeCache {
  inputValues: TensorValue[]
  outputValue: TensorValue
  localDerivatives: TensorValue[]
  error?: TensorValue
}

export interface GraphNode {
  id: string
  type: NodeType
  label: string
  position: Position
  dimensions?: NodeDimensions
  params: NodeParams
  value?: TensorValue
  grad?: TensorValue
  localDerivative?: TensorValue
  cache?: DerivativeCache
}

export interface GraphEdge {
  id: string
  source: string
  sourceSlot?: number
  target: string
  inputSlot?: number
  value?: TensorValue
  grad?: TensorValue
}

export interface GraphGroup {
  id: string
  label: string
  nodeIds: string[]
  position: Position
  dimensions: NodeDimensions
}

export interface GraphModel {
  nodes: GraphNode[]
  edges: GraphEdge[]
  groups?: GraphGroup[]
  learningRate: number
}

export interface ValidationIssue {
  code:
    | 'missing-loss'
    | 'multiple-losses'
    | 'cycle'
    | 'missing-input'
    | 'disconnected'
    | 'invalid-arity'
    | 'unknown-node'
    | 'shape-mismatch'
  message: string
  nodeId?: string
  edgeId?: string
}

export interface TensorValue {
  shape: number[]
  data: number[]
}

export interface EvaluationTraceStep {
  id: string
  phase: GraphPhase
  nodeId?: string
  edgeIds: string[]
  title: string
  explanation: string
  formula: string
  calculation: string
  pseudocode: string[]
}

export interface EvaluationResult {
  graph: GraphModel
  steps: EvaluationTraceStep[]
  loss?: number
}

export interface ParameterUpdate {
  nodeId: string
  label: string
  oldValue: TensorValue
  gradient: TensorValue
  learningRate: number
  newValue: TensorValue
}

export interface UpdateResult {
  graph: GraphModel
  steps: EvaluationTraceStep[]
  updates: ParameterUpdate[]
}

export interface ProjectDisplayState {
  showMath: boolean
  showGradient: boolean
  showCode: boolean
  showVisualization: boolean
}

export interface ProjectStateSnapshot {
  graph: GraphModel
  visualizationGraph: GraphModel
  initialParameterValues: Record<string, TensorValue>
  selectedNodeIds: string[]
  selectedGroupId?: string
  phase: GraphPhase
  traceSteps: EvaluationTraceStep[]
  traceIndex: number
  epoch: number
  currentLoss: number | null
  display: ProjectDisplayState
}

export interface ProjectStateFile {
  kind: 'backprop-builder-state'
  version: 1
  savedAt: string
  state: ProjectStateSnapshot
}

export type ProjectStateParseResult =
  | { ok: true; file: ProjectStateFile }
  | { ok: false; error: string }
