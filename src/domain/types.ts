export type NodeType =
  | 'dataset'
  | 'input'
  | 'weight'
  | 'bias'
  | 'multiply'
  | 'matmul'
  | 'add'
  | 'activation'
  | 'target'
  | 'loss'
  | 'embedding'
  | 'transpose'
  | 'slice'
  | 'concat'
  | 'softmax'
  | 'causal-mask'
  | 'layer-norm'
  | 'reshape'
  | 'mean'
  | 'cross-entropy'
  | 'conv2d'
  | 'avgpool2d'

export type ActivationKind = 'identity' | 'relu' | 'sigmoid' | 'tanh'
export type LossKind = 'squared-error' | 'mse' | 'mae' | 'binary-cross-entropy'
export type DatasetKind =
  | 'line-1d'
  | 'cubic-1d'
  | 'plane-2d'
  | 'threshold-1d'
  | 'circle-center'
  | 'parabola-boundary'
  | 'xor'
  | 'digits-8x8'
  | 'color-cycle'
  | 'counting'
  | 'attention-query'
  | 'attention-sequence'
  | 'class-scores'
  | 'neuron-basics'
export type DatasetTask = 'regression' | 'binary-classification' | 'classification' | 'sequence' | 'attention'

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
  datasetMode?: 'sample' | 'batch'
  datasetIndex?: number
  datasetSplit?: 'all' | 'train' | 'test'
  /** An editable experiment supplied by the dataset, e.g. a token prompt. */
  datasetValues?: TensorValue[]
  inputCount?: number
  axis?: number
  start?: number
  end?: number
  axes?: number[]
  shape?: number[]
  epsilon?: number
  keepDims?: boolean
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
  kind?: string
  /** Semantic inspection metadata; identifiers refer to the same executable graph. */
  detail?: Record<string, unknown>
  /** Parent module; nodeIds includes all descendant computation nodes. */
  parentId?: string
  id: string
  label: string
  nodeIds: string[]
  position: Position
  dimensions: NodeDimensions
}

export interface GraphViewState {
  canvasStyle?: 'builder' | 'architecture'
  semanticZoom?: boolean
  inspectedNeuron?: { groupId: string; unitIndex: number; row: number }
  /** User adjustments to semantic auto-layout; group keys use `visual-group:`. */
  layoutOffsets?: Record<string, Position>
  /** Manually placed calculations keep their scale and location in the continuous scene. */
  manualNodePlacements?: Record<string, { parentId?: string; offset: Position; scale?: number }>
  /** Connection reference for layout only. Live wiring can change independently. */
  layoutEdges?: Array<Pick<GraphEdge, 'id' | 'source' | 'target' | 'inputSlot' | 'sourceSlot'>>
  expandedGroupIds: string[]
  focusedGroupId?: string
  viewport?: { x: number; y: number; zoom: number }
}

export interface GraphModel {
  nodes: GraphNode[]
  edges: GraphEdge[]
  groups?: GraphGroup[]
  view?: GraphViewState
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
    | 'invalid-value'
  message: string
  nodeId?: string
  edgeId?: string
}

export interface TensorValue {
  shape: number[]
  data: number[]
  /** Exact masking metadata consumed by softmax, independent of display sentinels. */
  excluded?: boolean[]
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
