export type NodeType =
  | 'input'
  | 'weight'
  | 'bias'
  | 'multiply'
  | 'add'
  | 'activation'
  | 'target'
  | 'loss'

export type ActivationKind = 'identity' | 'relu' | 'sigmoid' | 'tanh'

export type GraphPhase = 'edit' | 'forward' | 'loss' | 'backward' | 'update'

export interface Position {
  x: number
  y: number
}

export interface NodeParams {
  value?: TensorValue | number
  activation?: ActivationKind
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

export interface LessonDefinition {
  id: string
  name: string
  prompt: string
  task: string
  completionCondition: string
  successMessage: string
  createGraph: () => GraphModel
}

export interface TrainingSessionSummary {
  timestamp: string
  lessonName: string
  graph: Pick<GraphModel, 'nodes' | 'edges'>
  initialParameterValues: Record<string, TensorValue>
  finalParameterValues: Record<string, TensorValue>
  learningRate: number
  trainingSteps: number
  finalLoss: number | null
  completedLessonActions: string[]
}
