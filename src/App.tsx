import { DatasetWorkbench } from './components/DatasetWorkbench'
import { datasetForNode } from './domain/datasets'
import { denseGroupDetail } from './domain/authoring'
import '@xyflow/react/dist/style.css'
import {
  BookOpen,
  Calculator,
  ChevronDown,
  Download,
  FastForward,
  Eye,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  StepForward,
  Upload,
} from 'lucide-react'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactElement,
} from 'react'
import './App.css'
import './unifiedStudio.css'
import { ModelInspector } from './components/ModelInspector'
import { DecoderControls } from './components/DecoderControls'
import { DatasetControls } from './components/DatasetControls'
import { CnnControls } from './components/CnnControls'
import { isHeldOutSample } from './domain/modelDatasets'
import { createModelPreset, type ModelPresetKind } from './domain/modelPresets'
import { placeCanvasNode } from './domain/nodePlacement'
import {
  activeProjectionEdges,
  projectDenseNeurons,
} from './domain/neuronProjection'
import { LESSONS } from './learning/presets'
import { GraphCanvas } from './components/GraphCanvas'
import { VisualizationPanel } from './components/VisualizationPanel'
import {
  copyGraphSelection,
  pasteGraphClipboard,
  type GraphClipboardFragment,
} from './domain/clipboard'
import {
  backwardPass,
  cloneGraph,
  datasetOutputValueForSlot,
  formatNumber,
  forwardPass,
  formulaForNode,
  parameterValues,
  runTrainingStep,
  remapDatasetOutputSlot,
  updateParameters,
  validateGraph,
  isLossNode,
} from './domain/engine'
import {
  createEmptyGraph,
  createStarterGraph,
  createNode,
} from './domain/examples'
import {
  collapsedGroupForNode,
  explodeVisualGroup,
  mergeNodesIntoVisualGroup,
  moveVisualGroup,
  setVisualGroupExpanded,
} from './domain/grouping'
import {
  createProjectStateFile,
  downloadProjectStateFile,
  parseProjectStateFile,
} from './domain/session'
import {
  cloneTensor,
  formatFullTensor,
  toTensor,
  zeroLike,
} from './domain/tensor'
import {
  visibleGraphForTrace,
  visibleStepEdgeIds,
} from './domain/traceVisibility'
import type {
  ActivationKind,
  DatasetKind,
  EvaluationTraceStep,
  GraphModel,
  GraphPhase,
  GraphViewState,
  LossKind,
  NodeType,
  NodeParams,
  TensorValue,
} from './domain/types'

const palette: Array<{ type: NodeType; label: string }> = [
  { type: 'dataset', label: 'Dataset' },
  { type: 'input', label: 'Input' },
  { type: 'weight', label: 'Weight' },
  { type: 'bias', label: 'Bias' },
  { type: 'multiply', label: 'Multiply' },
  { type: 'matmul', label: 'Matrix product' },
  { type: 'add', label: 'Add' },
  { type: 'activation', label: 'Activation' },
  { type: 'target', label: 'Target' },
  { type: 'loss', label: 'Loss' },
  { type: 'embedding', label: 'Embedding lookup' },
  { type: 'transpose', label: 'Transpose' },
  { type: 'slice', label: 'Slice tensor' },
  { type: 'concat', label: 'Concatenate' },
  { type: 'softmax', label: 'Softmax' },
  { type: 'causal-mask', label: 'Causal mask' },
  { type: 'layer-norm', label: 'Layer norm' },
  { type: 'reshape', label: 'Reshape' },
  { type: 'mean', label: 'Mean' },
  { type: 'cross-entropy', label: 'Cross-entropy' },
  { type: 'conv2d', label: 'Convolution' },
  { type: 'avgpool2d', label: 'Average pooling' },
]

const MIN_PLAY_DELAY_MS = 50
const MAX_PLAY_DELAY_MS = 1800
const DEFAULT_PLAY_DELAY_MS = 900
const DEFAULT_SPEED_SLIDER_VALUE =
  MIN_PLAY_DELAY_MS + MAX_PLAY_DELAY_MS - DEFAULT_PLAY_DELAY_MS
const HISTORY_LIMIT = 100
const PASTE_OFFSET_STEP = 36
const SHOW_MATH_LAYER = true
const SHOW_GRADIENT_LAYER = true
const SHOW_CODE_LAYER = false

interface HistorySnapshot {
  graph: GraphModel
  visualizationGraph: GraphModel
  initialParams: Record<string, TensorValue>
  selectedNodeIds: string[]
  selectedGroupId?: string
  phase: GraphPhase
  traceSteps: EvaluationTraceStep[]
  traceIndex: number
  epoch: number
  currentLoss: number | null
}

function speedSliderValueToDelay(value: number): number {
  return MIN_PLAY_DELAY_MS + MAX_PLAY_DELAY_MS - value
}

interface AppProps {
  initialGraph?: GraphModel
  presetKind?: ModelPresetKind
  onGallery?: () => void
}

function App({
  initialGraph,
  presetKind,
  onGallery,
}: AppProps = {}): ReactElement {
  const [graph, setGraph] = useState<GraphModel>(
    () => safeForward(initialGraph ?? createEmptyGraph()).graph,
  )
  const [visualizationGraph, setVisualizationGraph] = useState<GraphModel>(
    () => safeForward(initialGraph ?? createEmptyGraph()).graph,
  )
  const [initialParams, setInitialParams] = useState<
    Record<string, TensorValue>
  >(() => parameterValues(initialGraph ?? createEmptyGraph()))
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([])
  const [selectedGroupId, setSelectedGroupId] = useState<string | undefined>()
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | undefined>()
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [executionError, setExecutionError] = useState<string>()
  const [phase, setPhase] = useState<GraphPhase>('edit')
  const [traceSteps, setTraceSteps] = useState<EvaluationTraceStep[]>([])
  const [traceIndex, setTraceIndex] = useState(0)
  const [showVisualization, setShowVisualization] = useState(false)
  const [isFileMenuOpen, setIsFileMenuOpen] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const runCanvasAction = useCallback((action: () => void) => {
    setExecutionError(undefined)
    try {
      action()
    } catch (error) {
      setIsPlaying(false)
      setInspectorOpen(true)
      setExecutionError(
        error instanceof Error
          ? error.message
          : 'The model could not complete this calculation.',
      )
    }
  }, [])
  const [speedSliderValue, setSpeedSliderValue] = useState(
    DEFAULT_SPEED_SLIDER_VALUE,
  )
  const [epoch, setEpoch] = useState(0)
  const [currentLoss, setCurrentLoss] = useState<number | null>(() =>
    initialGraph ? (safeForward(initialGraph).loss ?? null) : null,
  )
  const [pendingNodeType, setPendingNodeType] = useState<NodeType | undefined>()
  const [undoStack, setUndoStack] = useState<HistorySnapshot[]>([])
  const [clipboard, setClipboard] = useState<
    GraphClipboardFragment | undefined
  >()
  const [clipboardPasteCount, setClipboardPasteCount] = useState(0)
  const [importError, setImportError] = useState<string | undefined>()
  const importInputRef = useRef<HTMLInputElement | null>(null)

  const validationIssues = useMemo(() => validateGraph(graph), [graph])
  const blockingIssues = validationIssues.filter(
    (issue) => issue.code !== 'disconnected',
  )
  const hasLoss = graph.nodes.some(isLossNode)
  const heldOutSample = isHeldOutSample(graph)
  const activeStep = traceSteps[traceIndex]
  const selectedNodeId =
    !selectedGroupId && selectedNodeIds.length === 1
      ? selectedNodeIds[0]
      : undefined
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId)
  const traceGraph = useMemo(
    () => visibleGraphForTrace(graph, traceSteps, traceIndex, phase),
    [graph, phase, traceIndex, traceSteps],
  )
  const projection = useMemo(
    () => projectDenseNeurons(traceGraph),
    [traceGraph],
  )
  const displayGraph = projection.graph
  const inspectedNode = displayGraph.nodes.find(
    (node) => node.id === selectedNodeId,
  )
  const selectedGroup = graph.groups?.find(
    (group) => group.id === selectedGroupId,
  )
  const preset = LESSONS.find((item) => item.id === presetKind)
  const inspectedEdge = displayGraph.edges.find(
    (edge) => edge.id === selectedEdgeId,
  )
  const canvasStep = activeStep
    ? {
        ...activeStep,
        edgeIds: [
          ...visibleStepEdgeIds(graph, traceSteps, traceIndex),
          ...activeProjectionEdges(
            graph,
            graph.view?.inspectedNeuron,
            activeStep,
          ),
        ],
      }
    : undefined

  const snapshotCurrentState = useCallback(
    (): HistorySnapshot => ({
      graph: cloneGraph(graph),
      visualizationGraph: cloneGraph(visualizationGraph),
      initialParams: cloneParameterValueMap(initialParams),
      selectedNodeIds: [...selectedNodeIds],
      selectedGroupId,
      phase,
      traceSteps: cloneTraceSteps(traceSteps),
      traceIndex,
      epoch,
      currentLoss,
    }),
    [
      currentLoss,
      epoch,
      graph,
      initialParams,
      phase,
      selectedGroupId,
      selectedNodeIds,
      traceIndex,
      traceSteps,
      visualizationGraph,
    ],
  )

  const pushHistory = useCallback(() => {
    const snapshot = snapshotCurrentState()
    setUndoStack((stack) => [...stack, snapshot].slice(-HISTORY_LIMIT))
  }, [snapshotCurrentState])

  const restoreSnapshot = useCallback((snapshot: HistorySnapshot) => {
    setGraph(cloneGraph(snapshot.graph))
    setVisualizationGraph(cloneGraph(snapshot.visualizationGraph))
    setInitialParams(cloneParameterValueMap(snapshot.initialParams))
    setSelectedNodeIds([...snapshot.selectedNodeIds])
    setSelectedGroupId(snapshot.selectedGroupId)
    setPhase(snapshot.phase)
    setTraceSteps(cloneTraceSteps(snapshot.traceSteps))
    setTraceIndex(snapshot.traceIndex)
    setEpoch(snapshot.epoch)
    setCurrentLoss(snapshot.currentLoss)
    setPendingNodeType(undefined)
    setIsPlaying(false)
  }, [])

  const undoLastAction = useCallback(() => {
    if (undoStack.length === 0) return
    const snapshot = undoStack[undoStack.length - 1]
    setUndoStack((stack) => stack.slice(0, -1))
    restoreSnapshot(snapshot)
  }, [restoreSnapshot, undoStack])

  const selectSingleNode = useCallback((nodeId?: string) => {
    setSelectedNodeIds(nodeId ? [nodeId] : [])
    setSelectedGroupId(undefined)
  }, [])

  const selectCanvasSelection = useCallback(
    (selection: { nodeIds: string[]; groupId?: string }) => {
      setSelectedEdgeId(undefined)
      setSelectedNodeIds((existing) =>
        stringArraysEqual(existing, selection.nodeIds)
          ? existing
          : selection.nodeIds,
      )
      setSelectedGroupId((existing) =>
        existing === selection.groupId ? existing : selection.groupId,
      )
      if (selection.nodeIds.length > 0 || selection.groupId)
        setPendingNodeType(undefined)
    },
    [],
  )

  const copySelectionToClipboard = useCallback((): boolean => {
    const fragment = copyGraphSelection(graph, {
      nodeIds: selectedNodeIds,
      groupId: selectedGroupId,
    })
    if (!fragment) return false

    setClipboard(fragment)
    setClipboardPasteCount(0)
    return true
  }, [graph, selectedGroupId, selectedNodeIds])

  const pasteClipboard = useCallback((): boolean => {
    if (!clipboard) return false

    pushHistory()
    const offset = PASTE_OFFSET_STEP * (clipboardPasteCount + 1)
    const result = pasteGraphClipboard(graph, clipboard, {
      x: offset,
      y: offset,
    })
    setGraph(result.graph)
    setSelectedNodeIds(result.selection.nodeIds)
    setSelectedGroupId(result.selection.groupId)
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setPendingNodeType(undefined)
    setIsPlaying(false)
    setClipboardPasteCount((count) => count + 1)
    return true
  }, [clipboard, clipboardPasteCount, graph, pushHistory])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isEditableShortcutTarget(event.target))
        return
      if ((!event.metaKey && !event.ctrlKey) || event.shiftKey) return

      const key = event.key.toLowerCase()

      if (key === 'z') {
        if (undoStack.length === 0) return
        event.preventDefault()
        undoLastAction()
        return
      }

      if (key === 'c') {
        if (!copySelectionToClipboard()) return
        event.preventDefault()
        return
      }

      if (key === 'v') {
        if (!pasteClipboard()) return
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [
    copySelectionToClipboard,
    pasteClipboard,
    undoLastAction,
    undoStack.length,
  ])

  const loadGraph = useCallback(
    (nextGraph: GraphModel) => {
      pushHistory()
      setSelectedEdgeId(undefined)
      const evaluated = safeForward(nextGraph)
      setGraph(evaluated.graph)
      setVisualizationGraph(evaluated.graph)
      setInitialParams(parameterValues(nextGraph))
      selectSingleNode(
        evaluated.graph.nodes.find((node) => node.type === 'activation')?.id,
      )
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setEpoch(0)
      setCurrentLoss(evaluated.loss ?? null)
      setPendingNodeType(undefined)
    },
    [pushHistory, selectSingleNode],
  )

  const stepForward = useCallback(() => {
    if (blockingIssues.length > 0) return

    if (traceSteps.length > 0 && traceIndex < traceSteps.length - 1) {
      pushHistory()
      const nextIndex = nextVisibleStepEnd(graph, traceSteps, traceIndex + 1)
      setTraceIndex(nextIndex)
      setPhase(traceSteps[nextIndex].phase)
      selectSingleNode(traceSteps[nextIndex].nodeId)
      if (traceSteps[nextIndex].phase === 'loss') {
        setVisualizationGraph(cloneGraph(graph))
      }
      return
    }

    if (phase === 'edit' || phase === 'update') {
      pushHistory()
      const forward = forwardPass(graph)
      setGraph(forward.graph)
      const nextIndex = nextVisibleStepEnd(graph, forward.steps, 0)
      setTraceSteps(forward.steps)
      setTraceIndex(nextIndex)
      setPhase(forward.steps[nextIndex]?.phase ?? 'forward')
      selectSingleNode(forward.steps[nextIndex]?.nodeId)
      setCurrentLoss(forward.loss ?? null)
      if (forward.steps.length === 0) {
        setVisualizationGraph(forward.graph)
      }
      return
    }

    if (phase === 'forward' || phase === 'loss') {
      if (!hasLoss) {
        setIsPlaying(false)
        setVisualizationGraph(graph)
        return
      }
      pushHistory()
      const backward = backwardPass(graph)
      setGraph(backward.graph)
      const nextIndex = nextVisibleStepEnd(graph, backward.steps, 0)
      setTraceSteps(backward.steps)
      setTraceIndex(nextIndex)
      setPhase('backward')
      selectSingleNode(backward.steps[nextIndex]?.nodeId)
      return
    }

    if (phase === 'backward') {
      if (heldOutSample) {
        setIsPlaying(false)
        return
      }
      pushHistory()
      const updated = updateParameters(graph, graph.learningRate)
      const refreshed = safeForward(updated.graph)
      setVisualizationGraph(refreshed.graph)
      setGraph({
        ...refreshed.graph,
        nodes: refreshed.graph.nodes.map((node) => ({
          ...node,
          grad: zeroLike(toTensor(node.value ?? node.params.value)),
        })),
      })
      setTraceSteps(updated.steps)
      setTraceIndex(0)
      setPhase('update')
      setEpoch((value) => value + 1)
      setCurrentLoss(refreshed.loss ?? null)
      selectSingleNode(updated.steps[0]?.nodeId)
    }
  }, [
    blockingIssues.length,
    graph,
    hasLoss,
    heldOutSample,
    phase,
    pushHistory,
    selectSingleNode,
    traceIndex,
    traceSteps,
  ])

  useEffect(() => {
    if (!isPlaying) return
    const timeout = window.setTimeout(
      () => runCanvasAction(stepForward),
      speedSliderValueToDelay(speedSliderValue),
    )
    return () => window.clearTimeout(timeout)
  }, [isPlaying, speedSliderValue, stepForward, runCanvasAction])

  const runOneTrainingStep = useCallback(() => {
    if (blockingIssues.length > 0 || !hasLoss || heldOutSample) return
    pushHistory()
    const result = runTrainingStep(graph, graph.learningRate)
    const updateSteps = result.steps.filter((step) => step.phase === 'update')
    setGraph(result.graph)
    setVisualizationGraph(result.graph)
    setTraceSteps(updateSteps)
    setTraceIndex(0)
    setPhase('update')
    setEpoch((value) => value + 1)
    setCurrentLoss(result.loss ?? null)
    selectSingleNode(updateSteps[0]?.nodeId)
  }, [
    blockingIssues.length,
    graph,
    hasLoss,
    heldOutSample,
    pushHistory,
    selectSingleNode,
  ])

  const runTenTrainingSteps = useCallback(() => {
    if (blockingIssues.length > 0 || !hasLoss || heldOutSample) return
    pushHistory()
    let nextGraph = graph
    const startingLoss = currentLoss
    let loss = currentLoss
    for (let index = 0; index < 10; index += 1) {
      const result = runTrainingStep(nextGraph, nextGraph.learningRate)
      nextGraph = result.graph
      loss = result.loss ?? loss
    }
    const summaryStep: EvaluationTraceStep = {
      id: `update-batch-${Date.now()}`,
      phase: 'update',
      nodeId: nextGraph.nodes.find((node) => node.type === 'loss')?.id,
      edgeIds: [],
      title: 'Ran 10 gradient descent steps',
      explanation:
        'Each step recomputed the loss, backpropagated gradients, and updated the trainable parameters.',
      formula: 'parameter = parameter - learning_rate * gradient',
      calculation: `loss ${formatNumber(startingLoss ?? undefined)} -> ${formatNumber(loss ?? undefined)}`,
      pseudocode: [
        'for step in range(10):',
        '  loss = forward()',
        '  loss.backward()',
        '  update_parameters()',
      ],
    }
    setGraph(nextGraph)
    setVisualizationGraph(nextGraph)
    setTraceSteps([summaryStep])
    setTraceIndex(0)
    setPhase('update')
    setEpoch((value) => value + 10)
    setCurrentLoss(loss)
    selectSingleNode(summaryStep.nodeId)
  }, [
    blockingIssues.length,
    currentLoss,
    graph,
    hasLoss,
    heldOutSample,
    pushHistory,
    selectSingleNode,
  ])

  const selectPaletteNode = (type: NodeType) => {
    setPendingNodeType(type)
    setPhase('edit')
    setTraceSteps([])
  }

  const placePaletteNode = useCallback(
    (type: NodeType, position: { x: number; y: number }) => {
      pushHistory()
      const nextNode = createNode(type, nextNodeIndexForType(graph, type))
      const placedNode = { ...nextNode, position }
      setGraph(placeCanvasNode(graph, placedNode, displayGraph))
      selectSingleNode(placedNode.id)
      setPendingNodeType(undefined)
      setPhase('edit')
    },
    [graph, displayGraph, pushHistory, selectSingleNode],
  )

  const clearPendingPlacement = useCallback(() => {
    setPendingNodeType(undefined)
  }, [])

  const updateNodeValue = useCallback(
    (nodeId: string, value: TensorValue) => {
      pushHistory()
      setGraph((existing) =>
        invalidateGraphResults({
          ...existing,
          nodes: existing.nodes.map((node) =>
            node.id === nodeId
              ? { ...node, params: { ...node.params, value }, value }
              : node,
          ),
        }),
      )
      setVisualizationGraph((existing) => ({
        ...existing,
        nodes: existing.nodes.map((node) =>
          node.id === nodeId &&
          (node.type === 'input' || node.type === 'target')
            ? { ...node, params: { ...node.params, value }, value }
            : node,
        ),
      }))
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setCurrentLoss(null)
      setIsPlaying(false)
    },
    [pushHistory],
  )

  const updateActivation = useCallback(
    (nodeId: string, activation: ActivationKind) => {
      pushHistory()
      setGraph((existing) =>
        invalidateGraphResults({
          ...existing,
          nodes: existing.nodes.map((node) =>
            node.id === nodeId
              ? { ...node, params: { ...node.params, activation } }
              : node,
          ),
        }),
      )
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setCurrentLoss(null)
      setIsPlaying(false)
    },
    [pushHistory],
  )

  const updateLoss = useCallback(
    (nodeId: string, loss: LossKind) => {
      pushHistory()
      setGraph((existing) =>
        invalidateGraphResults({
          ...existing,
          nodes: existing.nodes.map((node) =>
            node.id === nodeId
              ? { ...node, params: { ...node.params, loss } }
              : node,
          ),
        }),
      )
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setCurrentLoss(null)
      setIsPlaying(false)
    },
    [pushHistory],
  )

  const updateDataset = useCallback(
    (nodeId: string, dataset: DatasetKind) => {
      pushHistory()
      const updateNode = (node: GraphModel['nodes'][number]) => {
        if (node.id !== nodeId || node.type !== 'dataset') return node
        const updated = { ...node, params: { ...node.params, dataset, datasetIndex: 0, datasetValues: undefined } }
        const value = datasetOutputValueForSlot(updated, 0)
        return { ...updated, value, grad: zeroLike(value) }
      }
      setGraph((existing) =>
        invalidateGraphResults({
          ...existing,
          nodes: existing.nodes.map(updateNode),
          edges: remapDatasetOutgoingEdges(existing, nodeId, dataset),
        }),
      )
      setVisualizationGraph((existing) => ({
        ...existing,
        nodes: existing.nodes.map(updateNode),
        edges: remapDatasetOutgoingEdges(existing, nodeId, dataset),
      }))
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setCurrentLoss(null)
      setIsPlaying(false)
    },
    [pushHistory],
  )

  const updateLearningRate = (learningRate: number) => {
    pushHistory()
    setGraph((existing) => ({ ...existing, learningRate }))
  }

  const randomizeParameters = () => {
    pushHistory()
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setCurrentLoss(null)
    setIsPlaying(false)
    setGraph((existing) => {
      const next = {
        ...existing,
        nodes: existing.nodes.map((node) => {
          if (node.type !== 'weight' && node.type !== 'bias') return node
          const original = toTensor(node.params.value)
          const value = {
            ...original,
            data: original.data.map(() =>
              Number((Math.random() * 2 - 1).toFixed(2)),
            ),
          }
          return { ...node, params: { ...node.params, value }, value }
        }),
      }
      setInitialParams(parameterValues(next))
      const evaluated = safeForward(next)
      setVisualizationGraph(evaluated.graph)
      return evaluated.graph
    })
  }

  const clearRecordedExecution = useCallback(() => {
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setCurrentLoss(null)
    setIsPlaying(false)
  }, [])

  const applyGraphChange = useCallback(
    (nextGraph: GraphModel) => {
      pushHistory()
      const changed =
        computationSignature(graph) !== computationSignature(nextGraph)
      setGraph(changed ? invalidateGraphResults(nextGraph) : nextGraph)
      if (changed) {
        clearRecordedExecution()
        setVisualizationGraph(invalidateGraphResults(nextGraph))
      }
    },
    [graph, pushHistory, clearRecordedExecution],
  )

  const mergeSelectedNodes = useCallback(() => {
    const result = mergeNodesIntoVisualGroup(graph, selectedNodeIds)
    if (!result.group) return

    pushHistory()
    setGraph(result.graph)
    setSelectedNodeIds([])
    setSelectedGroupId(result.group.id)
    setPendingNodeType(undefined)
  }, [graph, pushHistory, selectedNodeIds])

  const explodeGroup = useCallback(
    (groupId: string) => {
      const group = graph.groups?.find((candidate) => candidate.id === groupId)
      if (!group) return

      pushHistory()
      setGraph(explodeVisualGroup(graph, groupId))
      setSelectedNodeIds(group.nodeIds)
      setSelectedGroupId(undefined)
      setPendingNodeType(undefined)
    },
    [graph, pushHistory],
  )

  const moveGroup = useCallback(
    (groupId: string, position: { x: number; y: number }) => {
      pushHistory()
      setGraph((existing) => moveVisualGroup(existing, groupId, position))
    },
    [pushHistory],
  )

  const openGroup = (groupId: string) => {
    pushHistory()
    let next = setVisualGroupExpanded(graph, groupId, true)
    let parent = graph.groups?.find((group) => group.id === groupId)?.parentId
    while (parent) {
      next = setVisualGroupExpanded(next, parent, true)
      parent = graph.groups?.find((group) => group.id === parent)?.parentId
    }
    setGraph({ ...next, view: { ...next.view!, focusedGroupId: groupId } })
    setSelectedGroupId(groupId)
    setSelectedNodeIds([])
    setSelectedEdgeId(undefined)
  }

  const inspectNeuron = (
    groupId: string,
    unitIndex: number,
    row = graph.view?.inspectedNeuron?.row ?? 0,
  ) => {
    pushHistory()
    const next = setVisualGroupExpanded(graph, groupId, true)
    const expanded = new Set(next.view!.expandedGroupIds)
    let parent = graph.groups?.find((group) => group.id === groupId)?.parentId
    while (parent) {
      expanded.add(parent)
      parent = graph.groups?.find((group) => group.id === parent)?.parentId
    }
    setGraph({
      ...next,
      view: {
        ...next.view!,
        expandedGroupIds: [...expanded],
        focusedGroupId: groupId,
        inspectedNeuron: { groupId, unitIndex, row },
      },
    })
    setSelectedGroupId(groupId)
    setSelectedNodeIds([])
  }

  const updateNodeParams = (nodeId: string, params: NodeParams) =>
    applyGraphChange({
      ...graph,
      nodes: graph.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, params: { ...node.params, ...params } }
          : node,
      ),
    })
  const duplicateSelection = () => {
    const fragment = copyGraphSelection(graph, {
      nodeIds: selectedNodeIds,
      groupId: selectedGroupId,
    })
    if (!fragment) return
    const result = pasteGraphClipboard(graph, fragment, { x: 64, y: 64 })
    applyGraphChange(result.graph)
    setSelectedNodeIds(result.selection.nodeIds)
    setSelectedGroupId(result.selection.groupId)
  }
  const evaluateModel = () => {
    if (blockingIssues.length) return
    pushHistory()
    const result = forwardPass(graph)
    setGraph(result.graph)
    setVisualizationGraph(result.graph)
    setTraceSteps(result.steps)
    setTraceIndex(Math.max(0, result.steps.length - 1))
    setPhase(hasLoss ? 'loss' : 'forward')
    setCurrentLoss(result.loss ?? null)
    setIsPlaying(false)
  }

  const saveProjectState = () => {
    setImportError(undefined)
    downloadProjectStateFile(
      createProjectStateFile({
        graph,
        visualizationGraph,
        initialParameterValues: initialParams,
        selectedNodeIds,
        selectedGroupId,
        phase,
        traceSteps,
        traceIndex,
        epoch,
        currentLoss,
        display: {
          showMath: SHOW_MATH_LAYER,
          showGradient: SHOW_GRADIENT_LAYER,
          showCode: SHOW_CODE_LAYER,
          showVisualization,
        },
      }),
    )
  }

  const chooseProjectStateFile = () => {
    setImportError(undefined)
    importInputRef.current?.click()
  }

  const runFileMenuAction = (action: () => void) => {
    setIsFileMenuOpen(false)
    action()
  }

  const importProjectState = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) return

    const result = parseProjectStateFile(await file.text())
    if (!result.ok) {
      setImportError(result.error)
      return
    }

    const nextState = result.file.state
    pushHistory()
    setGraph(cloneGraph(nextState.graph))
    setVisualizationGraph(cloneGraph(nextState.visualizationGraph))
    setInitialParams(cloneParameterValueMap(nextState.initialParameterValues))
    setSelectedNodeIds([...nextState.selectedNodeIds])
    setSelectedGroupId(nextState.selectedGroupId)
    setPhase(nextState.phase)
    setTraceSteps(cloneTraceSteps(nextState.traceSteps))
    setTraceIndex(nextState.traceIndex)
    setEpoch(nextState.epoch)
    setCurrentLoss(nextState.currentLoss)
    setShowVisualization(nextState.display.showVisualization)
    setPendingNodeType(undefined)
    setIsPlaying(false)
    setImportError(undefined)
  }

  return (
    <main
      className={`app-shell ${presetKind ? 'unified-studio' : ''} ${inspectorOpen ? 'inspector-open' : ''} ${paletteOpen ? 'palette-open' : ''}`}
    >
      <header className="top-bar">
        <div className="top-brand">
          <div className="brand-mark">BB</div>
          <div>
            <h1>Backprop Builder</h1>
            {presetKind && (
              <p className="model-title">
                {preset?.title ?? 'Your model, from scratch'}
              </p>
            )}
          </div>
        </div>

        <div className="top-actions">
          {presetKind && (
            <>
              <button
                type="button"
                className="topbar-button compact-panel-toggle"
                aria-pressed={paletteOpen}
                onClick={() => {
                  setPaletteOpen(!paletteOpen)
                  setInspectorOpen(false)
                }}
              >
                Build
              </button>
              <button
                type="button"
                className="topbar-button compact-panel-toggle"
                aria-pressed={inspectorOpen}
                onClick={() => {
                  setInspectorOpen(!inspectorOpen)
                  setPaletteOpen(false)
                }}
              >
                Inspect
              </button>
            </>
          )}
          {onGallery ? (
            <button type="button" className="topbar-button" onClick={onGallery}>
              <BookOpen size={16} /> Preset gallery
            </button>
          ) : null}
          <div className="file-menu">
            <button
              type="button"
              className="topbar-button"
              aria-haspopup="menu"
              aria-expanded={isFileMenuOpen}
              onClick={() => setIsFileMenuOpen((open) => !open)}
            >
              File
              <ChevronDown size={15} />
            </button>
            {isFileMenuOpen ? (
              <div className="file-menu-panel" role="menu" aria-label="File">
                <button
                  type="button"
                  role="menuitem"
                  onClick={() =>
                    runFileMenuAction(() => loadGraph(createEmptyGraph()))
                  }
                >
                  <RotateCcw size={15} />
                  New
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => runFileMenuAction(saveProjectState)}
                >
                  <Download size={15} />
                  Save
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => runFileMenuAction(chooseProjectStateFile)}
                >
                  <Upload size={15} />
                  Import
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() =>
                    runFileMenuAction(() => loadGraph(createStarterGraph(true)))
                  }
                >
                  <BookOpen size={15} />
                  Starter
                </button>
              </div>
            ) : null}
          </div>
          <button
            type="button"
            className={
              showVisualization
                ? 'topbar-button panel-toggle-button is-active'
                : 'topbar-button panel-toggle-button'
            }
            aria-pressed={showVisualization}
            onClick={() => setShowVisualization((visible) => !visible)}
          >
            <Eye size={16} />
            {showVisualization ? 'Hide visualization' : 'Show visualization'}
          </button>
          <button
            type="button"
            className="topbar-button"
            onClick={randomizeParameters}
          >
            <Shuffle size={16} />
            Randomize parameters
          </button>
          <input
            ref={importInputRef}
            type="file"
            accept="application/json,.json"
            aria-label="Import state file"
            style={{ display: 'none' }}
            onChange={importProjectState}
          />
          {importError ? (
            <p className="import-error" role="alert">
              {importError}
            </p>
          ) : null}
        </div>
      </header>

      <aside className="left-panel">
        <section className="panel-section">
          <p className="eyebrow">Build the model</p>
          <p className="palette-intro">
            Pick an operation. Place it. Connect it.
          </p>
          <div className="palette-grid">
            {palette.map((item) => (
              <button
                key={item.type}
                type="button"
                className={`palette-button ${pendingNodeType === item.type ? 'is-selected' : ''}`}
                aria-pressed={pendingNodeType === item.type}
                onClick={() => selectPaletteNode(item.type)}
              >
                <Calculator size={15} />
                {item.label}
              </button>
            ))}
          </div>
          {pendingNodeType ? (
            <p className="placement-hint">
              Click the graph canvas to place {labelForType(pendingNodeType)}.
            </p>
          ) : null}
        </section>
      </aside>

      <GraphCanvas
        graph={graph}
        displayGraph={displayGraph}
        activeStep={canvasStep}
        selectedNodeIds={selectedNodeIds}
        selectedGroupId={selectedGroupId}
        selectedEdgeId={selectedEdgeId}
        onInspectEdge={(edgeId) => {
          setSelectedEdgeId(edgeId)
          setSelectedNodeIds([])
          setSelectedGroupId(undefined)
          setInspectorOpen(true)
          setPaletteOpen(false)
        }}
        showMath={SHOW_MATH_LAYER}
        showGradient={presetKind ? phase === 'backward' : SHOW_GRADIENT_LAYER}
        phase={phase}
        pendingNodeType={pendingNodeType}
        onGraphChange={applyGraphChange}
        onInspectNeuron={inspectNeuron}
        onViewChange={(view: GraphViewState) => {
          if (
            JSON.stringify(graph.view?.expandedGroupIds ?? []) !==
              JSON.stringify(view.expandedGroupIds) ||
            graph.view?.focusedGroupId !== view.focusedGroupId ||
            JSON.stringify(graph.view?.layoutOffsets) !== JSON.stringify(view.layoutOffsets) ||
            JSON.stringify(graph.view?.layoutEdges) !== JSON.stringify(view.layoutEdges)
          )
            pushHistory()
          setGraph((existing) => ({
            ...existing,
            view: {
              ...view,
              expandedGroupIds: view.expandedGroupIds.filter(
                (id) => !id.startsWith('inspect:'),
              ),
              inspectedNeuron:
                existing.view?.inspectedNeuron &&
                view.expandedGroupIds.includes(
                  existing.view.inspectedNeuron.groupId,
                )
                  ? existing.view.inspectedNeuron
                  : undefined,
            },
          }))
        }}
        onSelectionChange={selectCanvasSelection}
        onCreateNode={placePaletteNode}
        onCancelPendingPlacement={clearPendingPlacement}
        onNodeValueChange={updateNodeValue}
        onActivationChange={updateActivation}
        onLossChange={updateLoss}
        onDatasetChange={updateDataset}
        onGroupCreate={mergeSelectedNodes}
        onGroupExplode={explodeGroup}
        onGroupMove={moveGroup}
      />

      <aside className="right-panel">
        {executionError && (
          <div className="graph-issues" role="alert">
            {executionError}
          </div>
        )}
        {heldOutSample && (
          <p className="coordinate-note">
            Held-out example: inspect its prediction and gradients. Parameter
            updates use training examples.
          </p>
        )}
        {(selectedNodeId || selectedGroupId || selectedEdgeId) && (
          <button
            type="button"
            className="inspection-return"
            onClick={() => {
              setSelectedNodeIds([])
              setSelectedGroupId(undefined)
              setSelectedEdgeId(undefined)
            }}
          >
            ← Model controls
          </button>
        )}
        {inspectedEdge && (
          <section className="connection-inspector">
            <p className="eyebrow">Information in motion</p>
            <h2>One connection</h2>
            <p>
              {
                displayGraph.nodes.find(
                  (node) => node.id === inspectedEdge.source,
                )?.label
              }{' '}
              →{' '}
              {
                displayGraph.nodes.find(
                  (node) => node.id === inspectedEdge.target,
                )?.label
              }
            </p>
            <div className="inspector-values">
              <span>
                Forward value →
                <strong>{formatFullTensor(inspectedEdge.value)}</strong>
              </span>
              <span>
                ← Gradient contribution
                <strong>{formatFullTensor(inspectedEdge.grad)}</strong>
              </span>
            </div>
            <p className="coordinate-note">
              The forward value travels to the next operation. During
              backpropagation, this connection returns its contribution to the
              source’s gradient.
            </p>
          </section>
        )}
        {!selectedNodeId &&
          !selectedGroupId &&
          !selectedEdgeId &&
          graph.nodes.some((node) => node.id === 'image-input') && (
            <CnnControls
              graph={graph}
              onGraphChange={(next) => {
                pushHistory()
                setGraph(next)
                setVisualizationGraph(next)
                setPhase('edit')
                setTraceSteps([])
                setTraceIndex(0)
                setCurrentLoss(
                  next.nodes.find(isLossNode)?.value?.data[0] ?? null,
                )
                setIsPlaying(false)
              }}
            />
          )}
        {!selectedNodeId &&
          !selectedGroupId &&
          !selectedEdgeId &&
          graph.groups?.some((group) => group.id === 'network') &&
          graph.nodes.some(node => node.type === 'dataset' && !datasetForNode(node).examples) && (
            <DatasetControls
              graph={graph}
              onGraphChange={(next, epochs = 0, loss) => {
                pushHistory()
                setGraph(next)
                setVisualizationGraph(next)
                setPhase(epochs ? 'update' : 'edit')
                setTraceSteps([])
                setTraceIndex(0)
                setCurrentLoss(loss ?? null)
                setEpoch((value) => value + epochs)
                setIsPlaying(false)
              }}
            />
          )}
        {(!selectedNodeId || graph.nodes.find(node => node.id === selectedNodeId)?.type === 'dataset') &&
          !selectedGroupId &&
          !selectedEdgeId &&
          graph.nodes.some(node => node.type === 'dataset' && datasetForNode(node).task === 'sequence') && (
            <DecoderControls
              graph={graph}
              onGraphChange={(next) => {
                pushHistory()
                setGraph(next)
                setVisualizationGraph(next)
                setPhase('forward')
                setTraceSteps([])
                setTraceIndex(0)
                setCurrentLoss(null)
                setIsPlaying(false)
              }}
              onReloadCheckpoint={['block', 'decoder'].includes(presetKind ?? '') ? (checkpoint) =>
                loadGraph(
                  createModelPreset(
                    presetKind === 'block' ? 'block' : 'decoder',
                    checkpoint,
                  ),
                ) : undefined
              }
            />
          )}
        {!selectedGroupId && !selectedEdgeId && graph.nodes.filter(node => node.type === 'dataset' && (selectedNodeId === node.id || (!selectedNodeId && (!graph.groups?.some(group => group.id === 'network') || Boolean(datasetForNode(node).examples))))).map(node => <DatasetWorkbench key={`${node.id}:${node.params.dataset}`} graph={graph} node={node} onParams={updateNodeParams} onGraphChange={(next, epochs = 0) => {
          pushHistory(); setGraph(next); setVisualizationGraph(next); setPhase(epochs ? 'update' : 'forward'); setTraceSteps([]); setTraceIndex(0); setCurrentLoss(next.nodes.find(isLossNode)?.value?.data[0] ?? null); setEpoch(value => value + epochs); setIsPlaying(false)
        }}/>) }
        <ModelInspector
          graph={graph}
          node={inspectedNode}
          binding={
            selectedNodeId ? projection.bindings[selectedNodeId] : undefined
          }
          group={selectedGroup}
          onParams={updateNodeParams}
          onRename={(id,label) => applyGraphChange({...graph,nodes:graph.nodes.map(node => node.id === id ? {...node,label} : node)})}
          onGroupChange={(id,changes) => applyGraphChange({...graph,groups:graph.groups?.map(group => group.id === id ? {...group,...changes,detail:denseGroupDetail(graph,group)} : group)})}
          onValue={updateNodeValue}
          onDataset={updateDataset}
          onOpen={openGroup}
          onInspectNeuron={inspectNeuron}
          onCopy={copySelectionToClipboard}
          onDuplicate={duplicateSelection}
          onGroup={mergeSelectedNodes}
          selectionCount={selectedNodeIds.length}
        />
        {blockingIssues.length > 0 && (
          <div className="graph-issues" role="status">
            <strong>Complete the connections</strong>
            {blockingIssues.slice(0, 4).map((issue, index) => (
              <p key={index}>{issue.message}</p>
            ))}
          </div>
        )}
        {showVisualization ? (
          <VisualizationPanel graph={visualizationGraph} />
        ) : null}

        <section className="inspector-card">
          <p className="eyebrow">Current step</p>
          <h3>{activeStep?.title ?? 'Ready to evaluate'}</h3>
          <div className="formula-box">
            <strong>Formula</strong>
            <span>
              {activeStep?.formula ??
                (selectedNode
                  ? formulaForNode(selectedNode, graph)
                  : 'Choose a node or press Step.')}
            </span>
          </div>
          <div className="formula-box">
            <strong>Mini calculation</strong>
            <span>
              {activeStep?.calculation ??
                'Numbers will appear here as each node evaluates.'}
            </span>
          </div>
        </section>
      </aside>

      <footer className="control-bar">
        <button
          type="button"
          onClick={() => runCanvasAction(evaluateModel)}
          disabled={blockingIssues.length > 0}
        >
          Run forward
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={() => runCanvasAction(stepForward)}
          disabled={blockingIssues.length > 0}
        >
          <StepForward size={16} />
          Step
        </button>
        <button
          type="button"
          disabled={
            !activeStep ||
            !collapsedGroupForNode(graph, activeStep.nodeId ?? '')
          }
          onClick={() => {
            const group = collapsedGroupForNode(graph, activeStep?.nodeId ?? '')
            if (group) setGraph(setVisualGroupExpanded(graph, group.id, true))
          }}
        >
          Step inside
        </button>
        <button
          type="button"
          disabled={!activeStep}
          onClick={() => {
            let end = traceIndex
            while (
              end + 1 < traceSteps.length &&
              traceSteps[end + 1].phase === phase
            )
              end += 1
            setTraceIndex(end)
            selectSingleNode(traceSteps[end]?.nodeId)
          }}
        >
          Finish phase
        </button>
        <button
          type="button"
          onClick={() => setIsPlaying((playing) => !playing)}
          disabled={blockingIssues.length > 0}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <label className="slider-control">
          Speed
          <input
            type="range"
            min={MIN_PLAY_DELAY_MS}
            max={MAX_PLAY_DELAY_MS}
            step="50"
            value={speedSliderValue}
            onChange={(event) =>
              setSpeedSliderValue(Number(event.target.value))
            }
          />
        </label>
        <button
          type="button"
          onClick={() => runCanvasAction(runOneTrainingStep)}
          disabled={blockingIssues.length > 0 || !hasLoss || heldOutSample}
        >
          <FastForward size={16} />
          Run one full training step
        </button>
        <button
          type="button"
          onClick={() => runCanvasAction(runTenTrainingSteps)}
          disabled={blockingIssues.length > 0 || !hasLoss || heldOutSample}
        >
          Run 10 training steps
        </button>
        <label className="slider-control">
          Learning rate eta ={' '}
          {graph.learningRate.toFixed(graph.learningRate < 0.01 ? 3 : 2)}
          <input
            type="range"
            min="0.001"
            max="0.5"
            step="0.001"
            value={graph.learningRate}
            onChange={(event) => updateLearningRate(Number(event.target.value))}
          />
        </label>
        <div className="metric-pill">Epoch {epoch}</div>
        <div className="metric-pill">
          Current loss {formatNumber(currentLoss ?? undefined)}
        </div>
      </footer>
    </main>
  )
}

function labelForType(type: NodeType): string {
  return palette.find((item) => item.type === type)?.label ?? type
}

function nextNodeIndexForType(graph: GraphModel, type: NodeType): number {
  const prefix = idPrefixForType(type)
  const matcher = new RegExp(`^${prefix}-(\\d+)$`)
  const existingIndexes = graph.nodes
    .map((node) => node.id.match(matcher)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number)
  return Math.max(0, ...existingIndexes) + 1
}

function idPrefixForType(type: NodeType): string {
  if (type === 'input') return 'input'
  if (type === 'weight') return 'weight'
  if (type === 'bias') return 'bias'
  if (type === 'target') return 'target'
  if (type === 'activation') return 'activation'
  return type
}

function cloneParameterValueMap(
  values: Record<string, TensorValue>,
): Record<string, TensorValue> {
  return Object.fromEntries(
    Object.entries(values).map(([label, value]) => [label, cloneTensor(value)]),
  )
}

function cloneTraceSteps(steps: EvaluationTraceStep[]): EvaluationTraceStep[] {
  return steps.map((step) => ({
    ...step,
    edgeIds: [...step.edgeIds],
    pseudocode: [...step.pseudocode],
  }))
}

function stringArraysEqual(first: string[], second: string[]): boolean {
  return (
    first.length === second.length &&
    first.every((value, index) => value === second[index])
  )
}

function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toLowerCase()
  return (
    target.isContentEditable ||
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select'
  )
}

function nextVisibleStepEnd(
  graph: GraphModel,
  steps: EvaluationTraceStep[],
  start: number,
): number {
  let index = start
  const module = collapsedGroupForNode(graph, steps[index]?.nodeId ?? '')
  if (module) {
    while (
      index + 1 < steps.length &&
      steps[index + 1].phase === steps[index].phase &&
      module.nodeIds.includes(steps[index + 1].nodeId ?? '')
    )
      index += 1
  }
  return index
}

function computationSignature(graph: GraphModel): string {
  return JSON.stringify({
    nodes: graph.nodes.map(({ id, type, params }) => ({ id, type, params })),
    edges: graph.edges.map(({ id, source, sourceSlot, target, inputSlot }) => ({
      id,
      source,
      sourceSlot,
      target,
      inputSlot,
    })),
  })
}

function invalidateGraphResults(graph: GraphModel): GraphModel {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => ({
      ...node,
      value: ['input', 'target', 'weight', 'bias', 'dataset'].includes(
        node.type,
      )
        ? node.value
        : undefined,
      grad: undefined,
      localDerivative: undefined,
      cache: undefined,
    })),
    edges: graph.edges.map((edge) => ({
      ...edge,
      value: undefined,
      grad: undefined,
    })),
  }
}

function safeForward(graph: GraphModel): { graph: GraphModel; loss?: number } {
  const issues = validateGraph(graph).filter(
    (issue) => issue.code !== 'disconnected',
  )
  if (issues.length > 0) return { graph: cloneGraph(graph) }
  const result = forwardPass(graph)
  return { graph: result.graph, loss: result.loss }
}

function remapDatasetOutgoingEdges(
  graph: GraphModel,
  nodeId: string,
  dataset: DatasetKind,
): GraphModel['edges'] {
  const existingNode = graph.nodes.find(
    (node) => node.id === nodeId && node.type === 'dataset',
  )
  if (!existingNode) return graph.edges

  const updatedNode = {
    ...existingNode,
    params: { ...existingNode.params, dataset },
  }
  return graph.edges.flatMap((edge) => {
    if (edge.source !== nodeId) return [edge]

    const nextSlot = remapDatasetOutputSlot(
      existingNode,
      updatedNode,
      edge.sourceSlot ?? 0,
    )
    if (nextSlot === undefined) return []

    return [edgeWithSourceSlot(edge, nextSlot)]
  })
}

function edgeWithSourceSlot(
  edge: GraphModel['edges'][number],
  sourceSlot: number,
): GraphModel['edges'][number] {
  return {
    id: edge.id,
    source: edge.source,
    ...(sourceSlot > 0 ? { sourceSlot } : {}),
    target: edge.target,
    inputSlot: edge.inputSlot,
    value: edge.value,
    grad: edge.grad,
  }
}

export default App
