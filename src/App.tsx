import {DEFAULT_TRAINING, type TrainingSettings} from './domain/trainingSettings'
import { mergePreservingLayout, ungroupPreservingLayout } from './domain/mergeLayout'
import { DatasetWorkbench } from './components/DatasetWorkbench'
import { datasetExamplesForNode, datasetForNode, datasetMode } from './domain/datasets'
import { parseCustomCsv } from './domain/customCsv'
import { TextImportDialog } from './components/TextImportDialog'
import type { TextDatasetData } from './domain/types'
import { generatePyTorchExport } from './domain/pytorchExport'
import { denseGroupDetail } from './domain/authoring'
import '@xyflow/react/dist/style.css'
import {
  BookOpen,
  ChevronDown,
  ClipboardPaste,
  Copy,
  CopyPlus,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Download,
  ExternalLink,
  FastForward,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  StepForward,
  Upload,
  Undo2,
} from 'lucide-react'
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactElement,
} from 'react'
import { flushSync } from 'react-dom'
import './App.css'
import './unifiedStudio.css'
import './workspacePanels.css'
import { CodeOutline, type CodeTarget } from './components/CodeOutline'
import { ModelInspector } from './components/ModelInspector'
import { DataInspector } from './components/DataInspector'
import { isHeldOutSample } from './domain/modelDatasets'
import { placeCanvasNode } from './domain/nodePlacement'
import { BlockPalette } from './components/BlockPalette'
import { blockPalette } from './domain/blockPalette'
import { arithmeticInputCount } from './domain/arithmetic'
import { compactVisualHierarchy } from './domain/continuousScene'
import {
  activeProjectionEdges,
  projectDenseNeurons,
} from './domain/neuronProjection'
import { GraphCanvas } from './components/GraphCanvas'
import { VisualizationPanel } from './components/VisualizationPanel'
import { LossReportPanel, type LossReport } from './components/LossReportPanel'
import { InferenceReportPanel } from './components/InferenceReportPanel'
import { evaluateDataset, evaluateDatasetAsync, supportsNumericBatches, trainDataset } from './domain/datasetTraining'
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
import { issueNodeIds, problemNodeIds } from './domain/validationPresentation'
import type {
  ActivationKind,
  CustomCsvData,
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

const CnnControls = lazy(() => import('./components/CnnControls').then(module => ({ default: module.CnnControls })))
const DecoderControls = lazy(() => import('./components/DecoderControls').then(module => ({ default: module.DecoderControls })))
const TextGenerationControls = lazy(() => import('./components/TextGenerationControls').then(module => ({ default: module.TextGenerationControls })))

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
  lossReports: LossReport[]
}

function speedSliderValueToDelay(value: number): number {
  return MIN_PLAY_DELAY_MS + MAX_PLAY_DELAY_MS - value
}

interface AppProps {
  initialGraph?: GraphModel
}

function App({
  initialGraph,
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
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [leftWidth, setLeftWidth] = useState(220)
  const [rightWidth, setRightWidth] = useState(340)
  const [rightTab, setRightTab] = useState<'details' | 'data' | 'code' | 'visualization'>('details')
  const [leftTab, setLeftTab] = useState<'build' | 'train' | 'test'>('build')
  const [inferenceSplit, setInferenceSplit] = useState<'train' | 'test'>('test')
  const [inferenceResult, setInferenceResult] = useState<{ split: 'train' | 'test'; metrics: ReturnType<typeof evaluateDataset> }>()
  const [testStatus, setTestStatus] = useState('')
  const [codeFocus, setCodeFocus] = useState<{ kind: 'group' | 'node'; id: string; serial: number }>()
  const nextCodeFocus = useRef(0)
  const resizeDrag = useRef<{ side: 'left' | 'right'; x: number; width: number } | undefined>(undefined)
  const [executionError, setExecutionError] = useState<string>()
  const [phase, setPhase] = useState<GraphPhase>('edit')
  const [traceSteps, setTraceSteps] = useState<EvaluationTraceStep[]>([])
  const [traceIndex, setTraceIndex] = useState(0)
  const [isFileMenuOpen, setIsFileMenuOpen] = useState(false)
  const [isEditMenuOpen, setIsEditMenuOpen] = useState(false)
  const [isAppMenuOpen, setIsAppMenuOpen] = useState(false)
  const [isAboutOpen, setIsAboutOpen] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const runCanvasAction = useCallback((action: () => void) => {
    setExecutionError(undefined)
    try {
      action()
    } catch (error) {
      setIsPlaying(false)
      setInspectorOpen(true)
      setRightOpen(true)
      setRightTab('details')
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
  const [epochsPerRun, setEpochsPerRun] = useState('10')
  const [reportEvery, setReportEvery] = useState('1')
  const [batchSizeInput, setBatchSizeInput] = useState('')
  const [shuffleEachEpoch, setShuffleEachEpoch] = useState(true)
  const [lossReports, setLossReports] = useState<LossReport[]>([])
  const [reportingWarning, setReportingWarning] = useState<string>()
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState('')
  const trainingController = useRef<AbortController | null>(null)
  const singleReportControllers = useRef(new Set<AbortController>())
  const singleReportQueue = useRef<Promise<void>>(Promise.resolve())
  const operationRevision = useRef(0)
  const cancelActiveRun = useCallback(() => {
    operationRevision.current++
    singleReportControllers.current.forEach(controller => controller.abort())
    singleReportControllers.current.clear()
    trainingController.current?.abort()
    trainingController.current = null
    setIsTraining(false)
    setTrainingStatus('')
    setTestStatus('')
  }, [])
  useEffect(() => () => {
    singleReportControllers.current.forEach(controller => controller.abort())
    singleReportControllers.current.clear()
    trainingController.current?.abort()
    trainingController.current = null
  }, [])
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
  const [exportNotice, setExportNotice] = useState<string | undefined>()
  useEffect(() => {
    if (!exportNotice) return
    const timeout = window.setTimeout(() => setExportNotice(undefined), 6000)
    return () => window.clearTimeout(timeout)
  }, [exportNotice])
  const [textImportNode, setTextImportNode] = useState<string>()
  const [csvPickerOpen, setCsvPickerOpen] = useState(false)
  const importInputRef = useRef<HTMLInputElement | null>(null)
  const customCsvInputRef = useRef<HTMLInputElement | null>(null)
  const pendingCustomCsvNodeId = useRef<string | undefined>(undefined)

  const dismissPendingImports = useCallback(() => {
    operationRevision.current++
    pendingCustomCsvNodeId.current = undefined
    setCsvPickerOpen(false)
    setTextImportNode(undefined)
    setImportError(undefined)
  }, [])

  const clearRecordedExecution = useCallback(() => {
    cancelActiveRun()
    dismissPendingImports()
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setCurrentLoss(null)
    setLossReports([])
    setInferenceResult(undefined)
    setReportingWarning(undefined)
    setExecutionError(undefined)
    setIsPlaying(false)
  }, [cancelActiveRun, dismissPendingImports])

  const validationIssues = useMemo(() => validateGraph(graph), [graph])
  const blockingIssues = useMemo(() => validationIssues.filter(
    (issue) => issue.code !== 'disconnected',
  ), [validationIssues])
  const problemNodes = useMemo(() => problemNodeIds(graph, blockingIssues), [graph, blockingIssues])
  const hasLoss = graph.nodes.some(isLossNode)
  const heldOutSample = isHeldOutSample(graph)
  const activeStep = traceSteps[traceIndex]
  const selectedNodeId =
    !selectedGroupId && selectedNodeIds.length === 1
      ? selectedNodeIds[0]
      : undefined
  const traceGraph = useMemo(
    () => visibleGraphForTrace(graph, traceSteps, traceIndex, phase),
    [graph, phase, traceIndex, traceSteps],
  )
  const projection = useMemo(
    () => projectDenseNeurons(traceGraph),
    [traceGraph],
  )
  const displayGraph = projection.graph
  const codeGraph = useMemo(() => compactVisualHierarchy(displayGraph), [displayGraph])
  const inspectedNode = displayGraph.nodes.find(
    (node) => node.id === selectedNodeId,
  )
  const selectedGroup = graph.groups?.find(
    (group) => group.id === selectedGroupId,
  )
  const selectedIssueNodeIds = new Set([
    ...selectedNodeIds,
    ...(selectedGroup?.nodeIds ?? []),
    ...(selectedNodeId && projection.bindings[selectedNodeId] ? [projection.bindings[selectedNodeId].nodeId] : []),
  ])
  const selectedBlockingIssues = blockingIssues.filter(issue =>
    (issue.edgeId && issue.edgeId === selectedEdgeId)
    || issueNodeIds(graph, issue).some(id => selectedIssueNodeIds.has(id)),
  ).filter(issue => issue.code !== 'invalid-arity' || issue.edgeId || !blockingIssues.some(other => other.code === 'missing-input' && other.nodeId === issue.nodeId))
  const globalBlockingIssues = blockingIssues.filter(issue => issueNodeIds(graph, issue).length === 0)
  const inspectedNeuron = graph.view?.inspectedNeuron
  const selectedCodeGroupId = selectedGroupId && (
    (inspectedNeuron?.groupId === selectedGroupId
      ? codeGraph.groups?.find(group => group.detail?.virtual && group.detail.layerId === selectedGroupId && group.detail.unitIndex === inspectedNeuron.unitIndex)?.id
      : undefined)
    ?? codeGraph.groups?.find(group => group.id === selectedGroupId)?.id
    ?? codeGraph.groups?.find(group => group.nodeIds.length === selectedGroup?.nodeIds.length && group.nodeIds.every(id => selectedGroup?.nodeIds.includes(id)))?.id
  )
  const selectedCodeTarget: CodeTarget | undefined = selectedNodeId ? { kind: 'node', id: selectedNodeId }
    : selectedCodeGroupId ? { kind: 'group', id: selectedCodeGroupId } : undefined
  const canCopySelection = selectedNodeIds.some(id => graph.nodes.some(node => node.id === id))
    || Boolean(selectedGroupId && graph.groups?.some(group => group.id === selectedGroupId))
  const inspectedEdge = displayGraph.edges.find(
    (edge) => edge.id === selectedEdgeId,
  ) ?? graph.edges.find((edge) => edge.id === selectedEdgeId)
  const numericalEdge = graph.edges.find((edge) => edge.id === selectedEdgeId) ?? inspectedEdge
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
      lossReports: [...lossReports],
    }),
    [
      currentLoss,
      lossReports,
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
    cancelActiveRun()
    dismissPendingImports()
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
    setLossReports([...snapshot.lossReports])
    setInferenceResult(undefined)
    setPendingNodeType(undefined)
    setIsPlaying(false)
  }, [cancelActiveRun, dismissPendingImports])

  const undoLastAction = useCallback(() => {
    if (undoStack.length === 0) return
    const snapshot = undoStack[undoStack.length - 1]
    setUndoStack((stack) => stack.slice(0, -1))
    restoreSnapshot(snapshot)
  }, [restoreSnapshot, undoStack])

  const selectSingleNode = useCallback((nodeId?: string) => {
    setCodeFocus(undefined)
    setSelectedNodeIds(nodeId ? [nodeId] : [])
    setSelectedGroupId(undefined)
  }, [])

  const selectCanvasSelection = useCallback(
    (selection: { nodeIds: string[]; groupId?: string }) => {
      setCodeFocus(undefined)
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
      const groupNodeIds = graph.groups?.find(group => group.id === selection.groupId)?.nodeIds ?? []
      if ([...selection.nodeIds, ...groupNodeIds].some(id => problemNodes.has(projection.bindings[id]?.nodeId ?? id))) {
        setRightOpen(true)
        setRightTab(current => current === 'data' ? current : 'details')
      }
    },
    [graph.groups, problemNodes, projection.bindings],
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
    clearRecordedExecution()
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
  }, [clipboard, clipboardPasteCount, graph, pushHistory, clearRecordedExecution])

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
      clearRecordedExecution()
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
      setLossReports([])
      setTrainingStatus('')
      setInferenceResult(undefined)
      setTestStatus('')
      setPendingNodeType(undefined)
    },
    [pushHistory, selectSingleNode, clearRecordedExecution],
  )

  const reportTrainingLoss = useCallback(async (sourceGraph: GraphModel, reportEpoch: number, controller: AbortController) => {
    const isCurrent = () => trainingController.current === controller || singleReportControllers.current.has(controller)
    const dataset = sourceGraph.nodes.find(node => node.type === 'dataset')
    const hasHeldOut = dataset && datasetExamplesForNode(dataset).some(example => example.split === 'test')
    const options = { signal: controller.signal, includePredictions: false }
    const loss = dataset ? (await evaluateDatasetAsync(sourceGraph, dataset.id, 'train', options)).loss : forwardPass(sourceGraph).loss
    controller.signal.throwIfAborted()
    if (loss === null || loss === undefined || !Number.isFinite(loss)) throw new Error('Training diverged. Lower the learning rate and try again.')
    let heldOutLoss: number | undefined
    if (dataset && hasHeldOut) {
      try { heldOutLoss = (await evaluateDatasetAsync(sourceGraph, dataset.id, 'test', options)).loss }
      catch (error) {
        controller.signal.throwIfAborted()
        setReportingWarning(`Held-out loss unavailable: ${error instanceof Error ? error.message : 'evaluation failed.'}`)
      }
      if (heldOutLoss !== undefined && !Number.isFinite(heldOutLoss)) {
        setReportingWarning('Held-out loss diverged. Reduce the learning rate or inspect the model inputs.')
        heldOutLoss = undefined
      } else if (heldOutLoss !== undefined) setReportingWarning(undefined)
    }
    controller.signal.throwIfAborted()
    if (isCurrent()) setLossReports(reports => appendLossReport(reports, { epoch: reportEpoch, loss, heldOutLoss }))
    return loss
  }, [])

  const recordSingleTrainingStep = useCallback((before: GraphModel, after: GraphModel) => {
    const controller = new AbortController()
    singleReportControllers.current.add(controller)
    const isCurrent = () => singleReportControllers.current.has(controller)
    // Keep rapid manual/Play updates in order without delaying the next step.
    singleReportQueue.current = singleReportQueue.current.then(async () => {
      try {
        controller.signal.throwIfAborted()
        setReportingWarning(undefined)
        if (!lossReports.some(report => report.epoch === epoch)) await reportTrainingLoss(before, epoch, controller)
        await reportTrainingLoss(after, epoch + 1, controller)
      } catch (error) {
        if (isCurrent() && !controller.signal.aborted)
          setReportingWarning(error instanceof Error ? error.message : 'Loss evaluation failed.')
      } finally {
        singleReportControllers.current.delete(controller)
      }
    })
  }, [epoch, lossReports, reportTrainingLoss])

  const stepForward = useCallback(() => {
    if (blockingIssues.length > 0 || trainingController.current) return
    setInferenceResult(undefined)

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
      const forward = forwardPass(graph, true, {training:hasLoss && !heldOutSample})
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
      const updateSummary = summarizeUpdateSteps(updated.steps)
      setTraceSteps([updateSummary, ...(refreshed.steps ?? [])])
      setTraceIndex(0)
      setPhase('update')
      setEpoch((value) => value + 1)
      setCurrentLoss(refreshed.loss ?? null)
      void recordSingleTrainingStep(graph, refreshed.graph)
      selectSingleNode(updateSummary.nodeId)
    }
  }, [
    blockingIssues,
    graph,
    hasLoss,
    heldOutSample,
    recordSingleTrainingStep,
    phase,
    pushHistory,
    selectSingleNode,
    traceIndex,
    traceSteps,
  ])

  useEffect(() => {
    if (!isPlaying || isTraining) return
    const timeout = window.setTimeout(
      () => runCanvasAction(stepForward),
      speedSliderValueToDelay(speedSliderValue),
    )
    return () => window.clearTimeout(timeout)
  }, [isPlaying, isTraining, speedSliderValue, stepForward, runCanvasAction])

  const runOneTrainingStep = useCallback(() => {
    if (blockingIssues.length > 0 || !hasLoss || trainingController.current) return
    if (heldOutSample) return
    setInferenceResult(undefined)
    pushHistory()
    const result = runTrainingStep(graph, graph.learningRate)
    const updateSummary = summarizeUpdateSteps(result.steps.filter((step) => step.phase === 'update'))
    const nextForward = forwardPass(result.graph)
    setGraph(result.graph)
    setVisualizationGraph(result.graph)
    setTraceSteps([updateSummary, ...nextForward.steps])
    setTraceIndex(0)
    setPhase('update')
    setEpoch((value) => value + 1)
    setCurrentLoss(result.loss ?? null)
    void recordSingleTrainingStep(graph, result.graph)
    selectSingleNode(updateSummary.nodeId)
  }, [
    blockingIssues,
    graph,
    hasLoss,
    heldOutSample,
    recordSingleTrainingStep,
    pushHistory,
    selectSingleNode,
  ])

  const epochCount = Number(epochsPerRun)
  const reportInterval = Number(reportEvery)
  const trainingDataset = graph.nodes.find(node => node.type === 'dataset')
  const trainingExampleCount = trainingDataset ? datasetExamplesForNode(trainingDataset).filter(example => example.split === 'train').length : 0
  const training = graph.training ?? DEFAULT_TRAINING
  const tensorTraining = training.engine === 'tensor' && !!trainingDataset
  const setTraining = (changes:Partial<TrainingSettings>) => {
    cancelActiveRun()
    setGraph(existing=>({...existing,training:{...(existing.training??DEFAULT_TRAINING),...changes}}))
  }
  const defaultBatchSize = trainingDataset && datasetMode(trainingDataset) === 'batch' ? trainingExampleCount : 1
  const batchSize = batchSizeInput === '' ? defaultBatchSize : Number(batchSizeInput)
  const validBatchSize = !trainingDataset || (Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= trainingExampleCount
    && (tensorTraining || batchSize === 1 || supportsNumericBatches(trainingDataset)))
  const validRunSettings = Number.isInteger(epochCount) && epochCount >= 1 && epochCount <= 100000
    && Number.isInteger(reportInterval) && reportInterval >= 1 && reportInterval <= 100000 && validBatchSize
  const canRunEpochs = validRunSettings && blockingIssues.length === 0 && hasLoss && !isTraining
    && (!heldOutSample || graph.nodes.some(node => node.type === 'dataset'))

  const runEpochs = useCallback(async () => {
    if (!canRunEpochs || trainingController.current) return
    dismissPendingImports()
    const controller = new AbortController()
    trainingController.current = controller
    const isCurrent = () => trainingController.current === controller
    setIsTraining(true)
    setIsPlaying(false)
    setTrainingStatus(`Training 0 / ${epochCount} epochs`)
    setReportingWarning(undefined)
    setInferenceResult(undefined)
    pushHistory()
    let nextGraph = graph, completed = 0, lastReported = 0
    const startEpoch = epoch
    const dataset = trainingDataset
    const reportLosses = (sourceGraph: GraphModel, reportEpoch: number) => reportTrainingLoss(sourceGraph, reportEpoch, controller)
    const publish = async () => {
      const loss = await reportLosses(nextGraph, startEpoch + completed)
      if (!isCurrent()) return
      setGraph(existing => mergeExecutionGraph(existing, nextGraph))
      setVisualizationGraph(nextGraph)
      setInferenceResult(undefined)
      setTraceSteps([])
      setTraceIndex(0)
      setPhase('update')
      setEpoch(startEpoch + completed)
      setCurrentLoss(loss)
      lastReported = completed
    }
    try {
      await singleReportQueue.current
      if (!isCurrent()) return
      controller.signal.throwIfAborted()
      if (tensorTraining && dataset) {
        const {trainTensorGraph} = await import('./domain/tensorTraining')
        if (!isCurrent()) return
        controller.signal.throwIfAborted()
        const result = await trainTensorGraph(graph, {
          epochs:epochCount,batchSize,settings:training,signal:controller.signal,epochOffset:startEpoch,shuffle:shuffleEachEpoch,
          onBackend:(backend,fallback)=>{if(!isCurrent())return;setTrainingStatus('Training on '+backend);if(fallback)setReportingWarning('Using '+backend+'. '+fallback)},
          onProgress:(done,total)=>{if(isCurrent())setTrainingStatus('Training '+done+' / '+total+' examples')},
          onReport:report=>{if(isCurrent())setLossReports(reports=>appendLossReport(reports,{epoch:report.epoch,loss:report.train.loss,heldOutLoss:report.validation.loss}))},
        })
        if (!isCurrent()) return
        const evaluated=forwardPass(result.graph)
        setGraph(existing=>mergeExecutionGraph(existing,evaluated.graph));setVisualizationGraph(evaluated.graph);setCurrentLoss(evaluated.loss??null)
        setInferenceResult(undefined);setTraceSteps([]);setTraceIndex(0);setPhase('update')
        setEpoch(startEpoch+(training.patience>0||result.stopped?result.bestEpoch:result.completed))
        setTrainingStatus((result.stopped?'Stopped':'Completed')+' after '+result.completed+' epochs on '+result.backend+'. '+(training.patience>0||result.stopped?'Restored best validation checkpoint at epoch '+(startEpoch+result.bestEpoch)+'.':'Kept final parameters.'))
        return
      }
      if (dataset) await reportLosses(graph, startEpoch)
      else {
        const initialLoss = forwardPass(graph).loss
        if (initialLoss !== undefined && Number.isFinite(initialLoss))
          setLossReports(reports => appendLossReport(reports, { epoch: startEpoch, loss: initialLoss }))
      }
      for (let index = 0; index < epochCount; index++) {
        controller.signal.throwIfAborted()
        if (dataset) {
          nextGraph = await trainDataset(nextGraph, dataset.id, 1, { signal: controller.signal, epochOffset: startEpoch + index, batchSize, shuffleEachEpoch })
        } else {
          const result = runTrainingStep(nextGraph)
          nextGraph = result.graph
          if (index % 8 === 7) await new Promise(resolve => setTimeout(resolve, 0))
        }
        if (!isCurrent()) return
        completed++
        setTrainingStatus(`Training ${completed} / ${epochCount} epochs`)
        if (completed % reportInterval === 0 || completed === epochCount) await publish()
      }
      if (completed > lastReported) await publish()
      if (!isCurrent()) return
      setTrainingStatus(controller.signal.aborted ? `Stopped after ${completed} ${completed === 1 ? 'epoch' : 'epochs'}.` : `Completed ${completed} ${completed === 1 ? 'epoch' : 'epochs'}.`)
    } catch (error) {
      if (!isCurrent()) return
      if (completed > lastReported) {
        if (controller.signal.aborted) {
          // Retain completed updates without starting another loss evaluation
          // after Stop. Partial epochs are discarded by trainDataset.
          setGraph(existing => mergeExecutionGraph(existing, nextGraph))
          setVisualizationGraph(nextGraph)
          setEpoch(startEpoch + completed)
          setCurrentLoss(null)
          setTraceSteps([])
          setTraceIndex(0)
          setPhase('update')
        } else {
          try { await publish() } catch { /* retain the last finite report */ }
        }
      }
      if (!isCurrent()) return
      const message = controller.signal.aborted ? `Stopped after ${completed} ${completed === 1 ? 'epoch' : 'epochs'}.` : error instanceof Error ? error.message : 'Training failed.'
      setTrainingStatus(message)
      if (!controller.signal.aborted) {
        setExecutionError(message)
        setRightOpen(true)
        setRightTab('details')
      }
    } finally {
      if (isCurrent()) {
        trainingController.current = null
        setIsTraining(false)
      }
    }
  }, [batchSize, canRunEpochs, reportTrainingLoss, dismissPendingImports, epoch, epochCount, graph, pushHistory, reportInterval, shuffleEachEpoch, trainingDataset, tensorTraining, training])

  const runInference = useCallback(async () => {
    const dataset = graph.nodes.find(node => node.type === 'dataset')
    if (!dataset || blockingIssues.length > 0 || trainingController.current) return
    setInferenceResult(undefined)
    setTestStatus('')
    const examples = datasetExamplesForNode(dataset)
    const first = examples.findIndex(example => example.split === inferenceSplit)
    if (first < 0) { setTestStatus(`This dataset has no ${inferenceSplit} examples.`); return }
    dismissPendingImports()
    const source = { ...dataset, params: { ...dataset.params, datasetSplit: inferenceSplit, datasetIndex: first, datasetValues: undefined } }
    const testGraph = { ...graph, nodes: graph.nodes.map(node => node.id === dataset.id ? source : node) }
    const controller = new AbortController()
    trainingController.current = controller
    const isCurrent = () => trainingController.current === controller
    setIsTraining(true)
    setIsPlaying(false)
    setTestStatus('Evaluating examples…')
    try {
      const inferred = forwardPass(testGraph).graph
      let metrics: ReturnType<typeof evaluateDataset>
      if (training.engine === 'tensor') {
        const { selectTensorBackend, TensorGraph } = await import('./domain/tensorTraining')
        if (!isCurrent()) return
        controller.signal.throwIfAborted()
        await selectTensorBackend(testGraph, training.backend, training)
        if (!isCurrent()) return
        controller.signal.throwIfAborted()
        const model = new TensorGraph(testGraph)
        try { metrics = await model.inference(model.examples.filter(row => row.split === inferenceSplit), 64, controller.signal) }
        finally { model.dispose() }
      } else metrics = await evaluateDatasetAsync(inferred, dataset.id, inferenceSplit, { signal: controller.signal })
      if (!isCurrent()) return
      controller.signal.throwIfAborted()
      pushHistory()
      setGraph(existing => mergeExecutionGraph(existing, inferred))
      setVisualizationGraph(inferred)
      setCurrentLoss(metrics.loss)
      setTraceSteps([])
      setTraceIndex(0)
      setPhase('forward')
      setInferenceResult({ split: inferenceSplit, metrics })
      setTestStatus('')
      setExecutionError(undefined)
    } catch (error) {
      if (!isCurrent()) return
      if (controller.signal.aborted) { setTestStatus('Evaluation stopped.'); return }
      const message = error instanceof Error ? error.message : 'Inference failed.'
      setTestStatus(message)
      setExecutionError(message)
      setRightOpen(true)
      setRightTab('details')
    }
    finally {
      if (isCurrent()) {
        trainingController.current = null
        setIsTraining(false)
      }
    }
  }, [blockingIssues.length, dismissPendingImports, graph, inferenceSplit, pushHistory, training])

  const openTrainTab = () => {
    setLeftTab('train')
    if (!heldOutSample) return
    const dataset = graph.nodes.find(node => node.type === 'dataset')
    if (!dataset) return
    if (datasetMode(dataset) === 'batch' && dataset.params.datasetSplit !== 'test') return
    const firstTrainingExample = datasetExamplesForNode(dataset).findIndex(example => example.split === 'train')
    if (firstTrainingExample < 0) return
    cancelActiveRun()
    pushHistory()
    const trainingGraph = {
      ...graph,
      nodes: graph.nodes.map(node => node.id === dataset.id
        ? { ...node, params: { ...node.params, datasetSplit: 'train' as const, datasetIndex: firstTrainingExample, datasetValues: undefined } }
        : node),
    }
    const evaluated = safeForward(trainingGraph)
    setGraph(evaluated.graph)
    setVisualizationGraph(evaluated.graph)
    setTraceSteps([])
    setTraceIndex(0)
    setPhase('edit')
    setCurrentLoss(evaluated.loss ?? null)
    setIsPlaying(false)
  }

  useEffect(() => {
    const handleRunShortcut = (event: KeyboardEvent) => {
      if (event.defaultPrevented || !event.shiftKey || event.metaKey || event.ctrlKey || event.altKey || isEditableShortcutTarget(event.target)) return
      if (event.code === 'Space' || event.key === ' ') {
        event.preventDefault()
        if (blockingIssues.length === 0 && !isTraining) runCanvasAction(stepForward)
      } else if (event.key === 'Enter') {
        event.preventDefault()
        void runEpochs()
      }
    }
    document.addEventListener('keydown', handleRunShortcut)
    return () => document.removeEventListener('keydown', handleRunShortcut)
  }, [blockingIssues.length, isTraining, runCanvasAction, runEpochs, stepForward])

  const selectPaletteNode = (type: NodeType) => {
    setPendingNodeType(type)
    setPhase('edit')
    setTraceSteps([])
  }

  const placePaletteNode = useCallback(
    (type: NodeType, position: { x: number; y: number }, parentGroupId?: string, sceneScale?: number) => {
      pushHistory()
      clearRecordedExecution()
      const nextNode = createNode(type, nextNodeIndexForType(graph, type))
      const placedNode = { ...nextNode, position }
      setGraph(placeCanvasNode(graph, placedNode, displayGraph, parentGroupId, sceneScale))
      selectSingleNode(placedNode.id)
      setPendingNodeType(undefined)
      setPhase('edit')
    },
    [graph, displayGraph, pushHistory, selectSingleNode, clearRecordedExecution],
  )

  const clearPendingPlacement = useCallback(() => {
    setPendingNodeType(undefined)
  }, [])

  const updateNodeValue = useCallback(
    (nodeId: string, value: TensorValue) => {
      const binding = projection.bindings[nodeId]
      if (binding) {
        if (!binding.editable || value.data.length !== 1) return
        const canonical = graph.nodes.find(node => node.id === binding.nodeId)
        if (!canonical) return
        const original = toTensor(canonical.params.value)
        nodeId = canonical.id
        value = { ...original, data: original.data.map((entry, index) => index === binding.index ? value.data[0] : entry) }
      }
      pushHistory()
      clearRecordedExecution()
      const next = invalidateGraphResults({
        ...graph,
        nodes: graph.nodes.map(node => node.id === nodeId ? { ...node, params: { ...node.params, value }, value } : node),
      })
      setGraph(next)
      // Parameter curves follow the last completed forward pass. Direct input
      // changes still update the experiment's displayed domain immediately.
      setVisualizationGraph(existing => ({
        ...existing,
        nodes: existing.nodes.map(node => node.id === nodeId && (node.type === 'input' || node.type === 'target')
          ? { ...node, params: { ...node.params, value }, value } : node),
      }))
    },
    [graph, projection.bindings, pushHistory, clearRecordedExecution],
  )

  const updateActivation = useCallback(
    (nodeId: string, activation: ActivationKind) => {
      pushHistory()
      clearRecordedExecution()
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
    [pushHistory, clearRecordedExecution],
  )

  const updateLoss = useCallback(
    (nodeId: string, loss: LossKind) => {
      pushHistory()
      clearRecordedExecution()
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
    [pushHistory, clearRecordedExecution],
  )

  const applyDatasetSelection = useCallback(
    (nodeId: string, dataset: DatasetKind, customCsv?: CustomCsvData, textData?: TextDatasetData) => {
      pushHistory()
      clearRecordedExecution()
      const updateNode = (node: GraphModel['nodes'][number]) => {
        if (node.id !== nodeId || node.type !== 'dataset') return node
        const updated = { ...node, params: { ...node.params, dataset, customCsv, textData, trainPercent: undefined, datasetIndex: dataset === 'custom-csv' ? 1 : 0, datasetMode: dataset === 'custom-csv' ? 'batch' as const : node.params.datasetMode, datasetSplit: dataset === 'custom-csv' ? 'train' as const : node.params.datasetSplit, datasetValues: undefined } }
        const value = datasetOutputValueForSlot(updated, 0)
        return { ...updated, value, grad: zeroLike(value) }
      }
      setGraph((existing) =>
        invalidateGraphResults({
          ...existing,
          nodes: existing.nodes.map(updateNode),
          edges: remapDatasetOutgoingEdges(existing, nodeId, dataset, customCsv, textData),
        }),
      )
      setVisualizationGraph((existing) => ({
        ...existing,
        nodes: existing.nodes.map(updateNode),
        edges: remapDatasetOutgoingEdges(existing, nodeId, dataset, customCsv, textData),
      }))
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setCurrentLoss(null)
      setIsPlaying(false)
    },
    [pushHistory, clearRecordedExecution],
  )

  const updateDataset = useCallback((nodeId: string, dataset: DatasetKind) => {
    if (dataset === 'custom-text' || dataset === 'custom-csv') {
      cancelActiveRun()
      dismissPendingImports()
    }
    if (dataset === 'custom-text') { setTextImportNode(nodeId); return }
    if (dataset === 'custom-csv') {
      pendingCustomCsvNodeId.current = nodeId
      flushSync(() => {
        setImportError(undefined)
        setCsvPickerOpen(true)
      })
      customCsvInputRef.current?.focus()
      customCsvInputRef.current?.click()
      return
    }
    applyDatasetSelection(nodeId, dataset)
  }, [applyDatasetSelection, cancelActiveRun, dismissPendingImports])

  const importCustomCsv = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    const nodeId = pendingCustomCsvNodeId.current
    if (!file || !nodeId) return
    cancelActiveRun()
    const revision = operationRevision.current
    try {
      const csv = parseCustomCsv(await file.text(), file.name)
      if (revision !== operationRevision.current) return
      applyDatasetSelection(nodeId, 'custom-csv', csv)
      pendingCustomCsvNodeId.current = undefined
      setCsvPickerOpen(false)
      setImportError(undefined)
    } catch (error) {
      if (revision !== operationRevision.current) return
      setImportError(error instanceof Error ? error.message : 'Could not read the CSV file.')
    }
  }

  const updateLearningRate = (learningRate: number) => {
    pushHistory()
    cancelActiveRun()
    setGraph((existing) => ({ ...existing, learningRate }))
  }

  const randomizeParameters = () => {
    pushHistory()
    clearRecordedExecution()
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setCurrentLoss(null)
    setLossReports([])
    setInferenceResult(undefined)
    setTrainingStatus('')
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

  const updateExpression = useCallback((nodeId: string, expression: string) => {
    const inputCount = arithmeticInputCount(expression)
    applyGraphChange({
      ...graph,
      nodes: graph.nodes.map(node => node.id === nodeId ? { ...node, params: { ...node.params, expression } } : node),
      edges: graph.edges.filter(edge => edge.target !== nodeId || (edge.inputSlot ?? 0) < inputCount),
    })
  }, [applyGraphChange, graph])

  const mergeSelectedNodes = useCallback(() => {
    const result = mergePreservingLayout(graph, selectedNodeIds, displayGraph)
    if (!result.group) return

    pushHistory()
    setGraph(result.graph)
    setSelectedNodeIds([])
    setSelectedGroupId(result.group.id)
    setPendingNodeType(undefined)
  }, [graph, displayGraph, pushHistory, selectedNodeIds])

  const explodeGroup = useCallback(
    (groupId: string) => {
      const group = graph.groups?.find((candidate) => candidate.id === groupId)
      if (!group) return

      pushHistory()
      setGraph(ungroupPreservingLayout(graph, groupId))
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

  const navigateFromCode = (target: CodeTarget) => {
    setSelectedEdgeId(undefined)
    if (target.kind === 'group') {
      const group = codeGraph.groups?.find(item => item.id === target.id)
      if (group?.detail?.virtual && typeof group.detail.layerId === 'string' && typeof group.detail.unitIndex === 'number') {
        inspectNeuron(group.detail.layerId, group.detail.unitIndex)
      } else if (graph.groups?.some(item => item.id === target.id)) {
        openGroup(target.id)
      }
    } else {
      setSelectedNodeIds([target.id])
      setSelectedGroupId(undefined)
    }
    setCodeFocus({ ...target, serial: ++nextCodeFocus.current })
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
    setInferenceResult(undefined)
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
    setExportNotice(undefined)
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
          showVisualization: rightTab === 'visualization',
        },
      }),
    )
  }

  const exportPyTorch = (format: 'notebook' | 'python') => {
    setImportError(undefined)
    setExportNotice(undefined)
    try {
      if (tensorTraining) throw new Error('Tensor training settings are not exported yet. Save the project to keep its weights and settings, or select Trace engine and batch size 1 to export a single-example SGD program.')
      const exported = generatePyTorchExport(graph, { batchSize: batchSizeInput === '' ? undefined : Number(batchSizeInput), shuffleEachEpoch, epochs: epochCount, reportEvery: reportInterval })
      const download = (content: string, name: string, type: string) => {
        const url = URL.createObjectURL(new Blob([content], { type }))
        const anchor = document.createElement('a')
        anchor.href = url
        anchor.download = name
        anchor.click()
        window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      }
      const name = format === 'notebook' ? 'neural-canvas-model.ipynb' : 'neural-canvas-model.py'
      download(format === 'notebook' ? exported.notebook : exported.script, name,
        format === 'notebook' ? 'application/x-ipynb+json' : 'text/x-python')
      if (exported.datasetFile) download(exported.datasetFile.content, exported.datasetFile.name, 'application/json')
      setExportNotice(exported.datasetFile
        ? `${name} and ${exported.datasetFile.name} downloaded. Keep them in the same folder.`
        : `${name} downloaded. Keep the source CSV beside it.`)
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'Could not export this model to PyTorch.')
    }
  }

  const chooseProjectStateFile = () => {
    dismissPendingImports()
    importInputRef.current?.click()
  }

  const runFileMenuAction = (action: () => void) => {
    setIsFileMenuOpen(false)
    action()
  }

  const runEditMenuAction = (action: () => void) => {
    setIsEditMenuOpen(false)
    action()
  }

  const closeAbout = () => {
    setIsAboutOpen(false)
    window.setTimeout(() => document.getElementById('app-menu-trigger')?.focus(), 0)
  }

  useEffect(() => {
    const closeOnPointerDown = (event: PointerEvent) => {
      if (event.target instanceof Element && event.target.closest('.topbar-menu')) return
      setIsFileMenuOpen(false)
      setIsEditMenuOpen(false)
      setIsAppMenuOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      setIsFileMenuOpen(false)
      setIsEditMenuOpen(false)
      setIsAppMenuOpen(false)
    }
    document.addEventListener('pointerdown', closeOnPointerDown)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnPointerDown)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [])

  const importProjectState = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) return
    cancelActiveRun()
    dismissPendingImports()
    const revision = operationRevision.current
    let contents: string
    try { contents = await file.text() }
    catch (error) {
      if (revision === operationRevision.current) setImportError(error instanceof Error ? error.message : 'Could not read the project file.')
      return
    }
    if (revision !== operationRevision.current) return
    const result = parseProjectStateFile(contents)
    if (!result.ok) {
      setImportError(result.error)
      return
    }

    const nextState = result.file.state
    pushHistory()
    clearRecordedExecution()
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
    setLossReports([])
    setTrainingStatus('')
    setInferenceResult(undefined)
    setTestStatus('')
    setRightTab(nextState.display.showVisualization ? 'visualization' : 'details')
    setPendingNodeType(undefined)
    setIsPlaying(false)
    setImportError(undefined)
  }

  const resizeLimit = (side: 'left' | 'right') => side === 'left' ? 430 : 620
  const handleResizeStart = (side: 'left' | 'right', event: ReactPointerEvent<HTMLDivElement>) => {
    resizeDrag.current = { side, x: event.clientX, width: side === 'left' ? leftWidth : rightWidth }
    event.currentTarget.setPointerCapture?.(event.pointerId)
    event.preventDefault()
  }
  const handleResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = resizeDrag.current
    if (!drag) return
    const delta = (event.clientX - drag.x) * (drag.side === 'left' ? 1 : -1)
    const width = Math.max(drag.side === 'left' ? 126 : 230, Math.min(resizeLimit(drag.side), drag.width + delta))
    if (drag.side === 'left') setLeftWidth(width)
    else setRightWidth(width)
  }
  const handleResizeKey = (side: 'left' | 'right', event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const minimum = side === 'left' ? 126 : 230
    const current = side === 'left' ? leftWidth : rightWidth
    const delta = event.key === 'Home' ? minimum - current : event.key === 'End' ? resizeLimit(side) - current
      : (event.key === 'ArrowRight' ? 16 : -16) * (side === 'left' ? 1 : -1)
    const value = Math.max(minimum, Math.min(resizeLimit(side), current + delta))
    if (side === 'left') setLeftWidth(value)
    else setRightWidth(value)
  }

  return (
    <main
      className={`app-shell workspace-split unified-studio theme-dark ${inspectorOpen ? 'inspector-open' : ''} ${paletteOpen ? 'palette-open' : ''} ${leftOpen ? '' : 'left-collapsed'} ${rightOpen ? '' : 'right-collapsed'}`}
      style={{ '--left-size': leftOpen ? `${leftWidth}px` : '42px', '--right-size': rightOpen ? `${rightWidth}px` : '42px' } as CSSProperties}
    >
      <header className="top-bar">
        <div className="top-brand topbar-menu">
          <div className="brand-mark" aria-hidden="true">NC</div>
          <h1><button
            id="app-menu-trigger"
            type="button"
            className="brand-menu-button"
            aria-haspopup="menu"
            aria-expanded={isAppMenuOpen}
            onClick={() => { setIsFileMenuOpen(false); setIsEditMenuOpen(false); setIsAppMenuOpen(open => !open) }}
          >Neural Canvas <ChevronDown size={15} aria-hidden="true" /></button></h1>
          {isAppMenuOpen ? <div className="topbar-menu-panel app-menu-panel" role="menu" aria-label="Neural Canvas">
            <button type="button" role="menuitem" onClick={() => { setIsAppMenuOpen(false); setIsAboutOpen(true) }}>About</button>
            <a role="menuitem" href="https://github.com/davbachman/NeuralCanvas#readme" target="_blank" rel="noopener noreferrer" onClick={() => setIsAppMenuOpen(false)}>Reference <ExternalLink size={14} aria-hidden="true" /></a>
          </div> : null}
        </div>

        <div className="top-actions">
            <>
              <button
                type="button"
                className="topbar-button compact-panel-toggle"
                aria-pressed={paletteOpen}
                onClick={() => {
                  setLeftOpen(true)
                  setPaletteOpen(!paletteOpen)
                  setInspectorOpen(false)
                }}
              >
                Controls
              </button>
              <button
                type="button"
                className="topbar-button compact-panel-toggle"
                aria-pressed={inspectorOpen}
                onClick={() => {
                  setRightOpen(true)
                  setInspectorOpen(!inspectorOpen)
                  setPaletteOpen(false)
                }}
              >
                Inspect
              </button>
            </>
          <div className="topbar-menu">
            <button
              type="button"
              className="topbar-menu-trigger"
              aria-haspopup="menu"
              aria-expanded={isFileMenuOpen}
              onClick={() => { setIsEditMenuOpen(false); setIsAppMenuOpen(false); setIsFileMenuOpen((open) => !open) }}
            >
              File
              <ChevronDown size={15} />
            </button>
            {isFileMenuOpen ? (
              <div className="topbar-menu-panel" role="menu" aria-label="File">
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
                <button type="button" role="menuitem" onClick={() => runFileMenuAction(() => exportPyTorch('notebook'))}>
                  <Download size={15} /> Export PyTorch notebook
                </button>
                <button type="button" role="menuitem" onClick={() => runFileMenuAction(() => exportPyTorch('python'))}>
                  <Download size={15} /> Export Python file
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
          <div className="topbar-menu">
            <button
              type="button"
              className="topbar-menu-trigger"
              aria-haspopup="menu"
              aria-expanded={isEditMenuOpen}
              onClick={() => { setIsFileMenuOpen(false); setIsAppMenuOpen(false); setIsEditMenuOpen((open) => !open) }}
            >
              Edit
              <ChevronDown size={15} />
            </button>
            {isEditMenuOpen ? (
              <div className="topbar-menu-panel" role="menu" aria-label="Edit">
                <button type="button" role="menuitem" aria-label="Undo" aria-keyshortcuts="Meta+Z Control+Z" disabled={undoStack.length === 0} onClick={() => runEditMenuAction(undoLastAction)}>
                  <Undo2 size={15} /> Undo <kbd aria-hidden="true">⌘/Ctrl Z</kbd>
                </button>
                <button type="button" role="menuitem" aria-label="Copy" aria-keyshortcuts="Meta+C Control+C" disabled={!canCopySelection} onClick={() => runEditMenuAction(copySelectionToClipboard)}>
                  <Copy size={15} /> Copy <kbd aria-hidden="true">⌘/Ctrl C</kbd>
                </button>
                <button type="button" role="menuitem" aria-label="Paste" aria-keyshortcuts="Meta+V Control+V" disabled={!clipboard} onClick={() => runEditMenuAction(pasteClipboard)}>
                  <ClipboardPaste size={15} /> Paste <kbd aria-hidden="true">⌘/Ctrl V</kbd>
                </button>
                <button type="button" role="menuitem" aria-label="Duplicate" disabled={!canCopySelection} onClick={() => runEditMenuAction(duplicateSelection)}>
                  <CopyPlus size={15} /> Duplicate
                </button>
              </div>
            ) : null}
          </div>
          <input
            ref={importInputRef}
            type="file"
            accept="application/json,.json"
            aria-label="Import state file"
            style={{ display: 'none' }}
            onChange={importProjectState}
          />
          {importError && !csvPickerOpen ? (
            <p className="import-error" role="alert">
              {importError}
            </p>
          ) : null}
          {exportNotice ? <p className="export-notice" role="status">{exportNotice}</p> : null}
        </div>
      </header>

      {isAboutOpen ? <div className="about-backdrop" onPointerDown={event => { if (event.target === event.currentTarget) closeAbout() }}>
        <section role="dialog" aria-modal="true" aria-labelledby="about-title" aria-describedby="about-description" className="about-dialog" onKeyDown={event => {
          if (event.key === 'Escape') { event.preventDefault(); closeAbout() }
          if (event.key !== 'Tab') return
          const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('a[href], button:not([disabled])')]
          const first = controls[0], last = controls[controls.length - 1]
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
        }}>
          <div className="about-dialog-mark" aria-hidden="true">NC</div>
          <h2 id="about-title">About Neural Canvas</h2>
          <p id="about-description">Created by David Bachman with Codex. Build and explore machine-learning models from individual calculations through neural networks, attention, and transformers.</p>
          <p>Learn more about <a href="https://pzacad.pitzer.edu/~dbachman/" target="_blank" rel="noopener noreferrer">David Bachman</a> and his AI podcast, <a href="https://profbachman.substack.com/" target="_blank" rel="noopener noreferrer"><em>Entropy Bonus</em></a>.</p>
          <button type="button" autoFocus onClick={closeAbout}>Close</button>
        </section>
      </div> : null}

      {csvPickerOpen && <div className="csv-picker-backdrop">
        <section role="dialog" aria-modal="true" aria-labelledby="csv-picker-title" className="csv-picker-dialog" onKeyDown={event => { if (event.key === 'Escape') dismissPendingImports() }}>
          <p className="eyebrow">Dataset source</p>
          <h2 id="csv-picker-title">Choose a CSV file</h2>
          <p>The CSV stays in this browser and becomes part of your saved project.</p>
          <input ref={customCsvInputRef} type="file" accept=".csv,text/csv" aria-label="Choose custom CSV file" onChange={importCustomCsv} />
          {importError && <p role="alert" className="csv-picker-error">{importError}</p>}
          <button type="button" onClick={dismissPendingImports}>Cancel</button>
        </section>
      </div>}

      {textImportNode && <TextImportDialog onCancel={dismissPendingImports} onImport={data => { applyDatasetSelection(textImportNode, 'custom-text', undefined, data); setTextImportNode(undefined) }}/>}
      <aside className="left-panel" aria-label="Build, train, and test sidebar">
        <div className="sidebar-heading">
          {leftOpen ? <div className="left-sidebar-tabs" role="tablist" aria-label="Left sidebar views">
            <button type="button" role="tab" aria-selected={leftTab === 'build'} onClick={() => setLeftTab('build')}>Build</button>
            <button type="button" role="tab" aria-selected={leftTab === 'train'} onClick={openTrainTab}>Train</button>
            <button type="button" role="tab" aria-selected={leftTab === 'test'} onClick={() => setLeftTab('test')}>Test</button>
          </div> : null}
          <button type="button" aria-label={leftOpen ? 'Collapse left sidebar' : 'Expand left sidebar'} title={leftOpen ? 'Collapse sidebar' : 'Expand sidebar'} onClick={() => { if (leftOpen) setPaletteOpen(false); setLeftOpen(value => !value) }}>
            {leftOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </button>
        </div>
        <section className="panel-section" role="tabpanel" aria-label="Build blocks" hidden={leftTab !== 'build'}>
          <p className="eyebrow">Build the model</p>
          <p className="palette-intro">
            Pick an operation. Place it. Connect it.
          </p>
          <BlockPalette selected={pendingNodeType ?? undefined} onSelect={selectPaletteNode} />
          {pendingNodeType ? (
            <p className="placement-hint">
              Click the graph canvas to place {labelForType(pendingNodeType)}.
            </p>
          ) : null}
        </section>
        <section className="panel-section run-panel" role="tabpanel" aria-label="Train controls" hidden={leftTab !== 'train'}>
          <p className="eyebrow">Trace the computation</p>
          <div className="run-button-grid">
            <button type="button" onClick={() => runCanvasAction(evaluateModel)} disabled={blockingIssues.length > 0 || isTraining}>Run forward</button>
            <button type="button" className="primary-button" aria-keyshortcuts="Shift+Space" onClick={() => runCanvasAction(stepForward)} disabled={blockingIssues.length > 0 || isTraining}><StepForward size={15} /> Step <kbd aria-hidden="true">⇧ Space</kbd></button>
            <button type="button" disabled={!activeStep || !collapsedGroupForNode(graph, activeStep.nodeId ?? '') || isTraining} onClick={() => {
              const group = collapsedGroupForNode(graph, activeStep?.nodeId ?? '')
              if (group) setGraph(setVisualGroupExpanded(graph, group.id, true))
            }}>Step inside</button>
            <button type="button" disabled={!activeStep || isTraining} onClick={() => {
              let end = traceIndex
              while (end + 1 < traceSteps.length && traceSteps[end + 1].phase === phase) end += 1
              setTraceIndex(end)
              selectSingleNode(traceSteps[end]?.nodeId)
            }}>Finish phase</button>
            <button type="button" onClick={() => setIsPlaying(playing => !playing)} disabled={blockingIssues.length > 0 || isTraining}>{isPlaying ? <Pause size={15} /> : <Play size={15} />}{isPlaying ? 'Pause' : 'Play'}</button>
          </div>
          <label className="run-field">Playback speed<input type="range" min={MIN_PLAY_DELAY_MS} max={MAX_PLAY_DELAY_MS} step="50" value={speedSliderValue} onChange={event => setSpeedSliderValue(Number(event.target.value))} /></label>
          <div className="run-section-divider" />
          <p className="eyebrow">Train the model</p>
          <button type="button" className="run-full-step randomize-button" onClick={randomizeParameters} disabled={isTraining}><Shuffle size={15} /> Randomize parameters</button>
          <button type="button" className="run-full-step" onClick={() => runCanvasAction(runOneTrainingStep)} disabled={blockingIssues.length > 0 || !hasLoss || heldOutSample || isTraining}><FastForward size={15} /> Run one full training step</button>
          {trainingDataset && <>
            <label className="run-field">Training execution<select aria-label="Training execution" disabled={isTraining} value={training.engine} onChange={event=>setTraining({engine:event.target.value as TrainingSettings['engine']})}><option value="trace">Trace engine · SGD</option><option value="tensor">Tensor engine · minibatches</option></select></label>
            {tensorTraining && <>
              <label className="run-field">Backend<select aria-label="Training backend" disabled={isTraining} value={training.backend} onChange={event=>setTraining({backend:event.target.value as TrainingSettings['backend']})}><option value="auto">Auto · WebGL, then CPU</option><option value="webgl">WebGL</option><option value="webgpu">WebGPU (experimental)</option><option value="cpu">Tensor CPU</option></select></label>
              <label className="run-field">Optimizer<select aria-label="Optimizer" disabled={isTraining} value={training.optimizer} onChange={event=>setTraining({optimizer:event.target.value as TrainingSettings['optimizer']})}><option value="sgd">SGD</option><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
              {training.optimizer==='adamw' && <label className="run-field">Weight decay<input aria-label="Weight decay" disabled={isTraining} type="number" min="0" step="0.001" value={training.weightDecay} onChange={event=>setTraining({weightDecay:Number(event.target.value)})}/></label>}
              <label className="run-field">Gradient norm limit (0 = off)<input aria-label="Gradient norm limit" disabled={isTraining} type="number" min="0" step="0.1" value={training.clipNorm} onChange={event=>setTraining({clipNorm:Number(event.target.value)})}/></label>
              <label className="run-field">Early stopping patience (0 = off)<input aria-label="Early stopping patience" disabled={isTraining} type="number" min="0" step="1" value={training.patience} onChange={event=>setTraining({patience:Number(event.target.value)})}/></label>
              <label className="run-field">Minimum validation improvement<input aria-label="Minimum validation improvement" disabled={isTraining} type="number" min="0" step="0.001" value={training.minDelta} onChange={event=>setTraining({minDelta:Number(event.target.value)})}/></label>
              <p className="run-intro">Tensor runs use padded batches and report every epoch. Early stopping uses the held-out split as validation and restores its best checkpoint. Adam moments start fresh for each run. Step remains a single-example SGD calculation.</p>
            </>}
          </>}
          <div className="run-number-grid">
            <label className="run-field">Epochs per run<input type="number" min="1" max="100000" step="1" value={epochsPerRun} onChange={event => setEpochsPerRun(event.target.value)} /></label>
            <label className="run-field">Report loss every<input type="number" min="1" max="100000" step="1" value={reportEvery} disabled={tensorTraining || isTraining} onChange={event => setReportEvery(event.target.value)} /><span>epochs</span></label>
          </div>
          {trainingDataset ? <>
            <label className="run-field">Examples per update<input type="number" min="1" max={trainingExampleCount} step="1" value={batchSizeInput} placeholder={String(defaultBatchSize)} onChange={event => setBatchSizeInput(event.target.value)} disabled={isTraining} /></label>
            <p className="run-intro">Batch size {validBatchSize ? batchSize : '—'} of {trainingExampleCount} training examples for Run epochs. Leave blank to use the Dataset’s current output mode; Step follows the canvas example.</p>
            <label className="run-field run-checkbox"><input type="checkbox" checked={shuffleEachEpoch} onChange={event => setShuffleEachEpoch(event.target.checked)} disabled={isTraining} /> Reshuffle training examples each epoch</label>
            {!validBatchSize && <p className="run-status" role="alert">{batchSize > 1 && !supportsNumericBatches(trainingDataset) ? 'This tensor-shaped dataset currently trains one example at a time. A larger batch needs a graph built with a batch dimension.' : `Choose a batch size from 1 to ${trainingExampleCount}.`}</p>}
          </> : null}
          {isTraining ? <button type="button" className="run-epochs-button" onClick={() => trainingController.current?.abort()}>Stop training</button>
            : <button type="button" className="run-epochs-button primary-button" aria-keyshortcuts="Shift+Enter" onClick={() => void runEpochs()} disabled={!canRunEpochs}>Run {validRunSettings ? epochCount : '—'} {epochCount === 1 ? 'epoch' : 'epochs'} <kbd aria-hidden="true">⇧ Return</kbd></button>}
          {tensorTraining && <label className="run-field">Exact learning rate<input aria-label="Exact learning rate" type="number" min="0.000001" step="0.0001" disabled={isTraining} value={graph.learningRate} onChange={event=>updateLearningRate(Number(event.target.value))}/></label>}
          <label className="run-field">Learning rate · η = {graph.learningRate.toString()}<input type="range" min={tensorTraining ? "0.0001" : "0.001"} max="0.5" step={tensorTraining ? "0.0001" : "0.001"} value={graph.learningRate} onChange={event => updateLearningRate(Number(event.target.value))} disabled={isTraining} /></label>
          <div className="run-metrics"><span>Epoch {epoch}</span><span>Current loss {formatNumber(currentLoss ?? undefined)}</span></div>
          {trainingStatus && <p className="run-status" role="status">{trainingStatus}</p>}
        </section>
        <section className="panel-section run-panel test-panel" role="tabpanel" aria-label="Test controls" hidden={leftTab !== 'test'}>
          <p className="eyebrow">Inference</p>
          <p className="run-intro">Run the trained model on examples without changing its parameters.</p>
          <label className="run-field">Examples to evaluate<select aria-label="Inference examples" value={inferenceSplit} onChange={event => setInferenceSplit(event.target.value as 'train' | 'test')}><option value="test">Held-out test set</option><option value="train">Training set</option></select></label>
          <button type="button" className="run-epochs-button primary-button" onClick={runInference} disabled={blockingIssues.length > 0 || isTraining || !graph.nodes.some(node => node.type === 'dataset')}>Run inference</button>
          {testStatus && <p className="run-status" role="status">{testStatus}</p>}
          {inferenceResult && <InferenceReportPanel metrics={inferenceResult.metrics} split={inferenceResult.split} task={graph.nodes.find(node => node.type === 'dataset') ? datasetForNode(graph.nodes.find(node => node.type === 'dataset')!).task : undefined} />}
        </section>
      </aside>

      <div className={`sidebar-splitter sidebar-splitter-left ${leftOpen ? '' : 'is-collapsed'}`} role="separator" aria-label="Resize left sidebar" aria-orientation="vertical" aria-valuemin={126} aria-valuemax={430} aria-valuenow={leftWidth} tabIndex={leftOpen ? 0 : -1}
        onPointerDown={event => { if (leftOpen) handleResizeStart('left', event) }} onPointerMove={handleResizeMove}
        onPointerUp={event => { resizeDrag.current = undefined; event.currentTarget.releasePointerCapture?.(event.pointerId) }} onLostPointerCapture={() => { resizeDrag.current = undefined }}
        onKeyDown={event => handleResizeKey('left', event)} onDoubleClick={() => setLeftWidth(220)} />

      <GraphCanvas
        graph={graph}
        displayGraph={displayGraph}
        problemNodeIds={problemNodes}
        activeStep={canvasStep}
        selectedNodeIds={selectedNodeIds}
        selectedGroupId={selectedGroupId}
        selectedEdgeId={selectedEdgeId}
        focusRequest={codeFocus}
        onInspectEdge={(edgeId) => {
          setSelectedEdgeId(edgeId)
          setSelectedNodeIds([])
          setSelectedGroupId(undefined)
          setInspectorOpen(true)
          setRightOpen(true)
          setRightTab('data')
          setRightWidth(width => Math.max(width, 390))
          setPaletteOpen(false)
        }}
        showMath={SHOW_MATH_LAYER}
        showGradient={SHOW_GRADIENT_LAYER}
        phase={phase}
        pendingNodeType={pendingNodeType}
        onGraphChange={applyGraphChange}
        onInspectNeuron={inspectNeuron}
        onViewChange={(view: GraphViewState) => {
          if (
            JSON.stringify(graph.view?.expandedGroupIds ?? []) !==
              JSON.stringify(view.expandedGroupIds) ||
            graph.view?.focusedGroupId !== view.focusedGroupId ||
            graph.view?.canvasStyle !== view.canvasStyle ||
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
        onExpressionChange={updateExpression}
        onTransformChange={(nodeId, transform) => updateNodeParams(nodeId, { transform })}
        onLossChange={updateLoss}
        onDatasetChange={updateDataset}
        onGroupCreate={mergeSelectedNodes}
        onGroupExplode={explodeGroup}
        onGroupRename={(id, label) => applyGraphChange({ ...graph, groups: graph.groups?.map(group => group.id === id ? { ...group, label } : group) })}
        onGroupMove={moveGroup}
      />

      <div className={`sidebar-splitter sidebar-splitter-right ${rightOpen ? '' : 'is-collapsed'}`} role="separator" aria-label="Resize right sidebar" aria-orientation="vertical" aria-valuemin={230} aria-valuemax={620} aria-valuenow={rightWidth} tabIndex={rightOpen ? 0 : -1}
        onPointerDown={event => { if (rightOpen) handleResizeStart('right', event) }} onPointerMove={handleResizeMove}
        onPointerUp={event => { resizeDrag.current = undefined; event.currentTarget.releasePointerCapture?.(event.pointerId) }} onLostPointerCapture={() => { resizeDrag.current = undefined }}
        onKeyDown={event => handleResizeKey('right', event)} onDoubleClick={() => setRightWidth(340)} />

      <aside className="right-panel" aria-label="Model sidebar">
        <div className="sidebar-heading right-sidebar-heading">
          {rightOpen ? <div className="right-sidebar-tabs" role="tablist" aria-label="Right sidebar views">
            <button type="button" role="tab" aria-selected={rightTab === 'details'} onClick={() => setRightTab('details')}>Details</button>
            <button type="button" role="tab" aria-selected={rightTab === 'data'} onClick={() => { setRightTab('data'); setRightWidth(width => Math.max(width, 390)) }}>Data</button>
            <button type="button" role="tab" aria-selected={rightTab === 'code'} onClick={() => { setRightTab('code'); setRightWidth(width => Math.max(width, 390)) }}>Code</button>
            <button type="button" role="tab" aria-selected={rightTab === 'visualization'} onClick={() => { setRightTab('visualization'); setRightWidth(width => Math.max(width, 390)) }}>Reporting</button>
          </div> : null}
          <button type="button" aria-label={rightOpen ? 'Collapse right sidebar' : 'Expand right sidebar'} title={rightOpen ? 'Collapse sidebar' : 'Expand sidebar'} onClick={() => { if (rightOpen) setInspectorOpen(false); setRightOpen(value => !value) }}>
            {rightOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
          </button>
        </div>
        <div className="right-panel-scroll" hidden={rightTab !== 'details'} role="tabpanel" aria-label="Model details">
        {executionError && (
          <div className="graph-issues" role="alert">
            {executionError}
          </div>
        )}
        {globalBlockingIssues.length > 0 && <section className="graph-issues" role="alert" aria-label="Model errors">
          <strong>Model issue</strong>
          {globalBlockingIssues.map((issue, index) => <p key={`${issue.code}-${index}`}>{issue.message}</p>)}
        </section>}
        {selectedBlockingIssues.length > 0 && <section className="graph-issues node-issues" role="alert" aria-label="Selected block errors">
          <strong>{selectedGroup ? 'Problem inside this block' : 'Fix this block'}</strong>
          {selectedBlockingIssues.map((issue, index) => <p key={`${issue.code}-${issue.nodeId ?? issue.edgeId}-${index}`}>{issue.message}</p>)}
        </section>}
        {problemNodes.size > 0 && !selectedNodeId && !selectedGroupId && !selectedEdgeId && <p className="validation-hint">Select a red block to see what needs fixing.</p>}
        {heldOutSample && (
          <p className="coordinate-note">
            Held-out example: inspect its prediction and gradients. Parameter
            updates use training examples.
          </p>
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
                <strong>{formatFullTensor(numericalEdge?.value)}</strong>
              </span>
              <span>
                ← Gradient contribution
                <strong>{formatFullTensor(numericalEdge?.grad)}</strong>
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
            <Suspense fallback={<p role="status">Loading digit controls…</p>}>
            <CnnControls
              graph={graph}
              onGraphChange={(next) => {
                pushHistory()
                clearRecordedExecution()
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
            </Suspense>
          )}
        {(!selectedNodeId || graph.nodes.find(node => node.id === selectedNodeId)?.type === 'dataset') &&
          !selectedGroupId &&
          !selectedEdgeId &&
          graph.nodes.some(node => node.type === 'dataset' && datasetForNode(node).task === 'sequence') && (
            <Suspense fallback={<p role="status">Loading generation controls…</p>}>
            {graph.nodes.some(node => node.params.textData?.task === 'language') ? <TextGenerationControls graph={graph}/> : <DecoderControls
              graph={graph}
              onGraphChange={(next) => {
                pushHistory()
                clearRecordedExecution()
                setGraph(next)
                setVisualizationGraph(next)
                setPhase('forward')
                setTraceSteps([])
                setTraceIndex(0)
                setCurrentLoss(null)
                setIsPlaying(false)
              }}
            />}
            </Suspense>
          )}
        {!selectedGroupId && !selectedEdgeId && graph.nodes.filter(node => node.type === 'dataset' && (selectedNodeId === node.id || (!selectedNodeId && (!graph.groups?.some(group => group.id === 'network') || Boolean(datasetForNode(node).examples))))).map(node => <DatasetWorkbench key={`${node.id}:${node.params.dataset}`} graph={graph} node={node} onParams={updateNodeParams} onDataset={updateDataset} onRename={selectedNodeId === node.id ? (id,label) => applyGraphChange({...graph,nodes:graph.nodes.map(candidate => candidate.id === id ? {...candidate,label} : candidate)}) : undefined} onChooseCustomCsv={id => updateDataset(id, 'custom-csv')} />) }
        {(selectedGroup || (selectedNodeIds.length > 0 && inspectedNode?.type !== 'dataset')) && <ModelInspector
          graph={graph}
          node={inspectedNode}
          binding={
            selectedNodeId ? projection.bindings[selectedNodeId] : undefined
          }
          group={selectedGroup}
          onParams={updateNodeParams}
          onRename={(id,label) => applyGraphChange({...graph,nodes:graph.nodes.map(node => node.id === id ? {...node,label} : node)})}
          onGroupChange={(id,changes) => applyGraphChange({...graph,groups:graph.groups?.map(group => group.id === id ? {...group,...changes,detail:denseGroupDetail(graph,{...group,...changes})} : group)})}
          onValue={updateNodeValue}
          onOpen={openGroup}
          onInspectNeuron={inspectNeuron}
          onGroup={mergeSelectedNodes}
          selectionCount={selectedNodeIds.length}
        />}
        {(activeStep || (!selectedNodeId && !selectedGroupId && !selectedEdgeId)) && <section className="inspector-card">
          <p className="eyebrow">Current step</p>
          <h3>{activeStep?.title ?? 'Ready to evaluate'}</h3>
          {activeStep && <div className="formula-box">
            {activeStep.phase === 'backward' && <>
              <strong>Derivative</strong>
              <span>{activeStep.formula}</span>
            </>}
            <span>{activeStep.calculation}</span>
          </div>}
        </section>}
        </div>
        <div className="right-panel-scroll right-data-scroll" hidden={rightTab !== 'data'} role="tabpanel" aria-label="Model data">
          {rightTab === 'data' ? <DataInspector key={selectedEdgeId ?? selectedGroupId ?? selectedNodeId ?? 'empty'} graph={graph} node={inspectedNode} edge={numericalEdge} group={selectedGroup} /> : null}
        </div>
        <div className="right-panel-scroll right-code-scroll" hidden={rightTab !== 'code'} role="tabpanel" aria-label="Model code">
          {rightTab === 'code' ? <CodeOutline graph={displayGraph} selected={selectedCodeTarget} active={rightOpen} onNavigate={navigateFromCode} /> : null}
        </div>
        <div className="right-panel-scroll right-visualization-scroll" hidden={rightTab !== 'visualization'} role="tabpanel" aria-label="Model reporting">
          {rightTab === 'visualization' ? <><VisualizationPanel graph={visualizationGraph} /><LossReportPanel reports={lossReports} warning={reportingWarning} /></> : null}
        </div>
      </aside>
    </main>
  )
}

function labelForType(type: NodeType): string {
  return blockPalette.find((item) => item.type === type)?.label ?? type
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

function appendLossReport(reports: LossReport[], next: LossReport): LossReport[] {
  // A restored checkpoint can rewind the epoch counter. Replace its old future
  // when a new run begins so the chart never joins incompatible trajectories.
  return [...reports.filter(report => report.epoch < next.epoch), next]
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

/** Parameter updates are applied together. Present them as one visible step so
 * the next press starts a fresh forward pass instead of replaying stale writes. */
function summarizeUpdateSteps(steps: EvaluationTraceStep[]): EvaluationTraceStep {
  const count = steps.length
  const shown = steps.slice(0, 3).map(step => step.calculation)
  return {
    id: 'update-parameters',
    phase: 'update',
    nodeId: steps[0]?.nodeId,
    edgeIds: [],
    title: count === 0 ? 'No trainable parameters to update' : `Update ${count} ${count === 1 ? 'parameter' : 'parameters'}`,
    explanation: count === 0
      ? 'Add and connect a Param block to make this model trainable.'
      : 'Gradient descent updates all trainable parameters together. The next Step begins another forward pass.',
    formula: 'parameter = parameter - learning_rate * gradient',
    calculation: count === 0 ? 'No weights or biases are connected.' : `${shown.join(' · ')}${count > shown.length ? ` · ${count - shown.length} more` : ''}`,
    pseudocode: ['for parameter in trainable_parameters:', '  parameter -= learning_rate * parameter.grad', '  parameter.grad = 0'],
  }
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

/** A run owns numerical state, while navigation and layout remain editable. */
function mergeExecutionGraph(current: GraphModel, evaluated: GraphModel): GraphModel {
  const nodes = new Map(evaluated.nodes.map(node => [node.id, node]))
  const edges = new Map(evaluated.edges.map(edge => [edge.id, edge]))
  return {
    ...current,
    nodes: current.nodes.map(node => {
      const result = nodes.get(node.id)
      return result ? { ...result, label: node.label, position: node.position, dimensions: node.dimensions } : node
    }),
    edges: current.edges.map(edge => ({ ...edge, value: edges.get(edge.id)?.value, grad: edges.get(edge.id)?.grad })),
  }
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

function safeForward(graph: GraphModel): { graph: GraphModel; loss?: number; steps?: EvaluationTraceStep[] } {
  const issues = validateGraph(graph).filter(
    (issue) => issue.code !== 'disconnected',
  )
  if (issues.length > 0) return { graph: cloneGraph(graph) }
  const result = forwardPass(graph)
  return { graph: result.graph, loss: result.loss, steps: result.steps }
}

function remapDatasetOutgoingEdges(
  graph: GraphModel,
  nodeId: string,
  dataset: DatasetKind,
  customCsv?: CustomCsvData,
  textData?: TextDatasetData,
): GraphModel['edges'] {
  const existingNode = graph.nodes.find(
    (node) => node.id === nodeId && node.type === 'dataset',
  )
  if (!existingNode) return graph.edges

  const updatedNode = {
    ...existingNode,
    params: { ...existingNode.params, dataset, customCsv, textData },
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
