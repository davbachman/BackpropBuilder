import '@xyflow/react/dist/style.css'
import {
  BookOpen,
  Calculator,
  Download,
  FastForward,
  Eye,
  GraduationCap,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  StepForward,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState, type ReactElement } from 'react'
import './App.css'
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
  formatNumber,
  forwardPass,
  formulaForNode,
  parameterValues,
  runTrainingStep,
  updateParameters,
  validateGraph,
} from './domain/engine'
import { createEmptyGraph, createNode, createStarterGraph } from './domain/examples'
import { explodeVisualGroup, mergeNodesIntoVisualGroup, moveVisualGroup } from './domain/grouping'
import { lessons } from './domain/lessons'
import { createSessionSummary, downloadSessionSummary } from './domain/session'
import { cloneTensor, scalarValue, toTensor, zeroLike } from './domain/tensor'
import { visibleGraphForTrace } from './domain/traceVisibility'
import type {
  ActivationKind,
  EvaluationTraceStep,
  GraphModel,
  GraphPhase,
  LessonDefinition,
  LossKind,
  NodeType,
  TensorValue,
} from './domain/types'

const palette: Array<{ type: NodeType; label: string }> = [
  { type: 'input', label: 'Input' },
  { type: 'weight', label: 'Weight' },
  { type: 'bias', label: 'Bias' },
  { type: 'multiply', label: 'Multiply' },
  { type: 'add', label: 'Add' },
  { type: 'activation', label: 'Activation' },
  { type: 'target', label: 'Target' },
  { type: 'loss', label: 'Loss' },
]

const MIN_PLAY_DELAY_MS = 250
const MAX_PLAY_DELAY_MS = 1800
const DEFAULT_PLAY_DELAY_MS = 900
const DEFAULT_SPEED_SLIDER_VALUE = MIN_PLAY_DELAY_MS + MAX_PLAY_DELAY_MS - DEFAULT_PLAY_DELAY_MS
const HISTORY_LIMIT = 100
const PASTE_OFFSET_STEP = 36

interface HistorySnapshot {
  graph: GraphModel
  initialParams: Record<string, TensorValue>
  selectedNodeIds: string[]
  selectedGroupId?: string
  phase: GraphPhase
  traceSteps: EvaluationTraceStep[]
  traceIndex: number
  epoch: number
  currentLoss: number | null
  currentLesson?: LessonDefinition
  completedActions: string[]
}

function speedSliderValueToDelay(value: number): number {
  return MIN_PLAY_DELAY_MS + MAX_PLAY_DELAY_MS - value
}

function App(): ReactElement {
  const [graph, setGraph] = useState<GraphModel>(() => createEmptyGraph())
  const [initialParams, setInitialParams] = useState<Record<string, TensorValue>>({})
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([])
  const [selectedGroupId, setSelectedGroupId] = useState<string | undefined>()
  const [phase, setPhase] = useState<GraphPhase>('edit')
  const [traceSteps, setTraceSteps] = useState<EvaluationTraceStep[]>([])
  const [traceIndex, setTraceIndex] = useState(0)
  const [showMath, setShowMath] = useState(true)
  const [showGradient, setShowGradient] = useState(true)
  const [showCode, setShowCode] = useState(false)
  const [showVisualization, setShowVisualization] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speedSliderValue, setSpeedSliderValue] = useState(DEFAULT_SPEED_SLIDER_VALUE)
  const [epoch, setEpoch] = useState(0)
  const [currentLoss, setCurrentLoss] = useState<number | null>(null)
  const [lessonOpen, setLessonOpen] = useState(false)
  const [currentLesson, setCurrentLesson] = useState<LessonDefinition | undefined>()
  const [completedActions, setCompletedActions] = useState<string[]>([])
  const [pendingNodeType, setPendingNodeType] = useState<NodeType | undefined>()
  const [undoStack, setUndoStack] = useState<HistorySnapshot[]>([])
  const [clipboard, setClipboard] = useState<GraphClipboardFragment | undefined>()
  const [clipboardPasteCount, setClipboardPasteCount] = useState(0)

  const validationIssues = useMemo(() => validateGraph(graph), [graph])
  const blockingIssues = validationIssues.filter((issue) => issue.code !== 'disconnected')
  const activeStep = traceSteps[traceIndex]
  const selectedNodeId = !selectedGroupId && selectedNodeIds.length === 1 ? selectedNodeIds[0] : undefined
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId)
  const displayGraph = useMemo(
    () => visibleGraphForTrace(graph, traceSteps, traceIndex, phase),
    [graph, phase, traceIndex, traceSteps],
  )

  const appendAction = useCallback((action: string) => {
    setCompletedActions((actions) => [...actions, action])
  }, [])

  const snapshotCurrentState = useCallback(
    (): HistorySnapshot => ({
      graph: cloneGraph(graph),
      initialParams: cloneParameterValueMap(initialParams),
      selectedNodeIds: [...selectedNodeIds],
      selectedGroupId,
      phase,
      traceSteps: cloneTraceSteps(traceSteps),
      traceIndex,
      epoch,
      currentLoss,
      currentLesson,
      completedActions: [...completedActions],
    }),
    [
      completedActions,
      currentLesson,
      currentLoss,
      epoch,
      graph,
      initialParams,
      phase,
      selectedGroupId,
      selectedNodeIds,
      traceIndex,
      traceSteps,
    ],
  )

  const pushHistory = useCallback(() => {
    const snapshot = snapshotCurrentState()
    setUndoStack((stack) => [...stack, snapshot].slice(-HISTORY_LIMIT))
  }, [snapshotCurrentState])

  const restoreSnapshot = useCallback((snapshot: HistorySnapshot) => {
    setGraph(cloneGraph(snapshot.graph))
    setInitialParams(cloneParameterValueMap(snapshot.initialParams))
    setSelectedNodeIds([...snapshot.selectedNodeIds])
    setSelectedGroupId(snapshot.selectedGroupId)
    setPhase(snapshot.phase)
    setTraceSteps(cloneTraceSteps(snapshot.traceSteps))
    setTraceIndex(snapshot.traceIndex)
    setEpoch(snapshot.epoch)
    setCurrentLoss(snapshot.currentLoss)
    setCurrentLesson(snapshot.currentLesson)
    setCompletedActions([...snapshot.completedActions])
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

  const selectCanvasSelection = useCallback((selection: { nodeIds: string[]; groupId?: string }) => {
    setSelectedNodeIds((existing) => (stringArraysEqual(existing, selection.nodeIds) ? existing : selection.nodeIds))
    setSelectedGroupId((existing) => (existing === selection.groupId ? existing : selection.groupId))
    if (selection.nodeIds.length > 0 || selection.groupId) setPendingNodeType(undefined)
  }, [])

  const copySelectionToClipboard = useCallback((): boolean => {
    const fragment = copyGraphSelection(graph, { nodeIds: selectedNodeIds, groupId: selectedGroupId })
    if (!fragment) return false

    setClipboard(fragment)
    setClipboardPasteCount(0)
    return true
  }, [graph, selectedGroupId, selectedNodeIds])

  const pasteClipboard = useCallback((): boolean => {
    if (!clipboard) return false

    pushHistory()
    const offset = PASTE_OFFSET_STEP * (clipboardPasteCount + 1)
    const result = pasteGraphClipboard(graph, clipboard, { x: offset, y: offset })
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
      if (event.defaultPrevented || isEditableShortcutTarget(event.target)) return
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
  }, [copySelectionToClipboard, pasteClipboard, undoLastAction, undoStack.length])

  const loadGraph = useCallback((nextGraph: GraphModel, lesson?: LessonDefinition) => {
    pushHistory()
    const evaluated = safeForward(nextGraph)
    setGraph(evaluated.graph)
    setInitialParams(parameterValues(nextGraph))
    selectSingleNode(evaluated.graph.nodes.find((node) => node.type === 'activation')?.id)
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
    setEpoch(0)
    setCurrentLoss(evaluated.loss ?? null)
    setCurrentLesson(lesson)
    setCompletedActions(lesson ? [`Started ${lesson.name}`] : [])
    setPendingNodeType(undefined)
  }, [pushHistory, selectSingleNode])

  const stepForward = useCallback(() => {
    if (blockingIssues.length > 0) return

    if (traceSteps.length > 0 && traceIndex < traceSteps.length - 1) {
      pushHistory()
      const nextIndex = traceIndex + 1
      setTraceIndex(nextIndex)
      setPhase(traceSteps[nextIndex].phase)
      selectSingleNode(traceSteps[nextIndex].nodeId)
      return
    }

    if (phase === 'edit' || phase === 'update') {
      pushHistory()
      const forward = forwardPass(graph)
      setGraph(forward.graph)
      setTraceSteps(forward.steps)
      setTraceIndex(0)
      setPhase(forward.steps[0]?.phase ?? 'forward')
      selectSingleNode(forward.steps[0]?.nodeId)
      setCurrentLoss(forward.loss ?? null)
      appendAction('Started forward pass')
      return
    }

    if (phase === 'forward' || phase === 'loss') {
      pushHistory()
      const backward = backwardPass(graph)
      setGraph(backward.graph)
      setTraceSteps(backward.steps)
      setTraceIndex(0)
      setPhase('backward')
      selectSingleNode(backward.steps[0]?.nodeId)
      appendAction('Started backward pass')
      return
    }

    if (phase === 'backward') {
      pushHistory()
      const updated = updateParameters(graph, graph.learningRate)
      const refreshed = safeForward(updated.graph)
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
      appendAction('Updated parameters')
    }
  }, [appendAction, blockingIssues.length, graph, phase, pushHistory, selectSingleNode, traceIndex, traceSteps])

  useEffect(() => {
    if (!isPlaying) return
    const timeout = window.setTimeout(stepForward, speedSliderValueToDelay(speedSliderValue))
    return () => window.clearTimeout(timeout)
  }, [isPlaying, speedSliderValue, stepForward])

  const runOneTrainingStep = useCallback(() => {
    if (blockingIssues.length > 0) return
    pushHistory()
    const result = runTrainingStep(graph, graph.learningRate)
    const updateSteps = result.steps.filter((step) => step.phase === 'update')
    setGraph(result.graph)
    setTraceSteps(updateSteps)
    setTraceIndex(0)
    setPhase('update')
    setEpoch((value) => value + 1)
    setCurrentLoss(result.loss ?? null)
    selectSingleNode(updateSteps[0]?.nodeId)
    appendAction('Ran one full training step')
  }, [appendAction, blockingIssues.length, graph, pushHistory, selectSingleNode])

  const runTenTrainingSteps = useCallback(() => {
    if (blockingIssues.length > 0) return
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
      explanation: 'Each step recomputed the loss, backpropagated gradients, and updated the trainable parameters.',
      formula: 'parameter = parameter - learning_rate * gradient',
      calculation: `loss ${formatNumber(startingLoss ?? undefined)} -> ${formatNumber(loss ?? undefined)}`,
      pseudocode: ['for step in range(10):', '  loss = forward()', '  loss.backward()', '  update_parameters()'],
    }
    setGraph(nextGraph)
    setTraceSteps([summaryStep])
    setTraceIndex(0)
    setPhase('update')
    setEpoch((value) => value + 10)
    setCurrentLoss(loss)
    selectSingleNode(summaryStep.nodeId)
    appendAction('Ran 10 training steps')
  }, [appendAction, blockingIssues.length, currentLoss, graph, pushHistory, selectSingleNode])

  const selectPaletteNode = (type: NodeType) => {
    setPendingNodeType(type)
    setPhase('edit')
    setTraceSteps([])
  }

  const placePaletteNode = useCallback((type: NodeType, position: { x: number; y: number }) => {
    pushHistory()
    const nextNode = createNode(type, nextNodeIndexForType(graph, type))
    const placedNode = { ...nextNode, position }
    setGraph((existing) => ({ ...existing, nodes: [...existing.nodes, placedNode] }))
    selectSingleNode(placedNode.id)
    setPendingNodeType(undefined)
    setPhase('edit')
  }, [graph, pushHistory, selectSingleNode])

  const clearPendingPlacement = useCallback(() => {
    setPendingNodeType(undefined)
  }, [])

  const updateNodeValue = useCallback((nodeId: string, value: TensorValue) => {
    pushHistory()
    setGraph((existing) => ({
      ...existing,
      nodes: existing.nodes.map((node) =>
        node.id === nodeId ? { ...node, params: { ...node.params, value }, value } : node,
      ),
    }))
    setPhase('edit')
  }, [pushHistory])

  const updateActivation = useCallback((nodeId: string, activation: ActivationKind) => {
    pushHistory()
    setGraph((existing) => ({
      ...existing,
      nodes: existing.nodes.map((node) =>
        node.id === nodeId ? { ...node, params: { ...node.params, activation } } : node,
      ),
    }))
    setPhase('edit')
  }, [pushHistory])

  const updateLoss = useCallback((nodeId: string, loss: LossKind) => {
    pushHistory()
    setGraph((existing) => ({
      ...existing,
      nodes: existing.nodes.map((node) =>
        node.id === nodeId ? { ...node, params: { ...node.params, loss } } : node,
      ),
    }))
    setPhase('edit')
  }, [pushHistory])

  const updateLearningRate = (learningRate: number) => {
    pushHistory()
    setGraph((existing) => ({ ...existing, learningRate }))
  }

  const randomizeParameters = () => {
    pushHistory()
    setGraph((existing) => {
      const next = {
        ...existing,
        nodes: existing.nodes.map((node) => {
          if (node.type !== 'weight' && node.type !== 'bias') return node
          const value = scalarValue(Number((Math.random() * 2 - 1).toFixed(2)))
          return { ...node, params: { ...node.params, value }, value }
        }),
      }
      setInitialParams(parameterValues(next))
      return safeForward(next).graph
    })
    appendAction('Randomized parameters')
  }

  const applyGraphChange = useCallback(
    (nextGraph: GraphModel) => {
      pushHistory()
      setGraph(nextGraph)
    },
    [pushHistory],
  )

  const mergeSelectedNodes = useCallback(() => {
    const result = mergeNodesIntoVisualGroup(graph, selectedNodeIds)
    if (!result.group) return

    pushHistory()
    setGraph(result.graph)
    setSelectedNodeIds([])
    setSelectedGroupId(result.group.id)
    setPhase('edit')
    setTraceSteps([])
    setTraceIndex(0)
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
      setPhase('edit')
      setTraceSteps([])
      setTraceIndex(0)
      setPendingNodeType(undefined)
    },
    [graph, pushHistory],
  )

  const moveGroup = useCallback(
    (groupId: string, position: { x: number; y: number }) => {
      pushHistory()
      setGraph((existing) => moveVisualGroup(existing, groupId, position))
      setPhase('edit')
    },
    [pushHistory],
  )

  const exportSummary = () => {
    downloadSessionSummary(
      createSessionSummary({
        lessonName: currentLesson?.name ?? 'Free exploration',
        graph,
        initialParameterValues: initialParams,
        trainingSteps: epoch,
        finalLoss: currentLoss,
        completedLessonActions: completedActions,
      }),
    )
  }

  return (
    <main className="app-shell">
      <aside className="left-panel">
        <div className="brand">
          <div className="brand-mark">BB</div>
          <div>
            <p className="eyebrow">Tensor autodiff lab</p>
            <h1>Backprop Builder</h1>
          </div>
        </div>

        <section className="panel-section">
          <p className="eyebrow">Node palette</p>
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
            <p className="placement-hint">Click the graph canvas to place {labelForType(pendingNodeType)}.</p>
          ) : null}
        </section>

        <section className="panel-section action-stack">
          <button type="button" className="primary-button" onClick={() => loadGraph(createStarterGraph(), lessons[0])}>
            <BookOpen size={16} />
            Load starter example
          </button>
          <button type="button" onClick={() => loadGraph(createEmptyGraph())}>
            <RotateCcw size={16} />
            Reset
          </button>
          <button type="button" onClick={randomizeParameters}>
            <Shuffle size={16} />
            Randomize parameters
          </button>
          <button
            type="button"
            className={showVisualization ? 'panel-toggle-button is-active' : 'panel-toggle-button'}
            aria-pressed={showVisualization}
            onClick={() => setShowVisualization((visible) => !visible)}
          >
            <Eye size={16} />
            {showVisualization ? 'Hide visualization' : 'Show visualization'}
          </button>
        </section>

        <section className="panel-section toggle-stack">
          <label>
            <input type="checkbox" checked={showMath} onChange={(event) => setShowMath(event.target.checked)} />
            Show math layer
          </label>
          <label>
            <input type="checkbox" checked={showGradient} onChange={(event) => setShowGradient(event.target.checked)} />
            Show gradient layer
          </label>
          <label>
            <input type="checkbox" checked={showCode} onChange={(event) => setShowCode(event.target.checked)} />
            Show code layer
          </label>
        </section>

        <section className="panel-section">
          <button type="button" className="lesson-toggle" onClick={() => setLessonOpen((open) => !open)}>
            <GraduationCap size={16} />
            Lesson drawer
          </button>
          {lessonOpen ? (
            <div className="lesson-list">
              {lessons.map((lesson, index) => (
                <article key={lesson.id} className={lesson.id === currentLesson?.id ? 'lesson-card is-current' : 'lesson-card'}>
                  <h3>{lesson.name}</h3>
                  <p>{lesson.prompt}</p>
                  <p className="lesson-task">{lesson.task}</p>
                  <button type="button" onClick={() => loadGraph(lesson.createGraph(), lesson)}>
                    Start Lesson {index + 1}
                  </button>
                </article>
              ))}
            </div>
          ) : null}
        </section>
      </aside>

      <GraphCanvas
        graph={graph}
        displayGraph={displayGraph}
        activeStep={activeStep}
        selectedNodeIds={selectedNodeIds}
        selectedGroupId={selectedGroupId}
        showMath={showMath}
        showGradient={showGradient}
        phase={phase}
        pendingNodeType={pendingNodeType}
        onGraphChange={applyGraphChange}
        onSelectionChange={selectCanvasSelection}
        onCreateNode={placePaletteNode}
        onCancelPendingPlacement={clearPendingPlacement}
        onNodeValueChange={updateNodeValue}
        onActivationChange={updateActivation}
        onLossChange={updateLoss}
        onGroupCreate={mergeSelectedNodes}
        onGroupExplode={explodeGroup}
        onGroupMove={moveGroup}
      />

      <aside className="right-panel">
        {showVisualization ? <VisualizationPanel graph={graph} /> : null}

        <section className="inspector-card">
          <p className="eyebrow">Inspector</p>
          <h2>Current phase: {phaseLabel(phase)}</h2>
          <p>{activeStep?.explanation ?? 'Edit the graph, load a lesson, or step through the tensor computation.'}</p>
          <div className={`phase-badge phase-${phase}`}>{phaseLabel(phase)}</div>
        </section>

        <section className="inspector-card">
          <p className="eyebrow">Current step</p>
          <h3>{activeStep?.title ?? 'Ready to evaluate'}</h3>
          <div className="formula-box">
            <strong>Formula</strong>
            <span>{activeStep?.formula ?? (selectedNode ? formulaForNode(selectedNode, graph) : 'Choose a node or press Step.')}</span>
          </div>
          <div className="formula-box">
            <strong>Mini calculation</strong>
            <span>{activeStep?.calculation ?? 'Numbers will appear here as each node evaluates.'}</span>
          </div>
          {showCode ? (
            <pre className="code-layer">
              {(activeStep?.pseudocode ?? starterPseudocode()).join('\n')}
            </pre>
          ) : null}
        </section>

        <section className="inspector-card">
          <p className="eyebrow">Validation</p>
          {validationIssues.length === 0 ? (
            <p className="valid-message">Graph is ready for a full forward and backward pass.</p>
          ) : (
            <ul className="issue-list">
              {validationIssues.slice(0, 5).map((issue, index) => (
                <li key={`${issue.code}-${issue.nodeId ?? issue.edgeId ?? 'graph'}-${index}`}>{issue.message}</li>
              ))}
            </ul>
          )}
        </section>

        <section className="inspector-card">
          <p className="eyebrow">Lesson progress</p>
          <h3>{currentLesson?.name ?? 'Free exploration'}</h3>
          <p>{currentLesson?.completionCondition ?? 'Build and inspect a graph at your own pace.'}</p>
          {completedActions.length > 0 ? <p className="success-message">{currentLesson?.successMessage}</p> : null}
          <button type="button" onClick={exportSummary}>
            <Download size={16} />
            Download session summary
          </button>
        </section>
      </aside>

      <footer className="control-bar">
        <button type="button" className="primary-button" onClick={stepForward} disabled={blockingIssues.length > 0}>
          <StepForward size={16} />
          Step
        </button>
        <button type="button" onClick={() => setIsPlaying((playing) => !playing)} disabled={blockingIssues.length > 0}>
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
            onChange={(event) => setSpeedSliderValue(Number(event.target.value))}
          />
        </label>
        <button type="button" onClick={runOneTrainingStep} disabled={blockingIssues.length > 0}>
          <FastForward size={16} />
          Run one full training step
        </button>
        <button type="button" onClick={runTenTrainingSteps} disabled={blockingIssues.length > 0}>
          Run 10 training steps
        </button>
        <label className="slider-control">
          Learning rate eta = {graph.learningRate.toFixed(2)}
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={graph.learningRate}
            onChange={(event) => updateLearningRate(Number(event.target.value))}
          />
        </label>
        <div className="metric-pill">Epoch {epoch}</div>
        <div className="metric-pill">Current loss {formatNumber(currentLoss ?? undefined)}</div>
      </footer>
    </main>
  )
}

function phaseLabel(phase: GraphPhase): string {
  if (phase === 'edit') return 'Edit graph'
  if (phase === 'forward') return 'Forward pass'
  if (phase === 'loss') return 'Loss computation'
  if (phase === 'backward') return 'Backward pass'
  return 'Update parameters'
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

function starterPseudocode(): string[] {
  return [
    'z = x * w',
    'a = z + b',
    'pred = sigmoid(a)',
    'loss = 0.5 * (pred - y)**2',
    'loss.backward()',
    'w -= lr * w.grad',
    'b -= lr * b.grad',
  ]
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

function stringArraysEqual(first: string[], second: string[]): boolean {
  return first.length === second.length && first.every((value, index) => value === second[index])
}

function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toLowerCase()
  return target.isContentEditable || tagName === 'input' || tagName === 'textarea' || tagName === 'select'
}

function safeForward(graph: GraphModel): { graph: GraphModel; loss?: number } {
  const issues = validateGraph(graph).filter((issue) => issue.code !== 'disconnected')
  if (issues.length > 0) return { graph: cloneGraph(graph) }
  const result = forwardPass(graph)
  return { graph: result.graph, loss: result.loss }
}

export default App
