import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  getViewportForBounds,
  Position as FlowPosition,
  useReactFlow,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import { ArrowUp, Combine, Maximize, Ungroup, ScanSearch, LayoutGrid, Search } from 'lucide-react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactElement } from 'react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  MAX_FLEX_INPUT_COUNT,
  NODE_WIDTH,
  formulaForNode,
  heightForInputCount,
  inputArityForNode,
  isFlexibleInputNodeType,
  lossKindForNode,
  lossOptionsForNode,
  outputArityForNode,
} from '../domain/engine'
import { canConnectGraphNodes, connectGraphNodes, type GraphConnection } from '../domain/graphEditing'
import {
  deleteVisualGroup,
  collapsedGroupForNode,
  expandedGroupRect,
  groupAncestors,
  setVisualGroupExpanded,
  visibleGroups,
  removeNodesFromVisualGroups,
  resolveVisualGroupInputHandle,
  resolveVisualGroupOutputHandle,
  visualGroupInterface,
  moveVisualGroup,
} from '../domain/grouping'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import { blockPalette } from '../domain/blockPalette'
import { codeForGroup } from '../domain/codeOutline'
import type {
  ActivationKind,
  CustomCsvData,
  DatasetKind,
  EvaluationTraceStep,
  GraphModel,
  GraphNode,
  GraphViewState,
  LossKind,
  NodeType,
  Position as GraphPosition,
  TensorValue,
  TensorTransformKind,
} from '../domain/types'
import { BuilderEdge, type BuilderEdgeData } from './BuilderEdge'
import { BuilderNode, type BuilderNodeData } from './BuilderNode'
import { GroupNode, type GroupNodeData } from './GroupNode'
import { SemanticNode } from './SemanticNode'
import { layoutSemanticGraph, semanticGroupDepth } from '../domain/semanticLayout'
import { preserveLayoutForWiring } from '../domain/layoutState'
import { routeDiagramWires, type DiagramWire, type WireEndpoint, type WireObstacle } from '../domain/wireRouting'
import { findWireCrossings } from '../domain/wireCrossings'
import { cardReveal, compactVisualHierarchy, continuousSceneMaxZoom, layoutContinuousScene, routeContinuousScene, sceneContentBounds } from '../domain/continuousScene'
import './modules.css'
import './semanticCanvas.css'
import './canvasAddMenu.css'

const nodeTypes = { builderNode: BuilderNode, groupNode: GroupNode, semanticNode: SemanticNode }
const edgeTypes = { builderEdge: BuilderEdge }
const GROUP_NODE_ID_PREFIX = 'visual-group:'
const EMPTY_SELECTED_NODE_IDS: string[] = []
const INITIAL_FIT_OPTIONS = { padding: 0.2 }

interface CanvasSelection {
  nodeIds: string[]
  groupId?: string
}

type CanvasNode = Node<BuilderNodeData | GroupNodeData>
type CanvasEdge = Edge<BuilderEdgeData>

interface GraphCanvasProps {
  graph: GraphModel
  displayGraph?: GraphModel
  activeStep?: EvaluationTraceStep
  selectedNodeIds?: string[]
  selectedGroupId?: string
  selectedEdgeId?: string
  focusRequest?: { kind: 'group' | 'node'; id: string; serial: number }
  showMath: boolean
  showGradient: boolean
  phase: string
  pendingNodeType?: NodeType
  onGraphChange: (graph: GraphModel) => void
  onViewChange?: (view: GraphViewState) => void
  onSelectionChange: (selection: CanvasSelection) => void
  onCreateNode: (type: NodeType, position: GraphPosition, parentGroupId?: string, sceneScale?: number) => void
  onCancelPendingPlacement: () => void
  onNodeValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, activation: ActivationKind) => void
  onExpressionChange?: (nodeId: string, expression: string) => void
  onTransformChange?: (nodeId: string, transform: TensorTransformKind) => void
  onLossChange?: (nodeId: string, loss: LossKind) => void
  onDatasetChange?: (nodeId: string, dataset: DatasetKind) => void
  onGroupCreate: () => void
  onGroupExplode: (groupId: string) => void
  onGroupRename?: (groupId: string, label: string) => void
  onGroupMove: (groupId: string, position: GraphPosition) => void
  onInspectNeuron?: (groupId: string, unitIndex: number) => void
  onInspectEdge?: (edgeId: string) => void
}

export function GraphCanvas(props: GraphCanvasProps): ReactElement {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner {...props} />
    </ReactFlowProvider>
  )
}

function GraphCanvasInner({
  graph,
  displayGraph,
  activeStep,
  selectedNodeIds = EMPTY_SELECTED_NODE_IDS,
  selectedGroupId,
  selectedEdgeId,
  focusRequest,
  showMath,
  showGradient,
  phase,
  pendingNodeType,
  onGraphChange,
  onViewChange,
  onSelectionChange,
  onCreateNode,
  onCancelPendingPlacement,
  onNodeValueChange,
  onActivationChange,
  onExpressionChange,
  onTransformChange,
  onLossChange,
  onDatasetChange,
  onGroupCreate,
  onGroupExplode,
  onGroupRename,
  onInspectNeuron,
  onInspectEdge,
}: GraphCanvasProps): ReactElement {
  const { screenToFlowPosition, fitView, setViewport, getViewport } = useReactFlow()
  const sourceGraph = displayGraph ?? graph
  const semantic = sourceGraph.view?.canvasStyle === 'architecture' ||
    (sourceGraph.view?.canvasStyle !== 'builder' &&
      (sourceGraph.view?.semanticZoom !== undefined || Boolean(sourceGraph.groups?.some((group) => group.kind))))
  const continuous = semantic && sourceGraph.view?.semanticZoom !== false
  const renderedGraph = useMemo(() => continuous ? compactVisualHierarchy(sourceGraph) : sourceGraph, [sourceGraph, continuous])
  const geometryKey = JSON.stringify({
    nodes: renderedGraph.nodes.map(({ id, type, label, position, dimensions, params }) => ({ id, type, label, position, dimensions, params: { inputCount: params.inputCount, expression: params.expression, dataset: params.dataset, customCsv: params.customCsv ? csvGeometrySample(params.customCsv) : undefined } })),
    edges: renderedGraph.edges.map(({ id, source, target, inputSlot, sourceSlot }) => ({ id, source, target, inputSlot, sourceSlot })),
    groups: renderedGraph.groups,
    view: { expandedGroupIds: [], layoutOffsets: renderedGraph.view?.layoutOffsets, layoutEdges: renderedGraph.view?.layoutEdges, manualNodePlacements: renderedGraph.view?.manualNodePlacements },
  })
  const geometryGraph = useMemo(() => JSON.parse(geometryKey) as GraphModel, [geometryKey])
  const scene = useMemo(() => continuous ? layoutContinuousScene(geometryGraph) : undefined, [geometryGraph, continuous])
  const layout = useMemo(() => scene ?? (semantic ? layoutSemanticGraph(renderedGraph) : undefined), [renderedGraph, semantic, scene])
  const shell = useRef<HTMLDivElement>(null)
  const [camera, setCamera] = useState(graph.view?.viewport ?? { x: 0, y: 0, zoom: 1 })
  const [addMenu, setAddMenu] = useState<{ x: number; y: number; position: GraphPosition; parentGroupId?: string; sceneScale?: number }>()
  const [renameGroup, setRenameGroup] = useState<{ id: string; x: number; y: number; label: string }>()
  const [addQuery, setAddQuery] = useState('')
  const [activeSuggestion, setActiveSuggestion] = useState(0)
  const addSearch = useRef<HTMLInputElement>(null)
  const emittedViewport = useRef<GraphViewState['viewport']>(undefined)
  const cameraZoom = camera.zoom
  const [canvasSize, setCanvasSize] = useState({ width: 900, height: 600 })
  const maxZoom = scene ? continuousSceneMaxZoom(scene) : semantic ? 3 : 1.6
  const zoomToGroup = useCallback((groupId?: string) => {
    if (!scene) return
    const frame = groupId ? scene.groups.get(groupId) : undefined
    const contents = groupId ? scene.levels.find(level => level.parentId === groupId) : undefined
    // A leaf block opens onto its actual calculations, using their bounds so
    // that the final zoom lands on the graph rather than the card's padding.
    const leaf = Boolean(frame && contents?.ids.length && contents.ids.every(id => scene.nodes.has(id)))
    const rects = frame ? leaf && contents
      ? contents.ids.map(id => scene.nodes.get(id)!) : [frame]
      : [...scene.nodes.values(), ...scene.groups.values()]
    if (leaf) {
      const internalEdges = new Set(geometryGraph.edges.filter(edge => scene.parents.get(edge.source) === groupId && scene.parents.get(edge.target) === groupId).map(edge => edge.id))
      for (const wire of routeContinuousScene(geometryGraph, scene)) {
        if (wire.parentId === groupId && internalEdges.has(wire.edgeId)) rects.push(...wire.route.map(point => ({ ...point, width: 0, height: 0 })))
      }
    }
    if (!rects.length) return
    const x = Math.min(...rects.map(rect => rect.x)), y = Math.min(...rects.map(rect => rect.y))
    const bounds = { x, y, width: Math.max(...rects.map(rect => rect.x + rect.width)) - x, height: Math.max(...rects.map(rect => rect.y + rect.height)) - y }
    void setViewport(getViewportForBounds(bounds, canvasSize.width, canvasSize.height, .01, maxZoom, frame ? .08 : .2), { duration: 650 })
  }, [scene, geometryGraph, canvasSize, maxZoom, setViewport])
  const lastCodeFocus = useRef(0)
  useEffect(() => {
    if (!focusRequest || lastCodeFocus.current === focusRequest.serial) return
    lastCodeFocus.current = focusRequest.serial
    if (focusRequest.kind === 'group') {
      if (scene) zoomToGroup(focusRequest.id)
      else void fitView({ nodes: [{ id: groupNodeId(focusRequest.id) }], padding: .18, duration: 650, maxZoom: 1.6 })
      return
    }
    const rect = scene?.nodes.get(focusRequest.id) ?? layout?.nodes.get(focusRequest.id)
    if (!rect) return
    void setViewport(getViewportForBounds(rect, canvasSize.width, canvasSize.height, .01, maxZoom, .32), { duration: 650 })
  }, [focusRequest, scene, layout, zoomToGroup, fitView, setViewport, canvasSize, maxZoom])
  useEffect(() => { if (addMenu) addSearch.current?.focus() }, [addMenu])
  useEffect(() => {
    if (!shell.current || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => setCanvasSize({ width: entry.contentRect.width, height: entry.contentRect.height }))
    observer.observe(shell.current)
    return () => observer.disconnect()
  }, [])
  const dragPositions = useRef<Map<string, GraphPosition> | undefined>(undefined)
  const secondaryPan = useRef<{ pointerId: number; x: number; y: number; viewport: { x: number; y: number; zoom: number } } | undefined>(undefined)
  const groupedNodeIds = useMemo(() => new Set(continuous ? [] : renderedGraph.nodes.filter((node) => collapsedGroupForNode(renderedGraph, node.id)).map((node) => node.id)), [renderedGraph, continuous])
  const changeView = useCallback((view: GraphViewState) => {
    if (onViewChange) onViewChange(view)
    else onGraphChange({ ...graph, view })
  }, [graph, onGraphChange, onViewChange])
  const compactLayout = useCallback(() => {
    const cleanGraph = { ...renderedGraph, view: { ...renderedGraph.view!, layoutOffsets: undefined, layoutEdges: undefined } }
    const clean = continuous ? layoutContinuousScene(cleanGraph) : layoutSemanticGraph(cleanGraph)
    const rects = [...clean.nodes.values(), ...clean.groups.values()]
    if (!rects.length) return
    const x = Math.min(...rects.map(rect => rect.x)), y = Math.min(...rects.map(rect => rect.y))
    const bounds = { x, y, width: Math.max(...rects.map(rect => rect.x + rect.width)) - x, height: Math.max(...rects.map(rect => rect.y + rect.height)) - y }
    changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], layoutOffsets: undefined, layoutEdges: undefined, focusedGroupId: undefined,
      viewport: getViewportForBounds(bounds, canvasSize.width, canvasSize.height, .01, 1.5, .2) })
    onSelectionChange({ nodeIds: [] })
  }, [renderedGraph, continuous, graph.view, canvasSize, changeView, onSelectionChange])
  useEffect(() => {
    if (!scene?.repairedOffsetIds.size || !graph.view?.layoutOffsets) return
    const removed = new Set(scene.repairedOffsetIds)
    // A compacted display group can inherit an offset from a skipped wrapper.
    for (const id of scene.repairedOffsetIds) {
      const groupId = groupIdFromNodeId(id)
      if (groupId) for (const ancestor of groupAncestors(graph, groupId)) {
        if (!scene.groups.has(ancestor.id)) removed.add(groupNodeId(ancestor.id))
      }
    }
    const layoutOffsets = Object.fromEntries(Object.entries(graph.view.layoutOffsets).filter(([id]) => !removed.has(id)))
    if (Object.keys(layoutOffsets).length !== Object.keys(graph.view.layoutOffsets).length) changeView({ ...graph.view, layoutOffsets })
  }, [scene, graph, changeView])
  const toggleGroup = useCallback((groupId: string) => {
    if (continuous && scene) {
      const displayed = renderedGraph.groups?.find(item => item.id === groupId)
      if (displayed?.detail?.virtual && displayed.detail.unitIndex !== graph.view?.inspectedNeuron?.unitIndex) {
        onInspectNeuron?.(String(displayed.detail.layerId), Number(displayed.detail.unitIndex))
        return
      }
      const rect = scene.groups.get(groupId)
      const revealed = rect && cardReveal(rect, getViewport().zoom, canvasSize.width, canvasSize.height) > .95
      const group = renderedGraph.groups?.find(item => item.id === groupId)
      const target = revealed ? group?.parentId : groupId
      const next = setVisualGroupExpanded(graph, groupId, !revealed)
      if (next.view) changeView(next.view)
      onSelectionChange({ nodeIds: [], groupId: target })
      zoomToGroup(target)
      return
    }
    const displayed = renderedGraph.groups?.find((group) => group.id === groupId)
    if (displayed?.detail?.virtual && typeof displayed.detail.layerId === 'string' && typeof displayed.detail.unitIndex === 'number') {
      if (renderedGraph.view?.expandedGroupIds.includes(groupId)) {
        const next = setVisualGroupExpanded(graph, displayed.detail.layerId, false)
        if (next.view) changeView(next.view)
        onSelectionChange({ nodeIds: [], groupId: displayed.detail.layerId })
      } else onInspectNeuron?.(displayed.detail.layerId, displayed.detail.unitIndex)
      return
    }
    const next = setVisualGroupExpanded(graph, groupId, !graph.view?.expandedGroupIds.includes(groupId))
    if (next.view) changeView(next.view)
    if (!selectedNodeIds.some(id => displayed?.nodeIds.includes(id))) onSelectionChange({ nodeIds: [], groupId })
  }, [graph, renderedGraph, changeView, onSelectionChange, onInspectNeuron, selectedNodeIds, continuous, scene, getViewport, canvasSize, zoomToGroup])
  const focusGroup = useCallback((groupId?: string) => {
    changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], focusedGroupId: groupId })
    onSelectionChange({ nodeIds: [], groupId })
    if (continuous) zoomToGroup(groupId)
    else void fitView({ nodes: groupId ? [{ id: groupNodeId(groupId) }] : undefined, padding: .18, duration: 650, maxZoom: 1.5 })
  }, [graph, changeView, fitView, onSelectionChange, continuous, zoomToGroup])
  const { x: savedX, y: savedY, zoom: savedZoom } = graph.view?.viewport ?? {}
  useEffect(() => {
    if (savedX === undefined || savedY === undefined || savedZoom === undefined) return
    // Persisting our own camera event must not replay it: an interrupted
    // transition can emit its old position after the next zoom has started.
    const emitted = emittedViewport.current
    if (emitted?.x === savedX && emitted.y === savedY && emitted.zoom === savedZoom) return
    void setViewport({ x: savedX, y: savedY, zoom: savedZoom })
  }, [savedX, savedY, savedZoom, setViewport])
  const focusId = renderedGraph.view?.focusedGroupId
  const inspectedNeuron = renderedGraph.view?.inspectedNeuron
  const focusNodeId = inspectedNeuron && inspectedNeuron.groupId === focusId ? `inspect:${inspectedNeuron.groupId}:${inspectedNeuron.unitIndex}` : focusId
  const expansionKey = renderedGraph.view?.expandedGroupIds.join(',')
  const lastFocusRequest = useRef<string | undefined>(undefined)
  useEffect(() => {
    if (!semantic || !focusNodeId) { lastFocusRequest.current = undefined; return }
    // A code-line navigation has already started this camera transition.
    // Do not restart it when the matching focused group enters graph state.
    if ((focusRequest?.kind === 'node' && focusRequest.id === selectedNodeIds[0])
      || (focusRequest?.kind === 'group' && focusRequest.id === focusNodeId)) return
    const request = `${focusNodeId}:${expansionKey}`
    if (lastFocusRequest.current === request) return
    const timeout = window.setTimeout(() => {
      lastFocusRequest.current = request
      if (continuous) zoomToGroup(focusNodeId)
      else void fitView({ nodes: [{ id: groupNodeId(focusNodeId) }], padding: .2, duration: 650, maxZoom: 1.6 })
    }, 120)
    return () => window.clearTimeout(timeout)
  }, [focusNodeId, expansionKey, fitView, semantic, continuous, zoomToGroup, focusRequest, selectedNodeIds])
  const selectedNodeIdSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds])
  const groupInterfaces = useMemo(
    () => new Map((renderedGraph.groups ?? []).map((group) => [group.id, visualGroupInterface(renderedGraph, group)])),
    [renderedGraph],
  )

  const addFlexibleInput = useCallback(
    (nodeId: string) => {
      onGraphChange({
        ...graph,
        nodes: graph.nodes.map((node) => {
          if (node.id !== nodeId || !isFlexibleInputNodeType(node.type)) return node

          const currentInputCount = Math.max(inputArityForNode(node), minimumInputCountForNode(graph, node))
          const nextInputCount = Math.min(MAX_FLEX_INPUT_COUNT, currentInputCount + 1)
          if (nextInputCount === currentInputCount) return node

          const currentHeight = node.dimensions?.height ?? heightForInputCount(currentInputCount)
          const nextHeight = Math.max(currentHeight, heightForInputCount(nextInputCount))

          return {
            ...node,
            dimensions: { ...node.dimensions, width: NODE_WIDTH, height: nextHeight },
            params: { ...node.params, inputCount: nextInputCount },
          }
        }),
      })
    },
    [graph, onGraphChange],
  )

  const reactNodes = useMemo(
    () => {
      const groupNodes = (continuous ? renderedGraph.groups ?? [] : visibleGroups(renderedGraph)).map<CanvasNode>((group) => {
        const groupInterface = groupInterfaces.get(group.id)
        const expanded = continuous || (renderedGraph.view?.expandedGroupIds.includes(group.id) ?? false)
        const rect = layout?.groups.get(group.id) ?? (expanded ? expandedGroupRect(renderedGraph, group) : undefined)
        const weightNode = renderedGraph.nodes.find((node) => node.id === group.detail?.weightNodeId)
        const weightValue = weightNode?.value ?? (typeof weightNode?.params.value === 'object' ? weightNode.params.value : undefined)
        const outputNode = renderedGraph.nodes.find((node) => node.id === group.detail?.outputNodeId)
        const attentionWeightsId = group.detail?.weightsNodeId ?? renderedGraph.groups?.find((child) => child.parentId === group.id && child.kind === 'head')?.detail?.weightsNodeId

        return {
          id: groupNodeId(group.id),
          type: 'groupNode',
          position: rect ? { x: rect.x, y: rect.y } : group.position,
          width: rect?.width ?? group.dimensions.width,
          height: rect?.height ?? group.dimensions.height,
          draggable: true,
          selectable: !expanded,
          dragHandle: expanded ? '.visual-group-title-row' : undefined,
          zIndex: continuous ? 100 - semanticGroupDepth(renderedGraph, group.id) : expanded ? -10 + semanticGroupDepth(renderedGraph, group.id) : 0,
          className: expanded ? 'expanded-module-frame' : undefined,
          selected: group.id === selectedGroupId || group.nodeIds.length > 0 && group.nodeIds.every(id => selectedNodeIdSet.has(id)),
          data: {
            group: rect ? { ...group, dimensions: { width: rect.width, height: rect.height } } : group,
            codeLine: codeForGroup(renderedGraph, group),
            expanded,
            continuous,
            sceneScale: scene?.scales.get(groupNodeId(group.id)),
            semantic,
            modelStage: !group.parentId && renderedGraph.groups?.some((candidate) => candidate.kind === 'transformer-block' || candidate.kind === 'cnn'),
            unitCount: weightValue?.shape[weightValue.shape.length - 1],
            unitValues: outputNode?.value?.data,
            attentionWeights: renderedGraph.nodes.find((node) => node.id === attentionWeightsId)?.value,
            onInspectNeuron,
            inputCount: groupInterface?.inputs.length ?? 0,
            outputCount: groupInterface?.outputs.length ?? 0,
            outputMetrics: (groupInterface?.outputs ?? []).map((output) => {
              const edgeIds = edgeIdsForVisualGroupHandle(output)
              const edge = renderedGraph.edges.find((candidate) => edgeIds.includes(candidate.id))
              const source = renderedGraph.nodes.find((node) => node.id === output.source)
              return { forward: edge?.value ?? source?.value, gradient: edge?.grad ?? source?.grad }
            }),
            showGradient: semantic ? showGradient && phase === 'backward' : showGradient,
            active: group.nodeIds.includes(activeStep?.nodeId ?? ''),
            onToggle: toggleGroup,
          },
        }
      })

      const builderNodes = renderedGraph.nodes
        .filter((node) => !groupedNodeIds.has(node.id))
        .map<CanvasNode>((node) => ({
          id: node.id,
          type: node.type === 'loss' ? 'builderNode' : semantic ? 'semanticNode' : 'builderNode',
          position: layout?.nodes.get(node.id) ?? node.position,
          draggable: true,
          width: semantic ? layout?.nodes.get(node.id)?.width ?? 176 : isFlexibleInputNodeType(node.type) || node.type === 'arithmetic' ? NODE_WIDTH : undefined,
          height: semantic ? layout?.nodes.get(node.id)?.height ?? 112 : isFlexibleInputNodeType(node.type) || node.type === 'arithmetic'
            ? node.dimensions?.height ?? heightForInputCount(inputArityForNode(node))
            : undefined,
          selected: selectedNodeIdSet.has(node.id),
          data: {
            graphNode: node,
            sceneScale: scene?.scales.get(node.id),
            displayInputCount: Math.max(0, ...renderedGraph.edges.filter(edge => edge.target === node.id).map(edge => (edge.inputSlot ?? 0) + 1)),
            coordinate: node.id.startsWith('inspect:'),
            vertical: node.type !== 'loss' && node.params.dataset !== 'custom-csv' && semantic && renderedGraph.groups?.some((group) => group.kind === 'transformer-block' || group.kind === 'cnn') && !renderedGraph.groups?.some((group) => group.nodeIds.includes(node.id)),
            compactStage: node.params.dataset !== 'custom-csv' && semantic && renderedGraph.groups?.some((group) => group.kind === 'cnn') && !renderedGraph.groups?.some((group) => group.nodeIds.includes(node.id)),
            showMath,
            showGradient: semantic ? showGradient && phase === 'backward' : showGradient,
            formula: formulaForNode(node, renderedGraph, formatCompactTensor),
            fullFormula: formulaForNode(node, renderedGraph, formatFullTensor),
            lossKind: node.type === 'loss' ? lossKindForNode(node, renderedGraph) : undefined,
            lossOptions: node.type === 'loss' ? lossOptionsForNode(node, renderedGraph) : undefined,
            active: activeStep?.nodeId === node.id,
            hasIncomingValue: renderedGraph.edges.some((edge) => edge.target === node.id),
            onFlexibleInputAdd: addFlexibleInput,
            onValueChange: onNodeValueChange,
            onActivationChange: (nodeId: string, value: string) => onActivationChange(nodeId, value as ActivationKind),
            onExpressionChange: onExpressionChange ?? (() => undefined),
            onTransformChange: onTransformChange ?? (() => undefined),
            onLossChange: onLossChange ?? (() => undefined),
            onDatasetChange: onDatasetChange ?? (() => undefined),
          },
        }))

      return [...groupNodes, ...builderNodes]
    },
    [
      activeStep?.nodeId,
      onActivationChange,
      onExpressionChange,
      onTransformChange,
      onLossChange,
      onDatasetChange,
      addFlexibleInput,
      onNodeValueChange,
      toggleGroup,
      groupInterfaces,
      groupedNodeIds,
      renderedGraph,
      selectedGroupId,
      selectedNodeIdSet,
      showGradient,
      showMath,
      layout,
      semantic,
      continuous,
      scene,
      onInspectNeuron,
      phase,
    ],
  )

  const reactEdges = useMemo(
    () =>
      renderedGraph.edges.flatMap<CanvasEdge>((edge) => {
        const sourceGroup = continuous ? undefined : collapsedGroupForNode(renderedGraph, edge.source)
        const targetGroup = continuous ? undefined : collapsedGroupForNode(renderedGraph, edge.target)
        if (sourceGroup?.id && sourceGroup.id === targetGroup?.id) return []
        const sourceHandle = sourceGroup
          ? groupInterfaces.get(sourceGroup.id)?.outputs.find((handle) => visualGroupHandleHasEdge(handle, edge.id))?.handleId
          : undefined
        const targetHandle = targetGroup
          ? groupInterfaces.get(targetGroup.id)?.inputs.find((handle) => handle.edgeId === edge.id)?.handleId
          : undefined

        return [
          {
            id: edge.id,
            source: sourceGroup ? groupNodeId(sourceGroup.id) : edge.source,
            target: targetGroup ? groupNodeId(targetGroup.id) : edge.target,
            sourceHandle: sourceHandle ?? sourceHandleForEdge(renderedGraph, edge),
            targetHandle: targetHandle ?? `in-${edge.inputSlot ?? 0}`,
            type: 'builderEdge',
            selected: selectedEdgeId === edge.id,
            // BuilderEdge animates its own directional bands. React Flow's
            // `animated` class would override their speed and backward motion.
            data: {
              forward: edge.value,
              gradient: edge.grad,
              showGradient,
              active: activeStep?.edgeIds.includes(edge.id) ?? false,
              phase,
              residual: semantic && /residual|skip/i.test(renderedGraph.nodes.find((node) => node.id === edge.target)?.label ?? '') && (edge.inputSlot ?? 0) === 0,
              label: `${renderedGraph.nodes.find((node) => node.id === edge.source)?.label ?? edge.source} → ${renderedGraph.nodes.find((node) => node.id === edge.target)?.label ?? edge.target}`,
              onInspect: () => onInspectEdge?.(edge.id),
            },
          },
        ]
      }),
    [activeStep?.edgeIds, groupInterfaces, phase, renderedGraph, showGradient, semantic, continuous, selectedEdgeId, onInspectEdge],
  )

  const [nodes, setNodes, onNodesChangeBase] = useNodesState(reactNodes)
  const [edges, setEdges, onEdgesChangeBase] = useEdgesState(reactEdges)
  // Cache geometry separately from numerical traces: stepping through a model
  // changes wire values, but should not redo its routing or shuffle its lanes.
  const routingKey = JSON.stringify(semantic && !continuous ? {
    blocks: nodes.map(node => {
      const group = isGroupNodeId(node.id) ? node.data as GroupNodeData : undefined
      const operation = group ? undefined : (node.data as BuilderNodeData).graphNode
      return { id: node.id, ...node.position, width: node.width ?? 176, height: node.height ?? 112,
        expanded: group?.expanded ?? false, vertical: Boolean(group ? group.modelStage : node.data.vertical),
        inputs: group?.inputCount ?? Math.max(inputArityForNode(operation!), Number(node.data.displayInputCount ?? 0)),
        outputs: group?.outputCount ?? outputArityForNode(operation!),
      }
    }),
    wires: edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target, sourceHandle: edge.sourceHandle, targetHandle: edge.targetHandle })),
  } : null)
  const wireRoutes = useMemo(() => {
    const geometry = JSON.parse(routingKey) as { blocks: RoutingBlock[]; wires: Pick<CanvasEdge, 'id' | 'source' | 'target' | 'sourceHandle' | 'targetHandle'>[] } | null
    if (!geometry) return new Map<string, GraphPosition[]>()
    const blocks = new Map(geometry.blocks.map(block => [block.id, block]))
    const wires = geometry.wires.flatMap<DiagramWire>(edge => {
      const source = blocks.get(edge.source), target = blocks.get(edge.target)
      return source && target ? [{ id: edge.id, source: routingEndpoint(source, edge.sourceHandle, true), target: routingEndpoint(target, edge.targetHandle, false) }] : []
    })
    return routeDiagramWires(wires, geometry.blocks.map(block => block.expanded
      ? { ...block, width: Math.min(block.width, 310), height: 46 }
      : block))
  }, [routingKey])
  const wireCrossings = useMemo(() => findWireCrossings(wireRoutes), [wireRoutes])
  const scenePositionKey = JSON.stringify(scene ? nodes.map(node => [node.id, node.position]) : [])
  const sceneRoutes = useMemo(() => scene ? routeContinuousScene(geometryGraph, scene, new Map(JSON.parse(scenePositionKey))) : [], [geometryGraph, scene, scenePositionKey])
  const routedEdges = useMemo(() => {
    if (scene) {
      const byId = new Map(edges.map(edge => [edge.id, edge]))
      return sceneRoutes.flatMap(wire => {
        const edge = byId.get(wire.edgeId)
        return edge ? [{ ...edge, id: wire.id, data: { ...edge.data!, route: wire.route, crossings: wire.crossings, absoluteRoute: true, sceneScale: wire.scale, parentId: wire.parentId, canonicalEdgeId: wire.edgeId } }] : []
      })
    }
    return edges.map(edge => wireRoutes.has(edge.id) ? { ...edge, data: { ...edge.data!, route: wireRoutes.get(edge.id), crossings: wireCrossings.get(edge.id) } } : edge)
  }, [edges, wireRoutes, wireCrossings, scene, sceneRoutes])
  const reveals = useMemo(() => new Map(scene ? [...scene.groups].map(([id, rect]) => [id, cardReveal(rect, cameraZoom, canvasSize.width, canvasSize.height)] as const) : []), [scene, cameraZoom, canvasSize.width, canvasSize.height])
  const accessible = useCallback((id?: string): boolean => {
    let current = id
    while (current) {
      if ((reveals.get(current) ?? 0) <= .92) return false
      current = scene?.parents.get(groupNodeId(current))
    }
    return true
  }, [reveals, scene])
  const insertionGroupAt = useCallback((position: GraphPosition): string | undefined => scene && [...scene.groups]
    .filter(([id, rect]) => graph.groups?.some(group => group.id === id) && accessible(id)
      && position.x >= rect.x && position.x <= rect.x + rect.width
      && position.y >= rect.y && position.y <= rect.y + rect.height)
    .sort(([, a], [, b]) => a.width * a.height - b.width * b.height)[0]?.[0], [scene, graph.groups, accessible])
  const insertionScaleAt = useCallback((position: GraphPosition): number | undefined => {
    if (!scene) return undefined
    const nearby = [...scene.nodes].flatMap(([id, rect]) => {
      const scale = scene.scales.get(id) ?? 1
      const screenWidth = 176 * scale * cameraZoom
      if (!accessible(scene.parents.get(id)) || screenWidth < 72 || screenWidth > 400) return []
      const dx = Math.max(rect.x - position.x, 0, position.x - rect.x - rect.width)
      const dy = Math.max(rect.y - position.y, 0, position.y - rect.y - rect.height)
      return [{ scale, distance: dx * dx + dy * dy }]
    }).sort((a, b) => a.distance - b.distance)[0]
    return nearby?.scale ?? Math.min(1, 1.25 / Math.max(.01, cameraZoom))
  }, [scene, cameraZoom, accessible])
  const presentedNodes = scene ? nodes.filter(node => scene.nodes.has(node.id) || scene.groups.has(groupIdFromNodeId(node.id) ?? '')).map(node => {
    const groupId = groupIdFromNodeId(node.id)
    const reveal = groupId ? reveals.get(groupId) ?? 0 : 1
    const available = accessible(scene.parents.get(node.id))
    const rect = groupId ? scene.groups.get(groupId)! : scene.nodes.get(node.id)!
    const parentId = scene.parents.get(node.id)
    const parentBounds = parentId ? sceneContentBounds(scene, parentId) : undefined
    const parentPosition = parentId ? nodes.find(candidate => candidate.id === groupNodeId(parentId))?.position : undefined
    const parentRect = parentId ? scene.groups.get(parentId) : undefined
    if (parentBounds && parentPosition && parentRect) {
      parentBounds.x += parentPosition.x - parentRect.x
      parentBounds.y += parentPosition.y - parentRect.y
    }
    const groupData = groupId ? node.data as GroupNodeData : undefined
    const operation = groupId ? undefined : (node.data as BuilderNodeData).graphNode
    const inputs = groupData?.inputCount ?? Math.max(inputArityForNode(operation!), Number(node.data.displayInputCount ?? 0))
    const outputs = groupData?.outputCount ?? outputArityForNode(operation!)
    const vertical = Boolean(groupData ? groupData.modelStage : node.data.vertical)
    // Deeply nested nodes can be smaller than one CSS pixel in world space.
    // Keep exact dimensions/ports instead of waiting for rounded DOM measures.
    const handles = [false, true].flatMap(source => Array.from({ length: source ? outputs : inputs }, (_, index) => ({
      id: source ? groupId || outputs > 1 ? `out-${index}` : 'out' : `in-${index}`,
      type: source ? 'source' as const : 'target' as const,
      position: vertical ? source ? FlowPosition.Bottom : FlowPosition.Top : source ? FlowPosition.Right : FlowPosition.Left,
      x: vertical ? rect.width * (index + 1) / ((source ? outputs : inputs) + 1) : source ? rect.width : 0,
      y: vertical ? source ? rect.height : 0 : rect.height * (index + 1) / ((source ? outputs : inputs) + 1),
      width: 0, height: 0,
    })))
    return { ...node, selectable: available && (!groupId || reveal < .95), focusable: available,
      extent: parentBounds ? [[parentBounds.x, parentBounds.y], [parentBounds.x + parentBounds.width, parentBounds.y + parentBounds.height]] as [[number, number], [number, number]] : undefined,
      width: rect.width, height: rect.height, measured: { width: rect.width, height: rect.height }, handles,
      dragHandle: groupId && reveal > .95 ? '.visual-group-title-row' : undefined,
      className: groupId ? 'continuous-module' : undefined,
      style: { ...node.style, pointerEvents: available && (!groupId || reveal < .95) ? 'all' as const : 'none' as const },
      ariaLabel: groupId ? (node.data as GroupNodeData).group.label : (node.data as BuilderNodeData).graphNode.label,
      data: { ...node.data, reveal, accessible: available, cameraZoom, frameHeaderVisible: groupId ? Math.max(0, (reveal - .95) / .05) * (1 - Math.max(0, Math.min(1, Math.max(node.width! * cameraZoom / canvasSize.width, node.height! * cameraZoom / canvasSize.height) - 1.1))) : 0 },
    }
  }) : nodes
  const presentedEdges = scene ? routedEdges.map(edge => ({ ...edge, focusable: accessible(edge.data?.parentId as string | undefined),
    data: { ...edge.data!, cameraZoom, accessible: accessible(edge.data?.parentId as string | undefined) },
  })) : routedEdges
  const nodeSyncKey = useMemo(
    () =>
      JSON.stringify({
        continuous,
        activeNodeId: activeStep?.nodeId,
        edges: renderedGraph.edges,
        groups: renderedGraph.groups ?? [],
        nodes: renderedGraph.nodes,
        view: renderedGraph.view,
        selectedGroupId,
        selectedNodeIds,
        showGradient,
        showMath,
      }),
    [
      activeStep?.nodeId,
      continuous,
      renderedGraph.edges,
      renderedGraph.groups,
      renderedGraph.nodes,
      renderedGraph.view,
      selectedGroupId,
      selectedNodeIds,
      showGradient,
      showMath,
    ],
  )
  const edgeSyncKey = useMemo(
    () =>
      JSON.stringify({
        continuous,
        activeEdgeIds: activeStep?.edgeIds ?? [],
        edges: renderedGraph.edges,
        groups: renderedGraph.groups ?? [],
        phase,
        view: renderedGraph.view,
        showGradient,
        selectedEdgeId,
      }),
    [activeStep?.edgeIds, phase, renderedGraph.edges, renderedGraph.groups, renderedGraph.view, showGradient, selectedEdgeId, continuous],
  )
  const previousNodeSyncKey = useRef(nodeSyncKey)
  const previousEdgeSyncKey = useRef(edgeSyncKey)

  useLayoutEffect(() => {
    if (dragPositions.current) return
    if (previousNodeSyncKey.current === nodeSyncKey) return
    previousNodeSyncKey.current = nodeSyncKey
    setNodes(reactNodes)
  }, [nodeSyncKey, reactNodes, setNodes])

  useLayoutEffect(() => {
    if (previousEdgeSyncKey.current === edgeSyncKey) return
    previousEdgeSyncKey.current = edgeSyncKey
    setEdges(reactEdges)
  }, [edgeSyncKey, reactEdges, setEdges])

  const onNodesChange = useCallback(
    (changes: NodeChange<CanvasNode>[]) => {
      const removedNodeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []))
      const removedGroupIds = removedNodeIds.map(groupIdFromNodeId).filter((groupId): groupId is string => Boolean(groupId))
      const removedGraphNodeIds = removedNodeIds.filter((nodeId) => !isGroupNodeId(nodeId) && graph.nodes.some((node) => node.id === nodeId))
      const selectionChanges = changes.filter((change) => change.type === 'select')
      let nextGraph: GraphModel | undefined

      if (selectionChanges.length > 0) {
        const selectedById = new Map(nodes.map((node) => [node.id, Boolean(node.selected)]))
        for (const change of selectionChanges) {
          selectedById.set(change.id, change.selected)
        }

        const nextSelectedNodes = nodes.filter((node) => selectedById.get(node.id))
        const selectedGroupIds = nextSelectedNodes
          .map((node) => groupIdFromNodeId(node.id))
          .filter((groupId): groupId is string => Boolean(groupId))
        const nodeIds = nextSelectedNodes.filter((node) => !isGroupNodeId(node.id)).map((node) => node.id)
        if (selectedGroupIds.length > 1 || nodeIds.length > 0) {
          for (const id of selectedGroupIds) nodeIds.push(...(graph.groups?.find((group) => group.id === id)?.nodeIds ?? []))
        }

        onSelectionChange({
          nodeIds,
          groupId: nodeIds.length === 0 && selectedGroupIds.length === 1 ? selectedGroupIds[0] : undefined,
        })
      }

      if (removedGroupIds.length > 0) {
        nextGraph = removedGroupIds.reduce(
          (currentGraph, groupId) => deleteVisualGroup(currentGraph, groupId),
          nextGraph ?? graph,
        )
        onSelectionChange({ nodeIds: [] })
        onCancelPendingPlacement()
      }

      if (removedGraphNodeIds.length > 0) {
        const removed = new Set(removedGraphNodeIds)
        const sourceGraph = nextGraph ?? graph
        nextGraph = {
          ...sourceGraph,
          nodes: sourceGraph.nodes.filter((node) => !removed.has(node.id)),
          edges: sourceGraph.edges.filter((edge) => !removed.has(edge.source) && !removed.has(edge.target)),
          groups: removeNodesFromVisualGroups(sourceGraph, removed),
        }
        onSelectionChange({ nodeIds: [] })
        onCancelPendingPlacement()
      }

      if (nextGraph) {
        onGraphChange(nextGraph)
      }
      onNodesChangeBase(changes)
    },
    [graph, nodes, onCancelPendingPlacement, onGraphChange, onNodesChangeBase, onSelectionChange],
  )

  const onEdgesChange = useCallback(
    (changes: EdgeChange<CanvasEdge>[]) => {
      const removedEdgeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id.split('::')[0]] : []))
      if (removedEdgeIds.length > 0) {
        const removed = new Set(removedEdgeIds)
        onGraphChange({
          ...preserveLayoutForWiring(graph),
          edges: graph.edges.filter((edge) => !removed.has(edge.id)),
        })
      }
      onEdgesChangeBase(changes)
    },
    [graph, onEdgesChangeBase, onGraphChange],
  )

  // React Flow stores these frames and their contents as sibling nodes. Move a
  // frame's visible descendants with it, but never apply a selected parent's
  // displacement twice to a selected child.
  const dragRoots = useCallback((draggedNodes: CanvasNode[]) => draggedNodes.filter(candidate =>
    !draggedNodes.some(parent => parent.id !== candidate.id && canvasGroupContains(renderedGraph, parent.id, candidate.id)),
  ), [renderedGraph])
  const startNodeDrag = useCallback((_: unknown, node: CanvasNode) => {
    dragPositions.current = new Map(nodes.map(node => [node.id, { ...node.position }]))
    const groupId = groupIdFromNodeId(node.id)
    if (groupId && !node.selected) onSelectionChange({ nodeIds: [], groupId })
  }, [nodes, onSelectionChange])
  const moveNodeContents = useCallback((_: unknown, node: CanvasNode, draggedNodes: CanvasNode[] = [node]) => {
    const starts = dragPositions.current
    if (!starts) return
    const roots = dragRoots(draggedNodes)
    setNodes(current => current.map(candidate => {
      const parent = roots.find(root => canvasGroupContains(renderedGraph, root.id, candidate.id))
      const start = starts.get(candidate.id)
      const parentStart = parent && starts.get(parent.id)
      return parent && start && parentStart ? { ...candidate, position: {
        x: start.x + parent.position.x - parentStart.x,
        y: start.y + parent.position.y - parentStart.y,
      } } : candidate
    }))
  }, [dragRoots, renderedGraph, setNodes])
  const commitNodePosition = useCallback((_: unknown, node: CanvasNode, draggedNodes: CanvasNode[] = [node]) => {
    const starts = dragPositions.current
    dragPositions.current = undefined
    if (!starts) return
    const movements = dragRoots(draggedNodes).flatMap(moved => {
      const start = starts.get(moved.id)
      if (!start) return []
      const delta = { x: moved.position.x - start.x, y: moved.position.y - start.y }
      return delta.x || delta.y ? [{ id: moved.id, delta, position: moved.position }] : []
    })
    if (!movements.length) return
    if (semantic) {
      const layoutOffsets = { ...graph.view?.layoutOffsets }
      for (const { id, delta } of movements) {
        const offset = layoutOffsets[id] ?? { x: 0, y: 0 }
        layoutOffsets[id] = { x: offset.x + delta.x, y: offset.y + delta.y }
      }
      changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], layoutOffsets })
    } else {
      let next = graph
      for (const { id, delta, position } of movements) {
        const groupId = groupIdFromNodeId(id)
        const group = next.groups?.find(candidate => candidate.id === groupId)
        if (group) next = moveVisualGroup(next, group.id, { x: group.position.x + delta.x, y: group.position.y + delta.y })
        else next = { ...next, nodes: next.nodes.map(candidate => candidate.id === id ? { ...candidate, position } : candidate) }
      }
      onGraphChange(next)
    }
  }, [changeView, dragRoots, graph, onGraphChange, semantic])

  const onConnect = useCallback(
    (connection: Connection) => {
      const graphConnection = normalizeCanvasConnection(graph, connection)
      const nextGraph = connectGraphNodes(
        graph,
        graphConnection,
        (source, target, inputSlot) => `${source}-${target}-${inputSlot}-${Date.now()}`,
      )
      if (!nextGraph) return
      onGraphChange(nextGraph)
    },
    [graph, onGraphChange],
  )

  const isValidConnection = useCallback(
    (connection: Connection | Edge<BuilderEdgeData>) => {
      return canConnectGraphNodes(graph, normalizeCanvasConnection(graph, connection))
    },
    [graph],
  )

  const handlePaneClick = useCallback(
    (event: ReactMouseEvent) => {
      if (pendingNodeType) {
        const position = screenToFlowPosition({ x: event.clientX, y: event.clientY })
        const parentGroupId = insertionGroupAt(position)
        if (parentGroupId) onCreateNode(pendingNodeType, position, parentGroupId)
        else if (scene) onCreateNode(pendingNodeType, position, undefined, insertionScaleAt(position))
        else onCreateNode(pendingNodeType, position)
        setAddMenu(undefined)
        return
      }
      onSelectionChange({ nodeIds: [] })
      setAddMenu(undefined)
      setRenameGroup(undefined)
    },
    [onCreateNode, onSelectionChange, pendingNodeType, screenToFlowPosition, insertionGroupAt, insertionScaleAt, scene],
  )

  const handlePaneDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (pendingNodeType || !(event.target as HTMLElement).classList.contains('react-flow__pane')) return
      event.preventDefault()
      const frame = shell.current?.getBoundingClientRect()
      if (!frame) return
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY })
      const parentGroupId = insertionGroupAt(position)
      setAddQuery('')
      setActiveSuggestion(0)
      setAddMenu({ x: Math.max(8, Math.min(event.clientX - frame.left, frame.width - 288)), y: Math.max(8, Math.min(event.clientY - frame.top, frame.height - 350)), position, parentGroupId, sceneScale: parentGroupId ? undefined : insertionScaleAt(position) })
    },
    [pendingNodeType, screenToFlowPosition, insertionGroupAt, insertionScaleAt],
  )

  const handleFlowPointerDownCapture = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button === 2) {
        const node = (event.target as HTMLElement).closest('.react-flow__node')
        if (node?.getAttribute('data-id')?.startsWith(GROUP_NODE_ID_PREFIX)) return
        setAddMenu(undefined)
        event.preventDefault()
        event.stopPropagation()
        secondaryPan.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, viewport: getViewport() }
        event.currentTarget.setPointerCapture(event.pointerId)
        return
      }
      if (!pendingNodeType) return
      const target = event.target as HTMLElement
      const isPlacementSurface =
        target.classList.contains('react-flow__pane') ||
        Boolean(target.closest('.react-flow__background'))

      if (!isPlacementSurface) {
        onCancelPendingPlacement()
      }
    },
    [getViewport, onCancelPendingPlacement, pendingNodeType],
  )

  const handleNodeClick = useCallback(
    (_: ReactMouseEvent, node: CanvasNode) => {
      setAddMenu(undefined)
      setRenameGroup(undefined)
      const groupId = groupIdFromNodeId(node.id)
      const group = renderedGraph.groups?.find((candidate) => candidate.id === groupId)
      if (group?.detail?.virtual && typeof group.detail.layerId === 'string' && typeof group.detail.unitIndex === 'number') {
        onInspectNeuron?.(group.detail.layerId, group.detail.unitIndex)
        return
      }
      onSelectionChange(groupId ? { nodeIds: [], groupId } : { nodeIds: [node.id] })
    },
    [onSelectionChange, onInspectNeuron, renderedGraph.groups],
  )

  const handleNodeDoubleClick = useCallback(
    (_: ReactMouseEvent, node: CanvasNode) => {
      const groupId = groupIdFromNodeId(node.id)
      if (groupId) toggleGroup(groupId)
    },
    [toggleGroup],
  )

  const selectedGroup = selectedGroupId ? graph.groups?.find((group) => group.id === selectedGroupId) : undefined
  const groupCount = renderedGraph.groups?.length ?? 0
  const center = { x: (canvasSize.width / 2 - camera.x) / camera.zoom, y: (canvasSize.height / 2 - camera.y) / camera.zoom }
  const cameraFocus = scene && [...scene.groups].filter(([id, rect]) => (reveals.get(id) ?? 0) > .95 && center.x >= rect.x && center.x <= rect.x + rect.width && center.y >= rect.y && center.y <= rect.y + rect.height)
    .sort(([a], [b]) => semanticGroupDepth(renderedGraph, b) - semanticGroupDepth(renderedGraph, a))[0]?.[0]
  const navigationFocus = continuous ? cameraFocus : graph.view?.focusedGroupId
  const breadcrumb = navigationFocus ? groupAncestors(renderedGraph, navigationFocus) : []
  const suggestions = blockPalette.filter(item => `${item.label} ${item.type}`.toLowerCase().includes(addQuery.trim().toLowerCase()))
  const placeSuggestion = (type: NodeType) => {
    if (!addMenu) return
    if (addMenu.parentGroupId) onCreateNode(type, addMenu.position, addMenu.parentGroupId)
    else if (addMenu.sceneScale) onCreateNode(type, addMenu.position, undefined, addMenu.sceneScale)
    else onCreateNode(type, addMenu.position)
    setAddMenu(undefined)
  }

  return (
    <section className="canvas-panel" aria-label="Graph canvas">
      <div className="canvas-header">
        <div>
          <p className="eyebrow">Graph canvas</p>
          <h2>{semantic ? 'Follow the computation' : 'Tensor computation graph'}</h2>
          {semantic && <div className="flow-legend"><span><i/>Forward →</span><span><i/>← Gradient</span><span>Color strength = magnitude · click a wire for values</span></div>}
        </div>
        <div className="canvas-header-actions">
          <p>
            {graph.nodes.length} nodes, {graph.edges.length} edges{groupCount > 0 ? `, ${groupCount} ${groupCount === 1 ? 'group' : 'groups'}` : ''}
          </p>
          <button type="button" className="canvas-action-button" onClick={() => changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], canvasStyle: semantic ? 'builder' : 'architecture', semanticZoom: graph.view?.semanticZoom ?? true })}>
            {semantic ? 'Builder cards' : 'Architecture cards'}
          </button>
          {selectedNodeIds.length >= 2 ? (
            <button type="button" className="canvas-action-button" onClick={onGroupCreate}>
              <Combine size={15} />
              Merge selection
            </button>
          ) : null}
          {selectedGroup ? (
            <button type="button" className="canvas-action-button" onClick={() => onGroupExplode(selectedGroup.id)}>
              <Ungroup size={15} />
              Ungroup module
            </button>
          ) : null}
        </div>
      </div>
      {groupCount > 0 ? <nav className="module-navigation" aria-label="Module navigation">
        <button type="button" onClick={() => focusGroup(undefined)}>Model</button>
        {breadcrumb.map((group) => <span key={group.id}><span aria-hidden="true"> / </span><button type="button" onClick={() => focusGroup(group.id)}>{group.label}</button></span>)}
        <button type="button" disabled={breadcrumb.length === 0} onClick={() => {
          const current = breadcrumb[breadcrumb.length - 1]
          if (current) {
            const next = setVisualGroupExpanded(graph, current.id, false)
            if (next.view) changeView({ ...next.view, focusedGroupId: current.parentId })
            if (continuous) zoomToGroup(current.parentId)
            else window.setTimeout(() => void fitView({ nodes: current.parentId ? [{ id: groupNodeId(current.parentId) }] : undefined, padding: .18, duration: 650, maxZoom: 1.5 }), 100)
          }
        }}><ArrowUp size={14} /> Up one level</button>
        <button type="button" onClick={() => focusGroup(undefined)}><Maximize size={14} /> Fit model</button>
        {semantic ? <button type="button" onClick={compactLayout} title="Rearrange all blocks and fit the model; keep the current weights and values"><LayoutGrid size={14} /> Compact layout</button> : null}
        {semantic ? <label className="semantic-zoom-switch"><input type="checkbox" checked={graph.view?.semanticZoom !== false} onChange={(event) => changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], semanticZoom: event.target.checked })}/> Zoom reveals detail</label> : null}
      </nav> : null}
      <div className={`flow-shell ${semantic ? 'semantic-flow' : ''}`} ref={shell} onPointerDownCapture={handleFlowPointerDownCapture} onDoubleClick={handlePaneDoubleClick}
        onContextMenu={(event) => event.preventDefault()}
        onPointerMove={(event) => {
          const pan = secondaryPan.current
          if (pan && pan.pointerId === event.pointerId) {
            event.preventDefault()
            void setViewport({ ...pan.viewport, x: pan.viewport.x + event.clientX - pan.x, y: pan.viewport.y + event.clientY - pan.y })
          }
        }}
        onPointerUp={(event) => {
          if (secondaryPan.current?.pointerId !== event.pointerId) return
          secondaryPan.current = undefined
          event.currentTarget.releasePointerCapture(event.pointerId)
        }}
        onLostPointerCapture={() => { secondaryPan.current = undefined }}>
        <ReactFlow
          nodes={presentedNodes}
          edges={presentedEdges}
          elevateNodesOnSelect={false}
          elevateEdgesOnSelect={false}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView={!graph.view?.viewport}
          fitViewOptions={INITIAL_FIT_OPTIONS}
          defaultViewport={graph.view?.viewport}
          onMove={(_, viewport) => setCamera(viewport)}
          onMoveEnd={(_, viewport) => {
            emittedViewport.current = viewport
            setCamera(viewport)
            const previous = graph.view?.viewport
            if (!previous || previous.x !== viewport.x || previous.y !== viewport.y || previous.zoom !== viewport.zoom) {
              changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], viewport })
            }
          }}
          minZoom={semantic ? 0.01 : 0.15}
          maxZoom={maxZoom}
          onConnect={onConnect}
          onEdgesChange={onEdgesChange}
          onNodesChange={onNodesChange}
          onNodeDragStart={startNodeDrag}
          onNodeDrag={moveNodeContents}
          onNodeDragStop={commitNodePosition}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onNodeContextMenu={(event, node) => {
            const groupId = groupIdFromNodeId(node.id)
            const group = graph.groups?.find(candidate => candidate.id === groupId)
            if (!group) return
            event.preventDefault()
            event.stopPropagation()
            const frame = shell.current?.getBoundingClientRect()
            if (!frame) return
            setAddMenu(undefined)
            onSelectionChange({ nodeIds: [], groupId })
            setRenameGroup({ id: group.id, label: group.label, x: Math.max(8, Math.min(event.clientX - frame.left, frame.width - 240)), y: Math.max(8, Math.min(event.clientY - frame.top, frame.height - 100)) })
          }}
          onEdgeClick={(_, edge) => onInspectEdge?.(edge.data?.canonicalEdgeId ?? edge.id)}
          onPaneClick={handlePaneClick}
          zoomOnDoubleClick={false}
          isValidConnection={isValidConnection}
          selectionOnDrag={!pendingNodeType}
          selectionMode={SelectionMode.Partial}
          panOnDrag={[1, 2]}
          autoPanOnNodeDrag={false}
          selectNodesOnDrag
          selectionKeyCode={null}
          className={pendingNodeType ? 'placement-mode' : undefined}
        >
          <Background color="var(--grid-dot)" gap={continuous ? 24 / cameraZoom : 24} size={continuous ? 1 / cameraZoom : 1} variant={BackgroundVariant.Dots} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {addMenu ? <div className="canvas-add-menu" role="dialog" aria-label="Add a block" style={{ left: addMenu.x, top: addMenu.y }} onPointerDown={event => event.stopPropagation()} onClick={event => event.stopPropagation()}>
          <label className="canvas-add-search"><Search size={15} /><input ref={addSearch} type="search" aria-label="Search blocks" autoComplete="off" placeholder="Type a block name…" value={addQuery}
            onChange={event => { setAddQuery(event.target.value); setActiveSuggestion(0) }}
            onKeyDown={event => {
              if (event.key === 'Escape') { event.preventDefault(); setAddMenu(undefined) }
              if (event.key === 'ArrowDown') { event.preventDefault(); setActiveSuggestion(index => Math.min(index + 1, suggestions.length - 1)) }
              if (event.key === 'ArrowUp') { event.preventDefault(); setActiveSuggestion(index => Math.max(0, index - 1)) }
              if (event.key === 'Enter' && suggestions.length) { event.preventDefault(); placeSuggestion(suggestions[activeSuggestion]?.type ?? suggestions[0].type) }
            }} /></label>
          <div className="canvas-add-results" role="listbox" aria-label="Block types">
            {suggestions.length ? suggestions.map((item, index) => <button type="button" key={item.type} role="option" aria-selected={index === activeSuggestion} className={index === activeSuggestion ? 'is-active' : ''} onMouseEnter={() => setActiveSuggestion(index)} onClick={() => placeSuggestion(item.type)}><span>{item.label}</span><small>{item.type}</small></button>) : <p>No matching blocks</p>}
          </div>
          <div className="canvas-add-hint">↑ ↓ choose · Enter place · Esc close</div>
        </div> : null}
        {renameGroup ? <form className="canvas-group-rename" role="dialog" aria-label="Name block group" style={{ left: renameGroup.x, top: renameGroup.y }} onPointerDown={event => event.stopPropagation()} onClick={event => event.stopPropagation()} onSubmit={event => { event.preventDefault(); const name = renameGroup.label.trim(); if (name) onGroupRename?.(renameGroup.id, name); setRenameGroup(undefined) }}>
          <label>Block name<input autoFocus aria-label="Group name" value={renameGroup.label} onChange={event => setRenameGroup({ ...renameGroup, label: event.target.value })} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); setRenameGroup(undefined) } }} /></label>
          <button type="submit">Rename</button>
        </form> : null}
        {semantic ? <div className="semantic-canvas-hint"><ScanSearch size={15}/><span>Drag to move or select · two-finger click-drag to pan · double-click blocks to explore or canvas to add</span></div> : null}
      </div>
    </section>
  )
}

interface RoutingBlock extends WireObstacle { expanded: boolean; vertical: boolean; inputs: number; outputs: number }
function routingEndpoint(block: RoutingBlock, handle: string | null | undefined, source: boolean): WireEndpoint {
  const slot = Number(handle?.match(/-(\d+)$/)?.[1] ?? 0)
  const fraction = (slot + 1) / (Math.max(1, source ? block.outputs : block.inputs) + 1)
  return block.vertical
    ? { x: block.x + block.width * fraction, y: block.y + (source ? block.height : 0), side: source ? 'bottom' : 'top' }
    : { x: block.x + (source ? block.width : 0), y: block.y + block.height * fraction, side: source ? 'right' : 'left' }
}

function canvasGroupContains(graph: GraphModel, parentId: string, childId: string): boolean {
  const groupId = groupIdFromNodeId(parentId)
  if (!groupId) return false
  const childGroupId = groupIdFromNodeId(childId)
  if (childGroupId) return groupAncestors(graph, childGroupId).some(group => group.id === groupId && group.id !== childGroupId)
  return graph.groups?.find(group => group.id === groupId)?.nodeIds.includes(childId) ?? false
}

function minimumInputCountForNode(graph: GraphModel, node: GraphNode): number {
  if (!isFlexibleInputNodeType(node.type)) return inputArityForNode(node)

  const highestConnectedSlot = graph.edges
    .filter((edge) => edge.target === node.id)
    .reduce((highest, edge) => Math.max(highest, edge.inputSlot ?? 0), -1)

  return Math.max(2, highestConnectedSlot + 1)
}

function sourceHandleForEdge(graph: GraphModel, edge: { source: string; sourceSlot?: number }): string {
  const source = graph.nodes.find((node) => node.id === edge.source)
  if (!source || outputArityForNode(source) <= 1) return 'out'
  return `out-${edge.sourceSlot ?? 0}`
}

function visualGroupHandleHasEdge(handle: { edgeId?: string; edgeIds?: string[] }, edgeId: string): boolean {
  return edgeIdsForVisualGroupHandle(handle).includes(edgeId)
}

function edgeIdsForVisualGroupHandle(handle: { edgeId?: string; edgeIds?: string[] }): string[] {
  if (handle.edgeIds) return handle.edgeIds
  return handle.edgeId ? [handle.edgeId] : []
}

function normalizeCanvasConnection(
  graph: GraphModel,
  connection: Connection | Edge<BuilderEdgeData>,
): GraphConnection {
  let normalizedConnection: GraphConnection = connection
  const sourceGroupId = connection.source ? groupIdFromNodeId(connection.source) : undefined
  if (sourceGroupId) {
    const resolvedSource = resolveVisualGroupOutputHandle(graph, sourceGroupId, connection.sourceHandle ?? undefined)
    if (resolvedSource) {
      normalizedConnection = {
        ...normalizedConnection,
        source: resolvedSource.source,
        sourceHandle: resolvedSource.sourceHandle,
        replaceEdgeId: resolvedSource.edgeId,
      }
    }
  }

  const targetGroupId = connection.target ? groupIdFromNodeId(connection.target) : undefined
  if (!targetGroupId) return normalizedConnection

  const resolvedTarget = resolveVisualGroupInputHandle(graph, targetGroupId, connection.targetHandle ?? undefined)
  if (!resolvedTarget) return normalizedConnection

  return {
    ...normalizedConnection,
    target: resolvedTarget.target,
    targetHandle: resolvedTarget.targetHandle,
  }
}

function groupNodeId(groupId: string): string {
  return `${GROUP_NODE_ID_PREFIX}${groupId}`
}

function groupIdFromNodeId(nodeId: string): string | undefined {
  return isGroupNodeId(nodeId) ? nodeId.slice(GROUP_NODE_ID_PREFIX.length) : undefined
}

function isGroupNodeId(nodeId: string): boolean {
  return nodeId.startsWith(GROUP_NODE_ID_PREFIX)
}

/** Layout needs CSV column names and count, but never the full table. Keep the
 * geometry dependency small even when the imported CSV has thousands of rows. */
function csvGeometrySample(csv: CustomCsvData): CustomCsvData {
  const width = csv.rows[0].length
  const first = Array.from({ length: width }, () => '0')
  const second = [...first]
  second[csv.targetColumn] = '1'
  return { ...csv, rows: csv.hasHeader ? [csv.rows[0], first, second] : [first, second] }
}
