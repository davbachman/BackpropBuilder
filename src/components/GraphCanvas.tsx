import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  useReactFlow,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import { ArrowUp, Combine, Maximize, Ungroup } from 'lucide-react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactElement } from 'react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
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
} from '../domain/grouping'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import type {
  ActivationKind,
  DatasetKind,
  EvaluationTraceStep,
  GraphModel,
  GraphNode,
  GraphViewState,
  LossKind,
  NodeType,
  Position as GraphPosition,
  TensorValue,
} from '../domain/types'
import { BuilderEdge, type BuilderEdgeData } from './BuilderEdge'
import { BuilderNode, type BuilderNodeData } from './BuilderNode'
import { GroupNode, type GroupNodeData } from './GroupNode'
import './modules.css'

const nodeTypes = { builderNode: BuilderNode, groupNode: GroupNode }
const edgeTypes = { builderEdge: BuilderEdge }
const GROUP_NODE_ID_PREFIX = 'visual-group:'
const EMPTY_SELECTED_NODE_IDS: string[] = []

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
  showMath: boolean
  showGradient: boolean
  phase: string
  pendingNodeType?: NodeType
  onGraphChange: (graph: GraphModel) => void
  onViewChange?: (view: GraphViewState) => void
  onSelectionChange: (selection: CanvasSelection) => void
  onCreateNode: (type: NodeType, position: GraphPosition) => void
  onCancelPendingPlacement: () => void
  onNodeValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, activation: ActivationKind) => void
  onLossChange?: (nodeId: string, loss: LossKind) => void
  onDatasetChange?: (nodeId: string, dataset: DatasetKind) => void
  onGroupCreate: () => void
  onGroupExplode: (groupId: string) => void
  onGroupMove: (groupId: string, position: GraphPosition) => void
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
  onLossChange,
  onDatasetChange,
  onGroupCreate,
  onGroupExplode,
  onGroupMove,
}: GraphCanvasProps): ReactElement {
  const { screenToFlowPosition, fitView, setViewport } = useReactFlow()
  const renderedGraph = displayGraph ?? graph
  const groupedNodeIds = useMemo(() => new Set(renderedGraph.nodes.filter((node) => collapsedGroupForNode(renderedGraph, node.id)).map((node) => node.id)), [renderedGraph])
  const changeView = useCallback((view: GraphViewState) => {
    if (onViewChange) onViewChange(view)
    else onGraphChange({ ...graph, view })
  }, [graph, onGraphChange, onViewChange])
  const toggleGroup = useCallback((groupId: string) => {
    const next = setVisualGroupExpanded(graph, groupId, !graph.view?.expandedGroupIds.includes(groupId))
    if (next.view) changeView(next.view)
  }, [graph, changeView])
  const focusGroup = useCallback((groupId?: string) => {
    changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], focusedGroupId: groupId })
    const group = graph.groups?.find((candidate) => candidate.id === groupId)
    void fitView({ nodes: group ? group.nodeIds.map((id) => ({ id })) : undefined, padding: 0.18 })
  }, [graph, changeView, fitView])
  const savedViewport = graph.view?.viewport
  useEffect(() => {
    if (savedViewport) void setViewport(savedViewport)
  }, [savedViewport, setViewport])
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
      const groupNodes = visibleGroups(renderedGraph).map<CanvasNode>((group) => {
        const groupInterface = groupInterfaces.get(group.id)
        const expanded = renderedGraph.view?.expandedGroupIds.includes(group.id) ?? false
        const rect = expanded ? expandedGroupRect(renderedGraph, group) : undefined

        return {
          id: groupNodeId(group.id),
          type: 'groupNode',
          position: rect ? { x: rect.x, y: rect.y } : group.position,
          width: rect?.width ?? group.dimensions.width,
          height: rect?.height ?? group.dimensions.height,
          draggable: !expanded,
          zIndex: expanded ? -1 : 0,
          className: expanded ? 'expanded-module-frame' : undefined,
          selected: group.id === selectedGroupId,
          data: {
            group: rect ? { ...group, dimensions: { width: rect.width, height: rect.height } } : group,
            expanded,
            inputCount: groupInterface?.inputs.length ?? 0,
            outputCount: groupInterface?.outputs.length ?? 0,
            outputMetrics: (groupInterface?.outputs ?? []).map((output) => {
              const edgeIds = edgeIdsForVisualGroupHandle(output)
              const edge = renderedGraph.edges.find((candidate) => edgeIds.includes(candidate.id))
              const source = renderedGraph.nodes.find((node) => node.id === output.source)
              return { forward: edge?.value ?? source?.value, gradient: edge?.grad ?? source?.grad }
            }),
            showGradient,
            active: group.nodeIds.includes(activeStep?.nodeId ?? ''),
            onToggle: toggleGroup,
          },
        }
      })

      const builderNodes = renderedGraph.nodes
        .filter((node) => !groupedNodeIds.has(node.id))
        .map<CanvasNode>((node) => ({
          id: node.id,
          type: 'builderNode',
          position: node.position,
          width: isFlexibleInputNodeType(node.type) ? NODE_WIDTH : undefined,
          height: isFlexibleInputNodeType(node.type)
            ? node.dimensions?.height ?? heightForInputCount(inputArityForNode(node))
            : undefined,
          selected: selectedNodeIdSet.has(node.id),
          data: {
            graphNode: node,
            showMath,
            showGradient,
            formula: formulaForNode(node, renderedGraph, formatCompactTensor),
            fullFormula: formulaForNode(node, renderedGraph, formatFullTensor),
            lossKind: node.type === 'loss' ? lossKindForNode(node, renderedGraph) : undefined,
            lossOptions: node.type === 'loss' ? lossOptionsForNode(node, renderedGraph) : undefined,
            active: activeStep?.nodeId === node.id,
            hasIncomingValue: renderedGraph.edges.some((edge) => edge.target === node.id),
            onFlexibleInputAdd: addFlexibleInput,
            onValueChange: onNodeValueChange,
            onActivationChange: (nodeId: string, value: string) => onActivationChange(nodeId, value as ActivationKind),
            onLossChange: onLossChange ?? (() => undefined),
            onDatasetChange: onDatasetChange ?? (() => undefined),
          },
        }))

      return [...groupNodes, ...builderNodes]
    },
    [
      activeStep?.nodeId,
      onActivationChange,
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
    ],
  )

  const reactEdges = useMemo(
    () =>
      renderedGraph.edges.flatMap<CanvasEdge>((edge) => {
        const sourceGroup = collapsedGroupForNode(renderedGraph, edge.source)
        const targetGroup = collapsedGroupForNode(renderedGraph, edge.target)
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
            animated: activeStep?.edgeIds.includes(edge.id),
            data: {
              forward: edge.value,
              gradient: edge.grad,
              showGradient,
              active: activeStep?.edgeIds.includes(edge.id) ?? false,
              phase,
            },
          },
        ]
      }),
    [activeStep?.edgeIds, groupInterfaces, phase, renderedGraph, showGradient],
  )

  const [nodes, setNodes, onNodesChangeBase] = useNodesState(reactNodes)
  const [edges, setEdges, onEdgesChangeBase] = useEdgesState(reactEdges)
  const nodeSyncKey = useMemo(
    () =>
      JSON.stringify({
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
        activeEdgeIds: activeStep?.edgeIds ?? [],
        edges: renderedGraph.edges,
        groups: renderedGraph.groups ?? [],
        phase,
        view: renderedGraph.view,
        showGradient,
      }),
    [activeStep?.edgeIds, phase, renderedGraph.edges, renderedGraph.groups, renderedGraph.view, showGradient],
  )
  const previousNodeSyncKey = useRef(nodeSyncKey)
  const previousEdgeSyncKey = useRef(edgeSyncKey)

  useEffect(() => {
    if (previousNodeSyncKey.current === nodeSyncKey) return
    previousNodeSyncKey.current = nodeSyncKey
    setNodes(reactNodes)
  }, [nodeSyncKey, reactNodes, setNodes])

  useEffect(() => {
    if (previousEdgeSyncKey.current === edgeSyncKey) return
    previousEdgeSyncKey.current = edgeSyncKey
    setEdges(reactEdges)
  }, [edgeSyncKey, reactEdges, setEdges])

  const onNodesChange = useCallback(
    (changes: NodeChange<CanvasNode>[]) => {
      const removedNodeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []))
      const removedGroupIds = removedNodeIds.map(groupIdFromNodeId).filter((groupId): groupId is string => Boolean(groupId))
      const removedGraphNodeIds = removedNodeIds.filter((nodeId) => !isGroupNodeId(nodeId))
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
      const removedEdgeIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []))
      if (removedEdgeIds.length > 0) {
        const removed = new Set(removedEdgeIds)
        onGraphChange({
          ...graph,
          edges: graph.edges.filter((edge) => !removed.has(edge.id)),
        })
      }
      onEdgesChangeBase(changes)
    },
    [graph, onEdgesChangeBase, onGraphChange],
  )

  const commitNodePosition = useCallback(
    (_: unknown, node: CanvasNode, draggedNodes: CanvasNode[] = [node]) => {
      const movedGroup = draggedNodes.find((draggedNode) => isGroupNodeId(draggedNode.id))
      if (movedGroup) {
        const groupId = groupIdFromNodeId(movedGroup.id)
        if (groupId) onGroupMove(groupId, movedGroup.position)
        return
      }

      const movedPositions = new Map(draggedNodes.map((draggedNode) => [draggedNode.id, draggedNode.position]))
      onGraphChange({
        ...graph,
        nodes: graph.nodes.map((graphNode) => {
          const position = movedPositions.get(graphNode.id)
          return position ? { ...graphNode, position } : graphNode
        }),
      })
    },
    [graph, onGraphChange, onGroupMove],
  )

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
        onCreateNode(
          pendingNodeType,
          screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
          }),
        )
        return
      }
      onSelectionChange({ nodeIds: [] })
    },
    [onCreateNode, onSelectionChange, pendingNodeType, screenToFlowPosition],
  )

  const handleFlowPointerDownCapture = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!pendingNodeType) return
      const target = event.target as HTMLElement
      const isPlacementSurface =
        target.classList.contains('react-flow__pane') ||
        Boolean(target.closest('.react-flow__background'))

      if (!isPlacementSurface) {
        onCancelPendingPlacement()
      }
    },
    [onCancelPendingPlacement, pendingNodeType],
  )

  const handleNodeClick = useCallback(
    (_: ReactMouseEvent, node: CanvasNode) => {
      const groupId = groupIdFromNodeId(node.id)
      onSelectionChange(groupId ? { nodeIds: [], groupId } : { nodeIds: [node.id] })
    },
    [onSelectionChange],
  )

  const handleNodeDoubleClick = useCallback(
    (_: ReactMouseEvent, node: CanvasNode) => {
      const groupId = groupIdFromNodeId(node.id)
      if (groupId) toggleGroup(groupId)
    },
    [toggleGroup],
  )

  const selectedGroup = selectedGroupId ? graph.groups?.find((group) => group.id === selectedGroupId) : undefined
  const groupCount = graph.groups?.length ?? 0
  const breadcrumb = graph.view?.focusedGroupId ? groupAncestors(graph, graph.view.focusedGroupId) : []

  return (
    <section className="canvas-panel" aria-label="Graph canvas">
      <div className="canvas-header">
        <div>
          <p className="eyebrow">Graph canvas</p>
          <h2>Tensor computation graph</h2>
        </div>
        <div className="canvas-header-actions">
          <p>
            {graph.nodes.length} nodes, {graph.edges.length} edges{groupCount > 0 ? `, ${groupCount} groups` : ''}
          </p>
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
          if (current) { const next = setVisualGroupExpanded(graph, current.id, false); if (next.view) changeView(next.view) }
        }}><ArrowUp size={14} /> Up one level</button>
        <button type="button" onClick={() => void fitView({ padding: 0.18 })}><Maximize size={14} /> Fit view</button>
      </nav> : null}
      <div className="flow-shell" onPointerDownCapture={handleFlowPointerDownCapture}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView={!graph.view?.viewport}
          defaultViewport={graph.view?.viewport}
          onMoveEnd={(_, viewport) => {
            const previous = graph.view?.viewport
            if (!previous || previous.x !== viewport.x || previous.y !== viewport.y || previous.zoom !== viewport.zoom) {
              changeView({ ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], viewport })
            }
          }}
          minZoom={0.35}
          maxZoom={1.4}
          onConnect={onConnect}
          onEdgesChange={onEdgesChange}
          onNodesChange={onNodesChange}
          onNodeDragStop={commitNodePosition}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onPaneClick={handlePaneClick}
          isValidConnection={isValidConnection}
          selectionOnDrag={!pendingNodeType}
          selectionMode={SelectionMode.Partial}
          panOnDrag={pendingNodeType ? true : [1, 2]}
          className={pendingNodeType ? 'placement-mode' : undefined}
        >
          <Background color="var(--grid-dot)" gap={24} variant={BackgroundVariant.Dots} />
          <MiniMap pannable zoomable nodeStrokeWidth={3} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </section>
  )
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
