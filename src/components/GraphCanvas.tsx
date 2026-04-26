import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  addEdge,
  useReactFlow,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import { Combine, Ungroup } from 'lucide-react'
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactElement } from 'react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import {
  MAX_FLEX_INPUT_COUNT,
  NODE_WIDTH,
  formulaForNode,
  heightForInputCount,
  inputArityForNode,
  isFlexibleInputNodeType,
} from '../domain/engine'
import { groupForNode, nodeIdsInGroups, removeNodesFromVisualGroups, visualGroupInterface } from '../domain/grouping'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import type {
  ActivationKind,
  EvaluationTraceStep,
  GraphModel,
  GraphNode,
  NodeType,
  Position as GraphPosition,
  TensorValue,
} from '../domain/types'
import { BuilderEdge, type BuilderEdgeData } from './BuilderEdge'
import { BuilderNode, type BuilderNodeData } from './BuilderNode'
import { GroupNode, type GroupNodeData } from './GroupNode'

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
  onSelectionChange: (selection: CanvasSelection) => void
  onCreateNode: (type: NodeType, position: GraphPosition) => void
  onCancelPendingPlacement: () => void
  onNodeValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, activation: ActivationKind) => void
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
  onSelectionChange,
  onCreateNode,
  onCancelPendingPlacement,
  onNodeValueChange,
  onActivationChange,
  onGroupCreate,
  onGroupExplode,
  onGroupMove,
}: GraphCanvasProps): ReactElement {
  const { screenToFlowPosition } = useReactFlow()
  const renderedGraph = displayGraph ?? graph
  const groupedNodeIds = useMemo(() => nodeIdsInGroups(renderedGraph), [renderedGraph])
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
      const groupNodes = (renderedGraph.groups ?? []).map<CanvasNode>((group) => {
        const groupInterface = groupInterfaces.get(group.id)

        return {
          id: groupNodeId(group.id),
          type: 'groupNode',
          position: group.position,
          width: group.dimensions.width,
          height: group.dimensions.height,
          selected: group.id === selectedGroupId,
          connectable: false,
          data: {
            group,
            inputCount: groupInterface?.inputs.length ?? 0,
            outputCount: groupInterface?.outputs.length ?? 0,
            outputMetrics: (groupInterface?.outputs ?? []).map((output) => {
              const edge = renderedGraph.edges.find((candidate) => candidate.id === output.edgeId)
              return { forward: edge?.value, gradient: edge?.grad }
            }),
            showGradient,
            active: group.nodeIds.includes(activeStep?.nodeId ?? ''),
            onExplode: onGroupExplode,
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
            active: activeStep?.nodeId === node.id,
            onFlexibleInputAdd: addFlexibleInput,
            onValueChange: onNodeValueChange,
            onActivationChange: (nodeId: string, value: string) => onActivationChange(nodeId, value as ActivationKind),
          },
        }))

      return [...groupNodes, ...builderNodes]
    },
    [
      activeStep?.nodeId,
      onActivationChange,
      addFlexibleInput,
      onNodeValueChange,
      onGroupExplode,
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
        const sourceGroup = groupForNode(renderedGraph, edge.source)
        const targetGroup = groupForNode(renderedGraph, edge.target)
        if (sourceGroup?.id && sourceGroup.id === targetGroup?.id) return []
        const sourceHandle = sourceGroup
          ? groupInterfaces.get(sourceGroup.id)?.outputs.find((handle) => handle.edgeId === edge.id)?.handleId
          : undefined
        const targetHandle = targetGroup
          ? groupInterfaces.get(targetGroup.id)?.inputs.find((handle) => handle.edgeId === edge.id)?.handleId
          : undefined

        return [
          {
            id: edge.id,
            source: sourceGroup ? groupNodeId(sourceGroup.id) : edge.source,
            target: targetGroup ? groupNodeId(targetGroup.id) : edge.target,
            sourceHandle: sourceHandle ?? 'out',
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
        showGradient,
      }),
    [activeStep?.edgeIds, phase, renderedGraph.edges, renderedGraph.groups, showGradient],
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

        onSelectionChange({
          nodeIds,
          groupId: nodeIds.length === 0 && selectedGroupIds.length === 1 ? selectedGroupIds[0] : undefined,
        })
      }

      if (removedGroupIds.length > 0) {
        onGroupExplode(removedGroupIds[0])
      }

      if (removedGraphNodeIds.length > 0) {
        const removed = new Set(removedGraphNodeIds)
        nextGraph = {
          ...graph,
          nodes: graph.nodes.filter((node) => !removed.has(node.id)),
          edges: graph.edges.filter((edge) => !removed.has(edge.source) && !removed.has(edge.target)),
          groups: removeNodesFromVisualGroups(graph, removed),
        }
        onSelectionChange({ nodeIds: [] })
        onCancelPendingPlacement()
      }

      if (nextGraph) {
        onGraphChange(nextGraph)
      }
      onNodesChangeBase(changes)
    },
    [graph, nodes, onCancelPendingPlacement, onGraphChange, onGroupExplode, onNodesChangeBase, onSelectionChange],
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
      if (!connection.source || !connection.target) return
      if (isGroupNodeId(connection.source) || isGroupNodeId(connection.target)) return
      const target = graph.nodes.find((node) => node.id === connection.target)
      if (!target) return
      const slot = Number(connection.targetHandle?.replace('in-', '') ?? nextOpenSlot(graph, target))
      const id = `${connection.source}-${connection.target}-${slot}-${Date.now()}`
      const graphEdge = { id, source: connection.source, target: connection.target, inputSlot: slot }
      onGraphChange({ ...graph, edges: [...graph.edges, graphEdge] })
      setEdges((existing) =>
        addEdge(
          {
            ...connection,
            id,
            type: 'builderEdge',
            sourceHandle: 'out',
            targetHandle: `in-${slot}`,
          },
          existing,
        ),
      )
    },
    [graph, onGraphChange, setEdges],
  )

  const isValidConnection = useCallback(
    (connection: Connection | Edge<BuilderEdgeData>) => {
      if (!connection.source || !connection.target || connection.source === connection.target) return false
      if (isGroupNodeId(connection.source) || isGroupNodeId(connection.target)) return false
      const target = graph.nodes.find((node) => node.id === connection.target)
      if (!target) return false
      const slot = Number(connection.targetHandle?.replace('in-', '') ?? nextOpenSlot(graph, target))
      const arity = inputArityForNode(target)
      if (slot < 0 || slot >= arity) return false
      if (graph.edges.some((edge) => edge.target === target.id && (edge.inputSlot ?? 0) === slot)) return false
      return !createsCycle(graph, connection.source, connection.target)
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
      if (groupId) onGroupExplode(groupId)
    },
    [onGroupExplode],
  )

  const selectedGroup = selectedGroupId ? graph.groups?.find((group) => group.id === selectedGroupId) : undefined
  const groupCount = graph.groups?.length ?? 0

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
              Explode group
            </button>
          ) : null}
        </div>
      </div>
      <div className="flow-shell" onPointerDownCapture={handleFlowPointerDownCapture}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
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

function nextOpenSlot(graph: GraphModel, target: GraphNode): number {
  const used = new Set(graph.edges.filter((edge) => edge.target === target.id).map((edge) => edge.inputSlot ?? 0))
  for (let slot = 0; slot < inputArityForNode(target); slot += 1) {
    if (!used.has(slot)) return slot
  }
  return 0
}

function minimumInputCountForNode(graph: GraphModel, node: GraphNode): number {
  if (!isFlexibleInputNodeType(node.type)) return inputArityForNode(node)

  const highestConnectedSlot = graph.edges
    .filter((edge) => edge.target === node.id)
    .reduce((highest, edge) => Math.max(highest, edge.inputSlot ?? 0), -1)

  return Math.max(2, highestConnectedSlot + 1)
}

function createsCycle(graph: GraphModel, sourceId: string, targetId: string): boolean {
  const outgoing = new Map<string, string[]>()
  for (const edge of graph.edges) {
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target])
  }
  outgoing.set(sourceId, [...(outgoing.get(sourceId) ?? []), targetId])
  const stack = [targetId]
  const visited = new Set<string>()
  while (stack.length > 0) {
    const id = stack.pop()!
    if (id === sourceId) return true
    if (visited.has(id)) continue
    visited.add(id)
    stack.push(...(outgoing.get(id) ?? []))
  }
  return false
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
