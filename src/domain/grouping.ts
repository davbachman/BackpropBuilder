import { denseGroupDetail } from './authoring'
import {
  MIN_NODE_HEIGHT,
  NODE_WIDTH,
  heightForInputCount,
  inputArityForNode,
  isFlexibleInputNodeType,
  outputArityForNode,
} from './engine'
import type { GraphEdge, GraphGroup, GraphModel, GraphNode, Position } from './types'

interface MergeResult {
  graph: GraphModel
  group?: GraphGroup
}

export interface VisualGroupHandle {
  edgeId?: string
  edgeIds?: string[]
  handleId: string
  source?: string
  sourceSlot?: number
  target?: string
  inputSlot?: number
}

export interface VisualGroupInterface {
  inputs: VisualGroupHandle[]
  outputs: VisualGroupHandle[]
}

interface NodeRect {
  x: number
  y: number
  width: number
  height: number
}

export function mergeNodesIntoVisualGroup(graph: GraphModel, nodeIds: string[]): MergeResult {
  const ids = unique(nodeIds).filter((id) => graph.nodes.some((node) => node.id === id))
  if (ids.length < 2) return { graph }
  const selected = new Set(ids)
  const groups = graph.groups ?? []
  // A selection can contain complete modules or sit inside a module, but cannot
  // cut across a module boundary. This keeps the hierarchy a genuine tree.
  if (groups.some((group) => {
    const overlap = group.nodeIds.filter((id) => selected.has(id)).length
    return overlap > 0 && overlap < group.nodeIds.length && overlap < selected.size
  })) return { graph }
  const parent = groups.filter((group) => ids.every((id) => group.nodeIds.includes(id)) && group.nodeIds.length > ids.length)
    .sort((a, b) => a.nodeIds.length - b.nodeIds.length)[0]
  const nodes = ids.map((id) => graph.nodes.find((node) => node.id === id)!)
  const id = nextGroupId(graph)
  const group: GraphGroup = {
    id,
    label: `Group ${groupIndexFromId(id)}`,
    kind: 'module',
    ...(parent ? { parentId: parent.id } : {}),
    nodeIds: ids,
    position: collapsedPositionForNodes(nodes),
    dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
  }
  group.detail = { ...denseGroupDetail(graph, group), userCreated: true }
  return {
    graph: { ...graph, view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], semanticZoom: true }, groups: [...groups.map((candidate) =>
      candidate.parentId === parent?.id && candidate.nodeIds.every((nodeId) => selected.has(nodeId))
        ? { ...candidate, parentId: id } : candidate), group] },
    group,
  }
}

/** Ungroup is an explicit structure edit. Opening a module uses setVisualGroupExpanded. */
export function explodeVisualGroup(graph: GraphModel, groupId: string): GraphModel {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  return {
    ...graph,
    groups: (graph.groups ?? []).filter((candidate) => candidate.id !== groupId)
      .map((candidate) => candidate.parentId === groupId ? { ...candidate, parentId: group?.parentId } : candidate),
    view: graph.view ? { ...graph.view, expandedGroupIds: graph.view.expandedGroupIds.filter((id) => id !== groupId),
      focusedGroupId: graph.view.focusedGroupId === groupId ? group?.parentId : graph.view.focusedGroupId } : undefined,
  }
}

export function groupAncestors(graph: GraphModel, groupId: string): GraphGroup[] {
  const ancestors: GraphGroup[] = []
  const seen = new Set<string>()
  let group = graph.groups?.find((candidate) => candidate.id === groupId)
  while (group && !seen.has(group.id)) {
    seen.add(group.id)
    ancestors.unshift(group)
    group = graph.groups?.find((candidate) => candidate.id === group?.parentId)
  }
  return ancestors
}

export function setVisualGroupExpanded(graph: GraphModel, groupId: string, expanded: boolean): GraphModel {
  const path = groupAncestors(graph, groupId)
  if (path.length === 0) return graph
  const expandedIds = new Set(graph.view?.expandedGroupIds ?? [])
  if (expanded) path.forEach((group) => expandedIds.add(group.id))
  else expandedIds.delete(groupId)
  return { ...graph, view: { ...graph.view, expandedGroupIds: [...expandedIds],
    focusedGroupId: expanded ? groupId : path[path.length - 1].parentId } }
}

export function visibleGroups(graph: GraphModel): GraphGroup[] {
  const expanded = new Set(graph.view?.expandedGroupIds ?? [])
  return (graph.groups ?? []).filter((group) => groupAncestors(graph, group.id).slice(0, -1).every((parent) => expanded.has(parent.id)))
}

export function collapsedGroupForNode(graph: GraphModel, nodeId: string): GraphGroup | undefined {
  const expanded = new Set(graph.view?.expandedGroupIds ?? [])
  return visibleGroups(graph).find((group) => !expanded.has(group.id) && group.nodeIds.includes(nodeId))
}

export function expandedGroupRect(graph: GraphModel, group: GraphGroup): NodeRect {
  const members = graph.nodes.filter((node) => group.nodeIds.includes(node.id)).map(nodeRect)
  const minX = Math.min(...members.map((rect) => rect.x)) - 28
  const minY = Math.min(...members.map((rect) => rect.y)) - 64
  return { x: minX, y: minY, width: Math.max(...members.map((rect) => rect.x + rect.width)) - minX + 28,
    height: Math.max(...members.map((rect) => rect.y + rect.height)) - minY + 28 }
}

export function deleteVisualGroup(graph: GraphModel, groupId: string): GraphModel {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return graph

  const removedNodeIds = new Set(group.nodeIds)

  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !removedNodeIds.has(node.id)),
    edges: graph.edges.filter((edge) => !removedNodeIds.has(edge.source) && !removedNodeIds.has(edge.target)),
    groups: (graph.groups ?? [])
      .filter((candidate) => candidate.id !== groupId)
      .map((candidate) => ({
        ...candidate,
        nodeIds: candidate.nodeIds.filter((nodeId) => !removedNodeIds.has(nodeId)),
      }))
      .filter((candidate) => candidate.nodeIds.length >= 2),
  }
}

export function moveVisualGroup(graph: GraphModel, groupId: string, position: Position): GraphModel {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return graph

  const delta = {
    x: position.x - group.position.x,
    y: position.y - group.position.y,
  }
  const groupNodeIds = new Set(group.nodeIds)

  return {
    ...graph,
    groups: (graph.groups ?? []).map((candidate) =>
      candidate.id === groupId || groupAncestors(graph, candidate.id).some((parent) => parent.id === groupId)
        ? { ...candidate, position: { x: candidate.position.x + delta.x, y: candidate.position.y + delta.y } } : candidate,
    ),
    nodes: graph.nodes.map((node) =>
      groupNodeIds.has(node.id)
        ? { ...node, position: { x: node.position.x + delta.x, y: node.position.y + delta.y } }
        : node,
    ),
  }
}

export function removeNodesFromVisualGroups(graph: GraphModel, removedNodeIds: Set<string>): GraphGroup[] {
  return (graph.groups ?? [])
    .map((group) => ({ ...group, nodeIds: group.nodeIds.filter((nodeId) => !removedNodeIds.has(nodeId)) }))
    .filter((group) => group.nodeIds.length >= 2)
}

export function groupForNode(graph: GraphModel, nodeId: string): GraphGroup | undefined {
  return graph.groups?.find((group) => group.nodeIds.includes(nodeId))
}

export function nodeIdsInGroups(graph: GraphModel): Set<string> {
  return new Set((graph.groups ?? []).flatMap((group) => group.nodeIds))
}

export function visualGroupInterface(graph: GraphModel, group: GraphGroup): VisualGroupInterface {
  const nodeOrder = new Map(group.nodeIds.map((nodeId, index) => [nodeId, index]))

  const inputs = exposedInputSlots(graph, group)
    .sort((first, second) => compareInputSlots(first, second, nodeOrder))
    .map((slot, index) => ({
      ...(slot.edge ? { edgeId: slot.edge.id } : {}),
      handleId: `in-${index}`,
      target: slot.target,
      inputSlot: slot.inputSlot,
    }))

  const outputs = exposedOutputHandles(graph, group)
    .sort((first, second) => compareOutputHandles(first, second, nodeOrder))
    .map((handle, index) => {
      const edgeIds = handle.edges?.map((edge) => edge.id) ?? []

      return {
        ...(edgeIds.length === 1 ? { edgeId: edgeIds[0] } : {}),
        ...(edgeIds.length > 1 ? { edgeIds } : {}),
        handleId: `out-${index}`,
        source: handle.source,
        ...(handle.sourceSlot && handle.sourceSlot > 0 ? { sourceSlot: handle.sourceSlot } : {}),
      }
    })

  return { inputs, outputs }
}

export function resolveVisualGroupInputHandle(
  graph: GraphModel,
  groupId: string,
  handleId: string | undefined,
): { target: string; targetHandle: string } | undefined {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return undefined

  const handle = visualGroupInterface(graph, group).inputs.find((candidate) => candidate.handleId === handleId)
  if (!handle?.target || handle.inputSlot === undefined) return undefined

  return {
    target: handle.target,
    targetHandle: `in-${handle.inputSlot}`,
  }
}

export function resolveVisualGroupOutputHandle(
  graph: GraphModel,
  groupId: string,
  handleId: string | undefined,
): { source: string; sourceHandle?: string; edgeId?: string } | undefined {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return undefined

  const handle = visualGroupInterface(graph, group).outputs.find((candidate) => candidate.handleId === handleId)
  if (!handle?.source) return undefined

  const sourceSlot = handle.sourceSlot ?? 0
  const edgeIds = edgeIdsForHandle(handle)
  return {
    source: handle.source,
    ...(sourceSlot > 0 ? { sourceHandle: `out-${sourceSlot}` } : {}),
    ...(edgeIds.length === 1 ? { edgeId: edgeIds[0] } : {}),
  }
}

interface ExposedInputSlot {
  target: string
  inputSlot: number
  edge?: GraphEdge
}

function exposedInputSlots(graph: GraphModel, group: GraphGroup): ExposedInputSlot[] {
  const groupNodeIds = new Set(group.nodeIds)
  const incomingByTargetSlot = new Map<string, GraphEdge>()

  for (const edge of graph.edges) {
    if (!groupNodeIds.has(edge.target)) continue
    incomingByTargetSlot.set(inputSlotKey(edge.target, edge.inputSlot ?? 0), edge)
  }

  return group.nodeIds.flatMap((nodeId) => {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) return []

    return Array.from({ length: inputArityForNode(node) }).flatMap<ExposedInputSlot>((_, inputSlot) => {
      const edge = incomingByTargetSlot.get(inputSlotKey(node.id, inputSlot))
      if (edge && groupNodeIds.has(edge.source)) return []
      return [{ target: node.id, inputSlot, edge }]
    })
  })
}

function inputSlotKey(target: string, inputSlot: number): string {
  return `${target}:${inputSlot}`
}

interface ExposedOutputHandle {
  source: string
  sourceSlot?: number
  edges?: GraphEdge[]
}

function exposedOutputHandles(graph: GraphModel, group: GraphGroup): ExposedOutputHandle[] {
  const groupNodeIds = new Set(group.nodeIds)
  const outgoingBySource = new Map<string, GraphEdge[]>()

  for (const edge of graph.edges) {
    outgoingBySource.set(edge.source, [...(outgoingBySource.get(edge.source) ?? []), edge])
  }

  return group.nodeIds.flatMap((nodeId) => {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node || node.type === 'loss') return []

    const outgoing = outgoingBySource.get(node.id) ?? []
    const boundaryEdges = outgoing.filter((edge) => !groupNodeIds.has(edge.target))
    if (boundaryEdges.length > 0) {
      return Array.from(groupEdgesBySourceSlot(boundaryEdges).entries()).map(([sourceSlot, edges]) => ({
        source: node.id,
        sourceSlot,
        edges,
      }))
    }

    if (outgoing.length === 0) {
      return Array.from({ length: outputArityForNode(node) }, (_, sourceSlot) => ({ source: node.id, sourceSlot }))
    }
    return []
  })
}

function collapsedPositionForNodes(nodes: GraphNode[]): Position {
  const rects = nodes.map(nodeRect)
  const minX = Math.min(...rects.map((rect) => rect.x))
  const minY = Math.min(...rects.map((rect) => rect.y))
  const maxX = Math.max(...rects.map((rect) => rect.x + rect.width))
  const maxY = Math.max(...rects.map((rect) => rect.y + rect.height))

  return {
    x: (minX + maxX - NODE_WIDTH) / 2,
    y: (minY + maxY - MIN_NODE_HEIGHT) / 2,
  }
}

function nodeRect(node: GraphNode): NodeRect {
  return {
    x: node.position.x,
    y: node.position.y,
    width: node.dimensions?.width ?? NODE_WIDTH,
    height: node.dimensions?.height ?? (isFlexibleInputNodeType(node.type) ? heightForInputCount(inputArityForNode(node)) : MIN_NODE_HEIGHT),
  }
}

function nextGroupId(graph: GraphModel): string {
  const indexes = (graph.groups ?? [])
    .map((group) => group.id.match(/^group-(\d+)$/)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number)
  return `group-${Math.max(0, ...indexes) + 1}`
}

function groupIndexFromId(groupId: string): number {
  return Number(groupId.match(/^group-(\d+)$/)?.[1] ?? 1)
}

function compareInputSlots(
  first: { target: string; inputSlot: number; edge?: GraphEdge },
  second: { target: string; inputSlot: number; edge?: GraphEdge },
  nodeOrder: Map<string, number>,
): number {
  return (
    (nodeOrder.get(first.target) ?? 0) - (nodeOrder.get(second.target) ?? 0) ||
    first.inputSlot - second.inputSlot ||
    (first.edge?.id ?? '').localeCompare(second.edge?.id ?? '')
  )
}

function compareOutputHandles(
  first: { source: string; sourceSlot?: number; edges?: GraphEdge[] },
  second: { source: string; sourceSlot?: number; edges?: GraphEdge[] },
  nodeOrder: Map<string, number>,
): number {
  return (
    (nodeOrder.get(first.source) ?? 0) - (nodeOrder.get(second.source) ?? 0) ||
    (first.sourceSlot ?? 0) - (second.sourceSlot ?? 0) ||
    (first.edges?.[0]?.id ?? '').localeCompare(second.edges?.[0]?.id ?? '')
  )
}

function groupEdgesBySourceSlot(edges: GraphEdge[]): Map<number, GraphEdge[]> {
  const edgesBySourceSlot = new Map<number, GraphEdge[]>()

  for (const edge of edges) {
    const sourceSlot = edge.sourceSlot ?? 0
    edgesBySourceSlot.set(sourceSlot, [...(edgesBySourceSlot.get(sourceSlot) ?? []), edge])
  }

  return edgesBySourceSlot
}

function edgeIdsForHandle(handle: VisualGroupHandle): string[] {
  if (handle.edgeIds) return handle.edgeIds
  return handle.edgeId ? [handle.edgeId] : []
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values))
}
