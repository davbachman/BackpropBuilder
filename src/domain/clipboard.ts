import { cloneGraph } from './engine'
import type { GraphEdge, GraphGroup, GraphModel, GraphNode, NodeType, Position } from './types'

export interface GraphClipboardSelection {
  nodeIds: string[]
  groupId?: string
}

export interface GraphClipboardFragment {
  nodes: GraphNode[]
  edges: GraphEdge[]
  group?: GraphGroup
}

export interface GraphClipboardPasteResult {
  graph: GraphModel
  selection: GraphClipboardSelection
}

export function copyGraphSelection(
  graph: GraphModel,
  selection: GraphClipboardSelection,
): GraphClipboardFragment | undefined {
  if (selection.groupId) {
    return copyGroupSelection(graph, selection.groupId)
  }

  const selectedNodeIds = unique(selection.nodeIds)
  if (selectedNodeIds.length === 0) return undefined

  const selectedNodeIdSet = new Set(selectedNodeIds)
  const nodes = selectedNodeIds
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId))
    .filter((node): node is GraphNode => Boolean(node))
  if (nodes.length === 0) return undefined

  return cloneClipboardFragment({
    nodes,
    edges: internalEdgesForNodeIds(graph, selectedNodeIdSet),
  })
}

export function pasteGraphClipboard(
  graph: GraphModel,
  fragment: GraphClipboardFragment,
  offset: Position,
): GraphClipboardPasteResult {
  const source = cloneClipboardFragment(fragment)
  const usedNodeIds = new Set(graph.nodes.map((node) => node.id))
  const usedEdgeIds = new Set(graph.edges.map((edge) => edge.id))
  const usedGroupIds = new Set((graph.groups ?? []).map((group) => group.id))
  const nodeIdMap = new Map<string, string>()

  const pastedNodes = source.nodes.map((node) => {
    const { id, index } = nextNodeId(node.type, usedNodeIds)
    nodeIdMap.set(node.id, id)
    return {
      ...node,
      id,
      label: labelForPastedNode(node, index),
      position: translatePosition(node.position, offset),
    }
  })

  const pastedEdges = source.edges.flatMap((edge) => {
    const sourceId = nodeIdMap.get(edge.source) ?? (usedNodeIds.has(edge.source) ? edge.source : undefined)
    const targetId = nodeIdMap.get(edge.target) ?? (usedNodeIds.has(edge.target) ? edge.target : undefined)
    if (!sourceId || !targetId) return []

    return [
      {
        ...edge,
        id: nextCopyId(edge.id, usedEdgeIds),
        source: sourceId,
        target: targetId,
      },
    ]
  })

  const pastedGroup = source.group ? createPastedGroup(source.group, usedGroupIds, nodeIdMap, offset) : undefined

  return {
    graph: {
      ...graph,
      nodes: [...graph.nodes, ...pastedNodes],
      edges: [...graph.edges, ...pastedEdges],
      groups: pastedGroup ? [...(graph.groups ?? []), pastedGroup] : graph.groups,
    },
    selection: pastedGroup
      ? { nodeIds: [], groupId: pastedGroup.id }
      : { nodeIds: pastedNodes.map((node) => node.id) },
  }
}

function copyGroupSelection(graph: GraphModel, groupId: string): GraphClipboardFragment | undefined {
  const group = graph.groups?.find((candidate) => candidate.id === groupId)
  if (!group) return undefined

  const selectedNodeIdSet = new Set(group.nodeIds)
  const nodes = group.nodeIds
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId))
    .filter((node): node is GraphNode => Boolean(node))
  if (nodes.length === 0) return undefined

  return cloneClipboardFragment({
    nodes,
    edges: boundaryAndInternalEdgesForNodeIds(graph, selectedNodeIdSet),
    group,
  })
}

function internalEdgesForNodeIds(graph: GraphModel, nodeIds: Set<string>): GraphEdge[] {
  return graph.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
}

function boundaryAndInternalEdgesForNodeIds(graph: GraphModel, nodeIds: Set<string>): GraphEdge[] {
  return graph.edges.filter((edge) => nodeIds.has(edge.source) || nodeIds.has(edge.target))
}

function cloneClipboardFragment(fragment: GraphClipboardFragment): GraphClipboardFragment {
  const clonedGraph = cloneGraph({
    learningRate: 0,
    nodes: fragment.nodes,
    edges: fragment.edges,
    groups: fragment.group ? [fragment.group] : undefined,
  })

  return {
    nodes: clonedGraph.nodes,
    edges: clonedGraph.edges,
    group: clonedGraph.groups?.[0],
  }
}

function translatePosition(position: Position, offset: Position): Position {
  return {
    x: position.x + offset.x,
    y: position.y + offset.y,
  }
}

function createPastedGroup(
  group: GraphGroup,
  usedGroupIds: Set<string>,
  nodeIdMap: Map<string, string>,
  offset: Position,
): GraphGroup {
  const { id, index } = nextIndexedId('group', usedGroupIds)

  return {
    ...group,
    id,
    label: `Group ${index}`,
    nodeIds: group.nodeIds.map((nodeId) => nodeIdMap.get(nodeId)).filter((nodeId): nodeId is string => Boolean(nodeId)),
    position: translatePosition(group.position, offset),
    dimensions: { ...group.dimensions },
  }
}

function nextNodeId(type: NodeType, usedIds: Set<string>): { id: string; index: number } {
  return nextIndexedId(idPrefixForType(type), usedIds)
}

function labelForPastedNode(node: GraphNode, index: number): string {
  const labelPrefix = labelPrefixForType(node.type)
  return labelPrefix ? `${labelPrefix}${index}` : node.label
}

function idPrefixForType(type: NodeType): string {
  if (type === 'input') return 'input'
  if (type === 'weight') return 'weight'
  if (type === 'bias') return 'bias'
  if (type === 'target') return 'target'
  if (type === 'activation') return 'activation'
  return type
}

function labelPrefixForType(type: NodeType): string | undefined {
  if (type === 'input') return 'x'
  if (type === 'weight') return 'w'
  if (type === 'bias') return 'b'
  if (type === 'target') return 'y'
  return undefined
}

function nextIndexedId(prefix: string, usedIds: Set<string>): { id: string; index: number } {
  const matcher = new RegExp(`^${escapeRegExp(prefix)}-(\\d+)$`)
  const indexes = Array.from(usedIds)
    .map((id) => id.match(matcher)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number)

  let index = Math.max(0, ...indexes) + 1
  let id = `${prefix}-${index}`
  while (usedIds.has(id)) {
    index += 1
    id = `${prefix}-${index}`
  }

  usedIds.add(id)
  return { id, index }
}

function nextCopyId(baseId: string, usedIds: Set<string>): string {
  const copyBase = `${baseId}-copy`
  if (!usedIds.has(copyBase)) {
    usedIds.add(copyBase)
    return copyBase
  }

  let index = 2
  while (usedIds.has(`${copyBase}-${index}`)) {
    index += 1
  }

  const id = `${copyBase}-${index}`
  usedIds.add(id)
  return id
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values))
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
