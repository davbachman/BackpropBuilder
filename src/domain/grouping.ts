import {
  MIN_NODE_HEIGHT,
  NODE_WIDTH,
  heightForInputCount,
  inputArityForNode,
  isFlexibleInputNodeType,
} from './engine'
import type { GraphGroup, GraphModel, GraphNode, Position } from './types'

interface MergeResult {
  graph: GraphModel
  group?: GraphGroup
}

export interface VisualGroupHandle {
  edgeId: string
  handleId: string
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
  const ids = unique(nodeIds)
  const groupedNodeIds = nodeIdsInGroups(graph)
  const nodes = ids
    .filter((id) => !groupedNodeIds.has(id))
    .map((id) => graph.nodes.find((node) => node.id === id))
    .filter((node): node is GraphNode => Boolean(node))

  if (nodes.length < 2) return { graph }

  const id = nextGroupId(graph)
  const group: GraphGroup = {
    id,
    label: `Group ${groupIndexFromId(id)}`,
    nodeIds: nodes.map((node) => node.id),
    position: collapsedPositionForNodes(nodes),
    dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
  }

  return {
    graph: {
      ...graph,
      groups: [...(graph.groups ?? []), group],
    },
    group,
  }
}

export function explodeVisualGroup(graph: GraphModel, groupId: string): GraphModel {
  return {
    ...graph,
    groups: (graph.groups ?? []).filter((group) => group.id !== groupId),
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
      candidate.id === groupId ? { ...candidate, position: { ...position } } : candidate,
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
  const groupNodeIds = new Set(group.nodeIds)
  const nodeOrder = new Map(group.nodeIds.map((nodeId, index) => [nodeId, index]))

  const inputs = graph.edges
    .filter((edge) => groupNodeIds.has(edge.target) && !groupNodeIds.has(edge.source))
    .sort((first, second) => compareInputBoundaryEdges(first, second, nodeOrder))
    .map((edge, index) => ({ edgeId: edge.id, handleId: `in-${index}` }))

  const outputs = graph.edges
    .filter((edge) => groupNodeIds.has(edge.source) && !groupNodeIds.has(edge.target))
    .sort((first, second) => compareOutputBoundaryEdges(first, second, nodeOrder))
    .map((edge, index) => ({ edgeId: edge.id, handleId: `out-${index}` }))

  return { inputs, outputs }
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

function compareInputBoundaryEdges(
  first: { id: string; target: string; inputSlot?: number },
  second: { id: string; target: string; inputSlot?: number },
  nodeOrder: Map<string, number>,
): number {
  return (
    (nodeOrder.get(first.target) ?? 0) - (nodeOrder.get(second.target) ?? 0) ||
    (first.inputSlot ?? 0) - (second.inputSlot ?? 0) ||
    first.id.localeCompare(second.id)
  )
}

function compareOutputBoundaryEdges(
  first: { id: string; source: string },
  second: { id: string; source: string },
  nodeOrder: Map<string, number>,
): number {
  return (
    (nodeOrder.get(first.source) ?? 0) - (nodeOrder.get(second.source) ?? 0) ||
    first.id.localeCompare(second.id)
  )
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values))
}
