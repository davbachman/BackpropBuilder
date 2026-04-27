import { inputArityForNode, outputArityForNode } from './engine'
import type { GraphModel, GraphNode } from './types'

export interface GraphConnection {
  source?: string | null
  sourceHandle?: string | null
  target?: string | null
  targetHandle?: string | null
  replaceEdgeId?: string | null
}

type EdgeIdFactory = (source: string, target: string, inputSlot: number) => string

export function connectGraphNodes(
  graph: GraphModel,
  connection: GraphConnection,
  createEdgeId: EdgeIdFactory = defaultEdgeId,
): GraphModel | undefined {
  const candidate = graphConnectionCandidate(graph, connection)
  if (!candidate) return undefined

  const graphWithSlotCleared = {
    ...graph,
    edges: graph.edges.filter(
      (edge) =>
        edge.id !== connection.replaceEdgeId &&
        (edge.target !== candidate.target.id || (edge.inputSlot ?? 0) !== candidate.inputSlot),
    ),
  }

  if (createsCycle(graphWithSlotCleared, candidate.source.id, candidate.target.id)) return undefined

  return {
    ...graphWithSlotCleared,
    edges: [
      ...graphWithSlotCleared.edges,
      {
        id: createEdgeId(candidate.source.id, candidate.target.id, candidate.inputSlot),
        source: candidate.source.id,
        ...(candidate.sourceSlot > 0 ? { sourceSlot: candidate.sourceSlot } : {}),
        target: candidate.target.id,
        inputSlot: candidate.inputSlot,
      },
    ],
  }
}

export function canConnectGraphNodes(graph: GraphModel, connection: GraphConnection): boolean {
  return Boolean(connectGraphNodes(graph, connection))
}

function graphConnectionCandidate(
  graph: GraphModel,
  connection: GraphConnection,
): { source: GraphNode; target: GraphNode; inputSlot: number; sourceSlot: number } | undefined {
  if (!connection.source || !connection.target || connection.source === connection.target) return undefined

  const source = graph.nodes.find((node) => node.id === connection.source)
  const target = graph.nodes.find((node) => node.id === connection.target)
  if (!source || !target) return undefined

  const sourceSlot = outputSlotForConnection(connection)
  const outputArity = outputArityForNode(source)
  if (sourceSlot < 0 || sourceSlot >= outputArity) return undefined

  const inputSlot = inputSlotForConnection(graph, target, connection)
  const arity = inputArityForNode(target)
  if (inputSlot < 0 || inputSlot >= arity) return undefined

  return { source, target, inputSlot, sourceSlot }
}

function inputSlotForConnection(graph: GraphModel, target: GraphNode, connection: GraphConnection): number {
  return handleSlotForPrefix(connection.targetHandle, 'in') ?? nextOpenSlot(graph, target)
}

function outputSlotForConnection(connection: GraphConnection): number {
  return handleSlotForPrefix(connection.sourceHandle, 'out') ?? 0
}

function handleSlotForPrefix(handle: string | null | undefined, prefix: string): number | undefined {
  if (!handle) return undefined
  if (handle === prefix) return 0
  const match = handle.match(new RegExp(`^${prefix}-(\\d+)$`))
  if (!match) return undefined
  return Number(match[1])
}

function nextOpenSlot(graph: GraphModel, target: GraphNode): number {
  const used = new Set(graph.edges.filter((edge) => edge.target === target.id).map((edge) => edge.inputSlot ?? 0))
  for (let slot = 0; slot < inputArityForNode(target); slot += 1) {
    if (!used.has(slot)) return slot
  }
  return 0
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

function defaultEdgeId(source: string, target: string, inputSlot: number): string {
  return `${source}-${target}-${inputSlot}-${Date.now()}`
}
