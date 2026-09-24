import { cloneGraph, isLossNode } from './engine'
import { collapsedGroupForNode } from './grouping'
import { addTensorsExact, cloneTensor, toTensor, zeroLike } from './tensor'
import type { EvaluationTraceStep, GraphModel, GraphPhase, NodeType, TensorValue } from './types'

const SOURCE_TYPES = new Set<NodeType>(['dataset', 'input', 'weight', 'bias', 'target'])

export function visibleGraphForTrace(
  graph: GraphModel,
  traceSteps: EvaluationTraceStep[],
  traceIndex: number,
  phase: GraphPhase,
): GraphModel {
  if (traceSteps.length === 0) {
    const next = cloneGraph(graph)
    if (phase !== 'backward') {
      next.nodes = next.nodes.map(node => ({ ...node, grad: undefined }))
      next.edges = next.edges.map(edge => ({ ...edge, grad: undefined }))
    }
    return next
  }
  if (phase === 'forward' || phase === 'loss') return visibleForwardGraph(graph, traceSteps, traceIndex)
  if (phase === 'backward') return visibleBackwardGraph(graph, traceSteps, traceIndex)
  return cloneGraph(graph)
}

function visibleForwardGraph(graph: GraphModel, traceSteps: EvaluationTraceStep[], traceIndex: number): GraphModel {
  const visibleNodeIds = new Set(
    visibleSteps(traceSteps, traceIndex)
      .filter((step) => step.phase === 'forward' || step.phase === 'loss')
      .map((step) => step.nodeId)
      .filter(Boolean),
  )
  const sourceOrReached = (nodeId: string): boolean => {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) return false
    if (!SOURCE_TYPES.has(node.type)) return visibleNodeIds.has(node.id)
    const incoming = graph.edges.find(edge => edge.target === nodeId)
    return incoming ? sourceOrReached(incoming.source) : true
  }

  const next = cloneGraph(graph)
  next.nodes = next.nodes.map((node) => {
    if (sourceOrReached(node.id)) {
      return { ...node, grad: undefined }
    }
    return { ...node, value: undefined, grad: undefined, localDerivative: undefined, cache: undefined }
  })
  next.edges = next.edges.map((edge) => ({
    ...edge,
    value: sourceOrReached(edge.source) ? edge.value : undefined,
    grad: undefined,
  }))
  return next
}

function visibleBackwardGraph(graph: GraphModel, traceSteps: EvaluationTraceStep[], traceIndex: number): GraphModel {
  const processedSteps = visibleSteps(traceSteps, traceIndex).filter((step) => step.phase === 'backward')
  const visibleEdgeIds = new Set(processedSteps.flatMap((step) => step.edgeIds))
  const processedNodeIds = new Set(processedSteps.map(step => step.nodeId))
  const accumulated = new Map<string, TensorValue>()
  for (const edge of graph.edges) {
    if (!visibleEdgeIds.has(edge.id) || !edge.grad) continue
    const source = graph.nodes.find(node => node.id === edge.source)
    // Dataset outputs can have different shapes; their per-output contributions
    // remain on the wires instead of being combined into one dataset adjoint.
    if (source?.type === 'dataset') continue
    const previous = accumulated.get(edge.source)
    accumulated.set(edge.source, previous ? addTensorsExact(previous, edge.grad) : cloneTensor(edge.grad))
  }

  const next = cloneGraph(graph)
  next.nodes = next.nodes.map((node) => ({
    ...node,
    grad: isLossNode(node) && processedSteps.length > 0
      ? node.grad
      : accumulated.get(node.id) ?? (processedNodeIds.has(node.id) ? zeroLike(toTensor(node.value ?? node.params.value)) : undefined),
  }))
  next.edges = next.edges.map((edge) => ({
    ...edge,
    grad: visibleEdgeIds.has(edge.id) ? edge.grad : undefined,
  }))
  return next
}

function visibleSteps(traceSteps: EvaluationTraceStep[], traceIndex: number): EvaluationTraceStep[] {
  const end = Math.min(Math.max(traceIndex, 0), traceSteps.length - 1)
  return traceSteps.slice(0, end + 1)
}

/** A collapsed region advances through several primitive steps at once. Keep
 * their boundary wires active so flow remains visible at the architecture scale. */
export function visibleStepEdgeIds(graph: GraphModel, steps: EvaluationTraceStep[], index: number): string[] {
  const step = steps[index]
  if (!step) return []
  const group = collapsedGroupForNode(graph, step.nodeId ?? '')
  if (!group) return [...step.edgeIds]
  const members = new Set(group.nodeIds)
  let start = index
  while (start > 0 && steps[start - 1].phase === step.phase && members.has(steps[start - 1].nodeId ?? '')) start--
  const ids = new Set(steps.slice(start, index + 1).flatMap(item => item.edgeIds))
  if (step.phase === 'forward' || step.phase === 'loss') {
    const evaluated = new Set(steps.slice(0, index + 1).filter(item => item.phase === 'forward' || item.phase === 'loss').map(item => item.nodeId))
    for (const edge of graph.edges) {
      if (!members.has(edge.source) || members.has(edge.target)) continue
      const source = graph.nodes.find(node => node.id === edge.source)
      if (source?.value !== undefined && (evaluated.has(edge.source) || SOURCE_TYPES.has(source.type))) ids.add(edge.id)
    }
  }
  return [...ids]
}
