import { cloneGraph } from './engine'
import type { EvaluationTraceStep, GraphModel, GraphPhase, NodeType } from './types'

const SOURCE_TYPES = new Set<NodeType>(['input', 'weight', 'bias', 'target'])

export function visibleGraphForTrace(
  graph: GraphModel,
  traceSteps: EvaluationTraceStep[],
  traceIndex: number,
  phase: GraphPhase,
): GraphModel {
  if (traceSteps.length === 0) return cloneGraph(graph)
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
    return Boolean(node && (SOURCE_TYPES.has(node.type) || visibleNodeIds.has(node.id)))
  }

  const next = cloneGraph(graph)
  next.nodes = next.nodes.map((node) => {
    if (SOURCE_TYPES.has(node.type) || visibleNodeIds.has(node.id)) {
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
  const visibleGradNodeIds = new Set<string>()
  const lossNode = graph.nodes.find((node) => node.type === 'loss')
  if (lossNode && processedSteps.length > 0) visibleGradNodeIds.add(lossNode.id)

  for (const step of processedSteps) {
    if (step.nodeId) visibleGradNodeIds.add(step.nodeId)
    for (const edgeId of step.edgeIds) {
      const edge = graph.edges.find((candidate) => candidate.id === edgeId)
      if (edge) visibleGradNodeIds.add(edge.source)
    }
  }

  const next = cloneGraph(graph)
  next.nodes = next.nodes.map((node) => ({
    ...node,
    grad: visibleGradNodeIds.has(node.id) ? node.grad : undefined,
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
