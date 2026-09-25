import type { GraphModel, ValidationIssue } from './types'

/** Associate validation messages with visible blocks without changing engine validation. */
export function issueNodeIds(graph: GraphModel, issue: ValidationIssue): string[] {
  if (issue.nodeId) return [issue.nodeId]
  if (issue.code === 'multiple-losses') {
    return graph.nodes.filter(node => node.type === 'loss' || node.type === 'cross-entropy').map(node => node.id)
  }
  if (issue.edgeId) {
    const edge = graph.edges.find(candidate => candidate.id === issue.edgeId)
    const target = graph.nodes.find(node => node.id === edge?.target)
    const source = graph.nodes.find(node => node.id === edge?.source)
    return target ? [target.id] : source ? [source.id] : []
  }
  return []
}

export function problemNodeIds(graph: GraphModel, issues: ValidationIssue[]): Set<string> {
  return new Set(issues.flatMap(issue => issueNodeIds(graph, issue)))
}
