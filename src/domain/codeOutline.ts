import { compactVisualHierarchy } from './continuousScene'
import { formulaForNode } from './engine'
import type { GraphGroup, GraphModel, GraphNode } from './types'

export interface CodeOutlineLine {
  id: string
  kind: 'group' | 'node'
  label: string
  code: string
  parentId?: string
  children: CodeOutlineLine[]
}

const identifier = (label: string) => label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'block'

function groupVariable(graph: GraphModel, group: GraphGroup): string {
  const base = identifier(group.label)
  const matches = (graph.groups ?? []).filter(candidate => identifier(candidate.label) === base)
  return matches.length <= 1 ? base : `${base}_${matches.findIndex(candidate => candidate.id === group.id) + 1}`
}

/** The visible code for a module is a single call. Its inputs are the values
 * crossing that module's boundary; expanding it shows the actual calculations. */
export function codeForGroup(graph: GraphModel, group: GraphGroup): string {
  const contained = new Set(group.nodeIds)
  const names = new Map(graph.nodes.map(node => [node.id, node]))
  const sources = [...new Set(graph.edges.filter(edge => contained.has(edge.target) && !contained.has(edge.source)).map(edge => edge.source))]
  const argumentsList = sources.map(id => names.get(id)?.label ?? id).map(identifier).join(', ')
  const name = groupVariable(graph, group)
  const operation = identifier(group.kind ?? 'module')
  return `${name} = ${operation}(${argumentsList})`
}

/** Build the same hierarchy shown on the continuous canvas. An operation is
 * listed once, under its deepest visible group, in dependency order. */
export function buildCodeOutline(source: GraphModel): CodeOutlineLine[] {
  const graph = compactVisualHierarchy(source)
  const groups = graph.groups ?? []
  const byId = new Map(groups.map(group => [group.id, group]))
  const incoming = new Map(graph.nodes.map(node => [node.id, [] as string[]]))
  for (const edge of graph.edges) incoming.get(edge.target)?.push(edge.source)
  const rank = new Map<string, number>()
  const visiting = new Set<string>()
  function visit(id: string) {
    if (rank.has(id) || visiting.has(id)) return
    visiting.add(id)
    for (const sourceId of incoming.get(id) ?? []) visit(sourceId)
    visiting.delete(id)
    rank.set(id, rank.size)
  }
  graph.nodes.forEach(node => visit(node.id))
  const depth = (group: GraphGroup): number => {
    let count = 0, parent = group.parentId
    const seen = new Set<string>()
    while (parent && !seen.has(parent)) { seen.add(parent); count++; parent = byId.get(parent)?.parentId }
    return count
  }
  const owner = (node: GraphNode) => groups.filter(group => group.nodeIds.includes(node.id))
    .sort((a, b) => depth(b) - depth(a))[0]?.id
  const ownNodes = new Map<string | undefined, GraphNode[]>()
  for (const node of graph.nodes) {
    const id = owner(node)
    ownNodes.set(id, [...(ownNodes.get(id) ?? []), node])
  }
  const groupRank = (group: GraphGroup) => Math.max(-1, ...group.nodeIds.map(id => rank.get(id) ?? -1))
  const build = (parentId?: string): CodeOutlineLine[] => [
    ...groups.filter(group => group.parentId === parentId).map(group => ({
      id: group.id, kind: 'group' as const, label: group.label, code: codeForGroup(graph, group), parentId,
      children: build(group.id), rank: groupRank(group),
    })),
    ...(ownNodes.get(parentId) ?? []).map(node => ({
      id: node.id, kind: 'node' as const, label: node.label, code: formulaForNode(node, graph), parentId,
      children: [], rank: rank.get(node.id) ?? 0,
    })),
  ].sort((a, b) => a.rank - b.rank || a.label.localeCompare(b.label))
    .map(({ rank: _rank, ...line }) => { void _rank; return line })
  return build()
}
