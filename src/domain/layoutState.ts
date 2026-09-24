import type { GraphEdge, GraphModel, GraphViewState } from './types'

export function layoutConnections(edges: GraphEdge[]): NonNullable<GraphViewState['layoutEdges']> {
  return edges.map(({ id, source, target, inputSlot, sourceSlot }) => ({ id, source, target, inputSlot, sourceSlot }))
}

/** Keep the arrangement based on the connections it was drawn with. Live edges
 * remain the sole source for execution, ports and routing; only Compact layout
 * replaces this layout reference with the edited topology. */
export function preserveLayoutForWiring(graph: GraphModel): GraphModel {
  const semantic = graph.view?.semanticZoom !== undefined || graph.groups?.some(group => group.kind)
  if (!semantic || graph.view?.layoutEdges) return graph
  return { ...graph, view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], layoutEdges: layoutConnections(graph.edges) } }
}
