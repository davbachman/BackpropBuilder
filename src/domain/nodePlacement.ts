import { compactVisualHierarchy, layoutContinuousScene, sceneGroupId } from './continuousScene'
import { layoutSemanticGraph } from './semanticLayout'
import type { GraphModel, GraphNode, Position } from './types'

/** Insertion is a manual canvas edit. Compensate for automatic layout changes
 * at the top level so existing blocks (and all their contents) stay still. */
export function placeCanvasNode(graph: GraphModel, node: GraphNode, displayGraph = graph): GraphModel {
  const next = { ...graph, nodes: [...graph.nodes, node] }
  const semantic = displayGraph.view?.semanticZoom !== undefined || displayGraph.groups?.some(group => group.kind)
  if (!semantic) return next

  const continuous = displayGraph.view?.semanticZoom !== false
  const rendered = continuous ? compactVisualHierarchy(displayGraph) : displayGraph
  const layout = continuous ? layoutContinuousScene : layoutSemanticGraph
  const before = layout(rendered)
  const after = layout({ ...rendered, nodes: [...rendered.nodes, node] })
  const layoutOffsets = { ...graph.view?.layoutOffsets }
  const hold = (id: string, desired: Position, arranged: Position) => {
    const offset = layoutOffsets[id] ?? { x: 0, y: 0 }
    layoutOffsets[id] = { x: offset.x + desired.x - arranged.x, y: offset.y + desired.y - arranged.y }
  }

  // A root group's offset carries every descendant with it. Applying the same
  // compensation to its children would move them twice.
  const roots = (rendered.groups ?? []).filter(group => !group.parentId)
  const groupedIds = new Set(roots.flatMap(group => group.nodeIds))
  for (const group of roots) hold(sceneGroupId(group.id), before.groups.get(group.id)!, after.groups.get(group.id)!)
  for (const [id, rect] of before.nodes) {
    if (!groupedIds.has(id)) hold(id, rect, after.nodes.get(id)!)
  }
  hold(node.id, node.position, after.nodes.get(node.id)!)

  return { ...next, view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], layoutOffsets } }
}
