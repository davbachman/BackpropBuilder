import { compactVisualHierarchy, layoutContinuousScene, sceneContentBounds, sceneGroupId } from './continuousScene'
import { layoutSemanticGraph } from './semanticLayout'
import { customCsvCardHeight, customCsvCardWidth } from './datasets'
import type { GraphModel, GraphNode, Position } from './types'

/** Place a new calculation in the clicked visual level. Existing blocks keep
 * their world coordinates even when the surrounding hierarchy is nested. */
export function placeCanvasNode(graph: GraphModel, node: GraphNode, displayGraph = graph, parentGroupId?: string, sceneScale?: number): GraphModel {
  const next = { ...graph, nodes: [...graph.nodes, node] }
  const semantic = displayGraph.view?.semanticZoom !== undefined || displayGraph.groups?.some(group => group.kind)
  if (!semantic) return next

  const continuous = displayGraph.view?.semanticZoom !== false
  const rendered = continuous ? compactVisualHierarchy(displayGraph) : displayGraph
  const layout = continuous ? layoutContinuousScene : layoutSemanticGraph
  const before = layout(rendered)
  if (continuous && parentGroupId && graph.groups?.some(group => group.id === parentGroupId) && before.groups.has(parentGroupId)) {
    const scene = before as ReturnType<typeof layoutContinuousScene>
    const frame = scene.groups.get(parentGroupId)!
    const bounds = sceneContentBounds(scene, parentGroupId)
    const scale = scene.levels.find(level => level.parentId === parentGroupId)?.scale ?? 1
    const position = {
      x: Math.max(bounds.x, Math.min(node.position.x, bounds.x + bounds.width - (node.type === 'dataset' && node.params.dataset === 'custom-csv' ? customCsvCardWidth(node) : 176) * scale)),
      y: Math.max(bounds.y, Math.min(node.position.y, bounds.y + bounds.height - (node.type === 'dataset' && node.params.dataset === 'custom-csv' ? customCsvCardHeight(node) : node.type === 'loss' ? 176 : 112) * scale)),
    }
    const ancestors = new Set<string>()
    let current = graph.groups.find(group => group.id === parentGroupId)
    while (current && !ancestors.has(current.id)) {
      ancestors.add(current.id)
      current = graph.groups.find(group => group.id === current?.parentId)
    }
    return {
      ...next,
      nodes: [...graph.nodes, { ...node, position }],
      groups: graph.groups.map(group => ancestors.has(group.id) ? { ...group, nodeIds: [...group.nodeIds, node.id] } : group),
      view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], manualNodePlacements: {
        ...graph.view?.manualNodePlacements,
        [node.id]: { parentId: parentGroupId, offset: { x: position.x - frame.x, y: position.y - frame.y } },
      } },
    }
  }
  if (continuous && sceneScale && sceneScale > 0) {
    return { ...next, view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], manualNodePlacements: {
      ...graph.view?.manualNodePlacements,
      [node.id]: { offset: { ...node.position }, scale: sceneScale },
    } } }
  }
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
