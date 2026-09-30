import { compactVisualHierarchy, layoutContinuousScene, sceneContentBounds } from './continuousScene'
import { builderCardHeight, builderCardWidth } from './builderGeometry'
import type { GraphModel, GraphNode } from './types'

/** Place a new calculation in the clicked visual level. Existing blocks keep
 * their world coordinates even when the surrounding hierarchy is nested. */
export function placeCanvasNode(graph: GraphModel, node: GraphNode, displayGraph = graph, parentGroupId?: string, sceneScale?: number): GraphModel {
  const next = { ...graph, nodes: [...graph.nodes, node] }
  const semantic = Boolean(displayGraph.groups?.length || displayGraph.view?.preservedLayouts)
  if (!semantic) return next

  const before = layoutContinuousScene(compactVisualHierarchy(displayGraph))
  if (parentGroupId && graph.groups?.some(group => group.id === parentGroupId) && before.groups.has(parentGroupId)) {
    const scene = before
    const frame = scene.groups.get(parentGroupId)!
    const bounds = sceneContentBounds(scene, parentGroupId)
    const scale = scene.levels.find(level => level.parentId === parentGroupId)?.scale ?? 1
    const position = {
      x: Math.max(bounds.x, Math.min(node.position.x, bounds.x + bounds.width - builderCardWidth(node) * scale)),
      y: Math.max(bounds.y, Math.min(node.position.y, bounds.y + bounds.height - builderCardHeight(node) * scale)),
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
  // New root blocks use the clicked position. Existing blocks never need a
  // compensating automatic layout when a block is inserted.
  return { ...next, view: { ...graph.view, expandedGroupIds: graph.view?.expandedGroupIds ?? [], manualNodePlacements: {
    ...graph.view?.manualNodePlacements,
    [node.id]: { offset: { ...node.position }, scale: sceneScale && sceneScale > 0 ? sceneScale : 1 },
  } } }
}
