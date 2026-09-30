import { compactVisualHierarchy, layoutContinuousScene } from './continuousScene'
import type { GraphModel, GraphViewState } from './types'

/** Save the current drawing so later edits cannot trigger an automatic reflow. */
export function captureCanvasLayout(graph: GraphModel): GraphViewState {
  return preserveLayoutAfterDeletion(graph, graph, graph, true).view!
}

/** Capture the visible arrangement across a deletion, including levels whose
 * parent disappeared. Surviving cards retain their world positions and scales. */
export function preserveLayoutAfterDeletion(before: GraphModel, after: GraphModel, displayGraph = before, capture = false): GraphModel {
  if (!capture && !displayGraph.groups?.length && !displayGraph.view?.preservedLayouts) return after
  const scene = layoutContinuousScene(compactVisualHierarchy(displayGraph))
  const survivingGroups = new Set(after.groups?.map(group => group.id))
  const oldGroups = new Map(before.groups?.map(group => [group.id, group]))
  const survivingParent = (parentId?: string): string | undefined => {
    const seen = new Set<string>()
    while (parentId && !survivingGroups.has(parentId) && !seen.has(parentId)) {
      seen.add(parentId)
      parentId = oldGroups.get(parentId)?.parentId
    }
    return parentId
  }
  const layouts: NonNullable<GraphViewState['preservedLayouts']> = {}
  const bounds: NonNullable<GraphViewState['preservedLayoutBounds']> = {}
  const remainingNodes = new Set(after.nodes.map(node => node.id))
  for (const level of scene.levels) {
    const parentId = survivingParent(level.parentId)
    const destination = scene.levels.find(candidate => candidate.parentId === parentId)!
    const key = parentId ?? ''
    const entries = layouts[key] ??= {}
    bounds[key] = destination.bounds
    for (const id of level.ids) {
      const isGroup = id.startsWith('visual-group:')
      if (isGroup ? !survivingGroups.has(id.slice(13)) : !remainingNodes.has(id)) continue
      const rect = isGroup ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
      entries[id] = {
        x: (rect.x - destination.x) / destination.scale,
        y: (rect.y - destination.y) / destination.scale,
        width: rect.width / destination.scale,
        height: rect.height / destination.scale,
        scale: scene.scales.get(id)! / destination.scale,
      }
    }
  }
  return {
    ...after,
    nodes: after.nodes.map(node => {
      const rect = scene.nodes.get(node.id)
      return rect ? { ...node, position: { x: rect.x, y: rect.y } } : node
    }),
    groups: after.groups?.map(group => ({ ...group, parentId: survivingParent(group.parentId) })),
    view: { ...after.view, expandedGroupIds: (after.view?.expandedGroupIds ?? []).filter(id => survivingGroups.has(id)),
      preservedLayouts: layouts, preservedLayoutBounds: bounds, layoutOffsets: undefined, manualNodePlacements: undefined,
    },
  }
}
