import { builderCardHeight, builderCardWidth } from './builderGeometry'
import { compactVisualHierarchy, layoutContinuousScene, sceneGroupId } from './continuousScene'
import { explodeVisualGroup, mergeNodesIntoVisualGroup } from './grouping'
import type { GraphModel, GraphViewState } from './types'

/** Capture what the user sees before changing ownership. A merge only scales
 * and translates this arrangement; it never assigns new dependency columns. */
export function mergePreservingLayout(graph: GraphModel, nodeIds: string[], displayGraph = graph) {
  const merged = mergeNodesIntoVisualGroup(graph, nodeIds)
  if (!merged.group) return merged
  const layouts: NonNullable<GraphViewState['preservedLayouts']> = {}
  if (displayGraph.groups?.length) {
    const scene = layoutContinuousScene(compactVisualHierarchy(displayGraph))
    for (const level of scene.levels) {
      const entries: typeof layouts[string] = {}
      for (const id of level.ids) {
        const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
        entries[id] = { x: (rect.x - level.x) / level.scale, y: (rect.y - level.y) / level.scale,
          width: rect.width / level.scale, height: rect.height / level.scale, scale: (scene.scales.get(id) ?? 1) / level.scale }
      }
      layouts[level.parentId ?? ''] = entries
    }
  } else {
    layouts[''] = Object.fromEntries(displayGraph.nodes.map(node => [node.id, {
      ...node.position, width: builderCardWidth(node), height: builderCardHeight(node), scale: 1,
    }]))
  }
  const bounds = Object.fromEntries(Object.entries(layouts).flatMap(([key, entries]) => {
    const rects = Object.values(entries)
    if (!rects.length) return []
    const x = Math.min(...rects.map(r => r.x)), y = Math.min(...rects.map(r => r.y))
    return [[key, graph.view?.preservedLayoutBounds?.[key] ?? { x, y, width: Math.max(...rects.map(r => r.x + r.width)) - x, height: Math.max(...rects.map(r => r.y + r.height)) - y }]]
  }))
  const group = merged.group
  const parent = layouts[group.parentId ?? '']
  // Display-only hierarchy projections can hide a canonical parent. Fall back
  // to the model level that actually contains the selected blocks.
  const level = parent ?? Object.values(layouts).find(entries => nodeIds.some(id => entries[id]))
  if (!level) return merged
  const selected = new Set(nodeIds)
  const selectedKeys = Object.keys(level).filter(key => selected.has(key) ||
    displayGraph.groups?.some(candidate => key === sceneGroupId(candidate.id) && candidate.nodeIds.every(id => selected.has(id))))
  if (!selectedKeys.length) return merged
  const content = Object.fromEntries(selectedKeys.map(key => [key, level[key]]))
  const rects = Object.values(content)
  const left = Math.min(...rects.map(rect => rect.x)), top = Math.min(...rects.map(rect => rect.y))
  const width = Math.max(...rects.map(rect => rect.x + rect.width)) - left
  const height = Math.max(...rects.map(rect => rect.y + rect.height)) - top
  const scale = Math.min(Math.max(...rects.map(rect => rect.scale)), width / 238, height / 136)
  selectedKeys.forEach(key => { delete level[key] })
  level[sceneGroupId(group.id)] = { x: left + (width - 238 * scale) / 2, y: top + (height - 136 * scale) / 2,
    width: 238 * scale, height: 136 * scale, scale }
  layouts[group.id] = content
  return { ...merged, graph: { ...merged.graph, view: { ...merged.graph.view!, preservedLayouts: layouts, preservedLayoutBounds: bounds,
    layoutOffsets: undefined, manualNodePlacements: undefined } } }
}

/** Keep the visible contents in place when removing a preserved container. */
export function ungroupPreservingLayout(graph: GraphModel, groupId: string) {
  const group = graph.groups?.find(candidate => candidate.id === groupId)
  if (!group || !graph.view?.preservedLayouts?.[groupId]) return explodeVisualGroup(graph, groupId)
  const scene = layoutContinuousScene(graph)
  const parent = scene.levels.find(level => level.parentId === group.parentId)
  const contents = scene.levels.find(level => level.parentId === groupId)
  if (!parent || !contents) return explodeVisualGroup(graph, groupId)
  const layouts = structuredClone(graph.view.preservedLayouts)
  const entries = layouts[group.parentId ?? ''] ?? {}
  delete entries[sceneGroupId(groupId)]
  for (const id of contents.ids) {
    const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
    entries[id] = { x: (rect.x - parent.x) / parent.scale, y: (rect.y - parent.y) / parent.scale,
      width: rect.width / parent.scale, height: rect.height / parent.scale, scale: scene.scales.get(id)! / parent.scale }
  }
  layouts[group.parentId ?? ''] = entries
  delete layouts[groupId]
  const result = explodeVisualGroup(graph, groupId)
  return { ...result, view: { ...result.view!, preservedLayouts: layouts } }
}
