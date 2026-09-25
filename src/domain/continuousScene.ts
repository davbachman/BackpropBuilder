import { inputArityForNode, outputArityForNode } from './engine'
import { visualGroupInterface } from './grouping'
import { layoutSemanticGraph, type SemanticLayout, type SemanticRect } from './semanticLayout'
import type { GraphGroup, GraphModel, Position } from './types'
import { routeDiagramWires, type DiagramWire, type WireEndpoint } from './wireRouting'
import { findWireCrossings, type WireCrossings } from './wireCrossings'

export const sceneGroupId = (id: string) => `visual-group:${id}`
export interface SceneLevel { parentId?: string; ids: string[]; scale: number; x: number; y: number }
export interface ContinuousScene extends SemanticLayout {
  scales: Map<string, number>
  parents: Map<string, string | undefined>
  levels: SceneLevel[]
  repairedOffsetIds: Set<string>
}
export interface SceneWire { id: string; edgeId: string; parentId?: string; scale: number; route: Position[]; crossings: WireCrossings }

/** A container with exactly one child and no calculations of its own adds no
 * visual detail. Keep the most specific block and skip those empty levels.
 * This is a display projection: the editable model and its groups stay intact. */
export function compactVisualHierarchy(graph: GraphModel): GraphModel {
  const groups = graph.groups ?? []
  const replacements = new Map<string, string>()
  const byId = new Map(groups.map(group => [group.id, group]))
  for (const group of groups) {
    const children = groups.filter(child => child.parentId === group.id)
    const child = children[0]
    if (children.length === 1 && child.nodeIds.length > 0 && child.nodeIds.length === group.nodeIds.length && child.nodeIds.every(id => group.nodeIds.includes(id))) {
      replacements.set(group.id, child.id)
    }
  }
  if (!replacements.size) return graph
  function resolve(id?: string): string | undefined {
    const seen = new Set<string>()
    while (id && replacements.has(id) && !seen.has(id)) {
      seen.add(id)
      id = replacements.get(id)
    }
    return id
  }
  const offsets = { ...graph.view?.layoutOffsets }
  const visible = groups.filter(group => !replacements.has(group.id)).map(group => {
    let parentId = group.parentId
    const seen = new Set<string>()
    while (parentId && replacements.has(parentId) && !seen.has(parentId)) {
      seen.add(parentId)
      const offset = graph.view?.layoutOffsets?.[sceneGroupId(parentId)]
      if (offset) {
        const current = offsets[sceneGroupId(group.id)] ?? { x: 0, y: 0 }
        offsets[sceneGroupId(group.id)] = { x: current.x + offset.x, y: current.y + offset.y }
      }
      parentId = byId.get(parentId)?.parentId
    }
    return { ...group, parentId }
  })
  return { ...graph, groups: visible, view: graph.view ? { ...graph.view,
    focusedGroupId: resolve(graph.view.focusedGroupId),
    expandedGroupIds: [...new Set(graph.view.expandedGroupIds.map(id => resolve(id)!))],
    layoutOffsets: Object.keys(offsets).length ? offsets : undefined,
  } : undefined }
}

/** Stop at a readable close-up of the smallest operation, rather than letting
 * the camera magnify empty space tens of thousands of times past the graph. */
export function continuousSceneMaxZoom(scene: ContinuousScene): number {
  const scales = [...scene.nodes.keys()].map(id => scene.scales.get(id)!).filter(scale => scale > 0)
  return 2.5 / Math.min(1, ...scales)
}

/** Each card reserves a permanent place for its contents. Camera movement never
 * changes this geometry: a child is simply a smaller drawing inside its parent. */
export function layoutContinuousScene(graph: GraphModel): ContinuousScene {
  const scene: ContinuousScene = { nodes: new Map(), groups: new Map(), scales: new Map(), parents: new Map(), levels: [], repairedOffsetIds: new Set() }
  const groups = graph.groups ?? []
  const groupIds = new Set(groups.map(group => group.id))
  const manualPlacements = graph.view?.manualNodePlacements ?? {}
  const activeManualPlacement = (id: string) => {
    const placement = manualPlacements[id]
    return placement && (!placement.parentId || groupIds.has(placement.parentId)) ? placement : undefined
  }
  function level(parent?: GraphGroup, ancestors = new Set<string>()) {
    if (parent && ancestors.has(parent.id)) return
    const children = groups.filter(group => group.parentId === parent?.id)
    const manualNodes = graph.nodes.filter(node => {
      const placement = activeManualPlacement(node.id)
      return placement && placement.parentId === parent?.id
    })
    const localGraph: GraphModel = { ...graph,
      nodes: graph.nodes.filter(node => (!parent || parent.nodeIds.includes(node.id)) && !activeManualPlacement(node.id)),
      groups: children.map(group => ({ ...group, parentId: parent?.id })),
      view: { expandedGroupIds: [], layoutEdges: graph.view?.layoutEdges },
    }
    const local = layoutSemanticGraph(localGraph, true, parent)
    const rects = [...local.nodes.values(), ...local.groups.values()]
    if (!rects.length && !manualNodes.length) return
    const left = rects.length ? Math.min(...rects.map(rect => rect.x)) : 0
    const top = rects.length ? Math.min(...rects.map(rect => rect.y)) : 0
    const width = rects.length ? Math.max(...rects.map(rect => rect.x + rect.width)) - left : 176
    const height = rects.length ? Math.max(...rects.map(rect => rect.y + rect.height)) - top : 112
    let scale = 1, x = 0, y = 0
    if (parent) {
      const frame = scene.groups.get(parent.id)!, outerScale = scene.scales.get(sceneGroupId(parent.id))!
      const availableWidth = frame.width - 40 * outerScale, availableHeight = frame.height - 60 * outerScale
      scale = Math.min(availableWidth / (width + 64), availableHeight / (height + 64))
      x = frame.x + (frame.width - width * scale) / 2 - left * scale
      y = frame.y + 40 * outerScale + (availableHeight - height * scale) / 2 - top * scale
    }
    const ids: string[] = []
    const entries = [...local.groups].map(([id, rect]) => ({ id, rect, group: true }))
      .concat([...local.nodes].map(([id, rect]) => ({ id, rect, group: false })))
    const bounds = parent && sceneContentBounds(scene, parent.id)
    // Offsets from an earlier layout can put a weight or bias completely
    // outside its neuron. Restore this level's connected arrangement, leaving
    // the rest of the model and all of its numerical parameters untouched.
    if (bounds && entries.some(({ id, rect, group }) => {
      const offset = graph.view?.layoutOffsets?.[group ? sceneGroupId(id) : id]
      const left = x + rect.x * scale + (offset?.x ?? 0), top = y + rect.y * scale + (offset?.y ?? 0)
      return left < bounds.x - 1e-7 || top < bounds.y - 1e-7 || left + rect.width * scale > bounds.x + bounds.width + 1e-7 || top + rect.height * scale > bounds.y + bounds.height + 1e-7
    })) {
      for (const { id, group } of entries) {
        const key = group ? sceneGroupId(id) : id
        if (graph.view?.layoutOffsets?.[key]) scene.repairedOffsetIds.add(key)
      }
    }
    function place(id: string, rect: SemanticRect, group: boolean) {
      const key = group ? sceneGroupId(id) : id
      const offset = scene.repairedOffsetIds.has(key) ? undefined : graph.view?.layoutOffsets?.[key]
      const placed = { x: x + rect.x * scale + (offset?.x ?? 0), y: y + rect.y * scale + (offset?.y ?? 0), width: rect.width * scale, height: rect.height * scale }
      if (group) scene.groups.set(id, placed)
      else scene.nodes.set(id, placed)
      scene.scales.set(key, scale)
      scene.parents.set(key, parent?.id)
      ids.push(key)
    }
    for (const [id, rect] of local.groups) place(id, rect, true)
    for (const [id, rect] of local.nodes) place(id, rect, false)
    for (const node of manualNodes) {
      const placement = activeManualPlacement(node.id)!
      const frame = placement.parentId ? scene.groups.get(placement.parentId) : undefined
      const offset = graph.view?.layoutOffsets?.[node.id]
      const nodeScale = placement.scale ?? scale
      scene.nodes.set(node.id, {
        x: (frame?.x ?? 0) + placement.offset.x + (offset?.x ?? 0),
        y: (frame?.y ?? 0) + placement.offset.y + (offset?.y ?? 0),
        width: 176 * nodeScale,
        height: (node.type === 'loss' ? 176 : 112) * nodeScale,
      })
      scene.scales.set(node.id, nodeScale)
      scene.parents.set(node.id, parent?.id)
      ids.push(node.id)
    }
    scene.levels.push({ parentId: parent?.id, ids, scale, x, y })
    for (const child of children) level(child, new Set([...ancestors, ...(parent ? [parent.id] : [])]))
  }
  level()
  return scene
}

/** Reserve room for incoming/outgoing wires and the frame heading. React Flow
 * uses these absolute bounds to keep a dragged calculation inside its block. */
export function sceneContentBounds(scene: ContinuousScene, groupId: string): SemanticRect {
  const frame = scene.groups.get(groupId)!, scale = scene.scales.get(sceneGroupId(groupId))!
  return { x: frame.x + 16 * scale, y: frame.y + 40 * scale, width: frame.width - 32 * scale, height: frame.height - 56 * scale }
}

const smooth = (t: number) => { const value = Math.max(0, Math.min(1, t)); return value * value * (3 - 2 * value) }
/** Use viewport occupancy, not discrete zoom thresholds. Wide and tall cards
 * both reveal before the user has to scroll past their boundary. */
export function cardReveal(rect: SemanticRect, zoom: number, width: number, height: number): number {
  const occupancy = Math.max(rect.width * zoom / Math.max(1, width), rect.height * zoom / Math.max(1, height))
  return smooth((occupancy - .38) / .48)
}

/** Route in each level's own units, then map into the shared canvas. Boundary
 * ports join the exact same signal on both sides of a translucent card. */
export function routeContinuousScene(graph: GraphModel, scene: ContinuousScene, positions?: Map<string, Position>): SceneWire[] {
  const groups = new Map((graph.groups ?? []).map(group => [group.id, group]))
  const nodes = new Map(graph.nodes.map(node => [node.id, node]))
  const interfaces = new Map((graph.groups ?? []).map(group => [group.id, visualGroupInterface(graph, group)]))
  const stage = (id: string) => scene.parents.get(id) === undefined && (graph.groups ?? []).some(group => !group.parentId && ['cnn', 'transformer-block'].includes(group.kind ?? ''))
  const result: SceneWire[] = []
  const rectFor = (id: string) => {
    const rect = id.startsWith('visual-group:') ? scene.groups.get(id.slice(13))! : scene.nodes.get(id)!
    return { ...rect, ...positions?.get(id) }
  }
  function endpoint(id: string, edge: GraphModel['edges'][number], output: boolean): WireEndpoint {
    const rect = rectFor(id), group = id.startsWith('visual-group:') ? groups.get(id.slice(13)) : undefined
    const ports = group ? interfaces.get(group.id)![output ? 'outputs' : 'inputs'] : undefined
    const index = ports ? ports.findIndex(port => port.edgeId === edge.id || 'edgeIds' in port && port.edgeIds?.includes(edge.id)) : output ? edge.sourceSlot ?? 0 : edge.inputSlot ?? 0
    const count = ports?.length ?? (output ? outputArityForNode(nodes.get(id)!) : Math.max(inputArityForNode(nodes.get(id)!), ...graph.edges.filter(item => item.target === id).map(item => (item.inputSlot ?? 0) + 1)))
    const fraction = (Math.max(0, index) + 1) / (Math.max(1, count) + 1)
    return stage(id) ? { x: rect.x + rect.width * fraction, y: rect.y + (output ? rect.height : 0), side: output ? 'bottom' : 'top' }
      : { x: rect.x + (output ? rect.width : 0), y: rect.y + rect.height * fraction, side: output ? 'right' : 'left' }
  }
  for (const level of scene.levels) {
    const parent = level.parentId ? groups.get(level.parentId) : undefined
    const owner = new Map<string, string>()
    for (const id of level.ids) {
      const group = id.startsWith('visual-group:') ? groups.get(id.slice(13)) : undefined
      for (const nodeId of group?.nodeIds ?? [id]) owner.set(nodeId, id)
    }
    const boundary = parent ? sceneGroupId(parent.id) : undefined
    const shift = boundary && positions?.get(boundary)
    const original = parent && scene.groups.get(parent.id)
    const origin = { x: level.x + (shift && original ? shift.x - original.x : 0), y: level.y + (shift && original ? shift.y - original.y : 0) }
    const localPoint = (point: Position) => ({ x: (point.x - origin.x) / level.scale, y: (point.y - origin.y) / level.scale })
    const reverse = { left: 'right', right: 'left', top: 'bottom', bottom: 'top' } as const
    const wires: DiagramWire[] = graph.edges.flatMap(edge => {
      const source = owner.get(edge.source), target = owner.get(edge.target)
      if ((!source && !target) || source === target || (!boundary && (!source || !target))) return []
      const from = endpoint(source ?? boundary!, edge, Boolean(source)), to = endpoint(target ?? boundary!, edge, !target)
      return [{ id: edge.id, source: { ...localPoint(from), side: source ? from.side : reverse[from.side] }, target: { ...localPoint(to), side: target ? to.side : reverse[to.side] } }]
    })
    const obstacles = level.ids.map(id => { const rect = rectFor(id); return { id, ...localPoint(rect), width: rect.width / level.scale, height: rect.height / level.scale } })
    const frame = boundary ? rectFor(boundary) : undefined
    const bounds = frame ? { ...localPoint(frame), width: frame.width / level.scale, height: frame.height / level.scale } : undefined
    const routes = routeDiagramWires(wires, obstacles, bounds)
    const crossings = findWireCrossings(routes)
    const worldPoint = (point: Position) => ({ x: origin.x + point.x * level.scale, y: origin.y + point.y * level.scale })
    for (const [edgeId, route] of routes) result.push({ id: `${edgeId}::${parent?.id ?? 'model'}`, edgeId, parentId: parent?.id, scale: level.scale, route: route.map(worldPoint),
      crossings: { points: crossings.get(edgeId)!.points.map(worldPoint), gaps: crossings.get(edgeId)!.gaps.map(worldPoint) },
    })
  }
  return result
}
