import type { GraphGroup, GraphModel, GraphNode } from './types'
import { inputArityForNode, isFlexibleInputNodeType, MAX_FLEX_INPUT_COUNT } from './engine'
import { builderCardHeight, builderCardWidth, builderInputPortY, builderOutputPortY } from './builderGeometry'

export interface SemanticRect { x: number; y: number; width: number; height: number }
export interface SemanticLayout {
  nodes: Map<string, SemanticRect>
  groups: Map<string, SemanticRect>
}

const ROW_GAP = 32
const COLUMN_GAP = 72
const WIRE_LANE = 24

export function expandedSemanticInputHeight(node: GraphNode): number | undefined {
  if (node.type !== 'arithmetic' && !isFlexibleInputNodeType(node.type)) return undefined
  const count = inputArityForNode(node)
  if (node.type === 'concat') return Math.max(148, count > 4 ? (count + 1) * 20 : 0)
  return count > 4 ? (count + 1) * 20 : undefined
}

export function semanticHasAddInput(node: GraphNode, count: number): boolean {
  return (node.type === 'arithmetic' || isFlexibleInputNodeType(node.type)) && count < MAX_FLEX_INPUT_COUNT
}

export function semanticInputFraction(index: number, count: number, hasAddInput: boolean): number {
  return (index + 1) / (count + 1 + Number(hasAddInput))
}

/** Layout only the currently visible hierarchy. Expansion creates room without
 * moving or duplicating a single parameter in the underlying model. */
export function layoutSemanticGraph(graph: GraphModel, nestedCards = false, rootGroup?: GraphGroup): SemanticLayout {
  const result: SemanticLayout = { nodes: new Map(), groups: new Map() }
  const groups = graph.groups ?? []
  const edges = graph.view?.layoutEdges ?? graph.edges
  const expanded = new Set(graph.view?.expandedGroupIds ?? [])
  type Item = { id: string; memberIds: string[]; width: number; height: number; group?: GraphGroup; node?: GraphNode; nested?: Item[]; x: number; y: number }

  function layoutLevel(parent?: GraphGroup): Item[] {
    const coordinates = parent?.kind === 'neuron' && parent.detail?.virtual === true
    const children = groups.filter((group) => group.parentId === parent?.id)
    const covered = new Set(children.flatMap((group) => group.nodeIds))
    const ownNodes = graph.nodes.filter((node) => (!parent || parent.nodeIds.includes(node.id)) && !covered.has(node.id))
    const items: Item[] = children.map((group) => {
      if (!expanded.has(group.id)) {
        const neuron = group.kind === 'neuron'
        return { id: group.id, group, memberIds: group.nodeIds, width: neuron && !nestedCards ? 126 : 238, height: neuron && !nestedCards ? 152 : group.detail?.userCreated && group.kind === 'module' ? 136 : 172, x: 0, y: 0 }
      }
      const nested = layoutLevel(group)
      const width = Math.max(240, ...nested.map((item) => item.x + item.width)) + 56
      const height = Math.max(130, ...nested.map((item) => item.y + item.height)) + 94
      return { id: group.id, group, memberIds: group.nodeIds, nested, width, height, x: 0, y: 0 }
    })
    items.push(...ownNodes.map((node) => {
      return { id: node.id, node, memberIds: [node.id], width: builderCardWidth(node), height: builderCardHeight(node), x: 0, y: 0 }
    }))
    const owner = new Map(items.flatMap((item) => item.memberIds.map((id) => [id, item.id] as const)))
    const incoming = new Map(items.map((item) => [item.id, new Set<string>()]))
    const outgoing = new Map(items.map((item) => [item.id, new Set<string>()]))
    const links = edges.flatMap(edge => {
      const source = owner.get(edge.source), target = owner.get(edge.target)
      return source && target && source !== target ? [{ ...edge, from: source, to: target }] : []
    })
    for (const edge of edges) {
      const source = owner.get(edge.source)
      const target = owner.get(edge.target)
      if (source && target && source !== target) {
        incoming.get(target)?.add(source)
        outgoing.get(source)?.add(target)
      }
    }
    const rank = new Map<string, number>()
    const pending = new Set(items.map((item) => item.id))
    while (pending.size) {
      let advanced = false
      for (const id of pending) {
        const sources = [...(incoming.get(id) ?? [])]
        if (!sources.every((source) => rank.has(source))) continue
        rank.set(id, sources.length ? Math.max(...sources.map((source) => rank.get(source)!)) + 1 : 0)
        pending.delete(id)
        advanced = true
      }
      // Incomplete user-built graphs can contain a cycle while being edited.
      if (!advanced) { for (const id of pending) rank.set(id, 0); break }
    }
    // Put learned constants beside the operation they feed, instead of sending
    // every parameter wire across the entire region from column zero.
    for (const item of items) {
      if (!item.node || !['weight', 'bias', ...(parent ? ['input'] : [])].includes(item.node.type) || incoming.get(item.id)?.size) continue
      const consumers = [...(outgoing.get(item.id) ?? [])]
      if (consumers.length) rank.set(item.id, Math.max(0, Math.min(...consumers.map(id => rank.get(id) ?? 1)) - 1))
    }
    const columns = new Map<number, Item[]>()
    for (const item of items) {
      const column = rank.get(item.id) ?? 0
      columns.set(column, [...(columns.get(column) ?? []), item])
    }
    if (coordinates) {
      const inputPairs = items.flatMap((item) => { const match = item.id.match(/:([xw])(\d+)$/); return match ? [{ item, column: match[1] === 'x' ? 0 : 1, row: Number(match[2]) }] : [] })
      const inputRows = Math.max(1, ...inputPairs.map((pair) => pair.row + 1))
      const coordinateRow = 230
      const height = (inputRows + 1) * coordinateRow - 20
      let x = 0
      for (const [columnRank, column] of [...columns.entries()].sort(([a], [b]) => a - b)) {
        if (columnRank === 0 && inputPairs.length) {
          for (const item of column) {
            const pair = inputPairs.find((candidate) => candidate.item.id === item.id)
            item.x = x + (pair?.column ?? 0) * 200
            item.y = (pair?.row ?? inputRows) * coordinateRow + (pair?.column ?? 0) * 110
          }
          x += 420
        } else {
          column.forEach((item, index) => { item.x = x; item.y = column.length === 1 ? (height - item.height) / 2 : index * coordinateRow })
          x += 248
        }
      }
      return items
    }
    const ordered = [...columns.entries()].sort(([a], [b]) => a - b)
    const order = new Map(items.map((item, index) => [item.id, index]))
    // Alternating barycenter sweeps order parallel branches by their actual
    // connections rather than creation order. Input slots break ties at joins.
    for (let pass = 0; pass < 8; pass++) {
      const forward = pass % 2 === 0
      for (const [, column] of forward ? ordered : [...ordered].reverse()) {
        const score = (item: Item) => {
          const neighbors = links.filter(link => forward ? link.to === item.id : link.from === item.id)
          return neighbors.length ? neighbors.reduce((sum, link) => {
            const neighbor = items.find(candidate => candidate.id === (forward ? link.from : link.to))!
            const port = terminalY(neighbor, forward ? link.source : link.target, forward ? undefined : link.inputSlot ?? 0) / neighbor.height
            return sum + (order.get(neighbor.id) ?? 0) + port * .8
          }, 0) / neighbors.length : order.get(item.id) ?? 0
        }
        column.sort((a, b) => score(a) - score(b))
        column.forEach((item, index) => order.set(item.id, index))
      }
    }
    const columnHeights = [...columns.values()].map((column) => column.reduce((sum, item) => sum + item.height, 0) + (column.length - 1) * ROW_GAP)
    const maxHeight = Math.max(0, ...columnHeights)
    let x = 0
    for (const [, column] of ordered) {
      const height = column.reduce((sum, item) => sum + item.height, 0) + (column.length - 1) * ROW_GAP
      let y = (maxHeight - height) / 2
      for (const item of column) { item.x = x; item.y = y; y += item.height + ROW_GAP }
      x += Math.max(...column.map((item) => item.width)) + COLUMN_GAP
    }
    const byId = new Map(items.map(item => [item.id, item]))
    const footerIds = new Set(!parent ? items.filter(item => item.node?.type === 'target').map(item => item.id) : [])
    for (const link of links) if (footerIds.has(link.from) && ['loss', 'cross-entropy'].includes(byId.get(link.to)?.node?.type ?? '')) footerIds.add(link.to)
    const footer = (item: Item) => footerIds.has(item.id)
    // Match wires at the real nested entry/exit, not the center of a large
    // enclosing frame. This aligns external inputs with a neuron's products.
    function terminalY(item: Item, nodeId: string, inputSlot?: number): number {
      const child = item.nested?.find(candidate => candidate.memberIds.includes(nodeId))
      if (child) return 66 + child.y + terminalY(child, nodeId, inputSlot)
      if (inputSlot === undefined) return item.node ? builderOutputPortY(item.node, 0) : item.height / 2
      if (item.node) return builderInputPortY(item.node, inputSlot)
      const count = item.node ? Math.max(1, ...edges.filter(edge => edge.target === nodeId).map(edge => (edge.inputSlot ?? 0) + 1)) : 2
      return item.height * (inputSlot + 1) / (count + 1)
    }
    for (let pass = 0; pass < 10; pass++) {
      const forward = pass % 2 === 0
      for (const [, column] of forward ? ordered : [...ordered].reverse()) {
        const core = column.filter(item => !footer(item))
        const desired = core.map(item => {
          const adjacent = links.filter(link => (forward ? link.to : link.from) === item.id && !footer(byId.get(forward ? link.from : link.to)!))
          const positions = adjacent.map(link => {
            const source = byId.get(link.from)!, target = byId.get(link.to)!
            return forward ? source.y + terminalY(source, link.source) - terminalY(item, link.target, link.inputSlot ?? 0)
              : target.y + terminalY(target, link.target, link.inputSlot ?? 0) - terminalY(item, link.source)
          }).sort((a, b) => a - b)
          return positions.length ? positions[Math.floor(positions.length / 2)] : item.y
        })
        // Align each column as a compact stack. Independent alignment of every
        // node can stretch small branches to match distant neighbors and leave
        // large holes between calculations that belong together.
        let offset = 0
        const offsets = core.map(item => { const start = offset; offset += item.height + ROW_GAP; return start })
        const anchors = desired.map((y, index) => y - offsets[index]).sort((a, b) => a - b)
        const anchor = anchors.length ? anchors[Math.floor(anchors.length / 2)] : 0
        core.forEach((item, index) => { item.y = anchor + offsets[index] })
      }
    }
    for (const item of items) {
      if (!item.node || !['weight', 'bias'].includes(item.node.type)) continue
      const link = links.find(candidate => candidate.from === item.id)
      const target = link && byId.get(link.to)
      if (link && target?.node && edges.some(edge => edge.target === target.node!.id && !owner.has(edge.source))) {
        item.y = Math.max(item.y, target.y + target.height / 3 + 24)
      }
    }
    for (const [, column] of ordered) {
      let bottom = -Infinity
      for (const item of [...column].filter(item => !footer(item)).sort((a, b) => a.y - b.y)) {
        item.y = Math.max(item.y, bottom)
        bottom = item.y + item.height + ROW_GAP
      }
    }
    const core = items.filter(item => !footer(item))
    const minY = core.length ? Math.min(...core.map(item => item.y)) : 0
    core.forEach(item => { item.y -= minY })
    // Pack the training target below its own column. Its wire needs a clear
    // lane below intervening blocks, but the entire target card does not need
    // to sit below the tallest block in the model.
    const targets = items.filter(item => item.node?.type === 'target' && !parent)
    const placed = [...core]
    for (const target of targets) {
      const loss = links.filter(link => link.from === target.id).map(link => byId.get(link.to)!).find(footer)
      const overlapsColumn = (a: Item, b: Item) => a.x < b.x + b.width && a.x + a.width > b.x
      const columnBottom = Math.max(0, ...placed.filter(item => overlapsColumn(item, target)).map(item => item.y + item.height))
      const intervening = placed.filter(item => item.x >= target.x + target.width && (!loss || item.x + item.width <= loss.x))
      const laneY = Math.max(0, ...intervening.map(item => item.y + item.height + WIRE_LANE))
      target.y = Math.max(columnBottom + ROW_GAP, laneY - target.height / 2)
      if (loss) {
        const targetLink = links.find(link => link.from === target.id && link.to === loss.id)!
        const portY = terminalY(loss, loss.id, targetLink.inputSlot ?? 1)
        const lossColumnBottom = Math.max(-ROW_GAP, ...placed.filter(item => overlapsColumn(item, loss)).map(item => item.y + item.height))
        loss.y = Math.max(target.y + target.height / 2 - portY, lossColumnBottom + ROW_GAP)
        target.y = loss.y + portY - target.height / 2
        placed.push(loss)
      }
      placed.push(target)
    }
    return items
  }
  function place(items: Item[], x: number, y: number): void {
    for (const item of items) {
      const offset = graph.view?.layoutOffsets?.[item.group ? `visual-group:${item.id}` : item.id]
      const rect = { x: x + item.x + (offset?.x ?? 0), y: y + item.y + (offset?.y ?? 0), width: item.width, height: item.height }
      if (item.group) result.groups.set(item.id, rect)
      else result.nodes.set(item.id, rect)
      if (item.nested) place(item.nested, rect.x + 28, rect.y + 66)
    }
  }
  place(layoutLevel(rootGroup), 40, 40)
  return result
}

export function semanticGroupDepth(graph: GraphModel, id: string): number {
  let group = graph.groups?.find((candidate) => candidate.id === id)
  let depth = 0
  const seen = new Set<string>()
  while (group?.parentId && !seen.has(group.id)) {
    seen.add(group.id)
    depth += 1
    group = graph.groups?.find((candidate) => candidate.id === group?.parentId)
  }
  return depth
}
