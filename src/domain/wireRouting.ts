import type { Position } from './types'

export interface WireObstacle extends Position { id: string; width: number; height: number }
export interface WireEndpoint extends Position { side: 'left' | 'right' | 'top' | 'bottom' }
export interface DiagramWire { id: string; source: WireEndpoint; target: WireEndpoint }
export interface RoutingBounds extends Position { width: number; height: number }
type Segment = { a: Position; b: Position }
const GAP = 12
const EPS = .01
const distance = (a: Position, b: Position) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y)

export function segmentCrossesRect(a: Position, b: Position, rect: WireObstacle): boolean {
  if (Math.abs(a.y - b.y) < EPS) return a.y > rect.y + EPS && a.y < rect.y + rect.height - EPS && Math.max(a.x, b.x) > rect.x + EPS && Math.min(a.x, b.x) < rect.x + rect.width - EPS
  if (Math.abs(a.x - b.x) < EPS) return a.x > rect.x + EPS && a.x < rect.x + rect.width - EPS && Math.max(a.y, b.y) > rect.y + EPS && Math.min(a.y, b.y) < rect.y + rect.height - EPS
  return true
}

function stub(point: WireEndpoint): Position {
  return { x: point.x + (point.side === 'right' ? 24 : point.side === 'left' ? -24 : 0), y: point.y + (point.side === 'bottom' ? 24 : point.side === 'top' ? -24 : 0) }
}

function simplify(points: Position[]): Position[] {
  const result: Position[] = []
  for (const point of points) {
    if (result.length && distance(result.at(-1)!, point) < EPS) continue
    while (result.length > 1) {
      const a = result.at(-2)!, b = result.at(-1)!
      if (Math.abs(a.x - b.x) < EPS && Math.abs(b.x - point.x) < EPS || Math.abs(a.y - b.y) < EPS && Math.abs(b.y - point.y) < EPS) result.pop()
      else break
    }
    result.push(point)
  }
  return result
}

function conflictCost(a: Position, b: Position, used: Segment[]): number {
  const horizontal = Math.abs(a.y - b.y) < EPS
  let cost = 0
  for (const segment of used) {
    const c = segment.a, d = segment.b, otherHorizontal = Math.abs(c.y - d.y) < EPS
    if (horizontal === otherHorizontal) {
      if (Math.abs((horizontal ? a.y : a.x) - (horizontal ? c.y : c.x)) > 3) continue
      const overlap = Math.min(Math.max(horizontal ? a.x : a.y, horizontal ? b.x : b.y), Math.max(horizontal ? c.x : c.y, horizontal ? d.x : d.y)) - Math.max(Math.min(horizontal ? a.x : a.y, horizontal ? b.x : b.y), Math.min(horizontal ? c.x : c.y, horizontal ? d.x : d.y))
      cost += Math.max(0, overlap) * .8
    } else {
      const [h1, h2, v1, v2] = horizontal ? [a, b, c, d] : [c, d, a, b]
      if (v1.x > Math.min(h1.x, h2.x) + EPS && v1.x < Math.max(h1.x, h2.x) - EPS && h1.y > Math.min(v1.y, v2.y) + EPS && h1.y < Math.max(v1.y, v2.y) - EPS) cost += 48
    }
  }
  return cost
}

/** Short connections stay simple. Long connections use clear channels around
 * blocks, with a penalty for sharing or crossing already occupied channels. */
export function routeDiagramWires(wires: DiagramWire[], obstacles: WireObstacle[], bounds?: RoutingBounds): Map<string, Position[]> {
  const padded = obstacles.map(rect => ({ ...rect, x: rect.x - GAP, y: rect.y - GAP, width: rect.width + GAP * 2, height: rect.height + GAP * 2 }))
  const used: Segment[] = []
  const routes = new Map<string, Position[]>()
  for (const [index, wire] of [...wires].sort((a, b) => distance(a.source, a.target) - distance(b.source, b.target) || a.id.localeCompare(b.id)).entries()) {
    const start = stub(wire.source), end = stub(wire.target)
    // A manually moved block can overlap another one. Keep its endpoint usable
    // and route around the remaining blocks rather than trapping the search.
    const rects = padded.filter(rect => ![start, end].some(p => p.x > rect.x && p.x < rect.x + rect.width && p.y > rect.y && p.y < rect.y + rect.height))
    const lane = (index % 7 - 3) * 7
    const xs = [...new Set([(start.x + end.x) / 2 + lane, start.x, end.x, ...rects.flatMap(r => [r.x - 7, r.x + r.width + 7])])]
    const ys = [...new Set([(start.y + end.y) / 2 + lane, start.y, end.y, ...rects.flatMap(r => [r.y - 7, r.y + r.height + 7])])]
    const candidates = [
      [start, { x: end.x, y: start.y }, end],
      [start, { x: start.x, y: end.y }, end],
      ...xs.map(x => [start, { x, y: start.y }, { x, y: end.y }, end]),
      ...ys.map(y => [start, { x: start.x, y }, { x: end.x, y }, end]),
    ]
    let best: Position[] | undefined, bestCost = Infinity
    for (const points of candidates) {
      if (bounds && points.some(point => !insideBounds(point, bounds))) continue
      if (points.some((point, i) => i > 0 && rects.some(rect => segmentCrossesRect(points[i - 1], point, rect)))) continue
      const path = simplify([wire.source, ...points, wire.target])
      const cost = path.reduce((sum, point, i) => i ? sum + distance(path[i - 1], point) + conflictCost(path[i - 1], point, used) : sum, 0) + path.length * 16
      if (cost < bestCost) { best = path; bestCost = cost }
    }
    if (!best) best = simplify([wire.source, ...searchChannels(start, end, rects, bounds), wire.target])
    routes.set(wire.id, best)
    best.forEach((point, i) => { if (i) used.push({ a: best![i - 1], b: point }) })
  }
  return routes
}

/** Rectilinear visibility-grid A*: used only when a wire needs several turns. */
function searchChannels(start: Position, end: Position, rects: WireObstacle[], bounds?: RoutingBounds): Position[] {
  const xs = [...new Set([start.x, end.x, ...rects.flatMap(r => [r.x, r.x + r.width]), ...(bounds ? [bounds.x, bounds.x + bounds.width] : [])])].filter(x => !bounds || x >= bounds.x - EPS && x <= bounds.x + bounds.width + EPS).sort((a, b) => a - b)
  const ys = [...new Set([start.y, end.y, ...rects.flatMap(r => [r.y, r.y + r.height]), ...(bounds ? [bounds.y, bounds.y + bounds.height] : [])])].filter(y => !bounds || y >= bounds.y - EPS && y <= bounds.y + bounds.height + EPS).sort((a, b) => a - b)
  const width = xs.length, startCell = ys.indexOf(start.y) * width + xs.indexOf(start.x), endCell = ys.indexOf(end.y) * width + xs.indexOf(end.x)
  const point = (cell: number) => ({ x: xs[cell % width], y: ys[Math.floor(cell / width)] })
  const costs = new Map<number, number>([[startCell * 3, 0]]), previous = new Map<number, number>()
  const heap: { state: number; score: number }[] = []
  const push = (entry: { state: number; score: number }) => {
    heap.push(entry)
    let i = heap.length - 1
    while (i > 0) { const parent = (i - 1) >> 1; if (heap[parent].score <= entry.score) break; heap[i] = heap[parent]; i = parent }
    heap[i] = entry
  }
  const pop = () => {
    const first = heap[0], last = heap.pop()!
    if (heap.length) {
      let i = 0
      while (i * 2 + 1 < heap.length) { let child = i * 2 + 1; if (child + 1 < heap.length && heap[child + 1].score < heap[child].score) child++; if (heap[child].score >= last.score) break; heap[i] = heap[child]; i = child }
      heap[i] = last
    }
    return first
  }
  push({ state: startCell * 3, score: distance(start, end) })
  const visited = new Set<number>(), blocked = new Map<string, boolean>()
  while (heap.length) {
    const { state } = pop()
    if (visited.has(state)) continue
    visited.add(state)
    const cell = Math.floor(state / 3), direction = state % 3, current = point(cell)
    if (cell === endCell) {
      const path: Position[] = [current]
      let cursor = state
      while (previous.has(cursor)) { cursor = previous.get(cursor)!; path.push(point(Math.floor(cursor / 3))) }
      return path.reverse()
    }
    const x = cell % width, y = Math.floor(cell / width)
    for (const [nextCell, nextDirection] of [[x > 0 ? cell - 1 : -1, 1], [x < width - 1 ? cell + 1 : -1, 1], [y > 0 ? cell - width : -1, 2], [y < ys.length - 1 ? cell + width : -1, 2]]) {
      if (nextCell < 0) continue
      const next = point(nextCell), key = `${Math.min(cell, nextCell)}:${Math.max(cell, nextCell)}`
      if (!blocked.has(key)) blocked.set(key, rects.some(rect => segmentCrossesRect(current, next, rect)))
      if (blocked.get(key)) continue
      const nextState = nextCell * 3 + nextDirection, cost = costs.get(state)! + distance(current, next) + (direction && direction !== nextDirection ? 24 : 0)
      if (cost >= (costs.get(nextState) ?? Infinity)) continue
      costs.set(nextState, cost); previous.set(nextState, state)
      push({ state: nextState, score: cost + distance(next, end) })
    }
  }
  return [start, { x: end.x, y: start.y }, end]
}

function insideBounds(point: Position, bounds: RoutingBounds): boolean {
  return point.x >= bounds.x - EPS && point.x <= bounds.x + bounds.width + EPS && point.y >= bounds.y - EPS && point.y <= bounds.y + bounds.height + EPS
}

export function roundedWirePath(points: Position[], radius = 9): string {
  const path = simplify(points)
  if (!path.length) return ''
  const n = (value: number) => Number(value.toFixed(3))
  // React Flow measures handles on CSS borders, which can differ from the
  // layout by a fraction of a pixel. Keep each corner's tangent cardinal.
  const towards = (corner: Position, point: Position, amount: number): Position => Math.abs(point.x - corner.x) > Math.abs(point.y - corner.y)
    ? { x: corner.x + Math.sign(point.x - corner.x) * amount, y: corner.y }
    : { x: corner.x, y: corner.y + Math.sign(point.y - corner.y) * amount }
  let result = `M${n(path[0].x)},${n(path[0].y)}`
  for (let i = 1; i < path.length - 1; i++) {
    const previous = path[i - 1], corner = path[i], next = path[i + 1]
    const before = Math.min(radius, distance(previous, corner) / 2), after = Math.min(radius, distance(corner, next) / 2)
    const a = towards(corner, previous, before)
    const b = towards(corner, next, after)
    result += ` L${n(a.x)},${n(a.y)} Q${n(corner.x)},${n(corner.y)} ${n(b.x)},${n(b.y)}`
  }
  const end = path.at(-1)!
  return `${result} L${n(end.x)},${n(end.y)}`
}
