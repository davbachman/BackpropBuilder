import type { Position } from './types'

export interface WireObstacle extends Position { id: string; width: number; height: number }
export interface WireEndpoint extends Position { side: 'left' | 'right' | 'top' | 'bottom' }
export interface DiagramWire { id: string; source: WireEndpoint; target: WireEndpoint }
export interface RoutingBounds extends Position { width: number; height: number }
type Segment = { a: Position; b: Position }
interface OccupiedTracks { horizontal: Segment[]; vertical: Segment[] }
const GAP = 12
// In the local units of each zoom level, not in world-space pixels.
export const WIRE_SPACING = 16
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

function leavesPort(port: WireEndpoint, point: Position): boolean {
  return port.side === 'right' ? point.x >= port.x - EPS : port.side === 'left' ? point.x <= port.x + EPS
    : port.side === 'bottom' ? point.y >= port.y - EPS : point.y <= port.y + EPS
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

function tracksBetween(tracks: Segment[], coordinate: 'x' | 'y', min: number, max: number): Segment[] {
  let low = 0, high = tracks.length
  while (low < high) { const middle = (low + high) >> 1; if (tracks[middle].a[coordinate] < min) low = middle + 1; else high = middle }
  const found: Segment[] = []
  for (let i = low; i < tracks.length && tracks[i].a[coordinate] <= max; i++) found.push(tracks[i])
  return found
}

function conflictCost(a: Position, b: Position, tracks: OccupiedTracks): number {
  const horizontal = Math.abs(a.y - b.y) < EPS
  const parallel = horizontal ? tracksBetween(tracks.horizontal, 'y', a.y - WIRE_SPACING, a.y + WIRE_SPACING) : tracksBetween(tracks.vertical, 'x', a.x - WIRE_SPACING, a.x + WIRE_SPACING)
  const crossing = horizontal ? tracksBetween(tracks.vertical, 'x', Math.min(a.x, b.x), Math.max(a.x, b.x)) : tracksBetween(tracks.horizontal, 'y', Math.min(a.y, b.y), Math.max(a.y, b.y))
  let cost = 0
  for (const segment of [...parallel, ...crossing]) {
    const c = segment.a, d = segment.b, otherHorizontal = Math.abs(c.y - d.y) < EPS
    if (horizontal === otherHorizontal) {
      const separation = Math.abs((horizontal ? a.y : a.x) - (horizontal ? c.y : c.x))
      if (separation >= WIRE_SPACING - EPS) continue
      const overlap = Math.min(Math.max(horizontal ? a.x : a.y, horizontal ? b.x : b.y), Math.max(horizontal ? c.x : c.y, horizontal ? d.x : d.y)) - Math.max(Math.min(horizontal ? a.x : a.y, horizontal ? b.x : b.y), Math.min(horizontal ? c.x : c.y, horizontal ? d.x : d.y))
      // Sharing a long track is much worse than a short detour or a crossing.
      cost += Math.max(0, overlap) * 32 * (1 - separation / WIRE_SPACING)
    } else {
      const [h1, h2, v1, v2] = horizontal ? [a, b, c, d] : [c, d, a, b]
      if (v1.x > Math.min(h1.x, h2.x) + EPS && v1.x < Math.max(h1.x, h2.x) - EPS && h1.y > Math.min(v1.y, v2.y) + EPS && h1.y < Math.max(v1.y, v2.y) - EPS) cost += 48
    }
  }
  return cost
}

function cornerCost(point: Position, tracks: OccupiedTracks): number {
  return [...tracksBetween(tracks.horizontal, 'y', point.y - WIRE_SPACING, point.y + WIRE_SPACING), ...tracksBetween(tracks.vertical, 'x', point.x - WIRE_SPACING, point.x + WIRE_SPACING)].reduce((cost, { a, b }) => {
    const separation = Math.abs(a.y - b.y) < EPS
      ? Math.abs(point.y - a.y) + Math.max(0, Math.min(a.x, b.x) - point.x, point.x - Math.max(a.x, b.x))
      : Math.abs(point.x - a.x) + Math.max(0, Math.min(a.y, b.y) - point.y, point.y - Math.max(a.y, b.y))
    // A turn on another wire creates a tangency instead of a clear crossing.
    return cost + Math.max(0, 1 - separation / WIRE_SPACING) * 160
  }, 0)
}

function channelCoordinates(start: Position, end: Position, rects: WireObstacle[], used: Segment[], bounds?: RoutingBounds) {
  const xs = [start.x, end.x, (start.x + end.x) / 2, ...rects.flatMap(r => [r.x, r.x + r.width])]
  const ys = [start.y, end.y, (start.y + end.y) / 2, ...rects.flatMap(r => [r.y, r.y + r.height])]
  // Allocate new tracks beside occupied ones. There is no repeating lane index
  // and the multi-turn search uses these same coordinates and congestion costs.
  for (const { a, b } of used) {
    const coordinates = Math.abs(a.y - b.y) < EPS ? ys : xs
    const value = Math.abs(a.y - b.y) < EPS ? a.y : a.x
    coordinates.push(value - WIRE_SPACING, value + WIRE_SPACING)
  }
  if (bounds) { xs.push(bounds.x, bounds.x + bounds.width); ys.push(bounds.y, bounds.y + bounds.height) }
  const unique = (values: number[], min = -Infinity, max = Infinity) => values
    .filter(value => value >= min - EPS && value <= max + EPS)
    .sort((a, b) => a - b).filter((value, i, array) => !i || value - array[i - 1] > EPS)
  return {
    xs: unique(xs, bounds?.x, bounds ? bounds.x + bounds.width : undefined),
    ys: unique(ys, bounds?.y, bounds ? bounds.y + bounds.height : undefined),
  }
}

/** Short connections stay simple. Long connections use clear channels around
 * blocks, with a penalty for sharing or crossing already occupied channels. */
export function routeDiagramWires(wires: DiagramWire[], obstacles: WireObstacle[], bounds?: RoutingBounds): Map<string, Position[]> {
  const padded = obstacles.map(rect => ({ ...rect, x: rect.x - GAP, y: rect.y - GAP, width: rect.width + GAP * 2, height: rect.height + GAP * 2 }))
  const used: Segment[] = []
  const routes = new Map<string, Position[]>()
  for (const wire of [...wires].sort((a, b) => distance(a.source, a.target) - distance(b.source, b.target) || a.id.localeCompare(b.id))) {
    const tracks = {
      horizontal: used.filter(({ a, b }) => Math.abs(a.y - b.y) < EPS).sort((a, b) => a.a.y - b.a.y),
      vertical: used.filter(({ a, b }) => Math.abs(a.x - b.x) < EPS).sort((a, b) => a.a.x - b.a.x),
    }
    const start = stub(wire.source), end = stub(wire.target)
    // A manually moved block can overlap another one. Keep its endpoint usable
    // and route around the remaining blocks rather than trapping the search.
    const rects = padded.filter(rect => ![start, end].some(p => p.x > rect.x && p.x < rect.x + rect.width && p.y > rect.y && p.y < rect.y + rect.height))
    const channels = channelCoordinates(start, end, rects, used, bounds)
    const { xs, ys } = channels
    const candidates = [
      [start, { x: end.x, y: start.y }, end],
      [start, { x: start.x, y: end.y }, end],
      ...xs.map(x => [start, { x, y: start.y }, { x, y: end.y }, end]),
      ...ys.map(y => [start, { x: start.x, y }, { x: end.x, y }, end]),
    ]
    let best: Position[] | undefined, bestCost = Infinity, bestConflict = Infinity
    const score = (path: Position[]) => {
      if (!leavesPort(wire.source, path[1]) || !leavesPort(wire.target, path.at(-2)!)) return
      const conflict = path.reduce((sum, point, i) => i ? sum + conflictCost(path[i - 1], point, tracks) : sum, 0)
        + path.slice(1, -1).reduce((sum, point) => sum + cornerCost(point, tracks), 0)
      const cost = path.reduce((sum, point, i) => i ? sum + distance(path[i - 1], point) : sum, 0) + conflict + path.length * 24
      if (cost < bestCost) { best = path; bestCost = cost; bestConflict = conflict }
    }
    for (const points of candidates) {
      if (bounds && points.some(point => !insideBounds(point, bounds))) continue
      if (points.some((point, i) => i > 0 && rects.some(rect => segmentCrossesRect(points[i - 1], point, rect)))) continue
      const path = simplify([wire.source, ...points, wire.target])
      score(path)
    }
    // More turns may be needed to separate two paths even without a blocking
    // node. Compare that alternative before accepting a congested dogleg.
    const portConflict = conflictCost(wire.source, start, tracks) + conflictCost(end, wire.target, tracks)
    if (!best || bestConflict > portConflict + 200) {
      const searched = searchChannels(start, end, rects, tracks, channels, bestCost, wire, wires.length < 48 ? 8000 : 1200)
      if (searched) score(simplify([wire.source, ...searched, wire.target]))
    }
    const route = best ?? simplify([wire.source, start, { x: end.x, y: start.y }, end, wire.target])
    routes.set(wire.id, route)
    route.forEach((point, i) => { if (i) used.push({ a: route[i - 1], b: point }) })
  }
  return routes
}

/** Rectilinear visibility-grid A*: used only when a wire needs several turns. */
function searchChannels(start: Position, end: Position, rects: WireObstacle[], tracks: OccupiedTracks, { xs, ys }: { xs: number[]; ys: number[] }, maxCost: number, wire: DiagramWire, refinementLimit: number): Position[] | undefined {
  const indexOf = (values: number[], value: number) => values.findIndex(item => Math.abs(item - value) < EPS)
  const width = xs.length, startCell = indexOf(ys, start.y) * width + indexOf(xs, start.x), endCell = indexOf(ys, end.y) * width + indexOf(xs, end.x)
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
  const visited = new Set<number>(), segmentCosts = new Map<string, number>(), turns = new Map<number, number>()
  while (heap.length) {
    // Refinement is optional once an obstacle-free route exists. Bound its
    // work so a neuron with dozens of nearly coincident ports stays draggable;
    // the initial obstacle search still runs to completion when needed.
    if (Number.isFinite(maxCost) && visited.size >= refinementLimit) return undefined
    const { state, score } = pop()
    if (score > maxCost) return undefined
    if (visited.has(state)) continue
    visited.add(state)
    const cell = Math.floor(state / 3), direction = state % 3, current = point(cell)
    if (cell === endCell) {
      const path: Position[] = [current]
      let cursor = state
      while (previous.has(cursor)) { cursor = previous.get(cursor)!; path.push(point(Math.floor(cursor / 3))) }
      path.reverse()
      // Preserve exact ports even when equivalent grid coordinates differ by
      // floating-point noise in a deeply nested scene.
      path[0] = start; path[path.length - 1] = end
      return path
    }
    const x = cell % width, y = Math.floor(cell / width)
    for (const [nextCell, nextDirection] of [[x > 0 ? cell - 1 : -1, 1], [x < width - 1 ? cell + 1 : -1, 1], [y > 0 ? cell - width : -1, 2], [y < ys.length - 1 ? cell + width : -1, 2]]) {
      if (nextCell < 0) continue
      const next = point(nextCell), key = `${Math.min(cell, nextCell)}:${Math.max(cell, nextCell)}`
      if (cell === startCell && !leavesPort({ ...start, side: wire.source.side }, next)) continue
      if (nextCell === endCell && !leavesPort({ ...end, side: wire.target.side }, current)) continue
      if (!segmentCosts.has(key)) segmentCosts.set(key, rects.some(rect => segmentCrossesRect(current, next, rect)) ? Infinity : distance(current, next) + conflictCost(current, next, tracks))
      const segmentCost = segmentCosts.get(key)!
      if (!Number.isFinite(segmentCost)) continue
      if (!turns.has(cell)) turns.set(cell, 24 + cornerCost(current, tracks))
      const nextState = nextCell * 3 + nextDirection, cost = costs.get(state)! + segmentCost + (direction && direction !== nextDirection ? turns.get(cell)! : 0)
      if (cost >= (costs.get(nextState) ?? Infinity)) continue
      costs.set(nextState, cost); previous.set(nextState, state)
      push({ state: nextState, score: cost + distance(next, end) })
    }
  }
  return undefined
}

function insideBounds(point: Position, bounds: RoutingBounds): boolean {
  return point.x >= bounds.x - EPS && point.x <= bounds.x + bounds.width + EPS && point.y >= bounds.y - EPS && point.y <= bounds.y + bounds.height + EPS
}

export function roundedWirePath(points: Position[], radius = 9, crossings: Position[] = []): string {
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
    // Finish rounding before an intersection, so crossing paths meet at right
    // angles instead of brushing tangentially along another rounded corner.
    const cornerRadius = Math.min(radius, ...crossings.map(point => distance(point, corner) / 2))
    const before = Math.min(cornerRadius, distance(previous, corner) / 2), after = Math.min(cornerRadius, distance(corner, next) / 2)
    const a = towards(corner, previous, before)
    const b = towards(corner, next, after)
    result += ` L${n(a.x)},${n(a.y)} Q${n(corner.x)},${n(corner.y)} ${n(b.x)},${n(b.y)}`
  }
  const end = path.at(-1)!
  return `${result} L${n(end.x)},${n(end.y)}`
}
