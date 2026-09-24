import type { Position } from './types'

export interface WireCrossings { points: Position[]; gaps: Position[] }
const EPS = .01
const close = (a: Position, b: Position) => Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS

/** Decorate one routing level in its own units. A gap belongs to the vertical
 * wire at a crossing; the horizontal wire continues over it. The paths and
 * animated bands remain continuous, and real shared ports are never cut. */
export function findWireCrossings(routes: Map<string, Position[]>): Map<string, WireCrossings> {
  const result = new Map([...routes.keys()].map(id => [id, { points: [], gaps: [] } as WireCrossings]))
  const entries = [...routes]
  const add = (list: Position[], point: Position) => { if (!list.some(other => close(other, point))) list.push(point) }
  for (let i = 0; i < entries.length; i++) {
    const [id, path] = entries[i]
    for (let j = 0; j < i; j++) {
      const [otherId, other] = entries[j]
      for (let a = 1; a < path.length; a++) for (let b = 1; b < other.length; b++) {
        const p = path[a - 1], q = path[a], r = other[b - 1], s = other[b]
        const horizontal = Math.abs(p.y - q.y) < EPS
        if (horizontal === (Math.abs(r.y - s.y) < EPS)) continue
        const [h1, h2, v1, v2] = horizontal ? [p, q, r, s] : [r, s, p, q]
        const point = { x: v1.x, y: h1.y }
        // Strict interiors: bends which merely touch are discouraged by the
        // router, not disguised as crossings by punching a hole at the bend.
        if (point.x <= Math.min(h1.x, h2.x) + EPS || point.x >= Math.max(h1.x, h2.x) - EPS || point.y <= Math.min(v1.y, v2.y) + EPS || point.y >= Math.max(v1.y, v2.y) - EPS) continue
        add(result.get(id)!.points, point)
        add(result.get(otherId)!.points, point)
        add(result.get(horizontal ? otherId : id)!.gaps, point)
      }
    }
  }
  return result
}
