import { describe, expect, it } from 'vitest'
import { roundedWirePath, routeDiagramWires, segmentCrossesRect, WIRE_SPACING, type DiagramWire, type WireObstacle } from './wireRouting'
import type { Position } from './types'

const wire: DiagramWire = { id: 'signal', source: { x: 100, y: 60, side: 'right' }, target: { x: 600, y: 260, side: 'left' } }

function expectSeparated(routes: Map<string, Position[]>) {
  const paths = [...routes.values()]
  for (let i = 0; i < paths.length; i++) for (let j = 0; j < i; j++) {
    for (let a = 1; a < paths[i].length; a++) for (let b = 1; b < paths[j].length; b++) {
      const [p, q, r, s] = [paths[i][a - 1], paths[i][a], paths[j][b - 1], paths[j][b]]
      const horizontal = Math.abs(p.y - q.y) < .01
      if (horizontal !== (Math.abs(r.y - s.y) < .01)) continue
      const [along, across] = horizontal ? ['x', 'y'] as const : ['y', 'x'] as const
      const overlap = Math.min(Math.max(p[along], q[along]), Math.max(r[along], s[along])) - Math.max(Math.min(p[along], q[along]), Math.min(r[along], s[along]))
      // Prefer a full track of clearance, with room to compress slightly at
      // fixed ports and obstacle corners without visually merging the wires.
      if (overlap > .01) expect(Math.abs(p[across] - r[across])).toBeGreaterThanOrEqual(WIRE_SPACING * .75 - .01)
    }
  }
}

describe('clear connection routing', () => {
  it('keeps rounded corners clean when browser handle measurements have subpixel differences', () => {
    const path = roundedWirePath([{ x: 0, y: 60.5 }, { x: 100, y: 60 }, { x: 100.5, y: 120 }])
    expect(path).toContain('L91,60 Q100,60 100,69')
  })
  it('routes around intervening nodes and preserves the exact endpoints', () => {
    const obstacles: WireObstacle[] = [
      { id: 'source', x: 0, y: 10, width: 100, height: 100 },
      { id: 'middle', x: 220, y: 0, width: 160, height: 300 },
      { id: 'target', x: 600, y: 210, width: 100, height: 100 },
    ]
    const route = routeDiagramWires([wire], obstacles).get(wire.id)!
    expect(route[0]).toMatchObject({ x: 100, y: 60 })
    expect(route.at(-1)).toMatchObject({ x: 600, y: 260 })
    route.forEach((point, i) => { if (i) for (const rect of obstacles) expect(segmentCrossesRect(route[i - 1], point, rect)).toBe(false) })
    expect(roundedWirePath(route)).toContain('Q')
  })

  it('finds several turns through staggered obstacles and mixed port directions', () => {
    const obstacles = [
      { id: 'first', x: 140, y: -100, width: 100, height: 240 },
      { id: 'second', x: 300, y: 80, width: 100, height: 220 },
      { id: 'third', x: 460, y: -80, width: 80, height: 200 },
    ]
    const mixed = { ...wire, source: { x: 80, y: 60, side: 'bottom' as const }, target: { x: 620, y: 240, side: 'top' as const } }
    const route = routeDiagramWires([mixed], obstacles).get(wire.id)!
    for (let i = 1; i < route.length; i++) for (const obstacle of obstacles) expect(segmentCrossesRect(route[i - 1], route[i], obstacle)).toBe(false)
    expect(route.every(point => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })

  it('keeps a clear straight wire straight and recomputes after an obstacle moves', () => {
    const straight = { ...wire, target: { ...wire.target, y: 60 } }
    const before = routeDiagramWires([straight], []).get(wire.id)!
    expect(before).toHaveLength(2)
    const obstacle = { id: 'moved', x: 240, y: 30, width: 120, height: 90 }
    const after = routeDiagramWires([straight], [obstacle]).get(wire.id)!
    expect(after.length).toBeGreaterThan(2)
    for (let i = 1; i < after.length; i++) expect(segmentCrossesRect(after[i - 1], after[i], obstacle)).toBe(false)
  })

  it('routes around a blocking calculation without leaving its enclosing neuron', () => {
    const connection = { ...wire, target: { ...wire.target, y: 60 } }
    const obstacle = { id: 'calculation', x: 220, y: 0, width: 160, height: 180 }
    const bounds = { x: 0, y: 0, width: 740, height: 340 }
    const route = routeDiagramWires([connection], [obstacle], bounds).get(connection.id)!
    expect(route[0]).toEqual(connection.source)
    expect(route.at(-1)).toEqual(connection.target)
    for (const [i, point] of route.entries()) {
      expect(point.x).toBeGreaterThanOrEqual(bounds.x)
      expect(point.x).toBeLessThanOrEqual(bounds.width)
      expect(point.y).toBeGreaterThanOrEqual(bounds.y)
      expect(point.y).toBeLessThanOrEqual(bounds.height)
      if (i) expect(segmentCrossesRect(route[i - 1], point, obstacle)).toBe(false)
    }
  })

  it.each(['one obstacle', 'staggered obstacles'])('allocates separate tracks for ten wires around %s without recycling lanes', kind => {
    const wires: DiagramWire[] = Array.from({ length: 10 }, (_, i) => ({ id: `wire-${i}`, source: { x: 0, y: i * 24, side: 'right' }, target: { x: 700, y: i * 24 + 80, side: 'left' } }))
    const obstacles = kind === 'one obstacle'
      ? [{ id: 'box', x: 200, y: -30, width: 180, height: 340 }]
      : [{ id: 'one', x: 140, y: -100, width: 100, height: 260 }, { id: 'two', x: 310, y: 80, width: 100, height: 260 }, { id: 'three', x: 480, y: -100, width: 100, height: 300 }]
    const bounds = { x: -10, y: -400, width: 720, height: 1100 }
    const routes = routeDiagramWires(wires, obstacles, bounds)
    expectSeparated(routes)
    for (const wire of wires) {
      const path = routes.get(wire.id)!
      expect(path[0]).toEqual(wire.source)
      expect(path.at(-1)).toEqual(wire.target)
      for (let i = 1; i < path.length; i++) {
        for (const rect of obstacles) expect(segmentCrossesRect(path[i - 1], path[i], rect)).toBe(false)
        expect(path[i].x).toBeGreaterThanOrEqual(bounds.x)
        expect(path[i].x).toBeLessThanOrEqual(bounds.x + bounds.width)
        expect(path[i].y).toBeGreaterThanOrEqual(bounds.y)
        expect(path[i].y).toBeLessThanOrEqual(bounds.y + bounds.height)
      }
    }
    expect(routeDiagramWires([...wires].reverse(), obstacles, bounds)).toEqual(routes)
  })

  it('crosses swapped parallel connections without sharing a vertical tangent', () => {
    const routes = routeDiagramWires([
      { id: 'upper', source: { x: 0, y: 0, side: 'right' }, target: { x: 700, y: 24, side: 'left' } },
      { id: 'lower', source: { x: 0, y: 24, side: 'right' }, target: { x: 700, y: 0, side: 'left' } },
    ], [])
    expectSeparated(routes)
  })

  it('finishes rounded corners before a nearby crossing', () => {
    const points = [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }]
    const path = roundedWirePath(points, 12, [{ x: 100, y: 8 }])
    expect(path).toContain('L96,0 Q100,0 100,4 L100,100')
  })
})
