import { describe, expect, it } from 'vitest'
import { roundedWirePath, routeDiagramWires, segmentCrossesRect, type DiagramWire, type WireObstacle } from './wireRouting'

const wire: DiagramWire = { id: 'signal', source: { x: 100, y: 60, side: 'right' }, target: { x: 600, y: 260, side: 'left' } }

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
})
