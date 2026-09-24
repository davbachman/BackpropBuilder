import { describe, expect, it } from 'vitest'
import { findWireCrossings } from './wireCrossings'

describe('unambiguous crossings', () => {
  it('cuts only the vertical underpass regardless of drawing order', () => {
    const routes = new Map([
      ['horizontal', [{ x: 0, y: 40 }, { x: 100, y: 40 }]],
      ['vertical', [{ x: 60, y: 0 }, { x: 60, y: 80 }]],
    ])
    const crossings = findWireCrossings(routes)
    expect(crossings.get('vertical')).toEqual({ points: [{ x: 60, y: 40 }], gaps: [{ x: 60, y: 40 }] })
    expect(crossings.get('horizontal')).toEqual({ points: [{ x: 60, y: 40 }], gaps: [] })
    expect(findWireCrossings(new Map([...routes].reverse()))).toEqual(crossings)
  })

  it('preserves shared ports and does not cut adjacent parallel wires', () => {
    const routes = new Map([
      ['first', [{ x: 0, y: 0 }, { x: 100, y: 0 }]],
      ['branch', [{ x: 0, y: 0 }, { x: 0, y: 50 }, { x: 100, y: 50 }]],
      ['parallel', [{ x: 0, y: 16 }, { x: 100, y: 16 }]],
    ])
    for (const crossing of findWireCrossings(routes).values()) expect(crossing.gaps).toEqual([])
  })
})
