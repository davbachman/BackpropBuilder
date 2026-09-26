import { fireEvent, render, screen } from '@testing-library/react'
import { Position, type EdgeProps } from '@xyflow/react'
import { describe, expect, it, vi } from 'vitest'
import { BuilderEdge, type BuilderEdgeData } from './BuilderEdge'
import { edgeSignalIntensity } from '../domain/edgeSignal'

function renderEdge(data: Partial<BuilderEdgeData> = {}, selected = false) {
  const props = { id: 'wire', source: 'a', target: 'b', sourceX: 0, sourceY: 80, targetX: 240, targetY: 80, sourcePosition: Position.Right, targetPosition: Position.Left, selected, data: { active: true, phase: 'forward', showGradient: false, label: 'Input → Product', ...data } } as EdgeProps
  return render(<svg><BuilderEdge {...props} /></svg>)
}

describe('clean directional connection flow', () => {
  it('shows forward bands without visible numerical labels', () => {
    const { container } = renderEdge({ forward: { shape: [], data: [2.5] } })
    expect(screen.getByRole('button', { name: /Input → Product · forward value: 2.500/ })).toBeInTheDocument()
    expect(container.querySelector('.builder-edge-bands')?.getAttribute('data-direction')).toBe('forward')
    expect(container.querySelector('text')).toBeNull()
  })

  it('uses backward contributions for intensity and reverses the bands', () => {
    const { container } = renderEdge({ phase: 'backward', forward: { shape: [], data: [9] }, gradient: { shape: [], data: [-0.75] } })
    expect(screen.getByRole('button', { name: /backward gradient: -0.750/ })).toBeInTheDocument()
    expect(container.querySelector('.builder-edge-bands')?.getAttribute('data-direction')).toBe('backward')
    expect(container.querySelector('.builder-edge-flow')?.getAttribute('style')).toContain(String(edgeSignalIntensity({ shape: [], data: [-0.75] })))
  })

  it('never animates a value or gradient that the trace has not reached', () => {
    const { container } = renderEdge({ phase: 'backward', forward: { shape: [], data: [9] } })
    expect(container.querySelector('.builder-edge-bands')).toBeNull()
    expect(screen.getByRole('button', { name: /not reached in this trace/ })).toBeInTheDocument()
  })

  it('supports whole-wire click and keyboard inspection with full tensor details', () => {
    const onInspect = vi.fn()
    renderEdge({ forward: { shape: [2], data: [1.2, 3.4] }, onInspect }, true)
    const wire = screen.getByRole('button', { name: /forward value: \[2\] \[1.200, 3.400\]/ })
    fireEvent.click(wire)
    fireEvent.keyDown(wire, { key: 'Enter' })
    expect(onInspect).toHaveBeenCalledTimes(2)
    expect(onInspect).toHaveBeenLastCalledWith('wire')
  })

  it('places bands on the exact residual path', () => {
    const { container } = renderEdge({ residual: true, forward: { shape: [], data: [1] } })
    const path = container.querySelector('path.builder-edge')!.getAttribute('d')
    expect(container.querySelector('.builder-edge-bands')!.getAttribute('d')).toBe(path)
    expect(path).toContain('L180,16')
  })

  it('keeps backward flow bands on the routed path around blocks', () => {
    const { container } = renderEdge({ phase: 'backward', gradient: { shape: [], data: [2] }, route: [
      { x: 0, y: 80 }, { x: 40, y: 80 }, { x: 40, y: 20 }, { x: 200, y: 20 }, { x: 200, y: 80 }, { x: 240, y: 80 },
    ] })
    const path = container.querySelector('path.builder-edge')!.getAttribute('d')
    expect(path).toContain('Q40,20')
    expect(container.querySelector('.builder-edge-bands')!.getAttribute('d')).toBe(path)
    expect(container.querySelector('.builder-edge-bands')!.getAttribute('data-direction')).toBe('backward')
  })

  it('draws grouped builder wires with broad bends and readable weight at overview zoom', () => {
    const { container } = renderEdge({ absoluteRoute: true, sceneScale: 1, cameraZoom: .6,
      route: [{ x: 0, y: 0 }, { x: 80, y: 0 }, { x: 80, y: 60 }, { x: 184, y: 60 }],
      forward: { shape: [], data: [1] },
    })
    const wire = container.querySelector('.builder-edge-flow')!
    expect(wire.getAttribute('style')).toContain('--edge-scale: 1.4')
    const path = container.querySelector('path.builder-edge')!.getAttribute('d')!
    expect(path).toContain('L44,0 Q80,0 80,30')
    expect(container.querySelector('.builder-edge-bands')!.getAttribute('d')).toBe(path)
  })

  it('masks the crossing gap on both the wire and its moving bands at deep zoom', () => {
    const { container } = renderEdge({ absoluteRoute: true, sceneScale: .01, cameraZoom: 150,
      route: [{ x: 12, y: 34 }, { x: 12, y: 35 }], crossings: { points: [{ x: 12, y: 34.5 }], gaps: [{ x: 12, y: 34.5 }] },
      phase: 'backward', gradient: { shape: [], data: [-.25] },
    })
    const mask = container.querySelector('mask')!, circle = mask.querySelector('circle')!
    expect(mask.getAttribute('maskUnits')).toBe('userSpaceOnUse')
    expect(circle.getAttribute('cx')).toBe('0')
    expect(circle.getAttribute('cy')).toBe('50')
    expect(circle.getAttribute('r')).toBe('4')
    const wire = container.querySelector('path.builder-edge')!, bands = container.querySelector('.builder-edge-bands')!
    expect(wire.closest('g[mask]')?.getAttribute('mask')).toBe(`url(#${mask.id})`)
    expect(bands.closest('g[mask]')).toBe(wire.closest('g[mask]'))
    expect(bands.getAttribute('d')).toBe(wire.getAttribute('d'))
  })

  it('encodes absolute magnitude consistently, excluding causal sentinels', () => {
    expect(edgeSignalIntensity({ shape: [], data: [-2] })).toBe(edgeSignalIntensity({ shape: [], data: [2] }))
    expect(edgeSignalIntensity({ shape: [], data: [2] })).toBeGreaterThan(edgeSignalIntensity({ shape: [], data: [0.2] }))
    expect(edgeSignalIntensity({ shape: [2], data: [1, -1e9], excluded: [false, true] })).toBe(edgeSignalIntensity({ shape: [], data: [1] }))
    expect(edgeSignalIntensity({ shape: [], data: [1000] })).toBe(1)
  })
})

it('keeps nested wire segments in their exact world coordinates and preserves gradient inspection', () => {
  const onInspect = vi.fn()
  const { container } = renderEdge({ absoluteRoute: true, sceneScale: .01, cameraZoom: 100,
    route: [{ x: 12, y: 34 }, { x: 12.5, y: 34 }, { x: 12.5, y: 34.5 }, { x: 13, y: 34.5 }],
    phase: 'backward', gradient: { shape: [], data: [-.25] }, onInspect,
  })
  const path = container.querySelector('path.builder-edge')!.getAttribute('d')
  expect(container.querySelector('.builder-edge-flow')!.getAttribute('transform')).toBe('translate(12 34) scale(0.01)')
  expect(path).toMatch(/^M0,0/)
  expect(path).toContain('100,50')
  expect(path).toContain('Q50,0')
  expect(container.querySelector('.builder-edge-bands')!.getAttribute('d')).toBe(path)
  fireEvent.click(screen.getByRole('button', { name: /backward gradient: -0.250/ }))
  expect(onInspect).toHaveBeenCalledWith('wire')
})
