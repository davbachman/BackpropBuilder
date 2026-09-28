import { useRef } from 'react'
import { act, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useCanvasSelectionRepaint } from './useCanvasSelectionRepaint'

function Harness({ selection = '', transform = 'translate(-200px, -300px) scale(12)' }) {
  const shell = useRef<HTMLDivElement>(null)
  useCanvasSelectionRepaint(shell, selection)
  return <div ref={shell}><div className="react-flow__viewport" style={{ transform }}><input defaultValue="editable" /></div></div>
}

function frames(webkit = true) {
  vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue(webkit
    ? 'Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15'
    : 'Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36')
  let serial = 0
  const callbacks = new Map<number, FrameRequestCallback>()
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => { callbacks.set(++serial, callback); return serial }))
  vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => callbacks.delete(id)))
  return () => act(() => { const next = [...callbacks.values()]; callbacks.clear(); next.forEach(callback => callback(0)) })
}

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

it('refreshes selecting and deselecting without remounting editors or changing the final camera', () => {
  const frame = frames()
  const { container, rerender } = render(<Harness />)
  const viewport = container.querySelector<HTMLElement>('.react-flow__viewport')!
  const input = container.querySelector('input')!
  const original = viewport.style.transform
  input.focus()
  for (const selection of ['weight', '']) {
    rerender(<Harness selection={selection} />)
    frame()
    expect(viewport.style.transform).not.toBe(original)
    frame()
    expect(viewport.style.transform).toBe(original)
    expect(container.querySelector('input')).toBe(input)
    expect(input).toHaveFocus()
  }
})

it('does not overwrite a real camera move during the repaint', () => {
  const frame = frames()
  const { container, rerender } = render(<Harness />)
  rerender(<Harness selection="weight" />)
  frame()
  const moved = 'translate(25px, 10px) scale(14)'
  rerender(<Harness selection="weight" transform={moved} />)
  frame()
  expect(container.querySelector<HTMLElement>('.react-flow__viewport')!.style.transform).toBe(moved)
})

it('cancels an outstanding repaint on unmount and restores its temporary transform', () => {
  const frame = frames()
  const { container, rerender, unmount } = render(<Harness />)
  const viewport = container.querySelector<HTMLElement>('.react-flow__viewport')!
  const original = viewport.style.transform
  rerender(<Harness selection="weight" />)
  frame()
  unmount()
  frame()
  expect(viewport.style.transform).toBe(original)
})

it.each([false, true])('leaves unscaled scenes alone (WebKit: %s)', webkit => {
  const frame = frames(webkit)
  const transform = 'translate(10px, 20px) scale(0.8)'
  const { container, rerender } = render(<Harness transform={transform} />)
  rerender(<Harness selection="weight" transform={transform} />)
  frame()
  expect(container.querySelector<HTMLElement>('.react-flow__viewport')!.style.transform).toBe(transform)
})

it('does not add repaint work to other browsers', () => {
  frames(false)
  const { rerender } = render(<Harness />)
  rerender(<Harness selection="weight" />)
  expect(requestAnimationFrame).not.toHaveBeenCalled()
})
