import { useLayoutEffect, useRef, type RefObject } from 'react'

/** WebKit can retain a low-resolution backing layer after selection repaints
 * a deeply scaled scene (https://bugs.webkit.org/show_bug.cgi?id=27684).
 * Refresh only its presentation transform; never move the React Flow camera,
 * remount editors, or publish a viewport/history change. */
export function useCanvasSelectionRepaint(shell: RefObject<HTMLDivElement | null>, selection: string) {
  const previous = useRef(selection)
  useLayoutEffect(() => {
    if (previous.current === selection) return
    previous.current = selection
    // Desktop Safari does not expose every iOS-only CSS feature, so detect
    // the engine rather than using -webkit-touch-callout as a proxy.
    const agent = navigator.userAgent
    if (!/AppleWebKit\//.test(agent) || /(?:Chrome|Chromium|Edg|OPR|Android)\//.test(agent)) return
    const viewport = shell.current?.querySelector<HTMLElement>('.react-flow__viewport')
    if (!viewport) return
    let original: string | undefined
    let temporary: string | undefined
    let frame = requestAnimationFrame(() => {
      original = viewport.style.transform
      const zoom = Number(original.match(/scale\(([^,)]+)/)?.[1])
      if (!(zoom > 1)) return
      // A subpixel scale change invalidates WebKit's raster scale. Keep it out
      // of React Flow's state, and restore it on the next frame. If a pan/zoom
      // arrives in between, its newer transform must win.
      viewport.style.transform = `${original} scale(1.00001)`
      temporary = viewport.style.transform
      frame = requestAnimationFrame(restore)
    })
    function restore() {
      if (original !== undefined && viewport && viewport.style.transform === temporary) viewport.style.transform = original
    }
    return () => {
      cancelAnimationFrame(frame)
      restore()
    }
  }, [shell, selection])
}
