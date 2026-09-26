import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react'
import { useId, type CSSProperties, type ReactElement } from 'react'
import { formatFullTensor } from '../domain/tensor'
import { edgeSignalIntensity } from '../domain/edgeSignal'
import type { Position, TensorValue } from '../domain/types'
import { roundedWirePath } from '../domain/wireRouting'
import type { WireCrossings } from '../domain/wireCrossings'
import './flowEdges.css'

export interface BuilderEdgeData extends Record<string, unknown> {
  forward?: TensorValue
  gradient?: TensorValue
  showGradient: boolean
  active: boolean
  phase: string
  residual?: boolean
  route?: Position[]
  crossings?: WireCrossings
  absoluteRoute?: boolean
  sceneScale?: number
  cameraZoom?: number
  accessible?: boolean
  parentId?: string
  canonicalEdgeId?: string
  label?: string
  onInspect?: (edgeId: string) => void
}

/** Only trace-visible values reach this component. Bands show direction;
 * numerical details remain in the connection inspector. */
export function BuilderEdge(props: EdgeProps): ReactElement {
  const maskId = `wire-crossing-${useId()}`
  const [defaultPath] = getBezierPath(props)
  const data = props.data as BuilderEdgeData | undefined
  const isBackward = data?.phase === 'backward'
  const top = Math.min(props.sourceY, props.targetY) - 64
  const geometryScale = data?.sceneScale ?? 1
  const origin = data?.absoluteRoute ? data.route?.[0] : undefined
  // Draw nested paths in their own units. Rounding microscopic world-space
  // coordinates would erase corners when the camera reaches a single neuron.
  const screenScale = geometryScale * (data?.cameraZoom ?? 1)
  const scale = Math.min(origin ? 1.4 : 1, 1.5 / screenScale)
  const localPoint = (point: Position) => origin ? { x: (point.x - origin.x) / geometryScale, y: (point.y - origin.y) / geometryScale } : point
  const route = origin ? data?.route?.map(localPoint) : data?.route?.length ? [
    { x: props.sourceX, y: props.sourceY }, ...data.route.slice(1, -1), { x: props.targetX, y: props.targetY },
  ] : undefined
  const crossings = data?.crossings?.points.map(localPoint) ?? []
  const gaps = data?.crossings?.gaps.map(localPoint) ?? []
  const maskBounds = route && gaps.length ? {
    x: Math.min(...route.map(point => point.x)) - 16, y: Math.min(...route.map(point => point.y)) - 16,
    width: Math.max(...route.map(point => point.x)) - Math.min(...route.map(point => point.x)) + 32,
    height: Math.max(...route.map(point => point.y)) - Math.min(...route.map(point => point.y)) + 32,
  } : undefined
  // A continuous scene still needs routed lanes, but its bends should read
  // like the broad curves of builder wires rather than right-angle tracks.
  const edgePath = route ? roundedWirePath(route, origin ? 36 : 9, crossings) : data?.residual && props.targetX - props.sourceX > 170
    ? `M${props.sourceX},${props.sourceY} C${props.sourceX + 40},${props.sourceY} ${props.sourceX + 30},${top} ${props.sourceX + 60},${top} L${props.targetX - 60},${top} C${props.targetX - 30},${top} ${props.targetX - 40},${props.targetY} ${props.targetX},${props.targetY}`
    : defaultPath
  const value = isBackward ? data?.gradient : data?.forward
  const description = `${data?.label ?? 'Connection'} · ${isBackward ? 'backward gradient' : 'forward value'}: ${value === undefined ? 'not reached in this trace' : formatFullTensor(value)}`
  const className = `builder-edge-flow ${isBackward ? 'is-backward' : 'is-forward'} ${value === undefined ? 'is-unreached' : 'has-signal'} ${data?.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''}`
  const intensity = edgeSignalIntensity(value)
  const inspect = () => data?.onInspect?.(props.id)

  return (
    <g
      className={className}
      transform={origin ? `translate(${origin.x} ${origin.y}) scale(${geometryScale})` : undefined}
      style={{ '--edge-intensity': intensity, '--edge-width': (1.5 + intensity * 0.85) * scale, '--edge-scale': scale, pointerEvents: data?.accessible === false ? 'none' : undefined } as CSSProperties}
      role="button"
      tabIndex={data?.accessible === false ? -1 : 0}
      aria-hidden={data?.accessible === false}
      aria-label={description}
      onClick={event => { event.stopPropagation(); inspect() }}
      onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); inspect() } }}
    >
      <title>{description}</title>
      {maskBounds && <defs><mask id={maskId} maskUnits="userSpaceOnUse" {...maskBounds}>
        <rect {...maskBounds} fill="white" />
        {gaps.map((point, index) => <circle key={index} cx={point.x} cy={point.y} r={4 * scale} fill="black" />)}
      </mask></defs>}
      <g mask={maskBounds ? `url(#${maskId})` : undefined}>
        <BaseEdge
          path={edgePath}
          markerEnd={props.markerEnd}
          interactionWidth={data?.cameraZoom ? Math.max(24 * scale, 14 / (geometryScale * data.cameraZoom)) : 24 * scale}
          className={`builder-edge ${data?.residual ? 'is-residual' : ''} ${data?.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''} ${isBackward ? 'is-backward' : ''}`}
        />
        {data?.active && value !== undefined && <path
          d={edgePath}
          className="builder-edge-bands"
          data-direction={isBackward ? 'backward' : 'forward'}
          aria-hidden="true"
        />}
      </g>
    </g>
  )
}
