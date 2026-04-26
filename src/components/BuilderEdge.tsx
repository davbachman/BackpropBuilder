import {
  BaseEdge,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react'
import type { ReactElement } from 'react'
import type { TensorValue } from '../domain/types'

export interface BuilderEdgeData extends Record<string, unknown> {
  forward?: TensorValue
  gradient?: TensorValue
  showGradient: boolean
  active: boolean
  phase: string
}

export function BuilderEdge(props: EdgeProps): ReactElement {
  const [edgePath] = getBezierPath(props)
  const data = props.data as BuilderEdgeData | undefined
  const isBackward = data?.phase === 'backward'

  return (
    <BaseEdge
      path={edgePath}
      markerEnd={props.markerEnd}
      className={`builder-edge ${data?.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''} ${isBackward ? 'is-backward' : ''}`}
    />
  )
}
