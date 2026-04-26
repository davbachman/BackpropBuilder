import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Boxes, Ungroup } from 'lucide-react'
import type { ReactElement } from 'react'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import type { GraphGroup, TensorValue } from '../domain/types'

export interface GroupOutputMetric {
  forward?: TensorValue
  gradient?: TensorValue
}

export interface GroupNodeData extends Record<string, unknown> {
  group: GraphGroup
  inputCount: number
  outputCount: number
  outputMetrics: GroupOutputMetric[]
  showGradient: boolean
  active: boolean
  onExplode: (groupId: string) => void
}

export function GroupNode(props: NodeProps): ReactElement {
  const data = props.data as GroupNodeData
  const group = data.group

  return (
    <div
      className={`visual-group-node ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''}`}
      style={{
        width: group.dimensions.width,
        height: group.dimensions.height,
      }}
    >
      {Array.from({ length: data.inputCount }).map((_, index) => (
        <Handle
          key={`in-${index}`}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          className="node-handle group-handle"
          style={{ top: handleTop(index, data.inputCount) }}
        />
      ))}
      <div className="visual-group-title-row">
        <span className="node-icon" aria-hidden="true">
          <Boxes size={14} />
        </span>
        <div>
          <strong>{group.label}</strong>
          <span>{data.inputCount} in / {data.outputCount} out</span>
        </div>
        <button
          type="button"
          className="group-explode-button nodrag nopan"
          aria-label={`Explode merged node ${group.label}`}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation()
            data.onExplode(group.id)
          }}
        >
          <Ungroup size={15} />
        </button>
      </div>
      <div className="node-metrics">
        <GroupMetrics outputs={data.outputMetrics} showGradient={data.showGradient} />
      </div>
      {Array.from({ length: data.outputCount }).map((_, index) => (
        <Handle
          key={`out-${index}`}
          id={`out-${index}`}
          type="source"
          position={Position.Right}
          className="node-handle source-handle group-handle"
          style={{ top: handleTop(index, data.outputCount) }}
        />
      ))}
    </div>
  )
}

function GroupMetrics({ outputs, showGradient }: { outputs: GroupOutputMetric[]; showGradient: boolean }): ReactElement {
  if (outputs.length === 0) {
    return (
      <>
        <TensorMetric label="out" value={undefined} />
        {showGradient ? <TensorMetric label="grad" value={undefined} /> : null}
      </>
    )
  }

  return (
    <>
      {outputs.map((output, index) => (
        <TensorMetric
          key={`out-${index}`}
          label={outputs.length === 1 ? 'out' : `out ${index + 1}`}
          value={output.forward}
        />
      ))}
      {showGradient
        ? outputs.map((output, index) => (
            <TensorMetric
              key={`grad-${index}`}
              label={outputs.length === 1 ? 'grad' : `grad ${index + 1}`}
              value={output.gradient}
            />
          ))
        : null}
    </>
  )
}

function TensorMetric({ label, value }: { label: string; value: TensorValue | undefined }): ReactElement {
  return (
    <HoverText
      text={`${label} ${formatCompactTensor(value)}`}
      tooltip={`${label} ${formatFullTensor(value)}`}
    />
  )
}

function HoverText({ text, tooltip }: { text: string; tooltip: string }): ReactElement {
  const hasTooltip = text !== tooltip
  return (
    <span
      className={hasTooltip ? 'tensor-hover-text' : undefined}
      data-tooltip={hasTooltip ? tooltip : undefined}
      tabIndex={hasTooltip ? 0 : undefined}
    >
      {text}
    </span>
  )
}

function handleTop(index: number, count: number): string {
  return `${((index + 1) / (count + 1)) * 100}%`
}
