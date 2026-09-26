import { Handle, Position, type NodeProps } from '@xyflow/react'
import { ChevronDown, ChevronUp, ArrowUpRight } from 'lucide-react'
import type { ReactElement } from 'react'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import type { GraphGroup, TensorValue } from '../domain/types'

export interface GroupOutputMetric {
  forward?: TensorValue
  gradient?: TensorValue
}

export interface GroupNodeData extends Record<string, unknown> {
  group: GraphGroup
  codeLine?: string
  inputCount: number
  outputCount: number
  outputMetrics: GroupOutputMetric[]
  showGradient: boolean
  active: boolean
  validationError?: boolean
  expanded: boolean
  onToggle: (groupId: string) => void
  semantic?: boolean
  unitCount?: number
  unitValues?: number[]
  attentionWeights?: TensorValue
  onInspectNeuron?: (groupId: string, unitIndex: number) => void
  continuous?: boolean
  sceneScale?: number
  reveal?: number
  accessible?: boolean
  cameraZoom?: number
  frameHeaderVisible?: number
}

export function GroupNode(props: NodeProps): ReactElement {
  const data = props.data as GroupNodeData
  const group = data.group
  const kind = group.kind ?? 'module'

  if (data.continuous) {
    const scale = data.sceneScale ?? 1, reveal = data.reveal ?? 0
    const dimensions = { width: (group.dimensions.width ?? 238) / scale, height: (group.dimensions.height ?? 172) / scale }
    const nestedData = { ...data, continuous: false, group: { ...group, dimensions } }
    const headingScale = Math.min(1, 1 / (scale * (data.cameraZoom ?? 1)))
    const headingAvailable = Boolean(data.accessible && (data.frameHeaderVisible ?? 0) > .1)
    const coverAvailable = Boolean(data.accessible && reveal < .95)
    return <div className="continuous-group-surface" style={{ ...dimensions, transform: `scale(${scale})` }}>
      <div className={`continuous-group-outline ${props.selected ? 'is-selected' : ''}`} style={{ borderWidth: Math.min(props.selected ? 3 : 1.5, (props.selected ? 2.5 : 1) / Math.max(0.01, scale * (data.cameraZoom ?? 1) || 1)), opacity: reveal }} />
      <div className="continuous-group-heading" style={{ opacity: data.frameHeaderVisible, transform: `scale(${headingScale})`, transformOrigin: 'top left', pointerEvents: 'none' }} aria-hidden={!headingAvailable} inert={!headingAvailable}>
        <GroupNode {...props} data={{ ...nestedData, expanded: true, group: { ...group, dimensions: { width: dimensions.width / headingScale, height: dimensions.height / headingScale } } }} />
      </div>
      <div className="continuous-card-cover" style={{ opacity: 1 - reveal, pointerEvents: coverAvailable ? 'auto' : 'none' }} aria-hidden={!coverAvailable} inert={!coverAvailable}>
        <GroupNode {...props} data={{ ...nestedData, expanded: false }} />
      </div>
    </div>
  }

  return (
    <div
      className={`visual-group-node ${data.semantic ? 'semantic-group' : ''} group-kind-${kind} ${data.expanded ? 'is-expanded' : ''} ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''} ${data.validationError && !data.expanded ? 'has-error' : ''}`}
      aria-invalid={data.validationError && !data.expanded || undefined}
      style={{
        width: group.dimensions.width,
        height: group.dimensions.height,
      }}
    >
      {Array.from({ length: data.expanded ? 0 : data.inputCount }).map((_, index) => (
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
        <div>
          <strong>{group.label}</strong>
          <span>{data.semantic ? kind.replaceAll('-', ' ') : `${group.nodeIds.length} calculations · ${data.inputCount} in / ${data.outputCount} out`}</span>
        </div>
        <button
          type="button"
          className="group-explode-button nodrag nopan"
          aria-label={`${data.expanded ? 'Zoom out of' : 'Zoom into'} ${group.label}`}
          aria-expanded={data.expanded}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation()
            data.onToggle(group.id)
          }}
        >
          {data.expanded ? <ChevronUp size={15} /> : data.semantic ? <ArrowUpRight size={15} /> : <ChevronDown size={15} />}
        </button>
      </div>
      {data.semantic && (!data.expanded || kind === 'layer' && data.onInspectNeuron && data.unitCount) ? <GroupPreview data={data} /> : null}
      {!data.expanded && data.codeLine ? <code className="group-code-line" title={data.codeLine}>{data.codeLine}</code> : null}
      <div className="node-metrics" hidden={data.expanded}>
        <GroupMetrics outputs={data.outputMetrics} showGradient={data.showGradient} />
      </div>
      {Array.from({ length: data.expanded ? 0 : data.outputCount }).map((_, index) => (
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

function GroupPreview({ data }: { data: GroupNodeData }): ReactElement | null {
  const kind = data.group.kind
  if (kind === 'neuron') return <div className="neuron-glyph" aria-hidden="true"><svg viewBox="0 0 112 64"><path d="M0 13 L40 32 M0 32 H40 M0 51 L40 32 M73 32 H112"/><circle cx="56" cy="32" r="21"/><text x="56" y="38">σ</text></svg></div>
  if (kind === 'layer' && data.onInspectNeuron && data.unitCount) return <div className="layer-neuron-preview"><span>Individual neurons <span>↗ inspect</span></span><div className="unit-grid" style={{ gridTemplateColumns: `repeat(${Math.min(16, data.unitCount)}, 1fr)` }}>{Array.from({ length: data.unitCount }, (_, i) => <button type="button" key={i} className="unit-dot nodrag nopan" style={{ opacity: 0.4 + Math.min(0.6, Math.abs(data.unitValues?.[i] ?? 1) * 0.25) }} title={`Neuron ${i + 1} · ${data.unitValues?.[i]?.toFixed(4) ?? 'run forward to see activation'}`} aria-label={`Inspect ${data.group.label} neuron ${i + 1}`} onClick={(event) => { event.stopPropagation(); data.onInspectNeuron?.(data.group.id, i) }} />)}</div></div>
  if (kind === 'attention' || kind === 'head') {
    const columns = data.attentionWeights?.shape.at(-1) ?? 3
    const rows = Math.min(6, data.attentionWeights?.shape.at(-2) ?? 3)
    const visibleColumns = Math.min(6, columns)
    const cell = 44 / Math.max(rows, visibleColumns)
    return <svg className="architecture-preview attention-preview" viewBox="0 0 190 62" aria-label="Query and key similarity weights mix value vectors"><path d="M7 31 H28 M44 10 H73 L103 31 M44 31 H103 M44 52 H73 L103 31 M103 31 H120 M166 31 H184"/><text x="31" y="14">Q</text><text x="31" y="35">K</text><text x="31" y="56">V</text>{Array.from({ length: rows * visibleColumns }, (_, i) => { const row = Math.floor(i / visibleColumns), col = i % visibleColumns; const value = data.attentionWeights?.data[row * columns + col]; return <rect key={i} x={121 + col * cell} y={9 + row * cell} width={cell - 2} height={cell - 2} rx="1.5" opacity={value === undefined ? .16 : .08 + Math.max(0, Math.min(1, value)) * .92}><title>{value === undefined ? 'Run forward to see attention' : `Token ${row + 1} → ${col + 1}: ${value.toFixed(4)}`}</title></rect> })}</svg>
  }
  if (kind === 'transformer-block') return <svg className="architecture-preview" viewBox="0 0 190 62" aria-hidden="true"><path d="M5 31 H29 M77 31 H104 M153 31 H185 M14 31 V4 H91 V31 M92 31 V58 H168 V31"/><rect x="28" y="18" width="49" height="27" rx="6"/><rect x="104" y="18" width="49" height="27" rx="6"/><text x="52" y="35" textAnchor="middle">Attn</text><text x="128" y="35" textAnchor="middle">MLP</text><circle cx="91" cy="31" r="4"/><circle cx="168" cy="31" r="4"/></svg>
  if (kind === 'normalization') return <svg className="architecture-preview" viewBox="0 0 190 62" aria-hidden="true"><path d="M6 31 H42 M148 31 H184"/><rect x="42" y="9" width="106" height="44" rx="8"/><text x="95" y="28" textAnchor="middle">center · normalize</text><text x="95" y="43" textAnchor="middle">× γ + β</text></svg>
  if (kind === 'convolution' || kind === 'cnn') return <svg className="architecture-preview" viewBox="0 0 190 62" aria-hidden="true"><path d="M62 29 H96 M144 29 H182"/>{Array.from({length:9},(_,i)=><rect key={`k-${i}`} x={20+i%3*12} y={11+Math.floor(i/3)*12} width="9" height="9" rx="1.5"/>)}{Array.from({length:16},(_,i)=><rect key={`f-${i}`} x={98+i%4*10} y={9+Math.floor(i/4)*10} width="7" height="7" rx="1"/>)}<text x="37" y="59" textAnchor="middle">3 × 3 filter</text><text x="117" y="59" textAnchor="middle">feature maps</text></svg>
  if (kind === 'embedding' || kind === 'projection') return <svg className="architecture-preview" viewBox="0 0 190 62" aria-hidden="true"><path d="M7 31 H34 M94 31 H128 M158 31 H184"/>{Array.from({ length: 24 }, (_, i) => <rect key={i} x={36 + i % 6 * 9} y={12 + Math.floor(i / 6) * 10} width="6" height="7" rx="1"/>)}{Array.from({ length: 4 }, (_, i) => <rect key={i} x="131" y={12 + i * 10} width="23" height="7" rx="1"/>)}</svg>
  return null
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
