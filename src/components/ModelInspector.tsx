import { useState } from 'react'
import { initializeTensor, operationHelp, type Initializer } from '../domain/authoring'
import { ConvolutionInspector } from './ConvolutionInspector'
import { TensorHeatmap } from './TensorHeatmap'
import { lossKindForNode, lossOptionsForNode, TENSOR_TRANSFORM_OPTIONS } from '../domain/engine'
import { formatFullTensor, toTensor } from '../domain/tensor'
import type { CoordinateBinding } from '../domain/neuronProjection'
import type {
  GraphGroup,
  GraphModel,
  GraphNode,
  NodeParams,
  TensorValue,
  TensorTransformKind,
} from '../domain/types'

interface Props {
  graph: GraphModel
  node?: GraphNode
  binding?: CoordinateBinding
  group?: GraphGroup
  onGroupChange?: (id: string, changes: Pick<GraphGroup, 'label' | 'kind'>) => void
  onRename?: (id: string, label: string) => void
  onParams: (id: string, params: NodeParams) => void
  onValue: (id: string, value: TensorValue) => void
  onOpen: (id: string) => void
  onInspectNeuron: (id: string, unit: number, row?: number) => void
  onGroup: () => void
  selectionCount: number
}

export function ModelInspector({
  graph,
  node,
  binding,
  group,
  onParams,
  onGroupChange,
  onRename,
  onValue,
  onOpen,
  onInspectNeuron,
  onGroup,
  selectionCount,
}: Props) {
  const canonical = binding
    ? graph.nodes.find((candidate) => candidate.id === binding.nodeId)
    : node
  const weight = group?.detail?.weightNodeId
    ? graph.nodes.find(
        (candidate) => candidate.id === group.detail?.weightNodeId,
      )
    : undefined
  const width = weight ? toTensor(weight.params.value).shape[1] : 0
  const children =
    graph.groups?.filter((candidate) => candidate.parentId === group?.id) ?? []
  return (
    <section className="model-inspector">
      <p className="eyebrow">
        {group
          ? (group.kind ?? 'Building block')
          : node
            ? 'Selected block'
            : 'Selection'}
      </p>
      <h2>{group?.label ?? node?.label ?? `${selectionCount} blocks selected`}</h2>
      {group && (
        <>
          {onGroupChange && <details className="inspector-disclosure"><summary>Group settings</summary><form key={group.id} onSubmit={event => {
            event.preventDefault()
            const form = new FormData(event.currentTarget)
            onGroupChange(group.id, {label:String(form.get('name')).trim() || group.label,kind:String(form.get('kind'))})
          }}><label className="inspector-field">Block name<input name="name" aria-label="Block name" defaultValue={group.label}/></label><label className="inspector-field">Block kind<select name="kind" aria-label="Block kind" defaultValue={group.kind ?? 'module'}>{['module','neuron','layer','mlp','head','attention','transformer-block','cnn','convolution','network','embedding','normalization','projection'].map(kind=><option key={kind}>{kind}</option>)}</select></label><button className="inspector-wide">Apply block details</button></form></details>}
          <button className="inspector-wide" onClick={() => onOpen(group.id)}>
            Zoom into {group.kind ?? 'block'} ↗
          </button>
          {children.length > 0 && (
            <div className="inspector-children">
              {children.map((child) => (
                <button key={child.id} onClick={() => onOpen(child.id)}>
                  {child.label}
                  <span>↗</span>
                </button>
              ))}
            </div>
          )}
          {width > 0 && (
            <label className="inspector-field">
              Inspect a neuron
              <select
                value={
                  graph.view?.inspectedNeuron?.groupId === group.id
                    ? graph.view.inspectedNeuron.unitIndex
                    : ''
                }
                onChange={(event) =>
                  onInspectNeuron(group.id, Number(event.target.value))
                }
              >
                <option value="" disabled>
                  Choose a neuron
                </option>
                {Array.from({ length: width }, (_, i) => (
                  <option value={i} key={i}>
                    Neuron {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
        </>
      )}
      {graph.view?.inspectedNeuron && (
        <label className="inspector-field">
          Token row
          <input
            aria-label="Inspected token row"
            type="number"
            min="1"
            max={
              toTensor(
                graph.nodes.find(candidate => candidate.id === graph.groups?.find(group => group.id === graph.view?.inspectedNeuron?.groupId)?.detail?.inputNodeId)?.value,
              ).shape[0] ?? 1
            }
            value={graph.view.inspectedNeuron.row + 1}
            onChange={(event) => {
              const focus = graph.view!.inspectedNeuron!
              onInspectNeuron(
                focus.groupId,
                focus.unitIndex,
                Math.max(0, Number(event.target.value) - 1),
              )
            }}
          />
        </label>
      )}
      {node && (
        <>
          {operationHelp[node.type] && <p className="coordinate-note">{operationHelp[node.type]}</p>}
          {node.type === 'conv2d' && <ConvolutionInspector graph={graph} node={node} onValue={onValue}/>}
          {node.type === 'loss' && <label className="inspector-field">Loss<select aria-label="Loss function" value={lossKindForNode(node, graph)} onChange={event => onParams(node.id,{loss:event.target.value as NodeParams['loss']})}>{lossOptionsForNode(node, graph).map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}</select></label>}
          {node.type === 'tensor-transform' && <label className="inspector-field">Transform<select aria-label="Tensor transform operation" value={node.params.transform ?? 'reshape'} onChange={event => onParams(node.id, { transform: event.target.value as TensorTransformKind })}>{TENSOR_TRANSFORM_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}</select></label>}
          {binding && canonical && (
            <p className="coordinate-note">
              Coordinate {binding.index + 1} of {canonical.label}. This is the
              model’s shared parameter
              {canonical.type === 'input' ? ' or input' : ''}; its gradient
              includes all uses.
            </p>
          )}
          {canonical && binding?.editable && (
            <label className="inspector-field">
              Edit coordinate
              <input
                aria-label="Edit tensor coordinate"
                type="number"
                step="0.05"
                value={toTensor(canonical.params.value).data[binding.index]}
                onChange={(event) => {
                  const value = toTensor(canonical.params.value)
                  onValue(canonical.id, {
                    ...value,
                    data: value.data.map((item, i) =>
                      i === binding.index ? Number(event.target.value) : item,
                    ),
                  })
                }}
              />
            </label>
          )}
          {canonical &&
            !binding &&
            !graph.edges.some(edge => edge.target === canonical.id) &&
            ['input', 'target', 'weight', 'bias'].includes(canonical.type) && (
              <TensorEditor
                key={canonical.id}
                node={canonical}
                onValue={onValue}
              />
            )}
          {node.type === 'activation' &&
            !binding &&
            !node.id.startsWith('inspect:') && (
              <label className="inspector-field">
                Activation
                <select
                  value={node.params.activation ?? 'identity'}
                  onChange={(event) =>
                    onParams(node.id, {
                      activation: event.target.value as NodeParams['activation'],
                    })
                  }
                >
                  {['identity', 'relu', 'sigmoid', 'tanh'].map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
            )}
          {[
            'slice',
            'transpose',
            'concat',
            'reshape',
            'mean',
            'layer-norm',
            'add',
            'tensor-transform',
          ].includes(node.type) &&
            !node.id.startsWith('inspect:') && (
              <OperationEditor
                key={`${node.id}:${JSON.stringify(node.params)}`}
                node={node}
                onParams={onParams}
              />
            )}
          {node.value && node.value.data.length > 1 && (
            <TensorHeatmap
              key={`${node.id}:${node.value.shape.join(',')}`}
              value={node.value}
              onEdit={
                canonical &&
                !binding &&
                !graph.edges.some(edge => edge.target === canonical.id) &&
                ['weight', 'bias', 'input', 'target'].includes(canonical.type)
                  ? (index, next) => {
                      const value = toTensor(canonical.params.value)
                      onValue(canonical.id, {
                        ...value,
                        data: value.data.map((item, i) =>
                          i === index ? next : item,
                        ),
                      })
                    }
                  : undefined
              }
            />
          )}
          {(node.value || node.grad) && <details className="inspector-disclosure"><summary>Exact values and gradients</summary><div className="inspector-values">
            <span>
              Value<strong>{formatFullTensor(node.value)}</strong>
            </span>
            <span>
              Gradient<strong>{formatFullTensor(node.grad)}</strong>
            </span>
          </div></details>}
          {onRename && !binding && <details className="inspector-disclosure"><summary>Rename block</summary><label className="inspector-field">Name<input key={node.id} aria-label="Node name" defaultValue={node.label} onBlur={event => { if (event.target.value.trim() && event.target.value !== node.label) onRename(node.id,event.target.value.trim()) }}/></label></details>}
        </>
      )}
      {selectionCount > 1 && (
        <div className="inspector-actions">
          <button onClick={onGroup}>Group selection</button>
        </div>
      )}
    </section>
  )
}

function TensorEditor({
  node,
  onValue,
}: {
  node: GraphNode
  onValue: Props['onValue']
}) {
  const value = toTensor(node.params.value)
  const [shape, setShape] = useState(value.shape.join(', ')),
    [data, setData] = useState(
      value.data.map((item) => String(item)).join(', '),
    ),
    [error, setError] = useState('')
  const [initializer, setInitializer] = useState<Initializer>(node.type === 'bias' ? 'zeros' : 'xavier')
  const [seed, setSeed] = useState(() => [...node.id].reduce((seed,letter) => (seed*31+letter.charCodeAt(0)) >>> 0,42))
  const signature = JSON.stringify(value)
  const [source, setSource] = useState(signature)
  if (source !== signature) {
    setSource(signature)
    setShape(value.shape.join(', '))
    setData(value.data.map(String).join(', '))
  }
  const initializeShape = () => {
    try {
      onValue(node.id, initializeTensor(parseTensorShape(shape), initializer, seed))
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Invalid shape.')
    }
  }
  return (
    <form
      className="tensor-editor"
      onSubmit={(event) => {
        event.preventDefault()
        let dimensions: number[]
        try {
          dimensions = parseTensorShape(shape)
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : 'Invalid shape.')
          return
        }
        const numbers = data
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(Number)
        if (
          dimensions.some((size) => !Number.isInteger(size) || size <= 0) ||
          dimensions.reduce((a, b) => a * b, 1) !== numbers.length ||
          numbers.some((item) => !Number.isFinite(item))
        ) {
          setError(
            'Use positive dimensions matching the number of finite values.',
          )
          return
        }
        onValue(node.id, { shape: dimensions, data: numbers })
        setError('')
      }}
    >
      <label className="inspector-field">
        Shape <span>e.g. 3, 1 or (3, 1); empty = scalar</span>
        <input
          aria-label="Tensor shape"
          value={shape}
          onChange={(event) => setShape(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); initializeShape() } }}
        />
      </label>
      <p className="coordinate-note">Enter a shape, then initialize its values or enter all values below and apply them.</p>
      <div className="tensor-initializer"><label className="inspector-field">Initialize<select aria-label="Tensor initializer" value={initializer} onChange={event => setInitializer(event.target.value as Initializer)}>{(['xavier','he','zeros','ones','uniform'] as const).map(kind => <option key={kind} value={kind}>{kind === 'xavier' ? 'Xavier · matrices' : kind === 'he' ? 'He · ReLU / filters' : kind}</option>)}</select></label><button type="button" onClick={initializeShape}>Initialize tensor</button></div>
      <label className="inspector-field">Random seed<input aria-label="Initializer seed" type="number" value={seed} onChange={event => setSeed(Number(event.target.value))}/></label>
      <label className="inspector-field">
        Values
        <textarea
          aria-label="Tensor values"
          rows={Math.min(6, Math.max(2, Math.ceil(value.data.length / 6)))}
          value={data}
          onChange={(event) => setData(event.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" className="inspector-wide">
        Apply values
      </button>
    </form>
  )
}

function parseTensorShape(text: string): number[] {
  const trimmed = text.trim()
  const unwrapped = trimmed.startsWith('(') && trimmed.endsWith(')') || trimmed.startsWith('[') && trimmed.endsWith(']')
    ? trimmed.slice(1, -1).trim()
    : trimmed
  if (/[()[\]]/.test(unwrapped)) throw new Error('Use dimensions such as 3, 1 or (3, 1).')
  const dimensions = unwrapped ? unwrapped.split(/[\s,×x]+/).filter(Boolean).map(Number) : []
  if (dimensions.some(size => !Number.isInteger(size) || size <= 0)) throw new Error('Use positive dimensions (whole numbers) such as 3, 1.')
  return dimensions
}

function OperationEditor({
  node,
  onParams,
}: {
  node: GraphNode
  onParams: Props['onParams']
}) {
  const operation = node.type === 'tensor-transform' ? node.params.transform ?? 'reshape' : node.type
  const fields =
    operation === 'slice'
      ? ['axis', 'start', 'end']
      : operation === 'transpose'
        ? ['axes']
        : operation === 'reshape'
          ? ['shape']
          : operation === 'layer-norm'
            ? ['epsilon']
            : operation === 'mean'
              ? ['axis']
              : operation === 'concat'
                ? ['axis', 'inputCount']
                : ['inputCount']
  const [draft, setDraft] = useState<Record<string, string>>(
    Object.fromEntries(
      fields.map((field) => [
        field,
        Array.isArray(node.params[field as keyof NodeParams])
          ? (node.params[field as keyof NodeParams] as number[]).join(', ')
          : String(
              node.params[field as keyof NodeParams] ??
                (field === 'inputCount'
                  ? 2
                  : field === 'epsilon'
                    ? 0.00001
                    : ['axis', 'start'].includes(field) && operation !== 'mean' ? (field === 'axis' && operation === 'concat' ? 1 : 0) : ''),
            ),
      ]),
    ),
  )
  const [error, setError] = useState('')
  const [keepDims, setKeepDims] = useState(node.params.keepDims ?? false)
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        const params = Object.fromEntries(
          fields.map((field) => [
            field,
            draft[field].trim() === '' && (field !== 'shape' || node.type === 'tensor-transform') ? undefined : ['axes', 'shape'].includes(field)
              ? draft[field]
                  .split(/[\s,]+/)
                  .filter(Boolean)
                  .map(Number)
              : Number(draft[field]),
          ]),
        )
        const numbers = Object.values(params).flat().filter(value => value !== undefined) as number[]
        if (numbers.some(value => !Number.isFinite(value) || (value < 0 && !(operation === 'reshape' && value === -1))) || (params.epsilon !== undefined && Number(params.epsilon) <= 0) || (params.inputCount !== undefined && (!Number.isInteger(params.inputCount) || Number(params.inputCount) < 2 || Number(params.inputCount) > 16)) || fields.some(field => field !== 'epsilon' && params[field] !== undefined && (Array.isArray(params[field]) ? params[field] as number[] : [params[field] as number]).some(n => !Number.isInteger(n))) || (Array.isArray(params.shape) && (params.shape.some(n => n < 1 && n !== -1) || params.shape.filter(n => n === -1).length > 1))) {
          setError('Use whole nonnegative indices, positive dimensions, 2–16 inputs, and a positive epsilon.'); return
        }
        onParams(node.id, {...params, ...(operation === 'mean' ? {keepDims} : {})})
        setError('')
      }}
    >
      <div className="operation-fields">
        {fields.map((field) => (
          <label className="inspector-field" key={field}>
            {field}
            <input
              aria-label={`Operation ${field}`}
              value={draft[field]}
              onChange={(event) =>
                setDraft({ ...draft, [field]: event.target.value })
              }
            />
          </label>
        ))}
      </div>
      {operation === 'mean' && <label className="inspector-field"><span><input type="checkbox" checked={keepDims} onChange={event => setKeepDims(event.target.checked)}/> Keep dimensions</span></label>}
      {error && <p role="alert">{error}</p>}
      <button className="inspector-wide">Apply operation</button>
    </form>
  )
}
