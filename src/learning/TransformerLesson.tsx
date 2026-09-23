import { useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ArrowDown,
  ChevronDown,
  ChevronRight,
  Download,
  Expand,
  Play,
  RotateCcw,
  Save,
  Undo2,
  Upload,
} from 'lucide-react'
import { causalMask, matmul, scale, softmax, tensor, transpose } from './math'
import {
  DECODER_CONFIG,
  decoderForward,
  getCheckpoint,
  loadCheckpoint,
  sampleToken,
  samplingDistribution,
  type DecoderTrace,
  type DecoderModel,
  type CheckpointMetadata,
} from './decoder'
import {
  downloadExploration,
  initialExploration,
  parseExploration,
  type ExplorationFile,
  type ExplorationState,
} from './exploration'
import { LESSONS, type LessonKind } from './presets'
import { MatrixView } from './MatrixView'
import { numberLabel as n } from './format'
import type { TensorValue } from '../domain/types'

type TokenLesson = Exclude<
  LessonKind,
  'linear' | 'neuron' | 'small-network' | 'playground'
>
interface AttentionTrace {
  q: TensorValue
  k: TensorValue
  v: TensorValue
  scores: TensorValue
  maskedScores: TensorValue
  weights: TensorValue
  output: TensorValue
}
interface Snapshot {
  decoder?: DecoderTrace
  attention?: AttentionTrace
  probability?: TensorValue
}
type Comparison = ExplorationFile['comparisons'][number]

function modelFor(state: Pick<ExplorationState, 'checkpoint' | 'overrides'>) {
  const model = loadCheckpoint(getCheckpoint(state.checkpoint))
  for (const [key, data] of Object.entries(state.overrides)) {
    const parameter = model.parameters[key]
    if (!parameter || data.length !== parameter.data.length)
      throw new Error(`Parameter ${key} does not match this architecture.`)
    parameter.data = [...data]
  }
  return model
}
function calculate(state: ExplorationState, model: DecoderModel): Snapshot {
  if (state.kind === 'probabilities')
    return { probability: softmax(tensor([1, 3], state.scores)).toValue() }
  if (state.kind === 'attention' || state.kind === 'causal') {
    const q = tensor([state.kind === 'causal' ? 3 : 1, 2], state.q),
      k = tensor([3, 2], state.k)
    const values = [...state.v]
    if (state.intervention) {
      values[2] = 0
      values[3] = 4
    }
    const v = tensor([3, 2], values),
      scores = scale(matmul(q, transpose(k)), 1 / Math.sqrt(2))
    const masked = state.kind === 'causal' ? causalMask(scores) : scores,
      weights = softmax(masked)
    return {
      attention: {
        q: q.toValue(),
        k: k.toValue(),
        v: v.toValue(),
        scores: scores.toValue(),
        maskedScores: masked.toValue(),
        weights: weights.toValue(),
        output: matmul(weights, v).toValue(),
      },
    }
  }
  return { decoder: decoderForward(model, state.ids) }
}
const enteredNumber = (text: string, min = -1000, max = 1000) => {
  const value = Number(text)
  return Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : 0
}
const rowValues = (value: TensorValue, row: number) =>
  value.data.slice(row * value.shape.at(-1)!, (row + 1) * value.shape.at(-1)!)
const vector = (data: number[]): TensorValue => ({
  shape: [1, data.length],
  data,
})
function outputOf(snapshot: Snapshot | null) {
  return (
    snapshot?.attention?.output.data ??
    snapshot?.probability?.data ??
    snapshot?.decoder?.logits.data.slice(-DECODER_CONFIG.vocab.length) ??
    []
  )
}

function modulePath(id: string): string[] {
  if (id === 'model') return ['model']
  const match = id.match(/^(block-\d)(.*)$/)
  if (!match) return ['model', id]
  const [, block, suffix] = match
  if (!suffix) return ['model', block]
  if (suffix.startsWith('-head-'))
    return ['model', block, `${block}-attention`, id]
  if (suffix.startsWith('-neuron-'))
    return ['model', block, `${block}-feedforward`, `${block}-hidden`, id]
  if (suffix === '-hidden' || suffix === '-output-layer')
    return ['model', block, `${block}-feedforward`, id]
  return ['model', block, id]
}
function moduleLabel(id: string) {
  const block = id.match(/^block-(\d)/)
  if (block && id === block[0]) return `Block ${Number(block[1]) + 1}`
  const head = id.match(/-head-(\d)$/)
  if (head) return `Head ${Number(head[1]) + 1}`
  return id
    .replace(/^block-\d-/, '')
    .replaceAll('-', ' ')
    .replace(/^\w/, (c) => c.toUpperCase())
}

export function TransformerLesson({ kind }: { kind: TokenLesson }) {
  const [state, setState] = useState(() => initialExploration(kind))
  const [snapshot, setSnapshot] = useState<Snapshot | null>(() => {
    const s = initialExploration(kind)
    return calculate(s, modelFor(s))
  })
  const [history, setHistory] = useState<ExplorationState[]>([])
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [message, setMessage] = useState('')
  const [cursor, setCursor] = useState(-1)
  const [generationPhase, setGenerationPhase] = useState<
    'scores' | 'distribution' | 'chosen' | 'appended'
  >('scores')
  const [choice, setChoice] = useState<number | null>(null)
  const [parameterKey, setParameterKey] = useState('blocks.0.q')
  const [parameterIndex, setParameterIndex] = useState(0)
  const upload = useRef<HTMLInputElement>(null)
  const pan = useRef<{
    x: number
    y: number
    left: number
    top: number
  } | null>(null)
  const model = useMemo(
    () =>
      modelFor({ checkpoint: state.checkpoint, overrides: state.overrides }),
    [state.checkpoint, state.overrides],
  )
  const lesson = LESSONS.find((item) => item.id === state.kind)!
  const isDecoder = ['embeddings', 'block', 'decoder'].includes(state.kind)
  const isAttention = state.kind === 'attention' || state.kind === 'causal'
  const trace = snapshot?.decoder
  const token = Math.min(state.token, state.ids.length - 1)
  const currentOutput = outputOf(snapshot)
  const phases =
    state.kind === 'probabilities'
      ? ['Read scores', 'Subtract largest', 'Exponentiate', 'Normalize']
      : isAttention
        ? ['Read Q / K / V', 'Dot products', 'Mask & normalize', 'Mix values']
        : ['Token & position lookup', 'Block 1', 'Block 2', 'Vocabulary scores']
  const patch = (update: Partial<ExplorationState>, invalidates = false) => {
    setHistory((old) => [...old.slice(-39), state])
    setState((old) => ({ ...old, ...update }))
    if (invalidates) {
      setSnapshot(null)
      setCursor(-1)
      setChoice(null)
      setGenerationPhase('scores')
    }
    setMessage('')
  }
  const run = () => {
    setSnapshot(calculate(state, model))
    setCursor(phases.length - 1)
    setGenerationPhase('scores')
    setChoice(null)
    setMessage('Forward calculation recorded. No parameters were updated.')
  }
  const toggle = (id: string) =>
    patch({
      opened: state.opened.includes(id)
        ? state.opened.filter((x) => x !== id)
        : [...state.opened, id],
      focus: state.opened.includes(id) ? 'model' : id,
    })
  const open = (id: string) => state.opened.includes(id)
  const path = modulePath(state.focus)
  const navigate = (id: string) => {
    const ancestors = modulePath(id).filter(
      (part) => part !== 'model' && !part.includes('-neuron-'),
    )
    patch({
      focus: id,
      opened: Array.from(new Set([...state.opened, ...ancestors])),
    })
    requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({ block: 'nearest' }),
    )
  }
  const stepInside = () => {
    const next =
      state.focus === 'model'
        ? cursor <= 0
          ? 'embeddings'
          : cursor === 1
            ? 'block-0'
            : cursor === 2
              ? 'block-1'
              : 'vocabulary'
        : /^block-\d$/.test(state.focus)
          ? `${state.focus}-attention`
          : state.focus.endsWith('-attention')
            ? state.focus.replace('-attention', '-head-0')
            : state.focus.endsWith('-feedforward')
              ? state.focus.replace('-feedforward', '-hidden')
              : state.focus
    navigate(next)
  }
  const step = () => {
    if (!snapshot) setSnapshot(calculate(state, model))
    if (!isDecoder) {
      setCursor((value) => Math.min(phases.length - 1, value + 1))
      return
    }
    const visible = ['embeddings']
    if (state.kind !== 'embeddings') {
      for (let index = 0; index < 2; index++) {
        const block = 'block-' + index
        visible.push(block)
        if (open(block)) {
          visible.push(block + '-norm1', block + '-attention')
          if (open(block + '-attention'))
            visible.push(block + '-head-0', block + '-head-1')
          visible.push(block + '-norm2', block + '-feedforward')
          if (open(block + '-feedforward'))
            visible.push(block + '-hidden', block + '-output-layer')
        }
      }
      visible.push('vocabulary')
    }
    const next =
      visible[Math.min(visible.length - 1, visible.indexOf(state.focus) + 1)]
    patch({ focus: next })
    setCursor(
      next === 'embeddings'
        ? 0
        : next.startsWith('block-0')
          ? 1
          : next.startsWith('block-1')
            ? 2
            : 3,
    )
    requestAnimationFrame(() =>
      document.getElementById(next)?.scrollIntoView({ block: 'nearest' }),
    )
  }
  const undo = () => {
    const previous = history.at(-1)
    if (!previous) return
    setState(previous)
    setHistory(history.slice(0, -1))
    setSnapshot(calculate(previous, modelFor(previous)))
    setChoice(null)
    setGenerationPhase('scores')
  }
  const restore = (next: ExplorationState) => {
    setHistory((old) => [...old.slice(-39), state])
    setState(next)
    setSnapshot(calculate(next, modelFor(next)))
    setChoice(null)
    setCursor(-1)
    setGenerationPhase('scores')
  }
  const load = async (file: File | undefined) => {
    if (!file) return
    try {
      const saved = parseExploration(await file.text())
      if (saved.state.kind !== kind) {
        const preset = LESSONS.find((item) => item.id === saved.state.kind)
        throw new Error(`Open the “${preset?.title}” preset before importing this experiment.`)
      }
      modelFor(saved.state)
      restore(saved.state)
      setComparisons(saved.comparisons)
      if (saved.execution) {
        setCursor(saved.execution.cursor)
        setGenerationPhase(saved.execution.generationPhase)
        setChoice(saved.execution.choice)
        if (!saved.execution.current) setSnapshot(null)
      }
      setMessage(
        'Saved inputs, parameters, seed, controls, and module view restored.',
      )
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : 'Could not load this file.',
      )
    }
  }
  const logits = trace ? rowValues(trace.logits, state.ids.length - 1) : []
  const distribution = logits.length
    ? samplingDistribution(logits, state.temperature, state.topK)
    : []
  const selectedParameter = model.parameters[parameterKey]
  const selectToken = (position: number) => patch({ token: position })
  const save = () =>
    downloadExploration({
      kind: 'backprop-exploration',
      version: 1,
      state,
      comparisons,
      execution: {
        cursor,
        generationPhase,
        choice,
        current: Boolean(snapshot),
      },
    })
  const moduleProps = (id: string) => ({
    id,
    expanded: open(id),
    onToggle: () => toggle(id),
    active: state.focus === id,
  })

  return (
    <div className="exploration">
      <section className="learning-prompt">
        <span className="prompt-marker">?</span>
        <div>
          <strong>{lesson.question}</strong>
          <p>{lesson.experiment}</p>
        </div>
        <details className="prediction-detail">
          <summary>Make a prediction</summary>
          <textarea
            aria-label="My prediction"
            placeholder="I think… because…"
            value={state.prediction}
            onChange={(event) => patch({ prediction: event.target.value })}
          />
          <p>
            Predict → run → explain what changed → try a new example. No
            completion gates.
          </p>
        </details>
      </section>
      <div className="experiment-toolbar">
        <div className="toolbar-group">
          <button className="primary-button" onClick={run}>
            <Play size={15} /> Run forward
          </button>
          <button onClick={step}>Next visible component</button>
          {isDecoder && <button onClick={stepInside}>Step inside</button>}
          <button
            onClick={() => {
              if (!snapshot) setSnapshot(calculate(state, model))
              setCursor(phases.length - 1)
            }}
          >
            Finish forward
          </button>
          <button
            onClick={undo}
            disabled={!history.length}
            aria-label="Undo exploration edit"
          >
            <Undo2 size={15} />
          </button>
        </div>
        <div className="toolbar-group">
          <button onClick={() => patch({ showControls: !state.showControls })}>
            {state.showControls ? 'Hide' : 'Show'} controls
          </button>
          <button onClick={save}>
            <Download size={15} /> Save state
          </button>
          <button onClick={() => upload.current?.click()}>
            <Upload size={15} /> Import
          </button>
          <input
            ref={upload}
            className="visually-hidden"
            type="file"
            accept="application/json,.json"
            aria-label="Import exploration"
            onChange={(event) => {
              void load(event.target.files?.[0])
              event.target.value = ''
            }}
          />
        </div>
      </div>
      <div className="phase-strip" aria-label="Forward execution stages">
        {phases.map((phase, index) => (
          <span
            key={phase}
            className={
              cursor === index
                ? 'current-phase'
                : cursor > index
                  ? 'completed-phase'
                  : ''
            }
          >
            <b>{index + 1}</b>
            {phase}
          </span>
        ))}
        <span className="phase-readonly">Inference · parameters fixed</span>
      </div>
      <div
        role="status"
        className={`experiment-status ${snapshot ? '' : 'invalidated'}`}
      >
        {!snapshot
          ? 'Inputs or parameters changed. Results are invalidated. Run forward to record current values.'
          : message ||
            'Recorded values are current. Open any module to inspect the same calculation.'}
      </div>
      {state.showControls && (
        <section className="experiment-controls">
          <div className="control-row">
            <label>
              <input
                type="checkbox"
                checked={state.showMath}
                onChange={(e) => patch({ showMath: e.target.checked })}
              />{' '}
              Formulas
            </label>
            <label>
              <input
                type="checkbox"
                checked={state.showCode}
                onChange={(e) => patch({ showCode: e.target.checked })}
              />{' '}
              Python reference
            </label>
            <label>
              Seed{' '}
              <input
                type="number"
                min="0"
                max="4294967295"
                value={state.seed}
                onChange={(e) =>
                  patch({
                    seed: Math.floor(
                      enteredNumber(e.target.value, 0, 4294967295),
                    ),
                  })
                }
              />
            </label>
            <button onClick={() => restore(initialExploration(state.kind))}>
              <RotateCcw size={14} /> Reset experiment
            </button>
            <button
              disabled={!snapshot}
              onClick={() => {
                setComparisons((old) => [
                  ...old,
                  {
                    label: `Comparison ${old.length + 1}`,
                    state: structuredClone(state),
                    output: [...currentOutput],
                  },
                ])
                setMessage(
                  'Comparison saved with its exact parameters and inputs.',
                )
              }}
            >
              <Save size={14} /> Keep comparison
            </button>
          </div>
          {isDecoder && (
            <div className="control-row">
              <label>
                Parameters{' '}
                <select
                  value={state.checkpoint}
                  onChange={(e) =>
                    patch(
                      {
                        checkpoint: e.target
                          .value as ExplorationState['checkpoint'],
                        overrides: {},
                      },
                      true,
                    )
                  }
                >
                  <option value="trained">Trained checkpoint</option>
                  <option value="untrained">Initial checkpoint · seed 7</option>
                </select>
              </label>
              <button onClick={() => patch({ overrides: {} }, true)}>
                Reset parameter edits
              </button>
              <span className="control-help">
                2 blocks · width 8 · 2 heads · 5 tokens · 12 positions
              </span>
            </div>
          )}
          {isDecoder && (
            <div className="token-editor">
              <span className="control-label">Prompt tokens</span>
              {state.ids.map((id, index) => (
                <label key={index}>
                  Position {index}
                  <select
                    aria-label={`Token at position ${index}`}
                    value={id}
                    onChange={(e) =>
                      patch(
                        {
                          ids: state.ids.map((v, i) =>
                            i === index ? Number(e.target.value) : v,
                          ),
                        },
                        true,
                      )
                    }
                  >
                    {DECODER_CONFIG.vocab.map((v, i) => (
                      <option key={v} value={i}>
                        {v} · ID {i}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
              <button
                disabled={state.ids.length >= 12}
                onClick={() => patch({ ids: [...state.ids, 1] }, true)}
              >
                + Token
              </button>
              <button
                disabled={state.ids.length <= 1}
                onClick={() =>
                  patch(
                    {
                      ids: state.ids.slice(0, -1),
                      token: Math.min(token, state.ids.length - 2),
                    },
                    true,
                  )
                }
              >
                − Token
              </button>
            </div>
          )}
          {state.kind === 'probabilities' && (
            <div className="control-row">
              {state.scores.map((value, index) => (
                <label key={index}>
                  Score {index}
                  <input
                    aria-label={`Score ${index}`}
                    type="number"
                    step="0.5"
                    value={value}
                    onChange={(e) =>
                      patch(
                        {
                          scores: state.scores.map((v, i) =>
                            index === i ? enteredNumber(e.target.value) : v,
                          ),
                        },
                        true,
                      )
                    }
                  />
                </label>
              ))}
              <button
                onClick={() =>
                  patch({ scores: state.scores.map((v) => v + 10) }, true)
                }
              >
                Add 10 to all scores
              </button>
            </div>
          )}
          {isAttention && (
            <>
              <div className="attention-inputs">
                {(['q', 'k', 'v'] as const).map((key) => (
                  <fieldset key={key}>
                    <legend>
                      {key.toUpperCase()} · input-dependent values
                    </legend>
                    {state[key].map((value, index) => (
                      <label key={index}>
                        {Math.floor(index / 2)},{index % 2}
                        <input
                          aria-label={`${key.toUpperCase()} row ${Math.floor(index / 2)} coordinate ${index % 2}`}
                          type="number"
                          step="0.25"
                          value={value}
                          disabled={
                            key === 'v' &&
                            (index === 2 || index === 3) &&
                            state.intervention
                          }
                          onChange={(e) =>
                            patch(
                              {
                                [key]: state[key].map((v, i) =>
                                  index === i ? Number(e.target.value) : v,
                                ),
                              },
                              true,
                            )
                          }
                        />
                      </label>
                    ))}
                  </fieldset>
                ))}
              </div>
              <div className="control-row">
                <button
                  onClick={() => patch({ intervention: true }, true)}
                  disabled={state.intervention}
                >
                  Intervene: middle V → [0, 4]
                </button>
                <button
                  onClick={() => patch({ intervention: false }, true)}
                  disabled={!state.intervention}
                >
                  Reset V intervention
                </button>
                <span className="control-help">
                  Q and K are held fixed by this intervention. Attention weights
                  depend on Q/K, not V.
                </span>
              </div>
            </>
          )}
        </section>
      )}
      <div className="workspace-navigation">
        <nav aria-label="Model breadcrumbs">
          {path.map((id, index) => (
            <span className="breadcrumb-item" key={id}>
              {index > 0 && <ChevronRight size={13} />}
              <button
                aria-current={
                  index === path.length - 1 ? 'location' : undefined
                }
                onClick={() => navigate(id)}
              >
                {id === 'model' ? lesson.subtitle : moduleLabel(id)}
              </button>
            </span>
          ))}
        </nav>
        <div>
          <button
            disabled={path.length === 1}
            onClick={() => navigate(path.at(-2) ?? 'model')}
          >
            ↑ Up a level
          </button>
          <button onClick={() => patch({ zoom: 1 })}>
            <Expand size={13} /> Fit view
          </button>
          <button
            aria-label="Zoom out"
            onClick={() => patch({ zoom: Math.max(0.5, state.zoom - 0.1) })}
          >
            −
          </button>
          <span>{Math.round(state.zoom * 100)}%</span>
          <button
            aria-label="Zoom in"
            onClick={() => patch({ zoom: Math.min(1.5, state.zoom + 0.1) })}
          >
            +
          </button>
        </div>
      </div>
      <div
        className="tensor-workspace"
        tabIndex={0}
        aria-label="Model visual workspace"
        onPointerDown={(event) => {
          if (
            event.button !== 0 ||
            event.pointerType === 'touch' ||
            (event.target as HTMLElement).closest(
              'button,input,select,textarea,summary,table,pre',
            )
          )
            return
          pan.current = {
            x: event.clientX,
            y: event.clientY,
            left: event.currentTarget.scrollLeft,
            top: event.currentTarget.scrollTop,
          }
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (!pan.current) return
          event.currentTarget.scrollLeft =
            pan.current.left + pan.current.x - event.clientX
          event.currentTarget.scrollTop =
            pan.current.top + pan.current.y - event.clientY
        }}
        onPointerUp={() => {
          pan.current = null
        }}
        onPointerCancel={() => {
          pan.current = null
        }}
      >
        <div className="tensor-workspace-content" style={{ zoom: state.zoom }}>
          {!snapshot ? (
            <div className="empty-execution">
              <RotateCcw size={28} />
              <h2>Ready for a new calculation</h2>
              <p>Your model structure and selection are preserved.</p>
              <button onClick={run}>Calculate current values</button>
            </div>
          ) : (
            <>
              {state.kind === 'probabilities' && snapshot.probability && (
                <ProbabilityDetail
                  scores={state.scores}
                  probabilities={snapshot.probability.data}
                  showMath={state.showMath}
                  showCode={state.showCode}
                />
              )}
              {snapshot.attention && (
                <AttentionDetail
                  selectedKey={state.selectedKeys.attention ?? 0}
                  selectKey={(key) =>
                    patch({
                      selectedKeys: { ...state.selectedKeys, attention: key },
                    })
                  }
                  trace={snapshot.attention}
                  row={state.kind === 'attention' ? 0 : Math.min(token, 2)}
                  coordinate={Math.min(state.coordinate, 1)}
                  select={(row, col) => patch({ token: row, coordinate: col })}
                  showMath={state.showMath}
                  showCode={state.showCode}
                />
              )}
              {trace && (
                <>
                  <div
                    className="token-flow"
                    aria-label="Select a token to follow"
                  >
                    <span>Follow a token</span>
                    {state.ids.map((id, index) => (
                      <button
                        key={index}
                        className={token === index ? 'selected' : ''}
                        aria-pressed={token === index}
                        onClick={() => selectToken(index)}
                      >
                        <span>{DECODER_CONFIG.vocab[id]}</span>
                        <small>
                          position {index} · ID {id}
                        </small>
                      </button>
                    ))}
                  </div>
                  <Module
                    {...moduleProps('embeddings')}
                    title="Token + position embeddings"
                    subtitle={`Lookup row ${state.ids[token]}, add position ${token}`}
                    summary={rowValues(trace.input, token)}
                  >
                    <div className="matrix-pair">
                      <MatrixView
                        label="Token lookup"
                        value={trace.embedding}
                        row={token}
                        onCell={(r, c) => patch({ token: r, coordinate: c })}
                      />
                      <span className="math-connector">+</span>
                      <MatrixView
                        label="Position lookup"
                        value={trace.position}
                        row={token}
                        onCell={(r, c) => patch({ token: r, coordinate: c })}
                      />
                    </div>
                    <MatrixView
                      label="Model input"
                      value={trace.input}
                      row={token}
                      onCell={(r, c) => patch({ token: r, coordinate: c })}
                    />
                    <div className="arithmetic-card">
                      <strong>Selected coordinate {state.coordinate}</strong>
                      <p>
                        {n(rowValues(trace.embedding, token)[state.coordinate])}{' '}
                        +{' '}
                        {n(rowValues(trace.position, token)[state.coordinate])}{' '}
                        ={' '}
                        <b>
                          {n(rowValues(trace.input, token)[state.coordinate])}
                        </b>
                      </p>
                      <p>
                        Token and position tables are learned parameters. The
                        sum is an intermediate value.
                      </p>
                    </div>
                    {state.showCode && (
                      <pre>
                        representation = token_table[token_id] +
                        position_table[position]
                      </pre>
                    )}
                  </Module>
                  {state.kind !== 'embeddings' &&
                    trace.blocks.map((block, index) => (
                      <div key={index}>
                        <FlowArrow />
                        <Module
                          {...moduleProps(`block-${index}`)}
                          title={`Transformer block ${index + 1}`}
                          subtitle="Normalize → attend → add residual → normalize → feedforward → add residual"
                          summary={rowValues(block.output, token)}
                        >
                          <div className="residual-track">
                            <span>Input shortcut</span>
                            <span>↘ + attention output = residual 1</span>
                            <span>↘ + feedforward output = residual 2</span>
                          </div>
                          <Module
                            {...moduleProps(`block-${index}-norm1`)}
                            title="Layer normalization 1"
                            subtitle="Center and rescale each token’s 8 coordinates"
                            summary={rowValues(block.norm1, token)}
                          >
                            <NormDetail
                              input={block.input}
                              output={block.norm1}
                              token={token}
                              scale={
                                model.parameters[`blocks.${index}.norm1.scale`]
                                  .data
                              }
                              bias={
                                model.parameters[`blocks.${index}.norm1.bias`]
                                  .data
                              }
                            />
                          </Module>
                          <Module
                            {...moduleProps(`block-${index}-attention`)}
                            title="Multi-head causal attention"
                            subtitle="Learned projections produce Q/K/V; input-dependent weights mix V"
                            summary={rowValues(block.attention, token)}
                          >
                            <div className="matrix-triple">
                              <MatrixView
                                label="Q projection"
                                value={block.q}
                                row={token}
                              />
                              <MatrixView
                                label="K projection"
                                value={block.k}
                                row={token}
                              />
                              <MatrixView
                                label="V projection"
                                value={block.v}
                                row={token}
                              />
                            </div>
                            {block.heads.map((head, h) => (
                              <Module
                                key={h}
                                {...moduleProps(`block-${index}-head-${h}`)}
                                title={`Attention head ${h + 1}`}
                                subtitle={`Query row ${token} · 4 coordinates per head`}
                                summary={rowValues(head.output, token)}
                              >
                                <AttentionDetail
                                  selectedKey={
                                    state.selectedKeys[
                                      `block-${index}-head-${h}`
                                    ] ?? 0
                                  }
                                  selectKey={(key) =>
                                    patch({
                                      selectedKeys: {
                                        ...state.selectedKeys,
                                        [`block-${index}-head-${h}`]: key,
                                      },
                                    })
                                  }
                                  trace={head}
                                  row={token}
                                  coordinate={Math.min(state.coordinate, 3)}
                                  select={(r, c) =>
                                    patch({
                                      token: r,
                                      coordinate: c,
                                      block: index,
                                      head: h,
                                    })
                                  }
                                  showMath={state.showMath}
                                  showCode={state.showCode}
                                />
                              </Module>
                            ))}
                            <p className="computation-note">
                              The two head outputs are concatenated and
                              multiplied by the learned output projection.
                            </p>
                          </Module>
                          <ResidualDetail
                            label="Residual addition 1"
                            left={block.input}
                            right={block.attention}
                            output={block.residual}
                            token={token}
                            coordinate={state.coordinate}
                          />
                          <Module
                            {...moduleProps(`block-${index}-norm2`)}
                            title="Layer normalization 2"
                            subtitle="Normalize residual 1 independently at every token"
                            summary={rowValues(block.norm2, token)}
                          >
                            <NormDetail
                              input={block.residual}
                              output={block.norm2}
                              token={token}
                              scale={
                                model.parameters[`blocks.${index}.norm2.scale`]
                                  .data
                              }
                              bias={
                                model.parameters[`blocks.${index}.norm2.bias`]
                                  .data
                              }
                            />
                          </Module>
                          <Module
                            {...moduleProps(`block-${index}-feedforward`)}
                            title="Feedforward network"
                            subtitle="8 inputs → 16 ReLU neurons → 8 outputs; shared across positions"
                            summary={rowValues(block.ffOut, token)}
                          >
                            <Module
                              {...moduleProps(`block-${index}-hidden`)}
                              title="Hidden layer · 16 neurons"
                              subtitle="Each neuron has its own weighted sum and ReLU gate"
                              summary={rowValues(block.ffHidden, token)}
                            >
                              <div className="neuron-strip">
                                {rowValues(block.ffHidden, token).map(
                                  (value, j) => (
                                    <button
                                      key={j}
                                      aria-label={`Inspect hidden neuron ${j}`}
                                      aria-pressed={state.neuron === j}
                                      onClick={() =>
                                        patch({
                                          neuron: j,
                                          block: index,
                                          focus: `block-${index}-neuron-${j}`,
                                        })
                                      }
                                    >
                                      <small>h{j}</small>
                                      <b>{n(value)}</b>
                                    </button>
                                  ),
                                )}
                              </div>
                              <NeuronDetail
                                input={rowValues(block.norm2, token)}
                                weights={
                                  model.parameters[`blocks.${index}.ff1`].data
                                }
                                bias={
                                  model.parameters[`blocks.${index}.ff1Bias`]
                                    .data[state.neuron]
                                }
                                width={16}
                                neuron={state.neuron}
                                pre={
                                  rowValues(block.ffPre, token)[state.neuron]
                                }
                                output={
                                  rowValues(block.ffHidden, token)[state.neuron]
                                }
                                activation="ReLU"
                                showCode={state.showCode}
                              />
                            </Module>
                            <Module
                              {...moduleProps(`block-${index}-output-layer`)}
                              title="Output layer · 8 neurons"
                              subtitle="Mix hidden activations into the residual stream"
                              summary={rowValues(block.ffOut, token)}
                            >
                              <div className="neuron-strip">
                                {rowValues(block.ffOut, token).map(
                                  (value, j) => (
                                    <button
                                      key={j}
                                      aria-label={`Inspect output neuron ${j}`}
                                      aria-pressed={state.coordinate === j}
                                      onClick={() => patch({ coordinate: j })}
                                    >
                                      <small>y{j}</small>
                                      <b>{n(value)}</b>
                                    </button>
                                  ),
                                )}
                              </div>
                              <NeuronDetail
                                input={rowValues(block.ffHidden, token)}
                                weights={
                                  model.parameters[`blocks.${index}.ff2`].data
                                }
                                bias={
                                  model.parameters[`blocks.${index}.ff2Bias`]
                                    .data[state.coordinate]
                                }
                                width={8}
                                neuron={state.coordinate}
                                pre={
                                  rowValues(block.ffOut, token)[
                                    state.coordinate
                                  ]
                                }
                                output={
                                  rowValues(block.ffOut, token)[
                                    state.coordinate
                                  ]
                                }
                                activation="identity"
                                showCode={state.showCode}
                              />
                            </Module>
                          </Module>
                          <ResidualDetail
                            label="Residual addition 2"
                            left={block.residual}
                            right={block.ffOut}
                            output={block.output}
                            token={token}
                            coordinate={state.coordinate}
                          />
                        </Module>
                      </div>
                    ))}
                  {state.kind !== 'embeddings' && (
                    <>
                      <FlowArrow />
                      <Module
                        {...moduleProps('vocabulary')}
                        title="Final normalization & vocabulary projection"
                        subtitle={`Selected token at position ${token} → 5 vocabulary scores`}
                        summary={rowValues(trace.logits, token)}
                      >
                        <MatrixView
                          label="Final normalized values"
                          value={trace.norm}
                          row={token}
                        />
                        <MatrixView
                          label="Vocabulary scores · base logits"
                          value={trace.logits}
                          row={token}
                          columnLabels={DECODER_CONFIG.vocab}
                          onCell={(r, c) => patch({ token: r, coordinate: c })}
                        />
                        <ProbabilityDetail
                          scores={rowValues(trace.logits, token)}
                          probabilities={rowValues(trace.probabilities, token)}
                          labels={DECODER_CONFIG.vocab}
                          showMath={state.showMath}
                          showCode={state.showCode}
                        />
                      </Module>
                    </>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>
      {state.kind === 'decoder' && (
        <section className="generation-panel">
          <div className="section-heading">
            <div>
              <p className="studio-kicker">One generation cycle</p>
              <h2>Scores → distribution → choose → append</h2>
            </div>
            <span className="stage-badge">{generationPhase}</span>
          </div>
          <p>
            Next-token selection uses the final prompt position. Sampling
            settings change the distribution, while the prompt’s base logits and
            learned parameters stay fixed.
          </p>
          <div className="control-row">
            <label>
              Selection
              <select
                value={state.mode}
                onChange={(e) => {
                  patch({ mode: e.target.value as ExplorationState['mode'] })
                  setChoice(null)
                  setGenerationPhase('scores')
                }}
              >
                <option value="greedy">Greedy · largest base logit</option>
                <option value="sample">Sample · positive temperature</option>
              </select>
            </label>
            <label>
              Temperature
              <input
                aria-label="Sampling temperature"
                type="number"
                min=".05"
                max="5"
                step=".05"
                value={state.temperature}
                disabled={state.mode === 'greedy'}
                onChange={(e) => {
                  patch({
                    temperature: Math.max(
                      0.05,
                      Math.min(5, Number(e.target.value)),
                    ),
                  })
                  setChoice(null)
                  setGenerationPhase('scores')
                }}
              />
            </label>
            <label>
              Top-k
              <select
                value={state.topK}
                disabled={state.mode === 'greedy'}
                onChange={(e) => {
                  patch({ topK: Number(e.target.value) })
                  setChoice(null)
                  setGenerationPhase('scores')
                }}
              >
                {[1, 2, 3, 4, 5].map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="generation-cycle">
            <button onClick={run}>1. Calculate scores</button>
            <button
              disabled={!trace}
              onClick={() => {
                setGenerationPhase('distribution')
                setChoice(null)
              }}
            >
              2. Form distribution
            </button>
            <button
              disabled={!trace || generationPhase !== 'distribution'}
              onClick={() => {
                const result = sampleToken(logits, {
                  mode: state.mode,
                  temperature: state.temperature,
                  topK: state.topK,
                  seed: state.seed,
                })
                setChoice(result.token)
                patch({ seed: result.seed })
                setGenerationPhase('chosen')
              }}
            >
              3. Choose token
            </button>
            <button
              disabled={
                choice === null ||
                generationPhase !== 'chosen' ||
                state.ids.length >= 12
              }
              onClick={() => {
                patch(
                  { ids: [...state.ids, choice!], token: state.ids.length },
                  true,
                )
                setGenerationPhase('appended')
              }}
            >
              4. Append token
            </button>
          </div>
          {trace && (
            <MatrixView
              label="Fixed prompt base logits"
              value={vector(logits)}
              columnLabels={DECODER_CONFIG.vocab}
            />
          )}
          {trace &&
            (generationPhase === 'distribution' ||
              generationPhase === 'chosen') && (
              <ProbabilityBars
                probabilities={
                  state.mode === 'greedy'
                    ? logits.map((_, i) =>
                        i === logits.indexOf(Math.max(...logits)) ? 1 : 0,
                      )
                    : distribution
                }
                labels={DECODER_CONFIG.vocab}
              />
            )}
          {choice !== null && (
            <p className="chosen-token">
              Selected <strong>{DECODER_CONFIG.vocab[choice]}</strong>. Append
              it to make the next prompt.
            </p>
          )}
          {state.ids.length >= 12 && (
            <p>
              Context limit reached (12 tokens). Remove a token or reset the
              prompt to continue.
            </p>
          )}
        </section>
      )}
      {isDecoder && (
        <details className="parameter-editor">
          <summary>Learned parameters & checkpoint evidence</summary>
          <p>
            This model learns the synthetic red → green → blue cycle. It is not
            a language model for ordinary text. The reserved EOS token and
            positions 10–11 were not trained. Both checkpoints share the
            architecture, initialization seed, and data recipe. Compare 0
            updates with 360 updates.
          </p>
          <CheckpointEvidence
            metadata={getCheckpoint(state.checkpoint).metadata}
          />
          {state.showControls && (
            <>
              <div className="control-row">
                <label>
                  Parameter identity
                  <select
                    value={parameterKey}
                    onChange={(e) => {
                      setParameterKey(e.target.value)
                      setParameterIndex(0)
                    }}
                  >
                    {Object.keys(model.parameters).map((key) => (
                      <option key={key}>{key}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Flat coordinate
                  <input
                    type="number"
                    min="0"
                    max={selectedParameter.data.length - 1}
                    value={parameterIndex}
                    onChange={(e) =>
                      setParameterIndex(
                        Math.min(
                          selectedParameter.data.length - 1,
                          Math.max(0, Math.floor(Number(e.target.value))),
                        ),
                      )
                    }
                  />
                </label>
                <label>
                  Value
                  <input
                    aria-label="Learned parameter value"
                    type="number"
                    step=".05"
                    value={selectedParameter.data[parameterIndex]}
                    onChange={(e) =>
                      patch(
                        {
                          overrides: {
                            ...state.overrides,
                            [parameterKey]: selectedParameter.data.map(
                              (v, i) =>
                                i === parameterIndex
                                  ? enteredNumber(e.target.value)
                                  : v,
                            ),
                          },
                        },
                        true,
                      )
                    }
                  />
                </label>
              </div>
              <MatrixView
                label={parameterKey}
                parameter
                value={selectedParameter.toValue()}
              />
            </>
          )}
        </details>
      )}
      {comparisons.length > 0 && (
        <section className="comparison-panel">
          <h3>Saved comparisons</h3>
          <p>
            Restore an exact input, parameter, seed, and view state. Values
            below are {isDecoder ? 'last-position base logits' : 'outputs'}.
          </p>
          {comparisons.map((comparison, index) => (
            <div className="comparison-row" key={index}>
              <strong>{comparison.label}</strong>
              <code>{comparison.output.map(n).join(', ')}</code>
              <button onClick={() => restore(comparison.state)}>Restore</button>
              <button
                onClick={() =>
                  setComparisons(comparisons.filter((_, i) => i !== index))
                }
              >
                Remove
              </button>
            </div>
          ))}
        </section>
      )}
      <footer className="lesson-legend">
        <span className="legend-parameter">◆ Learned parameter</span>
        <span className="legend-value">● Computed value</span>
        <span>
          Amber: negative · violet: positive · numbers carry the exact sign
        </span>
        <span>Opening modules never updates parameters.</span>
      </footer>
    </div>
  )
}

function Module({
  id,
  title,
  subtitle,
  summary,
  expanded,
  onToggle,
  active,
  children,
}: {
  id: string
  title: string
  subtitle: string
  summary: number[]
  expanded: boolean
  onToggle: () => void
  active: boolean
  children: ReactNode
}) {
  return (
    <section
      className={`model-module ${expanded ? 'module-open' : ''} ${active ? 'module-focused' : ''}`}
      id={id}
    >
      <button
        className="module-heading"
        aria-expanded={expanded}
        aria-controls={`${id}-content`}
        onClick={onToggle}
      >
        {expanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        <span>
          <strong>{title}</strong>
          <small>{subtitle}</small>
        </span>
        <span className="module-mini-summary" title={summary.map(n).join(', ')}>
          <span className="mini-activation-bars">
            {summary.slice(0, 8).map((value, index) => (
              <i
                key={index}
                style={{
                  background: value < 0 ? '#d8b37e' : '#b398cf',
                  height: `${Math.min(24, 7 + Math.abs(value) * 5)}px`,
                }}
              />
            ))}
          </span>
          <small>
            out {n(summary[0])}
            {summary.length > 1 ? ', …' : ''}
          </small>
        </span>
        <span className="module-summary">
          [{summary.slice(0, 4).map(n).join(', ')}
          {summary.length > 4 ? ', …' : ''}]
        </span>
        <span className="module-action">{expanded ? 'Close' : 'Open'}</span>
      </button>
      {expanded && (
        <div className="module-content" id={`${id}-content`}>
          {children}
        </div>
      )}
    </section>
  )
}
function FlowArrow() {
  return (
    <div className="flow-arrow">
      <ArrowDown size={18} />
    </div>
  )
}
function ProbabilityBars({
  probabilities,
  labels,
}: {
  probabilities: number[]
  labels?: string[]
}) {
  return (
    <div className="probability-bars" aria-label="Probability distribution">
      {probabilities.map((p, i) => (
        <div key={i}>
          <span>{labels?.[i] ?? `Score ${i}`}</span>
          <div className="probability-track">
            <div style={{ width: `${p * 100}%` }} />
          </div>
          <strong>{(p * 100).toFixed(2)}%</strong>
        </div>
      ))}
    </div>
  )
}
function ProbabilityDetail({
  scores,
  probabilities,
  labels,
  showMath,
  showCode,
}: {
  scores: number[]
  probabilities: number[]
  labels?: string[]
  showMath: boolean
  showCode: boolean
}) {
  const max = Math.max(...scores),
    exps = scores.map((v) => Math.exp(v - max)),
    sum = exps.reduce((a, b) => a + b, 0)
  return (
    <section className="probability-detail">
      <div>
        <p className="studio-kicker">Scores are relative</p>
        <h2>Turn scores into shares of one whole.</h2>
        <p>
          Subtract the largest score, exponentiate, and divide each result by
          their sum.
        </p>
      </div>
      <div className="probability-layout">
        <table className="calculation-table">
          <thead>
            <tr>
              <th>Choice</th>
              <th>Score</th>
              <th>Minus max</th>
              <th>Exponential</th>
              <th>Probability</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((score, i) => (
              <tr key={i}>
                <th>{labels?.[i] ?? i}</th>
                <td>{n(score)}</td>
                <td>{n(score - max)}</td>
                <td>{n(exps[i])}</td>
                <td>
                  {n(exps[i])} / {n(sum)} = {n(probabilities[i])}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th colSpan={4}>Total probability</th>
              <td>{n(probabilities.reduce((a, b) => a + b, 0))}</td>
            </tr>
          </tfoot>
        </table>
        <ProbabilityBars probabilities={probabilities} labels={labels} />
      </div>
      {showMath && (
        <p className="formula-line">
          pᵢ = exp(scoreᵢ − max(score)) / Σⱼ exp(scoreⱼ − max(score))
        </p>
      )}
      {showCode && (
        <pre>
          {
            'shifted = scores - max(scores)\nexp_scores = np.exp(shifted)\nprobabilities = exp_scores / exp_scores.sum()'
          }
        </pre>
      )}
    </section>
  )
}
function AttentionDetail({
  trace,
  row,
  coordinate,
  select,
  showMath,
  showCode,
  selectedKey,
  selectKey,
}: {
  trace: AttentionTrace
  row: number
  coordinate: number
  select: (row: number, col: number) => void
  showMath: boolean
  showCode: boolean
  selectedKey: number
  selectKey: (key: number) => void
}) {
  const key = Math.min(selectedKey, trace.k.shape[0] - 1),
    setKey = selectKey,
    d = trace.q.shape[1],
    count = trace.k.shape[0],
    query = Math.min(row, trace.q.shape[0] - 1),
    coord = Math.min(coordinate, trace.v.shape[1] - 1),
    q = rowValues(trace.q, query),
    k = rowValues(trace.k, key),
    weights = rowValues(trace.weights, query),
    masked = trace.q.shape[0] === trace.k.shape[0] && key > query
  const allowed = rowValues(trace.maskedScores, query).filter(
      (_, index) => trace.q.shape[0] !== trace.k.shape[0] || index <= query,
    ),
    rowMax = Math.max(...allowed),
    denominator = allowed.reduce(
      (total, score) => total + Math.exp(score - rowMax),
      0,
    )
  return (
    <div className="attention-detail">
      <p className="computation-note">
        Q, K, V and attention weights are computed from the input. The
        projection matrices that produce them are learned parameters.
      </p>
      <div className="matrix-triple">
        <MatrixView
          label="Q · queries"
          value={trace.q}
          row={query}
          onCell={(r) => select(r, coord)}
        />
        <MatrixView
          label="K · keys"
          value={trace.k}
          row={key}
          onCell={(r) => setKey(r)}
        />
        <MatrixView
          label="V · values"
          value={trace.v}
          onCell={(_, c) => select(query, c)}
        />
      </div>
      <div className="matrix-triple">
        <MatrixView
          label="Scaled dot products"
          value={trace.scores}
          row={query}
          onCell={(r, c) => {
            select(r, coord)
            setKey(c)
          }}
        />
        <MatrixView
          label="After causal mask"
          causal={trace.q.shape[0] === trace.k.shape[0]}
          value={trace.maskedScores}
          row={query}
          onCell={(r, c) => {
            select(r, coord)
            setKey(c)
          }}
        />
        <MatrixView
          label="Attention weights"
          value={trace.weights}
          row={query}
          selected={[query, key]}
          onCell={(r, c) => {
            select(r, coord)
            setKey(c)
          }}
        />
      </div>
      <div className="attention-calculations">
        <div className="arithmetic-card">
          <p className="studio-kicker">
            Query {query} → key {key}
          </p>
          <h3>A dot product measures the match.</h3>
          <p>
            ({q.map((v, i) => `${n(v)} × ${n(k[i])}`).join(' + ')}) / √{d} ={' '}
            <b>{n(trace.scores.data[query * count + key])}</b>
          </p>
          {masked ? (
            <p className="masked-explanation">
              Future position: excluded by the causal mask. Its attention weight
              is exactly 0.
            </p>
          ) : (
            <p>
              exp({n(trace.maskedScores.data[query * count + key])} −{' '}
              {n(rowMax)}) / {n(denominator)} = <b>{n(weights[key])}</b>
            </p>
          )}
          <p>Row weights sum to {n(weights.reduce((a, b) => a + b, 0))}.</p>
        </div>
        <div className="arithmetic-card">
          <p className="studio-kicker">
            Output row {query}, coordinate {coord}
          </p>
          <h3>Mix values using those shares.</h3>
          <p>
            {weights
              .map((w, i) => `${n(w)} × ${n(rowValues(trace.v, i)[coord])}`)
              .join(' + ')}{' '}
            = <b>{n(rowValues(trace.output, query)[coord])}</b>
          </p>
          <p>
            Select a different output coordinate below to inspect its
            calculation.
          </p>
        </div>
      </div>
      <MatrixView
        label="Weighted values · output"
        value={trace.output}
        row={query}
        selected={[query, coord]}
        onCell={select}
      />
      {showMath && (
        <p className="formula-line">
          scores = QKᵀ / √d · weights = softmax(mask(scores)) · output = weights
          × V
        </p>
      )}
      {showCode && (
        <pre>
          {
            'scores = q @ k.T / np.sqrt(head_width)\nscores[future_positions] = -np.inf  # causal attention only\nweights = stable_softmax(scores)\noutput = weights @ v'
          }
        </pre>
      )}
    </div>
  )
}
function NeuronDetail({
  input,
  weights,
  bias,
  width,
  neuron,
  pre,
  output,
  activation,
  showCode,
}: {
  input: number[]
  weights: number[]
  bias: number
  width: number
  neuron: number
  pre: number
  output: number
  activation: string
  showCode: boolean
}) {
  return (
    <div className="arithmetic-card">
      <p className="studio-kicker">
        Neuron {neuron} → weighted sum → individual arithmetic
      </p>
      <table className="calculation-table">
        <thead>
          <tr>
            <th>Input coordinate</th>
            <th>Input (computed)</th>
            <th>Weight (learned)</th>
            <th>Product</th>
          </tr>
        </thead>
        <tbody>
          {input.map((value, i) => (
            <tr key={i}>
              <th>{i}</th>
              <td>{n(value)}</td>
              <td>{n(weights[i * width + neuron])}</td>
              <td>{n(value * weights[i * width + neuron])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Sum of products + bias {n(bias)} = <b>{n(pre)}</b>
      </p>
      <p>
        {activation === 'ReLU'
          ? `ReLU keeps positive values: max(0, ${n(pre)})`
          : `Identity keeps the sum: ${n(pre)}`}{' '}
        = <b>{n(output)}</b>
      </p>
      {showCode && (
        <pre>{`weighted_sum = sum(x * w for x, w in zip(inputs, weights)) + bias\noutput = ${activation === 'ReLU' ? 'max(0, weighted_sum)' : 'weighted_sum'}`}</pre>
      )}
    </div>
  )
}
function ResidualDetail({
  label,
  left,
  right,
  output,
  token,
  coordinate,
}: {
  label: string
  left: TensorValue
  right: TensorValue
  output: TensorValue
  token: number
  coordinate: number
}) {
  return (
    <div className="residual-calculation">
      <strong>↳ + {label}</strong>
      <span>
        coordinate {coordinate}: {n(rowValues(left, token)[coordinate])} +{' '}
        {n(rowValues(right, token)[coordinate])} ={' '}
        <b>{n(rowValues(output, token)[coordinate])}</b>
      </span>
    </div>
  )
}
function NormDetail({
  input,
  output,
  token,
  scale,
  bias,
}: {
  input: TensorValue
  output: TensorValue
  token: number
  scale: number[]
  bias: number[]
}) {
  const row = rowValues(input, token),
    mean = row.reduce((a, b) => a + b, 0) / row.length,
    variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length
  return (
    <>
      <MatrixView label="Before normalization" value={input} row={token} />
      <p className="formula-line">
        Row {token}: mean = {n(mean)}, variance = {n(variance)}. Divide centered
        values by √(variance + 0.00001).
      </p>
      <table className="calculation-table">
        <thead>
          <tr>
            <th>Coordinate</th>
            <th>Centered value</th>
            <th>Learned scale</th>
            <th>Learned bias</th>
            <th>Output</th>
          </tr>
        </thead>
        <tbody>
          {row.map((v, i) => (
            <tr key={i}>
              <th>{i}</th>
              <td>{n(v - mean)}</td>
              <td>{n(scale[i])}</td>
              <td>{n(bias[i])}</td>
              <td>{n(rowValues(output, token)[i])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function CheckpointEvidence({ metadata }: { metadata: CheckpointMetadata }) {
  const history = metadata.history,
    maxStep = Math.max(1, ...history.map((p) => p.step))
  const x = (step: number) => 58 + (step / maxStep) * 500,
    y = (loss: number) =>
      152 - ((Math.log10(Math.max(loss, 0.0001)) + 4) / 5) * 130
  return (
    <div className="checkpoint-evidence">
      <div className="checkpoint-stats">
        <span>
          <b>{metadata.steps}</b>parameter updates
        </span>
        <span>
          <b>{metadata.trainingLoss.toFixed(6)}</b>training loss
        </span>
        <span>
          <b>{metadata.validationLoss.toFixed(6)}</b>validation loss
        </span>
        <span>
          <b>{metadata.seed}</b>fixed initialization seed
        </span>
      </div>
      <svg
        viewBox="0 0 610 190"
        role="img"
        aria-label="Checkpoint training and validation cross-entropy over parameter updates, logarithmic loss axis"
      >
        {[0.0001, 0.001, 0.01, 0.1, 1, 10].map((loss) => (
          <g key={loss}>
            <line x1="58" x2="558" y1={y(loss)} y2={y(loss)} stroke="#ece5f3" />
            <text
              x="49"
              y={y(loss) + 3}
              textAnchor="end"
              fill="#9a88ab"
              fontSize="9"
            >
              {loss}
            </text>
          </g>
        ))}
        {history.length > 1 && (
          <>
            <polyline
              fill="none"
              stroke="#9973be"
              strokeWidth="2.5"
              points={history
                .map((p) => `${x(p.step)},${y(p.trainingLoss)}`)
                .join(' ')}
            />
            <polyline
              fill="none"
              stroke="#79a68a"
              strokeWidth="2"
              strokeDasharray="4 3"
              points={history
                .map((p) => `${x(p.step)},${y(p.validationLoss)}`)
                .join(' ')}
            />
          </>
        )}
        {history.map((p) => (
          <circle
            key={p.step}
            cx={x(p.step)}
            cy={y(p.validationLoss)}
            r="2.5"
            fill="#79a68a"
          />
        ))}
        <text x="58" y="170" fontSize="9" fill="#9a88ab">
          0
        </text>
        <text x="558" y="170" textAnchor="end" fontSize="9" fill="#9a88ab">
          {maxStep}
        </text>
        <text x="305" y="185" textAnchor="middle" fontSize="10" fill="#8c769d">
          Parameter updates
        </text>
        <text
          x="13"
          y="90"
          textAnchor="middle"
          transform="rotate(-90 13 90)"
          fontSize="9"
          fill="#8c769d"
        >
          Cross-entropy (log scale)
        </text>
      </svg>
      <p>
        Violet: training · dashed green: validation. Learning rate{' '}
        {metadata.learningRate}. Validation uses held-out prefix lengths of the
        same repeating-color task.
      </p>
    </div>
  )
}
