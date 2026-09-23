import type { LessonKind } from './presets'
import { isLessonKind } from './presets'
import { getCheckpoint } from './decoder'

export interface ExplorationState {
  kind: LessonKind
  ids: number[]
  checkpoint: 'trained' | 'untrained'
  overrides: Record<string, number[]>
  opened: string[]
  focus: string
  selectedKeys: Record<string, number>
  token: number
  coordinate: number
  neuron: number
  head: number
  block: number
  scores: number[]
  q: number[]
  k: number[]
  v: number[]
  intervention: boolean
  seed: number
  temperature: number
  topK: number
  mode: 'greedy' | 'sample'
  showMath: boolean
  showCode: boolean
  showControls: boolean
  prediction: string
  zoom: number
}

export interface ExplorationExecution {
  cursor: number
  generationPhase: 'scores' | 'distribution' | 'chosen' | 'appended'
  choice: number | null
  current: boolean
}

export interface ExplorationFile {
  kind: 'backprop-exploration'
  version: 1
  state: ExplorationState
  execution?: ExplorationExecution
  comparisons: Array<{
    label: string
    state: ExplorationState
    output: number[]
  }>
}

export function initialExploration(kind: LessonKind): ExplorationState {
  return {
    kind,
    ids: [0, 1, 2],
    checkpoint: 'trained',
    overrides: {},
    opened: kind === 'block' ? ['block-0'] : [],
    focus: 'model',
    selectedKeys: {},
    token: 2,
    coordinate: 0,
    neuron: 0,
    head: 0,
    block: 0,
    scores: [1, 2, 3],
    q: kind === 'causal' ? [1, 0, 0, 1, 1, 1] : [Math.SQRT2, 0],
    k: kind === 'causal' ? [1, 0, 0, 1, 1, 1] : [0, 0, Math.log(2), 0, 0, 0],
    v: [2, 0, 0, 2, 2, 2],
    intervention: false,
    seed: 42,
    temperature: 1,
    topK: 5,
    mode: 'greedy',
    showMath: false,
    showCode: false,
    showControls: kind !== 'decoder' && kind !== 'block',
    prediction: '',
    zoom: 1,
  }
}

export function validateExplorationState(
  value: unknown,
): value is ExplorationState {
  if (!value || typeof value !== 'object') return false
  const v = value as ExplorationState
  const record = (x: unknown): x is Record<string, unknown> =>
    Boolean(x && typeof x === 'object' && !Array.isArray(x))
  const parameters = getCheckpoint('trained').parameters
  const numbers = (x: unknown, length?: number): x is number[] =>
    Array.isArray(x) &&
    (length === undefined || x.length === length) &&
    x.every(
      (n) => typeof n === 'number' && Number.isFinite(n) && Math.abs(n) < 1e6,
    )
  return (
    isLessonKind(v.kind) &&
    !['linear', 'neuron', 'small-network', 'playground'].includes(v.kind) &&
    numbers(v.ids) &&
    v.ids.length > 0 &&
    v.ids.length <= 12 &&
    v.ids.every((id) => Number.isInteger(id) && id >= 0 && id < 5) &&
    (v.checkpoint === 'trained' || v.checkpoint === 'untrained') &&
    record(v.overrides) &&
    Object.entries(v.overrides).every(
      ([key, data]) =>
        Object.hasOwn(parameters, key) &&
        numbers(data, parameters[key].data.length),
    ) &&
    Array.isArray(v.opened) &&
    v.opened.every((x) => typeof x === 'string') &&
    typeof v.focus === 'string' &&
    record(v.selectedKeys) &&
    Object.values(v.selectedKeys).every(
      (key) =>
        Number.isInteger(key) && (key as number) >= 0 && (key as number) < 12,
    ) &&
    [v.token, v.coordinate, v.neuron, v.head, v.block, v.seed].every(
      (x) => Number.isInteger(x) && x >= 0,
    ) &&
    v.seed <= 4294967295 &&
    v.token < 12 &&
    v.coordinate < 8 &&
    v.neuron < 16 &&
    v.head < 2 &&
    v.block < 2 &&
    numbers(v.scores, 3) &&
    numbers(v.q, v.kind === 'causal' ? 6 : 2) &&
    numbers(v.k, 6) &&
    numbers(v.v, 6) &&
    typeof v.intervention === 'boolean' &&
    Number.isFinite(v.temperature) &&
    v.temperature >= 0.05 &&
    v.temperature <= 5 &&
    Number.isInteger(v.topK) &&
    v.topK >= 1 &&
    v.topK <= 5 &&
    (v.mode === 'greedy' || v.mode === 'sample') &&
    [v.showMath, v.showCode, v.showControls].every(
      (x) => typeof x === 'boolean',
    ) &&
    typeof v.prediction === 'string' &&
    Number.isFinite(v.zoom) &&
    v.zoom >= 0.5 &&
    v.zoom <= 1.5
  )
}

function validateExecution(value: unknown): value is ExplorationExecution {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const execution = value as ExplorationExecution
  return (
    Number.isInteger(execution.cursor) &&
    execution.cursor >= -1 &&
    execution.cursor <= 3 &&
    ['scores', 'distribution', 'chosen', 'appended'].includes(
      execution.generationPhase,
    ) &&
    (execution.choice === null ||
      (Number.isInteger(execution.choice) &&
        execution.choice >= 0 &&
        execution.choice < 5)) &&
    typeof execution.current === 'boolean' &&
    (execution.generationPhase !== 'chosen' ||
      (execution.choice !== null && execution.current))
  )
}

export function parseExploration(text: string): ExplorationFile {
  const error =
    'Choose a valid saved exploration. Legacy graph files open from the blank builder’s Import menu.'
  try {
    const value = JSON.parse(text) as ExplorationFile
    if (
      !value ||
      typeof value !== 'object' ||
      value.kind !== 'backprop-exploration' ||
      value.version !== 1 ||
      !Array.isArray(value.comparisons)
    )
      throw new Error(error)
    if (
      Object.hasOwn(value, 'execution') &&
      !validateExecution(value.execution)
    )
      throw new Error(error)
    // Early v1 files predate persistent per-head key selection.
    const migrate = (state: ExplorationState) => {
      if (
        state &&
        typeof state === 'object' &&
        !Object.hasOwn(state, 'selectedKeys')
      )
        state.selectedKeys = {}
      return state
    }
    if (
      !validateExplorationState(migrate(value.state)) ||
      value.comparisons.some(
        (c) =>
          !c ||
          typeof c !== 'object' ||
          typeof c.label !== 'string' ||
          !validateExplorationState(migrate(c.state)) ||
          c.state.kind !== value.state.kind ||
          !Array.isArray(c.output) ||
          !c.output.every((n) => typeof n === 'number' && Number.isFinite(n)),
      )
    )
      throw new Error(error)
    return value
  } catch {
    throw new Error(error)
  }
}

export function downloadExploration(file: ExplorationFile) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' }),
  )
  const a = document.createElement('a')
  a.href = url
  a.download = `backprop-${file.state.kind}.json`
  a.click()
  URL.revokeObjectURL(url)
}
