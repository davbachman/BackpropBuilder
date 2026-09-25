import type { CustomCsvData, DatasetKind, DatasetTask, GraphNode, TensorValue } from './types'
import { tensorValue } from './tensor'
import { analyzeCustomCsv } from './customCsv'
import digits from '../learning/digits.json'

export interface DatasetExample {
  label: string
  split: 'train' | 'test'
  features: TensorValue[]
  target: TensorValue
}

export interface ToyDataset {
  kind: DatasetKind
  label: string
  task: DatasetTask
  featureLabels: string[]
  targetLabel: string
  featureValues: TensorValue[]
  targetValue: TensorValue
  examples?: DatasetExample[]
  description?: string
  vocabulary?: string[]
  classLabels?: string[]
  maxLength?: number
  source?: string
}

export const DATASET_OPTIONS: ToyDataset[] = [
  {
    kind: 'line-1d',
    label: 'Line regression (1D)',
    task: 'regression',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.3]),
    ],
    targetValue: tensorValue([20], [-3.62, -3.32, -2.51, -2.2, -1.26, -0.85, 0.01, 0.24, 1.04, 1.77, 2.09, 2.88, 3.22, 4.12, 4.53, 5.4, 5.65, 6.45, 7.13, 7.5]),
  },
  {
    kind: 'cubic-1d',
    label: 'Cubic regression (1D)',
    task: 'regression',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.3]),
    ],
    targetValue: tensorValue([20], [-11.92, -8.97, -6.13, -4.29, -2.32, -1.39, -0.31, 0.05, 0.62, 0.67, 1.12, 1.13, 1.62, 1.76, 2.57, 3.16, 4.71, 6.11, 8.38, 10.81]),
  },
  {
    kind: 'plane-2d',
    label: 'Plane regression (2D)',
    task: 'regression',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2, -1.8, -1.1, -0.3, 0.5, 1.3, 2.1, -1.5, 0.2, 1.7]),
      tensorValue([20], [-1.5, -0.9, -0.2, 0.4, 1.1, 1.7, -1.3, -0.6, 0.1, 0.8, 1.5, 1.4, 0.7, -1.1, -1.8, -0.4, 0.3, -1.7, 1.2, -1]),
    ],
    targetValue: tensorValue([20], [-0.71, -0.97, -0.71, -0.86, -0.55, -0.76, 2.05, 1.77, 1.92, 1.71, 2.03, -2.73, -1.19, 0.71, 2.44, 2.21, 3.02, -0.17, 0, 3.07]),
  },
  {
    kind: 'threshold-1d',
    label: 'Threshold classification (1D)',
    task: 'binary-classification',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.05, -1.76, -1.42, -1.18, -0.92, -0.71, -0.53, -0.31, -0.16, -0.04, 0.08, 0.19, 0.36, 0.57, 0.79, 1.02, 1.28, 1.51, 1.74, 2.03]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
  },
  {
    kind: 'circle-center',
    label: 'Circle vs center classification (2D)',
    task: 'binary-classification',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-0.18, 0.12, -0.08, 0.22, 0.05, -0.25, 0.16, -0.04, 0.28, -0.14, 1.68, 1.23, 0.46, -0.42, -1.18, -1.72, -1.2, -0.5, 0.44, 1.18]),
      tensorValue([20], [0.06, -0.15, 0.22, 0.14, -0.26, -0.12, 0.3, -0.31, 0.02, 0.18, 0.08, 1.14, 1.58, 1.61, 1.18, 0.05, -1.09, -1.53, -1.61, -1.13]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
  },
  {
    kind: 'parabola-boundary',
    label: 'Parabola boundary classification (2D)',
    task: 'binary-classification',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]),
      tensorValue([20], [-2, -2, -2, -2, -2, -0.6, -0.6, -0.6, -0.6, -0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 2, 2, 2, 2, 2]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]),
  },
]

// Tensor examples retain their image/sequence axes; they are not batch axes.
function tensorDataset(kind: DatasetKind, label: string, task: DatasetTask, featureLabels: string[], targetLabel: string, examples: DatasetExample[], metadata: Partial<ToyDataset> = {}): ToyDataset {
  return { kind, label, task, featureLabels, targetLabel, featureValues: examples[0].features, targetValue: examples[0].target, examples, ...metadata }
}
DATASET_OPTIONS.push(tensorDataset('neuron-basics', 'Neuron basics · binary labels', 'binary-classification', ['x'], 'y', Array.from({length:20},(_,i)=>({label:`Example ${i+1}`,split:i % 4 === 3 ? 'test' : 'train',features:[tensorValue([], [i === 0 ? 2 : (i-10)/5])],target:tensorValue([], [Number(i === 0 || i >= 10)])}))))
let xorSeed = 137
const random = () => { xorSeed = (1664525 * xorSeed + 1013904223) >>> 0; return xorSeed / 4294967296 }
const xorPoints = Array.from({ length: 64 }, () => [random() * 2 - 1, random() * 2 - 1])
DATASET_OPTIONS.push({ kind: 'xor', label: 'XOR · opposite quadrants', task: 'binary-classification', featureLabels: ['x1', 'x2'], targetLabel: 'class', featureValues: [0, 1].map(axis => tensorValue([64], xorPoints.map(p => p[axis]))), targetValue: tensorValue([64], xorPoints.map(p => Number(p[0] * p[1] < 0))) })
DATASET_OPTIONS.push(tensorDataset('digits-8x8', 'Handwritten digits · 8 × 8', 'classification', ['image'], 'digit', digits.map(digit => ({ label: `${digit.label} · ${digit.id}`, split: digit.split as 'train' | 'test', features: [tensorValue([8, 8, 1], digit.pixels.map(pixel => pixel / 16))], target: tensorValue([1], [digit.label]) })), { description: '500 UCI handwritten digits: 400 training images and 100 held-out images. Pixels are normalized to 0–1; labels are classes 0–9.', source: 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits' }))
for (const kind of ['color-cycle', 'counting'] as const) {
  const vocabulary = kind === 'color-cycle' ? ['<bos>', 'red', 'green', 'blue', '<eos>'] : ['<bos>', 'one', 'two', 'three', 'four', 'five', '<eos>']
  const period = vocabulary.length - 2
  // Match the checkpoint's task: BOS predicts the first symbol; prefixes
  // without BOS may begin at any phase. Lengths 5 and 8 stay held out, as in
  // the supplied checkpoint, so evaluation does not relabel its training data.
  const examples: DatasetExample[] = Array.from({ length: 8 }, (_, i) => i + 3).flatMap(length =>
    Array.from({ length: period + 1 }, (_, phase) => {
      const tokens = Array.from({ length }, (_, step) => phase === 0 ? (step === 0 ? 0 : (step - 1) % period + 1) : (step + phase - 1) % period + 1)
      const target = tokens.map(token => token % period + 1)
      return { label: tokens.map(id => vocabulary[id]).join(' '), split: length === 5 || length === 8 ? 'test' as const : 'train' as const, features: [tensorValue([length], tokens), tensorValue([length], tokens.map((_, i) => i))], target: tensorValue([length], target) }
    }))
  DATASET_OPTIONS.push(tensorDataset(kind, kind === 'color-cycle' ? 'Color cycle · next token' : 'Counting · next token', 'sequence', ['token IDs', 'position IDs'], 'next-token IDs', examples, { vocabulary, maxLength: 12, description: 'Synthetic repeating sequences with training and held-out examples. Outputs are integer token IDs, matching position IDs, and one next-token target per position.' }))
}
for (const causal of [false, true]) {
  const examples: DatasetExample[] = Array.from({ length: 12 }, (_, i) => ({ label: `Message lookup ${i + 1}`, split: i % 4 === 3 ? 'test' : 'train', features: [tensorValue(causal ? [3, 2] : [1, 2], causal ? [1, 0, 0, 1, 1, 1] : [1, .5 + i / 10]), tensorValue([3, 2], [1, 0, 0, 1, 1, 1]), tensorValue([3, 2], [1 + i / 10, 0, 0, 2, 1, 1])], target: tensorValue(causal ? [3, 2] : [1, 2], causal ? [1 + i / 10, 0, 0, 2, 1, 1] : [1, 1]) }))
  DATASET_OPTIONS.push(tensorDataset(causal ? 'attention-sequence' : 'attention-query', causal ? 'Attention · three-token messages' : 'Attention · query and messages', 'attention', ['queries', 'keys', 'values'], 'desired message', examples, { description: 'Query/key/value tensors for tracing attention. The target is a desired retrieved message for optional regression experiments.' }))
}
DATASET_OPTIONS.push(tensorDataset('class-scores', 'Class scores · softmax experiment', 'classification', ['scores'], 'class', Array.from({ length: 16 }, (_, i) => {
  const scores = [2, 1, .1, -1].map((_, j, values) => values[(j + i) % 4] * (1 + Math.floor(i / 4) / 5))
  return { label: `Class ${(4 - i % 4) % 4} · example ${i + 1}`, split: i >= 12 ? 'test' : 'train', features: [tensorValue([1, 4], scores)], target: tensorValue([1], [scores.indexOf(Math.max(...scores))]) }
})))

// The menu is separate from the built-in data catalog: selecting Custom CSV
// opens a file picker rather than installing a placeholder dataset.
export const DATASET_MENU_OPTIONS: { kind: DatasetKind; label: string }[] = [
  ...DATASET_OPTIONS.map(({ kind, label }) => ({ kind, label })),
  { kind: 'custom-csv', label: 'Custom CSV…' },
]

export function customCsvCardWidth(node: GraphNode): number {
  const longest = Math.max(0, ...Array.from({ length: datasetOutputCountForNode(node) }, (_, slot) => datasetOutputLabelForSlot(node, slot).length))
  return Math.min(420, Math.max(280, 194 + longest * 6))
}
export function customCsvLabelWidth(node: GraphNode): number {
  return customCsvCardWidth(node) - 176
}
export function customCsvOutputTop(index: number, detailed = false): number {
  return (detailed ? 165 : 110) + index * 30
}
export function customCsvCardHeight(node: GraphNode, detailed = false): number {
  return customCsvOutputTop(datasetOutputCountForNode(node), detailed)
}

const customDatasetCache = new WeakMap<CustomCsvData, ToyDataset>()

export function datasetForNode(node: GraphNode): ToyDataset {
  if (node.params.dataset === 'custom-csv' && node.params.customCsv) {
    const csv = node.params.customCsv
    const cached = customDatasetCache.get(csv)
    if (cached) return cached
    const parsed = analyzeCustomCsv(csv)
    const featureLabels = parsed.headers.filter((_, index) => index !== csv.targetColumn)
    const count = parsed.targets.length
    const dataset: ToyDataset = {
      kind: 'custom-csv', label: `Custom CSV · ${csv.fileName}`, task: csv.task,
      featureLabels, targetLabel: parsed.headers[csv.targetColumn],
      featureValues: parsed.features.map(values => tensorValue([count], values)),
      targetValue: tensorValue([count], parsed.targets), classLabels: parsed.classLabels,
      description: `${count} rows · ${featureLabels.length} numeric ${featureLabels.length === 1 ? 'feature' : 'features'}. ${parsed.classLabels ? `Classes: ${parsed.classLabels.map((label, index) => `${label} = ${index}`).join(', ')}.` : 'Connect a column to a Target block to designate the target.'}`,
    }
    customDatasetCache.set(csv, dataset)
    return dataset
  }
  return DATASET_OPTIONS.find(dataset => dataset.kind === node.params.dataset) ?? DATASET_OPTIONS[0]
}
export function datasetExamples(dataset: ToyDataset): DatasetExample[] {
  return dataset.examples ?? dataset.targetValue.data.map((target, i) => ({ label: `Example ${i + 1}`, split: i % 4 === 0 ? 'test' : 'train', features: dataset.featureValues.map(value => tensorValue([], [value.data[i]])), target: tensorValue(dataset.task.includes('classification') ? [1] : [], [target]) }))
}
const splitCache = new WeakMap<ToyDataset, Map<number, DatasetExample[]>>()
/** An optional, deterministic split overrides the included examples' default
 * split. Classification examples are divided within each class. */
export function datasetExamplesForNode(node: GraphNode): DatasetExample[] {
  const dataset = datasetForNode(node)
  const percent = node.params.trainPercent
  if (percent === undefined) return datasetExamples(dataset)
  const cached = !dataset.examples && splitCache.get(dataset)?.get(percent)
  if (cached) return cached
  const examples = datasetExamples(dataset)
  const groups = new Map<string, number[]>()
  examples.forEach((example, index) => {
    const key = dataset.task.includes('classification') ? String(example.target.data[0]) : 'all'
    groups.set(key, [...(groups.get(key) ?? []), index])
  })
  const training = new Set<number>()
  for (const [key, indexes] of groups) {
    const shuffled = [...indexes]
    let seed = [...key].reduce((value, character) => Math.imul(value ^ character.charCodeAt(0), 16777619) >>> 0, 2166136261)
    for (let index = shuffled.length - 1; index > 0; index--) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      const swap = seed % (index + 1)
      ;[shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]]
    }
    const trainCount = shuffled.length === 1 ? 1 : Math.max(1, Math.min(shuffled.length - 1, Math.round(shuffled.length * percent / 100)))
    shuffled.slice(0, trainCount).forEach(index => training.add(index))
  }
  const split = examples.map((example, index) => ({ ...example, split: training.has(index) ? 'train' as const : 'test' as const }))
  if (!dataset.examples) {
    const byPercent = splitCache.get(dataset) ?? new Map<number, DatasetExample[]>()
    byPercent.set(percent, split)
    splitCache.set(dataset, byPercent)
  }
  return split
}
export function datasetExampleIndex(node: GraphNode): number {
  return Math.min(datasetExamplesForNode(node).length - 1, Math.max(0, node.params.datasetIndex ?? 0))
}
export function datasetMode(node: GraphNode): 'sample' | 'batch' {
  return datasetForNode(node).examples ? 'sample' : node.params.datasetMode ?? 'batch'
}
export function datasetOutputCountForNode(node: GraphNode): number {
  return datasetForNode(node).featureValues.length + 1
}
export function datasetTargetSlotForNode(node: GraphNode): number {
  return node.params.dataset === 'custom-csv' && node.params.customCsv
    ? node.params.customCsv.targetColumn : datasetForNode(node).featureValues.length
}
export function datasetOutputLabelForSlot(node: GraphNode, slot: number): string {
  const dataset = datasetForNode(node)
  if (node.params.dataset === 'custom-csv' && node.params.customCsv) {
    const csv = node.params.customCsv
    return csv.hasHeader ? csv.rows[0][slot]?.trim() || `Column ${slot + 1}` : `Column ${slot + 1}`
  }
  return dataset.featureLabels[slot] ?? dataset.targetLabel
}
export function datasetOutputValueForSlot(node: GraphNode, slot: number): TensorValue {
  if (node.params.datasetValues?.[slot]) return node.params.datasetValues[slot]
  const examples = datasetExamplesForNode(node)
  const targetSlot = datasetTargetSlotForNode(node)
  const featureSlot = slot < targetSlot ? slot : slot - 1
  const fromExample = (example: DatasetExample) => slot === targetSlot ? example.target : example.features[featureSlot] ?? example.target
  if (datasetMode(node) === 'sample') {
    const example = examples[datasetExampleIndex(node)]
    return fromExample(example)
  }
  const selected = examples.filter(example => !node.params.datasetSplit || node.params.datasetSplit === 'all' || example.split === node.params.datasetSplit)
  return tensorValue([selected.length], selected.map(example => fromExample(example).data[0]))
}
export function remapDatasetOutputSlot(fromNode: GraphNode, toNode: GraphNode, fromSlot: number): number | undefined {
  const from = datasetForNode(fromNode), to = datasetForNode(toNode)
  const fromTarget = datasetTargetSlotForNode(fromNode), toTarget = datasetTargetSlotForNode(toNode)
  if (fromSlot === fromTarget) return toTarget
  const featureIndex = fromSlot < fromTarget ? fromSlot : fromSlot - 1
  if (featureIndex < 0 || featureIndex >= Math.min(from.featureValues.length, to.featureValues.length)) return undefined
  return featureIndex < toTarget ? featureIndex : featureIndex + 1
}
export function isDatasetKind(value: string): value is DatasetKind {
  return value === 'custom-csv' || DATASET_OPTIONS.some(dataset => dataset.kind === value)
}
