/** Tiny pre-norm decoder. This is an actual trained local numerical model,
 * not a language model: its task is continuing the cycle red → green → blue.
 * Learned absolute positions, two causal heads per block, ReLU FFN, no dropout.
 * Vocabulary projection is untied; all named parameters keep stable identities. */
import type { TensorValue } from '../domain/types'
import { Tensor, tensor, add, matmul, scale, transpose, slice, concat, embedding, relu, softmax, causalMask, layerNorm, crossEntropy } from './math.ts'
import checkpointData from './decoder-checkpoints.json' with { type: 'json' }

export interface DecoderConfig { vocab: string[]; width: number; heads: number; blocks: number; ffWidth: number; maxLength: number }
export const DECODER_CONFIG: DecoderConfig = { vocab: ['<bos>', 'red', 'green', 'blue', '<eos>'], width: 8, heads: 2, blocks: 2, ffWidth: 16, maxLength: 12 }
export interface DecoderModel { config: DecoderConfig; parameters: Record<string, Tensor> }
export interface HeadTrace { q: TensorValue; k: TensorValue; v: TensorValue; scores: TensorValue; maskedScores: TensorValue; weights: TensorValue; output: TensorValue }
export interface BlockTrace {
  input: TensorValue; norm1: TensorValue; q: TensorValue; k: TensorValue; v: TensorValue; heads: HeadTrace[]
  attention: TensorValue; residual: TensorValue; norm2: TensorValue; ffPre: TensorValue; ffHidden: TensorValue; ffOut: TensorValue; output: TensorValue
}
export interface DecoderTrace { ids: number[]; embedding: TensorValue; position: TensorValue; input: TensorValue; blocks: BlockTrace[]; norm: TensorValue; logits: TensorValue; probabilities: TensorValue; logitsTensor: Tensor }
export interface ValueOverride { block: number; head: number; row: number; coordinate: number; value: number }
export interface CheckpointMetadata {
  name: string; seed: number; steps: number; learningRate: number; trainingLoss: number; validationLoss: number
  task: string; architecture: string; history: { step: number; trainingLoss: number; validationLoss: number }[]
}
export interface DecoderCheckpoint { version: 1; config: DecoderConfig; parameters: Record<string, TensorValue>; metadata: CheckpointMetadata }
export const CHECKPOINTS = checkpointData as unknown as Record<'trained' | 'untrained', DecoderCheckpoint>
export const getCheckpoint = (name: 'trained' | 'untrained') => CHECKPOINTS[name]

export function seededRandom(seed: number) {
  let state = seed >>> 0
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296 }
}
export function createDecoder(seed = 7): DecoderModel {
  const config = { ...DECODER_CONFIG, vocab: [...DECODER_CONFIG.vocab] }, random = seededRandom(seed), parameters: Record<string, Tensor> = {}
  const parameter = (name: string, shape: number[], magnitude: number, constant?: number) => {
    parameters[name] = tensor(shape, Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => constant ?? (random() * 2 - 1) * magnitude), true)
  }
  const norm = (name: string) => { parameter(`${name}.scale`, [config.width], 0, 1); parameter(`${name}.bias`, [config.width], 0, 0) }
  parameter('tokenEmbedding', [config.vocab.length, config.width], 0.35)
  parameter('positionEmbedding', [config.maxLength, config.width], 0.1)
  for (let i = 0; i < config.blocks; i++) {
    const prefix = `blocks.${i}`
    norm(`${prefix}.norm1`); norm(`${prefix}.norm2`)
    for (const name of ['q', 'k', 'v', 'o']) parameter(`${prefix}.${name}`, [config.width, config.width], 0.35)
    parameter(`${prefix}.ff1`, [config.width, config.ffWidth], 0.3)
    parameter(`${prefix}.ff1Bias`, [config.ffWidth], 0, 0)
    parameter(`${prefix}.ff2`, [config.ffWidth, config.width], 0.25)
    parameter(`${prefix}.ff2Bias`, [config.width], 0, 0)
  }
  norm('finalNorm')
  parameter('unembedding', [config.width, config.vocab.length], 0.35)
  parameter('outputBias', [config.vocab.length], 0, 0)
  return { config, parameters }
}

export function decoderForward(model: DecoderModel, ids: number[], options: { valueOverride?: ValueOverride } = {}): DecoderTrace {
  const { config: c, parameters: p } = model
  if (!ids.length || ids.length > c.maxLength) throw new Error(`Use between 1 and ${c.maxLength} tokens.`)
  const width = c.width / c.heads, blocks: BlockTrace[] = []
  const tokenEmbedding = embedding(p.tokenEmbedding, ids), position = embedding(p.positionEmbedding, ids.map((_, i) => i))
  let x = add(tokenEmbedding, position)
  const input = x.toValue()
  const norm = (value: Tensor, name: string) => layerNorm(value, p[`${name}.scale`], p[`${name}.bias`])
  for (let block = 0; block < c.blocks; block++) {
    const prefix = `blocks.${block}`, blockInput = x.toValue(), norm1 = norm(x, `${prefix}.norm1`)
    const q = matmul(norm1, p[`${prefix}.q`]), k = matmul(norm1, p[`${prefix}.k`]), v = matmul(norm1, p[`${prefix}.v`])
    const heads: HeadTrace[] = [], outputs: Tensor[] = []
    for (let head = 0; head < c.heads; head++) {
      const qh = slice(q, 1, head * width, (head + 1) * width), kh = slice(k, 1, head * width, (head + 1) * width)
      let vh = slice(v, 1, head * width, (head + 1) * width)
      const override = options.valueOverride
      if (override?.block === block && override.head === head) {
        if (!Number.isInteger(override.row) || override.row < 0 || override.row >= ids.length || !Number.isInteger(override.coordinate) || override.coordinate < 0 || override.coordinate >= width || !Number.isFinite(override.value)) throw new Error('Value intervention coordinate is out of range.')
        const delta = Array(vh.data.length).fill(0), index = override.row * width + override.coordinate
        delta[index] = override.value - vh.data[index]
        vh = add(vh, tensor(vh.shape, delta))
      }
      const scores = scale(matmul(qh, transpose(kh)), 1 / Math.sqrt(width)), maskedScores = causalMask(scores), weights = softmax(maskedScores), output = matmul(weights, vh)
      heads.push({ q: qh.toValue(), k: kh.toValue(), v: vh.toValue(), scores: scores.toValue(), maskedScores: maskedScores.toValue(), weights: weights.toValue(), output: output.toValue() })
      outputs.push(output)
    }
    const attention = matmul(concat(outputs, 1), p[`${prefix}.o`]), residual = add(x, attention), norm2 = norm(residual, `${prefix}.norm2`)
    const ffPre = add(matmul(norm2, p[`${prefix}.ff1`]), p[`${prefix}.ff1Bias`]), ffHidden = relu(ffPre)
    const ffOut = add(matmul(ffHidden, p[`${prefix}.ff2`]), p[`${prefix}.ff2Bias`])
    x = add(residual, ffOut)
    blocks.push({ input: blockInput, norm1: norm1.toValue(), q: q.toValue(), k: k.toValue(), v: v.toValue(), heads, attention: attention.toValue(), residual: residual.toValue(), norm2: norm2.toValue(), ffPre: ffPre.toValue(), ffHidden: ffHidden.toValue(), ffOut: ffOut.toValue(), output: x.toValue() })
  }
  const finalNorm = norm(x, 'finalNorm'), logits = add(matmul(finalNorm, p.unembedding), p.outputBias)
  return { ids: [...ids], embedding: tokenEmbedding.toValue(), position: position.toValue(), input, blocks, norm: finalNorm.toValue(), logits: logits.toValue(), probabilities: softmax(logits).toValue(), logitsTensor: logits }
}
export function exportCheckpoint(model: DecoderModel, metadata: CheckpointMetadata): DecoderCheckpoint {
  return { version: 1, config: { ...model.config, vocab: [...model.config.vocab] }, parameters: Object.fromEntries(Object.entries(model.parameters).map(([name, p]) => [name, p.toValue()])), metadata: structuredClone(metadata) }
}
export function loadCheckpoint(checkpoint: DecoderCheckpoint): DecoderModel {
  const expected = createDecoder(), config = checkpoint?.config
  if (checkpoint?.version !== 1 || !config || ['width', 'heads', 'blocks', 'ffWidth', 'maxLength'].some(key => config[key as keyof DecoderConfig] !== expected.config[key as keyof DecoderConfig]) || !Array.isArray(config.vocab) || config.vocab.join('\0') !== expected.config.vocab.join('\0')) throw new Error('Unsupported decoder checkpoint architecture.')
  if (Object.keys(checkpoint.parameters).length !== Object.keys(expected.parameters).length) throw new Error('Decoder checkpoint parameter count mismatch.')
  for (const [name, parameter] of Object.entries(expected.parameters)) {
    const value = checkpoint.parameters[name]
    if (!value || value.shape.join() !== parameter.shape.join()) throw new Error(`Checkpoint parameter ${name} is missing or has the wrong shape.`)
    expected.parameters[name] = tensor(value.shape, value.data, true)
  }
  return expected
}

/** The train and validation examples are distinct prefix lengths of the same
 * specified synthetic cycle. Validation checks this rule, not natural language
 * or out-of-distribution generalization. BOS predicts red; EOS is reserved and
 * never a training target. Prompts with EOS lie outside the training task. */
export function sequenceExamples(split: 'training' | 'validation'): { input: number[]; targets: number[] }[] {
  const examples: { input: number[]; targets: number[] }[] = []
  const lengths = split === 'training' ? [4, 7, 10] : [5, 8]
  for (const length of lengths) for (let phase = 0; phase < 3; phase++) {
    const sequence = Array.from({ length: length + 1 }, (_, i) => 1 + (i + phase) % 3)
    examples.push({ input: sequence.slice(0, -1), targets: sequence.slice(1) })
  }
  for (const length of lengths) {
    const sequence = [0, ...Array.from({ length }, (_, i) => 1 + i % 3)]
    examples.push({ input: sequence.slice(0, -1), targets: sequence.slice(1) })
  }
  return examples
}
export function decoderLoss(model: DecoderModel, split: 'training' | 'validation') {
  const examples = sequenceExamples(split)
  return examples.reduce((total, example) => total + crossEntropy(decoderForward(model, example.input).logitsTensor, example.targets).data[0], 0) / examples.length
}
export interface AdamState { step: number; first: Record<string, number[]>; second: Record<string, number[]> }
export const createAdamState = (): AdamState => ({ step: 0, first: {}, second: {} })
export function trainDecoderStep(model: DecoderModel, input: number[], targets: number[], state: AdamState, learningRate = 0.01) {
  Object.values(model.parameters).forEach(p => p.zeroGrad())
  const loss = crossEntropy(decoderForward(model, input).logitsTensor, targets)
  loss.backward(); state.step++
  let norm = 0
  for (const p of Object.values(model.parameters)) for (const g of p.grad) norm += g * g
  const clip = Math.min(1, 1 / (Math.sqrt(norm) + 1e-12))
  for (const [name, p] of Object.entries(model.parameters)) {
    const first = state.first[name] ??= p.data.map(() => 0), second = state.second[name] ??= p.data.map(() => 0)
    p.data.forEach((_, i) => {
      const g = p.grad[i] * clip
      first[i] = 0.9 * first[i] + 0.1 * g; second[i] = 0.999 * second[i] + 0.001 * g * g
      p.data[i] -= learningRate * (first[i] / (1 - 0.9 ** state.step)) / (Math.sqrt(second[i] / (1 - 0.999 ** state.step)) + 1e-8)
    })
  }
  return loss.data[0]
}
export function samplingDistribution(logits: number[], temperature = 1, topK = logits.length) {
  if (!logits.length || logits.some(n => !Number.isFinite(n)) || !Number.isFinite(temperature) || temperature <= 0 || !Number.isInteger(topK) || topK < 1) throw new Error('Sampling needs finite logits, positive temperature, and positive integer top-k.')
  const selected = logits.map((value, id) => ({ value, id })).sort((a, b) => b.value - a.value || a.id - b.id).slice(0, topK)
  const maximum = selected[0].value, probabilities = logits.map(() => 0)
  for (const { value, id } of selected) probabilities[id] = Math.exp((value - maximum) / temperature)
  const total = probabilities.reduce((a, b) => a + b, 0)
  return probabilities.map(v => v / total)
}
export function sampleToken(logits: number[], options: { mode: 'greedy' | 'sample'; temperature: number; topK: number; seed: number }) {
  if (!logits.length || logits.some(n => !Number.isFinite(n))) throw new Error('Token selection needs finite logits.')
  if (options.mode === 'greedy') {
    const token = logits.indexOf(Math.max(...logits))
    return { token, probabilities: logits.map((_, i) => i === token ? 1 : 0), seed: options.seed }
  }
  const probabilities = samplingDistribution(logits, options.temperature, options.topK), seed = (Math.imul(options.seed >>> 0, 1664525) + 1013904223) >>> 0
  const random = seed / 4294967296; let cumulative = 0, token = probabilities.length - 1
  for (let i = 0; i < probabilities.length; i++) { cumulative += probabilities[i]; if (random < cumulative) { token = i; break } }
  return { token, probabilities, seed }
}
