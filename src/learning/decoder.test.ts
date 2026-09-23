import { describe, expect, it } from 'vitest'
import { createDecoder, decoderForward, decoderLoss, loadCheckpoint, getCheckpoint, exportCheckpoint, sampleToken, samplingDistribution, createAdamState, trainDecoderStep, sequenceExamples, seededRandom } from './decoder'
import type { DecoderModel } from './decoder'
import { tensor, matmul, softmax, crossEntropy } from './math'

// Independent reference: nested ordinary arrays and hand-written arithmetic.
// It imports neither tensor operations nor production decoder internals.
function referenceForward(model: DecoderModel, ids: number[]) {
  type Matrix = number[][]
  const p = (key: string): Matrix => {
    const value = model.parameters[key]
    return value.shape.length === 1 ? [value.data] : Array.from({ length: value.shape[0] }, (_, i) => value.data.slice(i * value.shape[1], (i + 1) * value.shape[1]))
  }
  const mm = (a: Matrix, b: Matrix) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
  const plus = (a: Matrix, b: Matrix) => a.map((row, i) => row.map((v, j) => v + b[i][j]))
  const bias = (a: Matrix, b: number[]) => a.map(row => row.map((v, j) => v + b[j]))
  const ln = (a: Matrix, key: string) => a.map(row => {
    const avg = row.reduce((s, v) => s + v, 0) / row.length, variance = row.reduce((s, v) => s + (v - avg) ** 2, 0) / row.length
    return row.map((v, j) => (v - avg) / Math.sqrt(variance + 1e-5) * p(`${key}.scale`)[0][j] + p(`${key}.bias`)[0][j])
  })
  const probability = (row: number[]) => { const max = Math.max(...row), exps = row.map(v => Math.exp(v - max)), total = exps.reduce((s, v) => s + v, 0); return exps.map(v => v / total) }
  let x = ids.map((id, i) => p('tokenEmbedding')[id].map((v, j) => v + p('positionEmbedding')[i][j]))
  const blocks: Matrix[] = [], attentions: Matrix[][] = []
  const headWidth = model.config.width / model.config.heads
  for (let block = 0; block < model.config.blocks; block++) {
    const key = `blocks.${block}`, n = ln(x, `${key}.norm1`), q = mm(n, p(`${key}.q`)), k = mm(n, p(`${key}.k`)), v = mm(n, p(`${key}.v`))
    const headOutputs: Matrix[] = [], headWeights: Matrix[] = []
    for (let head = 0; head < model.config.heads; head++) {
      const start = head * headWidth, weights = q.map((row, i) => {
        const scores = k.slice(0, i + 1).map(keyRow => row.slice(start, start + headWidth).reduce((s, value, d) => s + value * keyRow[start + d], 0) / Math.sqrt(headWidth))
        return [...probability(scores), ...Array(ids.length - i - 1).fill(0)]
      })
      headWeights.push(weights)
      headOutputs.push(weights.map(row => Array.from({ length: headWidth }, (_, d) => row.reduce((s, w, j) => s + w * v[j][start + d], 0))))
    }
    const combined = x.map((_, row) => headOutputs.flatMap(output => output[row]))
    x = plus(x, mm(combined, p(`${key}.o`)))
    const normalized = ln(x, `${key}.norm2`), hidden = bias(mm(normalized, p(`${key}.ff1`)), p(`${key}.ff1Bias`)[0]).map(row => row.map(v => Math.max(0, v)))
    x = plus(x, bias(mm(hidden, p(`${key}.ff2`)), p(`${key}.ff2Bias`)[0]))
    blocks.push(x); attentions.push(headWeights)
  }
  const logits = bias(mm(ln(x, 'finalNorm'), p('unembedding')), p('outputBias')[0])
  return { logits, probabilities: logits.map(probability), blocks, attentions }
}
const expectClose = (a: number[], b: number[]) => { expect(a.length).toBe(b.length); a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 10)) }

describe('actual miniature decoder', () => {
  it('matches an independent array reference at every block, head, logit and probability', () => {
    for (const checkpoint of ['untrained', 'trained'] as const) {
      const model = loadCheckpoint(getCheckpoint(checkpoint)), ids = [0, 1, 2, 3, 1], actual = decoderForward(model, ids), reference = referenceForward(model, ids)
      expectClose(actual.logits.data, reference.logits.flat()); expectClose(actual.probabilities.data, reference.probabilities.flat())
      actual.blocks.forEach((block, i) => {
        expectClose(block.output.data, reference.blocks[i].flat())
        block.heads.forEach((head, j) => expectClose(head.weights.data, reference.attentions[i][j].flat()))
      })
    }
  })
  it('leaves earlier causal outputs unchanged when future tokens change', () => {
    const model = loadCheckpoint(getCheckpoint('trained')), first = decoderForward(model, [0, 1, 2, 3]), second = decoderForward(model, [0, 1, 3, 2])
    expect(first.logits.data.slice(0, 10)).toEqual(second.logits.data.slice(0, 10))
    first.blocks.forEach((block, i) => expect(block.output.data.slice(0, 16)).toEqual(second.blocks[i].output.data.slice(0, 16)))
    for (const block of first.blocks) for (const head of block.heads) for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) expect(head.weights.data[i * 4 + j]).toBe(0)
  })
  it('supports changing V while exactly holding the selected head Q/K and weights fixed', () => {
    const model = loadCheckpoint(getCheckpoint('untrained')), base = decoderForward(model, [0, 1, 2]), changed = decoderForward(model, [0, 1, 2], { valueOverride: { block: 0, head: 0, row: 1, coordinate: 0, value: 4 } })
    for (const key of ['q', 'k', 'weights'] as const) expect(changed.blocks[0].heads[0][key]).toEqual(base.blocks[0].heads[0][key])
    expect(changed.blocks[0].heads[0].output).not.toEqual(base.blocks[0].heads[0].output)
    const weights = softmax(tensor([1, 3], [0, Math.log(2), 0]))
    expect(matmul(weights, tensor([3, 2], [2, 0, 0, 2, 2, 2])).data).toEqual([1, 1.5])
    expect(matmul(weights, tensor([3, 2], [2, 0, 0, 4, 2, 2])).data).toEqual([1, 2.5])
  })
  it('finite-difference checks gradients through a complete decoder with shared token embeddings', () => {
    const model = createDecoder(7), ids = [1, 2, 1], targets = [2, 3, 2]
    crossEntropy(decoderForward(model, ids).logitsTensor, targets).backward()
    for (const [key, index] of [['tokenEmbedding', 8], ['positionEmbedding', 3], ['blocks.0.q', 5], ['blocks.1.v', 12], ['blocks.0.norm1.scale', 2], ['blocks.1.ff1', 33], ['blocks.1.ff2', 15], ['unembedding', 3]] as const) {
      const parameter = model.parameters[key], value = parameter.data[index], analytic = parameter.grad[index], epsilon = 1e-5
      parameter.data[index] = value + epsilon; const plus = crossEntropy(decoderForward(model, ids).logitsTensor, targets).data[0]
      parameter.data[index] = value - epsilon; const minus = crossEntropy(decoderForward(model, ids).logitsTensor, targets).data[0]
      parameter.data[index] = value
      expect(Math.abs(analytic - (plus - minus) / (2 * epsilon))).toBeLessThan(2e-5)
    }
  })
  it('keeps inference and sampling immutable, and training updates parameters', () => {
    const model = loadCheckpoint(getCheckpoint('trained')), before = JSON.stringify(exportCheckpoint(model, getCheckpoint('trained').metadata))
    const trace = decoderForward(model, [0, 1, 2]), logits = trace.logits.data.slice(-5)
    const first = sampleToken(logits, { mode: 'sample', temperature: 0.7, topK: 3, seed: 19 })
    expect(sampleToken(logits, { mode: 'sample', temperature: 0.7, topK: 3, seed: 19 })).toEqual(first)
    sampleToken(logits, { mode: 'greedy', temperature: 0, topK: 1, seed: 19 })
    expect(JSON.stringify(exportCheckpoint(model, getCheckpoint('trained').metadata))).toBe(before)
    expect(decoderForward(model, [0, 1, 2]).logits).toEqual(trace.logits)
    trainDecoderStep(model, [0, 1, 2], [1, 2, 3], createAdamState())
    expect(JSON.stringify(exportCheckpoint(model, getCheckpoint('trained').metadata))).not.toBe(before)
  })
  it('temperature/top-k change only the sampling distribution; greedy needs no positive temperature', () => {
    expect(samplingDistribution([1, 2, 3], 1, 1)).toEqual([0, 0, 1])
    expect(samplingDistribution([1, 2, 3], 0.2, 3)[2]).toBeGreaterThan(samplingDistribution([1, 2, 3], 2, 3)[2])
    expect(sampleToken([1, 3, 2], { mode: 'greedy', temperature: 0, topK: 1, seed: 3 }).token).toBe(1)
    expect(() => samplingDistribution([1, 2], 0)).toThrow(/positive/)
  })
  it('reproduces the checkpoint exactly from the fixed seed, dataset and update budget', () => {
    const trained = getCheckpoint('trained'), model = createDecoder(trained.metadata.seed), state = createAdamState(), examples = sequenceExamples('training'), random = seededRandom(trained.metadata.seed + 1)
    for (let step = 0; step < trained.metadata.steps; step++) {
      const example = examples[Math.floor(random() * examples.length)]
      trainDecoderStep(model, example.input, example.targets, state, trained.metadata.learningRate)
    }
    for (const [name, parameter] of Object.entries(model.parameters)) expect(parameter.data).toEqual(trained.parameters[name].data)
    expect(decoderLoss(model, 'validation')).toBeLessThan(0.01)
    expect(decoderLoss(model, 'validation')).toBeCloseTo(trained.metadata.validationLoss, 12)
    const restored = loadCheckpoint(JSON.parse(JSON.stringify(trained)))
    expect(decoderForward(restored, [0, 1, 2]).logits).toEqual(decoderForward(model, [0, 1, 2]).logits)
    expect(sampleToken(decoderForward(restored, [0, 1, 2]).logits.data.slice(-5), { mode: 'greedy', temperature: 1, topK: 5, seed: 7 }).token).toBe(3)
  })
  it('rejects invalid checkpoint shapes and out-of-budget inputs', () => {
    const checkpoint = structuredClone(getCheckpoint('trained')); checkpoint.parameters['blocks.0.q'].shape = [4, 16]
    expect(() => loadCheckpoint(checkpoint)).toThrow(/wrong shape/)
    expect(() => decoderForward(createDecoder(), Array(13).fill(1))).toThrow(/12/)
  })
})
