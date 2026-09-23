import { describe, expect, it } from 'vitest'
import { initialExploration, parseExploration, validateExplorationState } from './exploration'
import type { ExplorationFile } from './exploration'
import { decoderForward, getCheckpoint, loadCheckpoint, sampleToken } from './decoder'
import { softmax, tensor } from './math'

function file(): ExplorationFile {
  const state = initialExploration('decoder')
  return { kind: 'backprop-exploration', version: 1, state, comparisons: [{ label: 'Initial', state: structuredClone(state), output: [1, 2, 3] }] }
}

describe('prepared exploration save/restore', () => {
  it('round-trips exact parameters, seeds, inputs, sampling controls and expanded module selections', () => {
    const saved = file(), parameter = [...getCheckpoint('trained').parameters['blocks.0.q'].data]
    parameter[3] += 0.25
    Object.assign(saved.state, { ids: [0, 1, 2, 1], overrides: { 'blocks.0.q': parameter }, seed: 419, mode: 'sample', temperature: 0.7, topK: 3, opened: ['block-0', 'block-0-attention', 'block-0-head-0'], selectedKeys: { 'block-0-head-0': 2 }, focus: 'block-0-head-0', token: 1, coordinate: 2, prediction: 'The second token follows the first.', showControls: false, zoom: 1.2 })
    const restored = parseExploration(JSON.stringify(saved))
    expect(restored).toEqual(saved)
    const model = loadCheckpoint(getCheckpoint(restored.state.checkpoint))
    for (const [key, data] of Object.entries(restored.state.overrides)) model.parameters[key].data = [...data]
    const before = decoderForward(model, saved.state.ids), after = decoderForward(model, restored.state.ids)
    expect(after.logits).toEqual(before.logits)
  })
  it('reproduces softmax after state restore and preserves the supplied preset fixtures', () => {
    const saved = file(); saved.state = initialExploration('probabilities'); saved.comparisons = []
    const restored = parseExploration(JSON.stringify(saved))
    expect(softmax(tensor([1, 3], restored.state.scores)).data).toEqual(softmax(tensor([1, 3], saved.state.scores)).data)
    const attention = initialExploration('attention')
    expect(attention.v).toEqual([2, 0, 0, 2, 2, 2]); expect(attention.q).toEqual([Math.SQRT2, 0])
    expect(initialExploration('causal').q).toHaveLength(6)
    for (const kind of ['probabilities', 'embeddings', 'attention', 'causal', 'block', 'decoder'] as const) expect(validateExplorationState(initialExploration(kind))).toBe(true)
  })
  it('migrates early version-one files that omitted key selections', () => {
    const saved = file()
    delete (saved.state as Partial<typeof saved.state>).selectedKeys
    delete (saved.comparisons[0].state as Partial<typeof saved.state>).selectedKeys
    const parsed = parseExploration(JSON.stringify(saved))
    expect(parsed.state.selectedKeys).toEqual({}); expect(parsed.comparisons[0].state.selectedKeys).toEqual({})
  })
  it('restores a chosen token and advanced random seed without sampling it again', () => {
    const saved = file(), model = loadCheckpoint(getCheckpoint('trained'))
    Object.assign(saved.state, { mode: 'sample', temperature: 2, topK: 5, seed: 73 })
    const logits = decoderForward(model, saved.state.ids).logits.data.slice(-5)
    const chosen = sampleToken(logits, { mode: 'sample', temperature: saved.state.temperature, topK: saved.state.topK, seed: saved.state.seed })
    saved.state.seed = chosen.seed
    saved.execution = { cursor: 3, generationPhase: 'chosen', choice: chosen.token, current: true }
    const restored = parseExploration(JSON.stringify(saved))
    expect(restored.execution).toEqual(saved.execution)
    expect(restored.state.seed).toBe(chosen.seed)
    const nextLogits = decoderForward(model, [...restored.state.ids, restored.execution!.choice!]).logits.data.slice(-5)
    const next = sampleToken(nextLogits, { mode: 'sample', temperature: restored.state.temperature, topK: restored.state.topK, seed: restored.state.seed })
    const uninterruptedLogits = decoderForward(model, [...saved.state.ids, chosen.token]).logits.data.slice(-5)
    expect(next).toEqual(sampleToken(uninterruptedLogits, { mode: 'sample', temperature: saved.state.temperature, topK: saved.state.topK, seed: chosen.seed }))
  })
  it('supports optional execution state and each legal forward/generation phase', () => {
    expect(parseExploration(JSON.stringify(file())).execution).toBeUndefined()
    for (const execution of [
      { cursor: -1, generationPhase: 'scores', choice: null, current: false },
      { cursor: 1, generationPhase: 'distribution', choice: null, current: true },
      { cursor: 3, generationPhase: 'chosen', choice: 4, current: true },
      { cursor: -1, generationPhase: 'appended', choice: null, current: false },
    ] as const) expect(parseExploration(JSON.stringify({ ...file(), execution })).execution).toEqual(execution)
  })
  it('rejects malformed execution fields before restoring buttons or a chosen token', () => {
    const execution = { cursor: 3, generationPhase: 'chosen', choice: 2, current: true }
    for (const invalid of [null, [], {}, { ...execution, cursor: 4 }, { ...execution, cursor: -2 }, { ...execution, cursor: 1.5 }, { ...execution, generationPhase: 'training' }, { ...execution, choice: 5 }, { ...execution, choice: -1 }, { ...execution, choice: null }, { ...execution, current: 'true' }, { ...execution, current: false }]) {
      expect(() => parseExploration(JSON.stringify({ ...file(), execution: invalid }))).toThrow(/valid saved exploration/)
    }
  })
  it('rejects malformed files and legacy graph files with a useful message', () => {
    for (const text of ['', 'null', '{}', '{', '[]', '{"kind":"backprop-project","version":1}', JSON.stringify({ ...file(), version: 99 }), JSON.stringify({ ...file(), comparisons: [null] })]) expect(() => parseExploration(text)).toThrow(/valid saved exploration.*Legacy graph/)
  })
  it('keeps saved comparisons within the same architecture and lesson', () => {
    const saved = file()
    saved.comparisons[0].state = initialExploration('attention')
    expect(() => parseExploration(JSON.stringify(saved))).toThrow(/valid saved/)
  })
  it('rejects malformed parameters in the current state and in saved comparisons', () => {
    for (const invalid of [{ 'unknown-key': [1] }, { 'blocks.0.q': [1] }, [], { 'blocks.0.q': Array(64).fill(null) }]) {
      const saved = file(); saved.state.overrides = invalid as unknown as typeof saved.state.overrides
      expect(() => parseExploration(JSON.stringify(saved))).toThrow(/valid saved/)
      const comparison = file(); comparison.comparisons[0].state.overrides = invalid as unknown as typeof comparison.state.overrides
      expect(() => parseExploration(JSON.stringify(comparison))).toThrow(/valid saved/)
    }
  })
  it('bounds inputs, selected coordinates, key selections and sampling controls before rendering', () => {
    for (const invalid of [{ ids: [] }, { ids: Array(13).fill(0) }, { ids: [6] }, { q: [1] }, { v: [1, 2] }, { selectedKeys: { attention: 12 } }, { selectedKeys: [] }, { coordinate: 8 }, { neuron: 16 }, { head: 2 }, { seed: 4294967296 }, { temperature: 0 }, { topK: 0 }, { kind: 'playground' }, { zoom: 2 }]) {
      const saved = file(); Object.assign(saved.state, invalid)
      expect(() => parseExploration(JSON.stringify(saved))).toThrow(/valid saved/)
    }
  })
})
