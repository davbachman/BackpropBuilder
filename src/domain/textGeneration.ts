import { forwardPass, isLossNode } from './engine'
import { predictionNode } from './datasetTraining'
import { encodeText, decodeTokens } from './textData'
import { samplingDistribution } from '../learning/decoder'
import type { GraphModel } from './types'

/** Prompts are inference-only: evaluate a copy without Loss, never invent targets. */
export function predictText(graph: GraphModel, prompt: string) {
  return predictTokenIds(graph, prompt)
}

function predictTokenIds(graph: GraphModel, prompt: string | number[]) {
  const source = graph.nodes.find(node => node.type === 'dataset' && node.params.textData?.task === 'language')
  const data = source?.params.textData
  if (!source || !data) throw new Error('Import a next-token text dataset first.')
  const output = predictionNode(graph)
  if (!output) throw new Error('Connect vocabulary logits to a Loss block first.')
  const visited = new Set<string>()
  const checkInputs = (id: string) => {
    if (visited.has(id)) return
    visited.add(id)
    for (const edge of graph.edges.filter(edge => edge.target === id)) {
      if (edge.source === source.id && edge.sourceSlot === 2) throw new Error('Generation output must not depend on dataset targets.')
      checkInputs(edge.source)
    }
  }
  checkInputs(output.id)
  const allIds = typeof prompt === 'string' ? encodeText(prompt, data) : prompt
  if (!allIds.length) throw new Error('Enter a nonempty prompt.')
  let ids = allIds.slice(-data.maxLength)
  if (data.fixedLength && ids.length<data.maxLength) ids=[...Array(data.maxLength-ids.length).fill(1),...ids]
  const predictionGraph: GraphModel = { ...graph, nodes: graph.nodes.filter(node => !isLossNode(node)).map(node => node.id === source.id ? {...node, params: {...node.params, datasetValues: [{shape: [ids.length], data: ids}, {shape: [ids.length], data: ids.map((_, i) => i)}]}} : node), edges: graph.edges.filter(edge => !graph.nodes.some(node => isLossNode(node) && (node.id === edge.target || node.id === edge.source))) }
  const evaluated = forwardPass(predictionGraph, false).graph
  const value = evaluated.nodes.find(node => node.id === output.id)?.value
  if (!value || value.shape.length !== 2 || value.shape[0] !== (data.targetMode==='last'?1:ids.length) || value.shape[1] !== data.vocabulary.length || value.data.some(n => !Number.isFinite(n))) throw new Error('Generation needs logits shaped [context tokens, vocabulary size].')
  return { logits: value.data.slice(-data.vocabulary.length), ids, vocabulary: data.vocabulary, data }
}

export function* textGenerationSteps(graph: GraphModel, prompt: string, count: number, options: { temperature?: number; topK?: number; sample?: boolean; random?: () => number } = {}): Generator<string> {
  if (!Number.isInteger(count) || count < 1 || count > 256) throw new Error('Generate 1–256 tokens at a time.')
  let text = prompt
  let context: string | number[] = prompt
  for (let i = 0; i < count; i++) {
    const {logits, data, ids} = predictTokenIds(graph, context)
    const probabilities = samplingDistribution(logits, options.temperature ?? 1, options.topK ?? logits.length)
    let id = probabilities.indexOf(Math.max(...probabilities))
    if (options.sample) {
      let remaining = (options.random ?? Math.random)()
      id = probabilities.length - 1
      for (let j = 0; j < probabilities.length; j++) { remaining -= probabilities[j]; if (remaining < 0) {id = j; break} }
    }
    text += (data.tokenizer === 'word' ? ' ' : '') + decodeTokens([id], data)
    // Preserve IDs: decoding <unk> and tokenizing its spelling changes the context.
    context = [...ids, id]
    yield text
  }
}

export function generateText(graph: GraphModel, prompt: string, count: number, options: { temperature?: number; topK?: number; sample?: boolean; random?: () => number } = {}): string {
  let text = prompt
  for (const next of textGenerationSteps(graph, prompt, count, options)) text = next
  return text
}
