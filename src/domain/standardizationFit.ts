import type { GraphModel } from './types'
import { forwardPass } from './engine'
import { datasetExamplesForNode, datasetTargetSlotForNode } from './datasets'
import { isStandardizationStats, type StandardizationStats } from './standardization'

/** Fit a fixed preprocessing graph, never model parameters or held-out rows.
 * Dataset handles become small constant inputs so fitting does not repeatedly copy the full CSV.
 */
export async function fitStandardizer(graph: GraphModel, id: string): Promise<StandardizationStats> {
  const input = graph.edges.find(edge => edge.target === id)
  if (!input) throw Error('Connect numeric features before fitting.')
  const ancestors = new Set<string>()
  const visit = (key: string) => {
    if (ancestors.has(key)) return
    ancestors.add(key)
    graph.edges.filter(edge => edge.target === key).forEach(edge => visit(edge.source))
  }
  visit(input.source)
  const nodes = graph.nodes.filter(node => ancestors.has(node.id))
  const sources = nodes.filter(node => node.type === 'dataset')
  if (sources.length !== 1) throw Error('Connect features from exactly one dataset.')
  const allowed = ['dataset', 'input', 'arithmetic', 'add', 'multiply', 'concat', 'reshape', 'slice', 'transpose', 'tensor-transform', 'standardize']
  if (nodes.some(node => !allowed.includes(node.type))) throw Error('Fit only fixed input features, before trainable parameters or activations.')
  const source = sources[0], targetSlot = datasetTargetSlotForNode(source)
  const edges = graph.edges.filter(edge => ancestors.has(edge.source) && ancestors.has(edge.target))
  const uses = [...edges.filter(edge => edge.source === source.id), ...(input.source === source.id ? [input] : [])]
  if (uses.some(edge => (edge.sourceSlot ?? 0) === targetSlot)) throw Error('Standardize input features, not the target column.')
  const slots = [...new Set(uses.map(edge => edge.sourceSlot ?? 0))]
  const key = (slot: number) => `fit:${source.id}:${slot}`
  const output = input.source === source.id ? key(input.sourceSlot ?? 0) : input.source
  const prefix: GraphModel = { ...graph, groups: [], nodes: nodes.filter(node => node.id !== source.id), edges: edges.map(edge => edge.source === source.id ? { ...edge, source: key(edge.sourceSlot ?? 0), sourceSlot: 0 } : edge) }
  const rows = datasetExamplesForNode(source).filter(row => row.split === 'train')
  if (!rows.length) throw Error('The dataset needs training rows.')
  let count = 0, width = 0
  let mean: number[] = [], m2: number[] = []
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index]
    const constants = slots.map(slot => ({ id: key(slot), type: 'input' as const, label: key(slot), position: { x: 0, y: 0 }, params: { value: row.features[slot < targetSlot ? slot : slot - 1] } }))
    const value = forwardPass({ ...prefix, nodes: [...constants, ...prefix.nodes] }, false).graph.nodes.find(node => node.id === output)!.value!
    if (index === 0) { width = value.shape.at(-1) ?? 1; mean = Array(width).fill(0); m2 = Array(width).fill(0) }
    if ((value.shape.at(-1) ?? 1) !== width || value.data.some(x => !Number.isFinite(x))) throw Error('Features must have a fixed width and finite values.')
    for (let offset = 0; offset < value.data.length; offset += width) {
      count++
      for (let j = 0; j < width; j++) {
        const delta = value.data[offset + j] - mean[j]
        mean[j] += delta / count
        m2[j] += delta * (value.data[offset + j] - mean[j])
      }
    }
    if (index % 100 === 0) await new Promise(resolve => setTimeout(resolve, 0))
  }
  const stats = { mean, scale: m2.map(x => Math.sqrt(Math.max(0, x / count)) || 1), count }
  if (!isStandardizationStats(stats)) throw Error('Standardization supports up to 129 finite numeric features.')
  return stats
}
