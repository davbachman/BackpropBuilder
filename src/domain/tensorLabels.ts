import { datasetOutputValueForSlot } from './datasets'
import type { GraphModel } from './types'

/** Labels follow graph connections, not names or coincidentally matching sizes. */
export function tensorAxisLabels(graph: GraphModel, id: string, slot = 0, visited = new Set<string>()): Array<string[] | undefined> {
  if (visited.has(id)) return []
  const seen = new Set(visited).add(id)
  const node = graph.nodes.find(node => node.id === id)
  if (!node) return []
  const edges = graph.edges.filter(edge => edge.target === id).sort((a,b) => (a.inputSlot ?? 0) - (b.inputSlot ?? 0))
  const labels = (index: number) => edges[index] ? tensorAxisLabels(graph, edges[index].source, edges[index].sourceSlot ?? 0, seen) : []
  if (node.type === 'dataset' && node.params.textData) {
    const data = node.params.textData
    if (slot === 0 && data.representation === 'counts') return [undefined, data.vocabulary]
    if (slot < 2 && data.representation === 'tokens') return [datasetOutputValueForSlot(node, 0).data.map((token, position) => position + ': ' + JSON.stringify(data.vocabulary[token]))]
    return []
  }
  if (node.type === 'weight' || node.type === 'bias') {
    for (const use of graph.edges.filter(edge => edge.source === id)) {
      const consumer = graph.nodes.find(node => node.id === use.target)
      if (consumer?.type === 'embedding' && (use.inputSlot ?? 0) === 0) {
        const ids = graph.edges.find(edge => edge.target === consumer.id && edge.inputSlot === 1)
        const source = graph.nodes.find(node => node.id === ids?.source)
        if (source?.params.textData) return [(ids?.sourceSlot ?? 0) === 0 ? source.params.textData.vocabulary : Array.from({length: source.params.textData.maxLength}, (_, i) => 'position ' + i)]
      }
      if (consumer?.type === 'matmul' && use.inputSlot === 1) {
        const input = graph.edges.find(edge => edge.target === consumer.id && (edge.inputSlot ?? 0) === 0)
        if (input) return [tensorAxisLabels(graph, input.source, input.sourceSlot ?? 0, seen)[1]]
      }
    }
    return []
  }
  const kind = node.type === 'tensor-transform' ? node.params.transform : node.type
  if (kind === 'embedding') return [labels(1)[0], undefined]
  if (kind === 'one-hot') {
    const dataset = graph.nodes.find(node => node.params.textData)
    return [labels(0)[0], dataset?.params.textData?.vocabulary]
  }
  if (kind === 'matmul') return [labels(0)[0], labels(1)[1]]
  if (kind === 'transpose') { const input = labels(0); return (node.params.axes ?? [1, 0]).map(axis => input[axis]) }
  if (kind === 'mean') {
    const input = [...labels(0)]
    if (node.params.axis === undefined) return []
    if (node.params.keepDims) input[node.params.axis] = undefined
    else input.splice(node.params.axis, 1)
    return input
  }
  if (kind === 'slice') { const input = [...labels(0)]; const axis = node.params.axis ?? 0; input[axis] = input[axis]?.slice(node.params.start ?? 0, node.params.end); return input }
  if (['standardize', 'dropout', 'softmax', 'causal-mask', 'layer-norm', 'activation', 'input', 'arithmetic', 'add', 'multiply'].includes(kind ?? '')) return labels(0)
  return []
}
