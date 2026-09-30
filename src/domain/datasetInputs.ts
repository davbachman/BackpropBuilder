import { datasetOutputLabelForSlot, datasetTargetSlotForNode } from './datasets'
import type { GraphEdge, GraphModel, GraphNode } from './types'

export interface DatasetInputPort { nodeId: string; sourceSlot: number }

/** Dataset columns that contribute to a loss prediction, excluding its targets. */
export function datasetInputPorts(graph: GraphModel): DatasetInputPort[] {
  const nodes = new Map(graph.nodes.map(node => [node.id, node]))
  const incoming = new Map<string, GraphEdge[]>()
  for (const edge of graph.edges) incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge])
  const losses = graph.nodes.filter(node => node.type === 'loss' || node.type === 'cross-entropy')
  const collect = (inputSlot: number) => {
    const ports = new Map<string, DatasetInputPort>()
    const visited = new Set<string>()
    const visit = (edge: GraphEdge) => {
      const node = nodes.get(edge.source)
      if (!node) return
      if (node.type === 'dataset') {
        const sourceSlot = edge.sourceSlot ?? 0
        ports.set(JSON.stringify([node.id, sourceSlot]), { nodeId: node.id, sourceSlot })
        return
      }
      if (visited.has(node.id)) return
      visited.add(node.id)
      for (const parent of incoming.get(node.id) ?? []) visit(parent)
    }
    for (const loss of losses) for (const edge of incoming.get(loss.id) ?? []) {
      if ((edge.inputSlot ?? 0) === inputSlot) visit(edge)
    }
    return ports
  }
  const targets = collect(1)
  return [...collect(0)].filter(([key, port]) => !targets.has(key) &&
    port.sourceSlot !== datasetTargetSlotForNode(nodes.get(port.nodeId)!)).map(([, port]) => port)
}

/** Add temporary input aliases for plotting; the user's graph is never changed. */
export function withDatasetInputAliases(graph: GraphModel): GraphModel {
  const nodes = [...graph.nodes]
  let edges = [...graph.edges]
  const ids = new Set([...nodes.map(node => node.id), ...edges.map(edge => edge.id)])
  let serial = 0
  const nextId = () => {
    let id: string
    do { id = `dataset-input-preview:${serial++}` } while (ids.has(id))
    ids.add(id)
    return id
  }
  for (const port of datasetInputPorts(graph)) {
    const source = graph.nodes.find(node => node.id === port.nodeId)!
    const alias: GraphNode = { id: nextId(), type: 'input', label: datasetOutputLabelForSlot(source, port.sourceSlot), position: source.position, params: {} }
    edges = edges.map(edge => edge.source === source.id && (edge.sourceSlot ?? 0) === port.sourceSlot
      ? { ...edge, source: alias.id, sourceSlot: 0 } : edge)
    edges.push({ id: nextId(), source: source.id, sourceSlot: port.sourceSlot, target: alias.id, inputSlot: 0 })
    nodes.push(alias)
  }
  return { ...graph, nodes, edges }
}
