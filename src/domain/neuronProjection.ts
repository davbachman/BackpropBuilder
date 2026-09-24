import { scalarValue, tensorValue, toTensor } from './tensor'
import { layoutConnections } from './layoutState'
import type { GraphGroup, GraphModel, GraphNode, GraphViewState, TensorValue, EvaluationTraceStep } from './types'

export interface CoordinateBinding { nodeId: string; index: number; editable: boolean }
export interface NeuronProjection { graph: GraphModel; bindings: Record<string, CoordinateBinding> }

/** Expand a matrix layer into its neurons without changing the executable graph.
 * Every displayed number and parameter coordinate belongs to the current run. */
export function projectDenseNeurons(graph: GraphModel, focus = graph.view?.inspectedNeuron): NeuronProjection {
  const bindings: Record<string, CoordinateBinding> = {}
  const layer = graph.groups?.find(group => group.id === focus?.groupId)
  if (!focus || !layer?.detail) return { graph, bindings }
  const { inputNodeId, weightNodeId, biasNodeId, preNodeId, outputNodeId } = layer.detail
  const input = graph.nodes.find(node => node.id === inputNodeId)
  const weight = graph.nodes.find(node => node.id === weightNodeId)
  const bias = graph.nodes.find(node => node.id === biasNodeId)
  const pre = graph.nodes.find(node => node.id === preNodeId)
  const output = graph.nodes.find(node => node.id === outputNodeId)
  if (!input || !weight || !bias || !pre || !output) return { graph, bindings }
  const matrixProduct = graph.nodes.find(node => node.type === 'matmul' && graph.edges.some(edge => edge.source === node.id && edge.target === pre.id))
  const canonicalEdge = (sourceId: string | undefined, targetId: string | undefined) => graph.edges.find(edge => edge.source === sourceId && edge.target === targetId)
  const weights = toTensor(weight.params.value), width = weights.shape[1], inputs = weights.shape[0]
  if (!width || !inputs) return { graph, bindings }
  const inputValue = input.value, rows = inputValue?.shape[0] ?? 1
  const row = Math.max(0, Math.min(rows - 1, focus.row)), unit = Math.max(0, Math.min(width - 1, focus.unitIndex))
  const nodes: GraphNode[] = [], groups: GraphGroup[] = [], edges: GraphModel['edges'] = []
  const originalIds = new Set(layer.nodeIds)
  const wire = (source: string, target: string, inputSlot = 0, value?: TensorValue, grad?: TensorValue) => edges.push({ id: `inspect:${source}:${target}:${inputSlot}`, source, target, inputSlot, value, grad })
  const coordinate = (value: TensorValue | undefined, index: number) => value ? scalarValue(value.data[index]) : undefined
  const make = (id: string, type: GraphNode['type'], label: string, value?: TensorValue, grad?: TensorValue): GraphNode => {
    const node: GraphNode = { id, type, label, position: { x: 0, y: 0 }, params: {}, value, grad }
    nodes.push(node)
    return node
  }
  for (let n = 0; n < width; n++) {
    const prefix = `inspect:${layer.id}:${n}`, start = nodes.length, offset = row * width + n
    const activation = make(`${prefix}:output`, 'activation', `Neuron ${n + 1}`, coordinate(output.value, offset), coordinate(output.grad, offset))
    activation.params.activation = output.params.activation
    activation.localDerivative = coordinate(output.localDerivative, offset)
    const productGradient = coordinate(matrixProduct?.grad, offset)
    const preGradient = coordinate(pre.grad, offset)
    if (n === unit) {
      const products: string[] = []
      for (let i = 0; i < inputs; i++) {
        const xIndex = row * inputs + i, wIndex = i * width + n
        const x = make(`${prefix}:x${i}`, 'input', `x${i + 1} · token ${row + 1}`, coordinate(inputValue, xIndex), coordinate(input.grad, xIndex))
        bindings[x.id] = { nodeId: input.id, index: xIndex, editable: ['weight', 'bias'].includes(input.type) || (input.type === 'input' && !graph.edges.some(edge => edge.target === input.id)) }
        const w = make(`${prefix}:w${i}`, 'weight', `w${i + 1}`, scalarValue(weights.data[wIndex]), coordinate(weight.grad, wIndex))
        w.params.value = w.value
        bindings[w.id] = { nodeId: weight.id, index: wIndex, editable: true }
        const product = make(`${prefix}:product${i}`, 'multiply', `x${i + 1} × w${i + 1}`, matrixProduct?.value && x.value ? scalarValue(x.value.data[0] * weights.data[wIndex]) : undefined, productGradient)
        const xGradient = canonicalEdge(input.id, matrixProduct?.id)?.grad && productGradient ? scalarValue(productGradient.data[0] * weights.data[wIndex]) : undefined
        const wGradient = canonicalEdge(weight.id, matrixProduct?.id)?.grad && productGradient && x.value ? scalarValue(productGradient.data[0] * x.value.data[0]) : undefined
        wire(input.id, x.id, 0, x.value, xGradient)
        wire(x.id, product.id, 0, x.value, xGradient)
        wire(w.id, product.id, 1, w.value, wGradient)
        products.push(product.id)
      }
      const biasIndex = n
      const b = make(`${prefix}:bias`, 'bias', 'Bias', coordinate(toTensor(bias.params.value), biasIndex), coordinate(bias.grad, biasIndex))
      b.params.value = b.value
      bindings[b.id] = { nodeId: bias.id, index: biasIndex, editable: true }
      const sum = make(`${prefix}:sum`, 'add', 'Weighted sum + bias', coordinate(pre.value, offset), coordinate(pre.grad, offset))
      sum.params.inputCount = inputs + 1
      products.forEach((productId, index) => wire(productId, sum.id, index, nodes.find(node => node.id === productId)?.value, canonicalEdge(matrixProduct?.id, pre.id)?.grad ? preGradient : undefined))
      wire(b.id, sum.id, inputs, b.value, canonicalEdge(bias.id, pre.id)?.grad ? preGradient : undefined)
      wire(sum.id, activation.id, 0, sum.value, coordinate(canonicalEdge(pre.id, output.id)?.grad, offset))
    } else {
      const rowValue = inputValue ? tensorValue([inputs], inputValue.data.slice(row * inputs, (row + 1) * inputs)) : undefined
      const rowGradient = canonicalEdge(input.id, matrixProduct?.id)?.grad && productGradient ? tensorValue([inputs], Array.from({ length: inputs }, (_, i) => productGradient.data[0] * weights.data[i * width + n])) : undefined
      wire(input.id, activation.id, 0, rowValue, rowGradient)
    }
    wire(activation.id, output.id, n, activation.value, activation.grad)
    groups.push({ id: prefix, parentId: layer.id, label: `Neuron ${n + 1}${n === unit ? ` · token ${row + 1}` : ''}`, kind: 'neuron', detail: { virtual: true, layerId: layer.id, unitIndex: n }, nodeIds: nodes.slice(start).map(node => node.id), position: { x: 0, y: n * 200 }, dimensions: { width: 210, height: 168 } })
  }
  const ids = nodes.map(node => node.id)
  const nextGroups = graph.groups!.map(group => {
    if (group.id !== layer.id && !group.nodeIds.some(id => originalIds.has(id))) return group
    return { ...group, nodeIds: [...group.nodeIds.filter(id => !originalIds.has(id) || id === output.id), ...ids] }
  })
  const view: GraphViewState = { ...graph.view, expandedGroupIds: [...new Set([...(graph.view?.expandedGroupIds ?? []), layer.id, `inspect:${layer.id}:${unit}`])] }
  const outsideLayer = (edge: GraphModel['edges'][number]) => !originalIds.has(edge.target) && (!originalIds.has(edge.source) || edge.source === output.id)
  if (view.layoutEdges) view.layoutEdges = [...view.layoutEdges.filter(outsideLayer), ...layoutConnections(edges)]
  return { bindings, graph: { ...graph,
    nodes: [...graph.nodes.filter(node => !originalIds.has(node.id) || node.id === output.id).map(node => node.id === output.id ? { ...node, label: 'Layer output · all neurons' } : node), ...nodes],
    edges: [...graph.edges.filter(outsideLayer), ...edges],
    groups: [...nextGroups, ...groups], view,
  } }
}


/** Project one matrix operation's trace onto the visible scalar connections.
 * The executable step remains unchanged; this only exposes the same work at
 * the currently inspected spatial scale. */
export function activeProjectionEdges(
  graph: GraphModel,
  focus: GraphViewState['inspectedNeuron'],
  step: EvaluationTraceStep | undefined,
): string[] {
  if (!focus || !step?.nodeId || (step.phase !== 'forward' && step.phase !== 'backward')) return []
  const layer = graph.groups?.find(group => group.id === focus.groupId)
  if (!layer?.detail) return []
  const { inputNodeId, weightNodeId, preNodeId, outputNodeId } = layer.detail
  const weight = graph.nodes.find(node => node.id === weightNodeId)
  const width = weight ? toTensor(weight.params.value).shape[1] : 0
  if (!width) return []
  const unit = Math.max(0, Math.min(width - 1, focus.unitIndex))
  const prefix = `inspect:${layer.id}:${unit}`
  const isProduct = graph.nodes.some(node => node.id === step.nodeId && node.type === 'matmul' && graph.edges.some(edge => edge.source === node.id && edge.target === preNodeId))
  const projected = projectDenseNeurons(graph, focus).graph
  return projected.edges.filter(edge => {
    if (!edge.id.startsWith('inspect:')) return false
    if (isProduct) {
      return (edge.target.startsWith(`${prefix}:product`) || edge.target.startsWith(`${prefix}:x`)) ||
        (edge.source === inputNodeId && edge.target.startsWith(`inspect:${layer.id}:`) && edge.target.endsWith(':output'))
    }
    if (step.nodeId === preNodeId) return edge.target === `${prefix}:sum`
    if (step.nodeId === outputNodeId) return (edge.source === `${prefix}:sum` && edge.target === `${prefix}:output`) || (edge.target === outputNodeId && edge.source.startsWith(`inspect:${layer.id}:`))
    return false
  }).map(edge => edge.id)
}
