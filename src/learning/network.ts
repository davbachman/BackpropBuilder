import type { GraphModel } from '../domain/types'
import { add, crossEntropy, matmul, mean, mul, relu, softmax, sub, tensor } from './math'

export type NetworkKind = 'linear' | 'neuron' | 'small-network' | 'playground'
export type NetworkActivation = 'identity' | 'relu'
export interface Example { id: string; x: number[]; y: number; split: 'train' | 'validation' }
export interface DenseLayer { id: string; inputs: number; outputs: number; weights: number[]; biases: number[]; activation: NetworkActivation }
export interface NetworkModel { kind: NetworkKind; seed: number; dataSeed: number; layers: DenseLayer[]; learningRate: number }
export interface LayerCalculation { id: string; input: number[]; weightedSum: number[]; output: number[] }
export interface NetworkCalculation { layers: LayerCalculation[]; output: number[]; prediction: number; probabilities: number[] }
export interface NetworkGradient { id: string; weights: number[]; biases: number[] }
export interface LossPoint { epoch: number; train: number; validation: number }

export function seededRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => { state += 0x6d2b79f5; let t = Math.imul(state ^ state >>> 15, 1 | state); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296 }
}

export function createNetwork(kind: NetworkKind, seed = 42, hidden = [4, 3]): NetworkModel {
  if (kind === 'linear' || kind === 'neuron') return { kind, seed, dataSeed: 137, learningRate: .05, layers: [{ id: 'layer-0', inputs: 1, outputs: 1, weights: [3], biases: [1], activation: 'identity' }] }
  if (kind === 'small-network') return { kind, seed, dataSeed: 137, learningRate: .03, layers: [
    { id: 'layer-0', inputs: 2, outputs: 2, weights: [.5, -.5, 1, 1], biases: [0, 0], activation: 'relu' },
    { id: 'layer-1', inputs: 2, outputs: 1, weights: [.5, -.5], biases: [.5], activation: 'identity' },
  ] }
  const random = seededRandom(seed)
  const widths = [2, ...hidden, 2]
  return { kind, seed, dataSeed: 137, learningRate: .1, layers: widths.slice(1).map((outputs, index) => ({
    id: `layer-${index}`, inputs: widths[index], outputs,
    weights: Array.from({ length: widths[index] * outputs }, () => (random() * 2 - 1) * Math.sqrt(2 / widths[index])),
    biases: Array.from({ length: outputs }, () => .05), activation: index === widths.length - 2 ? 'identity' : 'relu',
  })) }
}

/** Fixed, independently labeled points. Validation is held out of all updates. */
export function createNetworkData(kind: NetworkKind, seed = 137): Example[] {
  if (kind === 'linear' || kind === 'neuron') return Array.from({ length: 13 }, (_, i) => {
    const x = -2 + i / 3
    return { id: `point-${i}`, x: [x], y: 4 * x + 1, split: i % 3 === 1 ? 'validation' : 'train' }
  })
  const random = seededRandom(seed)
  return Array.from({ length: 96 }, (_, i) => {
    const x = [random() * 2 - 1, random() * 2 - 1]
    return { id: `point-${i}`, x, y: kind === 'playground' ? Number(x[0] * x[1] > 0) : Math.max(0, x[0] + x[1]), split: i % 4 === 0 ? 'validation' : 'train' }
  })
}

function execute(model: NetworkModel, inputs: number[][], trainable = false) {
  let value = tensor([inputs.length, model.layers[0].inputs], inputs.flat())
  const parameters = model.layers.map(layer => ({ weights: tensor([layer.inputs, layer.outputs], layer.weights, trainable), biases: tensor([layer.outputs], layer.biases, trainable) }))
  const activations = model.layers.map((layer, index) => {
    const input = value
    const weightedSum = add(matmul(value, parameters[index].weights), parameters[index].biases)
    value = layer.activation === 'relu' ? relu(weightedSum) : weightedSum
    return { input, weightedSum, output: value }
  })
  return { value, activations, parameters }
}

export function runNetwork(model: NetworkModel, x: number[]): NetworkCalculation {
  const result = execute(model, [x])
  const probabilities = model.kind === 'playground' ? softmax(result.value).data : []
  return {
    layers: result.activations.map((layer, i) => ({ id: model.layers[i].id, input: [...layer.input.data], weightedSum: [...layer.weightedSum.data], output: [...layer.output.data] })),
    output: [...result.value.data], prediction: probabilities.length ? (probabilities[1] >= .5 ? 1 : 0) : result.value.data[0], probabilities: [...probabilities],
  }
}

export function networkLoss(model: NetworkModel, examples: Example[], backward = false): { loss: number; gradients: NetworkGradient[] } {
  if (!examples.length) throw new Error('Choose at least one example.')
  const result = execute(model, examples.map(point => point.x), backward)
  const difference = model.kind === 'playground' ? null : sub(result.value, tensor([examples.length, 1], examples.map(point => point.y)))
  const loss = difference ? mean(mul(difference, difference)) : crossEntropy(result.value, examples.map(point => point.y))
  if (backward) loss.backward()
  return { loss: loss.data[0], gradients: result.parameters.map((parameter, i) => ({ id: model.layers[i].id, weights: [...parameter.weights.grad], biases: [...parameter.biases.grad] })) }
}

export function applyNetworkGradients(model: NetworkModel, gradients: NetworkGradient[]): NetworkModel {
  const layers = model.layers.map((layer, i) => ({ ...layer,
    weights: layer.weights.map((weight, j) => weight - model.learningRate * gradients[i].weights[j]),
    biases: layer.biases.map((bias, j) => bias - model.learningRate * gradients[i].biases[j]),
  }))
  if (layers.some(layer => [...layer.weights, ...layer.biases].some(value => !Number.isFinite(value)))) throw new Error('The update diverged. Reset parameters and lower the learning rate.')
  return { ...model, layers }
}

export function trainNetwork(model: NetworkModel, data: Example[], epochs: number): { model: NetworkModel; losses: LossPoint[] } {
  let next = model
  const train = data.filter(point => point.split === 'train')
  const validation = data.filter(point => point.split === 'validation')
  const losses: LossPoint[] = []
  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    next = applyNetworkGradients(next, networkLoss(next, train, true).gradients)
    losses.push({ epoch, train: networkLoss(next, train).loss, validation: networkLoss(next, validation).loss })
  }
  return { model: next, losses }
}

/** Every scalar shown here is a view into the same matrix product used by inference. */
export function neuronCalculation(model: NetworkModel, calculation: NetworkCalculation, layerIndex: number, neuronIndex: number) {
  const layer = model.layers[layerIndex]
  const trace = calculation.layers[layerIndex]
  return { id: `${layer.id}/neuron-${neuronIndex}`, input: trace.input, weights: trace.input.map((_, i) => layer.weights[i * layer.outputs + neuronIndex]),
    products: trace.input.map((input, i) => input * layer.weights[i * layer.outputs + neuronIndex]), bias: layer.biases[neuronIndex],
    weightedSum: trace.weightedSum[neuronIndex], activation: layer.activation, output: trace.output[neuronIndex] }
}

/** Export the selected scalar network, retaining learned parameter identities. */
export function networkToGraph(model: NetworkModel, example: Example): GraphModel {
  const graph: GraphModel = { nodes: [], edges: [], groups: [], learningRate: model.learningRate }
  const source = (id: string, type: 'input' | 'weight' | 'bias' | 'target', value: number, x: number, y: number) => graph.nodes.push({ id, type, label: id, position: { x, y }, params: { value } })
  const edge = (from: string, to: string, inputSlot: number) => graph.edges.push({ id: `${from}:${to}:${inputSlot}`, source: from, target: to, inputSlot })
  let previous = example.x.map((x, i) => { source(`input-${i}`, 'input', x, 0, i * 180); return `input-${i}` })
  model.layers.forEach((layer, l) => {
    const nodeIds: string[] = []
    const next = Array.from({ length: layer.outputs }, (_, n) => {
      const prefix = `${layer.id}/neuron-${n}`
      const x = 300 + l * 1050, y = n * 400
      const products = previous.map((input, i) => {
        const w = `${layer.id}/weight-${i}-${n}`, p = `${prefix}/product-${i}`
        source(w, 'weight', layer.weights[i * layer.outputs + n], x, y + i * 90)
        graph.nodes.push({ id: p, type: 'multiply', label: `x${i} × w${i}`, position: { x: x + 200, y: y + i * 90 }, params: {} })
        edge(input, p, 0); edge(w, p, 1); nodeIds.push(w, p); return p
      })
      const bias = `${layer.id}/bias-${n}`, sum = `${prefix}/sum`, activation = `${prefix}/activation`
      source(bias, 'bias', layer.biases[n], x + 200, y + 270)
      // The legacy builder allows at most eight add ports, so add bias separately for width eight.
      if (products.length >= 8) {
        const productSum = `${prefix}/products-sum`
        graph.nodes.push({ id: productSum, type: 'add', label: 'Sum of products', position: { x: x + 400, y }, params: { inputCount: products.length } })
        products.forEach((id, i) => edge(id, productSum, i))
        graph.nodes.push({ id: sum, type: 'add', label: 'Add bias', position: { x: x + 580, y }, params: { inputCount: 2 } })
        edge(productSum, sum, 0); edge(bias, sum, 1); nodeIds.push(productSum)
      } else {
        graph.nodes.push({ id: sum, type: 'add', label: 'Weighted sum', position: { x: x + 440, y }, params: { inputCount: products.length + 1 } })
        products.forEach((id, i) => edge(id, sum, i)); edge(bias, sum, products.length)
      }
      graph.nodes.push({ id: activation, type: 'activation', label: `Neuron ${n + 1}: ${layer.activation}`, position: { x: x + 700, y }, params: { activation: layer.activation } }); edge(sum, activation, 0)
      nodeIds.push(bias, sum, activation)
      return activation
    })
    graph.groups!.push({ id: layer.id, label: `Layer ${l + 1}`, nodeIds, position: { x: 270 + l * 1050, y: -45 }, dimensions: { width: 220, height: 150 } })
    previous = next
  })
  if (model.kind !== 'playground') {
    source('target', 'target', example.y, 300 + model.layers.length * 1050, 180)
    // Legacy squared-error is half-squared error; scalar MSE preserves this lesson's full square.
    graph.nodes.push({ id: 'loss', type: 'loss', label: 'Squared error (full square)', position: { x: 550 + model.layers.length * 1050, y: 0 }, params: { loss: 'mse' } })
    edge(previous[0], 'loss', 0); edge('target', 'loss', 1)
  }
  return graph
}
