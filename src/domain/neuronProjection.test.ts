import { describe, expect, it } from 'vitest'
import { backwardPass, forwardPass } from './engine'
import { createModelPreset } from './modelPresets'
import { activeProjectionEdges, projectDenseNeurons } from './neuronProjection'
import { visibleGraphForTrace } from './traceVisibility'
import type { GraphModel, TensorValue } from './types'

function evaluatedDecoder(): GraphModel {
  const graph = createModelPreset('decoder')
  return backwardPass(forwardPass(graph).graph).graph
}
const node = (graph: GraphModel, id: string) => graph.nodes.find(candidate => candidate.id === id)!

describe('same-canvas dense neuron projection', () => {
  it('preserves the executable graph, its tensor output and downstream wires', () => {
    const graph = evaluatedDecoder(), before = JSON.stringify(graph)
    const { graph: display } = projectDenseNeurons(graph, { groupId: 'blocks.0.ff1.layer', unitIndex: 3, row: 1 })
    expect(JSON.stringify(graph)).toBe(before)
    expect(node(display, 'blocks.0.ff1.output').value).toEqual(node(graph, 'blocks.0.ff1.output').value)
    expect(node(display, 'blocks.0.ff1.output').grad).toEqual(node(graph, 'blocks.0.ff1.output').grad)
    expect(display.edges.some(edge => edge.source === 'blocks.0.ff1.output' && edge.target === 'blocks.0.ff2.product')).toBe(true)
    expect(display.groups!.filter(group => group.parentId === 'blocks.0.ff1.layer')).toHaveLength(16)
    expect(display.view?.expandedGroupIds).toContain('inspect:blocks.0.ff1.layer:3')
    expect(display.nodes.some(candidate => candidate.id === 'blocks.0.ff1')).toBe(false)
  })

  it('uses the selected token row and parameter coordinates with actual accumulated adjoints', () => {
    const graph = evaluatedDecoder(), row = 1, unitIndex = 3
    const { graph: display, bindings } = projectDenseNeurons(graph, { groupId: 'blocks.0.ff1.layer', unitIndex, row })
    const prefix = `inspect:blocks.0.ff1.layer:${unitIndex}`
    const input = node(graph, 'blocks.0.norm2.output'), weight = node(graph, 'blocks.0.ff1'), pre = node(graph, 'blocks.0.ff1.pre')
    const weights = weight.params.value as TensorValue
    const width = weights.shape[1], inputs = weights.shape[0]
    let sum = (node(graph, 'blocks.0.ff1Bias').params.value as TensorValue).data[unitIndex]
    for (let i = 0; i < inputs; i++) {
      const x = node(display, `${prefix}:x${i}`), w = node(display, `${prefix}:w${i}`), product = node(display, `${prefix}:product${i}`)
      expect(bindings[x.id]).toEqual({ nodeId: input.id, index: row * inputs + i, editable: false })
      expect(bindings[w.id]).toEqual({ nodeId: weight.id, index: i * width + unitIndex, editable: true })
      expect(x.value!.data[0]).toBe(input.value!.data[row * inputs + i])
      expect(w.value!.data[0]).toBe(weights.data[i * width + unitIndex])
      expect(w.grad!.data[0]).toBe(weight.grad!.data[i * width + unitIndex])
      expect(product.value!.data[0]).toBe(x.value!.data[0] * w.value!.data[0])
      expect(product.grad!.data[0]).toBe(pre.grad!.data[row * width + unitIndex])
      sum += product.value!.data[0]
    }
    expect(node(display, `${prefix}:sum`).value!.data[0]).toBeCloseTo(sum, 12)
    expect(node(display, `${prefix}:output`).value!.data[0]).toBeCloseTo(Math.max(0, sum), 12)
    expect(bindings[`${prefix}:bias`]).toEqual({ nodeId: 'blocks.0.ff1Bias', index: unitIndex, editable: true })
  })

  it('shows per-use connection contributions while preserving accumulated parameter adjoints', () => {
    const graph = evaluatedDecoder(), focus = { groupId: 'blocks.0.ff1.layer', unitIndex: 3, row: 1 }
    const display = projectDenseNeurons(graph, focus).graph, prefix = 'inspect:blocks.0.ff1.layer:3'
    const gradient = node(graph, 'blocks.0.ff1.product').grad!.data[19]
    const x = node(display, `${prefix}:x0`).value!.data[0], w = node(display, `${prefix}:w0`).value!.data[0]
    const xWire = display.edges.find(edge => edge.source === `${prefix}:x0` && edge.target === `${prefix}:product0`)!
    const wWire = display.edges.find(edge => edge.source === `${prefix}:w0` && edge.target === `${prefix}:product0`)!
    const boundaryWire = display.edges.find(edge => edge.source === 'blocks.0.norm2.output' && edge.target === `${prefix}:x0`)!
    expect(xWire.grad!.data[0]).toBeCloseTo(gradient * w, 12)
    expect(wWire.grad!.data[0]).toBeCloseTo(gradient * x, 12)
    expect(boundaryWire.value).toEqual({ shape: [], data: [x] })
    expect(boundaryWire.grad).toEqual(xWire.grad)
    let accumulated = 0
    for (let row = 0; row < 3; row++) {
      const rowGraph = projectDenseNeurons(graph, { ...focus, row }).graph
      accumulated += rowGraph.edges.find(edge => edge.source === `${prefix}:w0` && edge.target === `${prefix}:product0`)!.grad!.data[0]
    }
    expect(accumulated).toBeCloseTo(node(graph, 'blocks.0.ff1').grad!.data[3], 12)
  })

  it('reveals scalar arithmetic and gradients at the corresponding matrix trace stage', () => {
    const forward = forwardPass(createModelPreset('decoder')), backward = backwardPass(forward.graph)
    const focus = { groupId: 'blocks.0.ff1.layer', unitIndex: 3, row: 1 }, prefix = 'inspect:blocks.0.ff1.layer:3'
    const productIndex = forward.steps.findIndex(step => step.nodeId === 'blocks.0.ff1.product')
    const beforeProduct = projectDenseNeurons(visibleGraphForTrace(forward.graph, forward.steps, productIndex - 1, 'forward'), focus).graph
    expect(node(beforeProduct, `${prefix}:product0`).value).toBeUndefined()
    const duringProduct = projectDenseNeurons(visibleGraphForTrace(forward.graph, forward.steps, productIndex, 'forward'), focus).graph
    expect(node(duringProduct, `${prefix}:product0`).value).toBeDefined()
    const active = activeProjectionEdges(forward.graph, focus, forward.steps[productIndex])
    expect(active).toContain(duringProduct.edges.find(edge => edge.target === `${prefix}:product0` && edge.source === `${prefix}:w0`)!.id)
    expect(active).not.toContain(duringProduct.edges.find(edge => edge.target === `${prefix}:sum`)!.id)
    const activationIndex = backward.steps.findIndex(step => step.nodeId === 'blocks.0.ff1.output')
    const activation = projectDenseNeurons(visibleGraphForTrace(backward.graph, backward.steps, activationIndex, 'backward'), focus).graph
    expect(activation.edges.find(edge => edge.source === `${prefix}:sum` && edge.target === `${prefix}:output`)!.grad).toBeDefined()
    expect(activation.edges.find(edge => edge.source === `${prefix}:w0` && edge.target === `${prefix}:product0`)!.grad).toBeUndefined()
    const backwardProduct = backward.steps.findIndex(step => step.nodeId === 'blocks.0.ff1.product')
    const afterProduct = projectDenseNeurons(visibleGraphForTrace(backward.graph, backward.steps, backwardProduct, 'backward'), focus).graph
    expect(afterProduct.edges.find(edge => edge.source === `${prefix}:w0` && edge.target === `${prefix}:product0`)!.grad).toBeDefined()
  })

  it('clamps inspected rows and units to available coordinates, leaving absent focus untouched', () => {
    const graph = evaluatedDecoder()
    expect(projectDenseNeurons(graph).graph).toBe(graph)
    expect(projectDenseNeurons(graph, { groupId: 'missing', row: 0, unitIndex: 0 }).graph).toBe(graph)
    const display = projectDenseNeurons(graph, { groupId: 'blocks.0.ff1.layer', unitIndex: 999, row: 999 }).graph
    expect(node(display, 'inspect:blocks.0.ff1.layer:15:x0').value!.data[0]).toBe(node(graph, 'blocks.0.norm2.output').value!.data[16])
    expect(node(display, 'inspect:blocks.0.ff1.layer:15:output').value!.data[0]).toBe(node(graph, 'blocks.0.ff1.output').value!.data[47])
  })
})
