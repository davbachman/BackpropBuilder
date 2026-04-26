import type { ActivationKind, GraphModel, GraphNode, NodeType } from './types'
import { scalarValue } from './tensor'

const LR = 0.1

export function createStarterGraph(): GraphModel {
  return createSingleNeuronGraph('sigmoid', {
    x: 2,
    w: 0.5,
    b: -0.3,
    target: 1,
  })
}

export function createReluGateLessonGraph(): GraphModel {
  return createSingleNeuronGraph('relu', {
    x: 2,
    w: -1.1,
    b: -0.4,
    target: 1,
  })
}

export function createEmptyGraph(): GraphModel {
  return { nodes: [], edges: [], learningRate: LR }
}

export function createSingleNeuronGraph(
  activation: ActivationKind,
  values: { x: number; w: number; b: number; target: number },
): GraphModel {
  const nodes: GraphNode[] = [
    sourceNode('x', 'input', 'x', values.x, 40, 60),
    sourceNode('w', 'weight', 'w', values.w, 40, 260),
    sourceNode('b', 'bias', 'b', values.b, 320, 360),
    opNode('mul', 'multiply', 'x * w', 320, 160),
    opNode('add', 'add', 'xw + b', 600, 260),
    {
      ...opNode(activation === 'relu' ? 'relu' : 'pred', 'activation', 'activation', 880, 160),
      params: { activation },
    },
    sourceNode('target', 'target', 'y', values.target, 880, 360),
    opNode('loss', 'loss', 'loss', 1160, 260),
  ]

  return {
    learningRate: LR,
    nodes,
    edges: [
      { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
      { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
      { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
      { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
      { id: 'add-act', source: 'add', target: activation === 'relu' ? 'relu' : 'pred', inputSlot: 0 },
      { id: 'act-loss', source: activation === 'relu' ? 'relu' : 'pred', target: 'loss', inputSlot: 0 },
      { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
    ],
  }
}

export function createNode(type: NodeType, index: number): GraphNode {
  const baseX = 160 + (index % 4) * 180
  const baseY = 120 + Math.floor(index / 4) * 150
  if (type === 'input') return sourceNode(`input-${index}`, type, `x${index}`, 1, baseX, baseY)
  if (type === 'weight') return sourceNode(`weight-${index}`, type, `w${index}`, 0.5, baseX, baseY)
  if (type === 'bias') return sourceNode(`bias-${index}`, type, `b${index}`, 0, baseX, baseY)
  if (type === 'target') return sourceNode(`target-${index}`, type, `y${index}`, 1, baseX, baseY)
  if (type === 'activation') {
    return { ...opNode(`activation-${index}`, type, 'activation', baseX, baseY), params: { activation: 'sigmoid' } }
  }
  return opNode(`${type}-${index}`, type, type, baseX, baseY)
}

function sourceNode(
  id: string,
  type: Extract<NodeType, 'input' | 'weight' | 'bias' | 'target'>,
  label: string,
  value: number,
  x: number,
  y: number,
): GraphNode {
  return {
    id,
    type,
    label,
    position: { x, y },
    params: { value: scalarValue(value) },
    value: scalarValue(value),
    grad: scalarValue(0),
  }
}

function opNode(
  id: string,
  type: Exclude<NodeType, 'input' | 'weight' | 'bias' | 'target'>,
  label: string,
  x: number,
  y: number,
): GraphNode {
  return {
    id,
    type,
    label,
    position: { x, y },
    params: {},
    grad: scalarValue(0),
  }
}
