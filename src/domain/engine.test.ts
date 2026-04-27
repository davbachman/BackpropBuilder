import { describe, expect, it } from 'vitest'
import {
  backwardPass,
  DATASET_OPTIONS,
  formulaForNode,
  forwardPass,
  heightForInputCount,
  inputArityForNode,
  outputArityForNode,
  remapDatasetOutputSlot,
  runTrainingStep,
  updateParameters,
  validateGraph,
} from './engine'
import {
  createNode,
  createReluGateGraph,
  createStarterGraph,
} from './examples'
import { scalarFromTensor, scalarValue, tensorValue, toTensor } from './tensor'
import type { GraphModel, TensorValue } from './types'

function scalarOf(value: TensorValue | number | undefined): number {
  return scalarFromTensor(toTensor(value))
}

function maxLinearResidual(features: number[][], targets: number[]): number {
  const rows = targets.map((target, index) => ({
    inputs: [1, ...features.map((feature) => feature[index])],
    target,
  }))
  const size = rows[0].inputs.length
  const normalMatrix = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) =>
      rows.reduce((sum, entry) => sum + entry.inputs[row] * entry.inputs[column], 0),
    ),
  )
  const normalTarget = Array.from({ length: size }, (_, row) =>
    rows.reduce((sum, entry) => sum + entry.inputs[row] * entry.target, 0),
  )
  const coefficients = solveLinearSystem(normalMatrix, normalTarget)
  return Math.max(
    ...rows.map((entry) =>
      Math.abs(entry.target - entry.inputs.reduce((sum, input, index) => sum + input * coefficients[index], 0)),
    ),
  )
}

function maxPolynomialResidual(feature: number[], targets: number[], degree: number): number {
  const rows = targets.map((target, index) => ({
    inputs: Array.from({ length: degree + 1 }, (_, power) => feature[index] ** power),
    target,
  }))
  const size = degree + 1
  const normalMatrix = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) =>
      rows.reduce((sum, entry) => sum + entry.inputs[row] * entry.inputs[column], 0),
    ),
  )
  const normalTarget = Array.from({ length: size }, (_, row) =>
    rows.reduce((sum, entry) => sum + entry.inputs[row] * entry.target, 0),
  )
  const coefficients = solveLinearSystem(normalMatrix, normalTarget)
  return Math.max(
    ...rows.map((entry) =>
      Math.abs(entry.target - entry.inputs.reduce((sum, input, index) => sum + input * coefficients[index], 0)),
    ),
  )
}

function solveLinearSystem(matrix: number[][], target: number[]): number[] {
  const rows = matrix.map((row, index) => [...row, target[index]])
  for (let pivot = 0; pivot < rows.length; pivot += 1) {
    const pivotRow = rows.slice(pivot).reduce((bestRow, row, offset) => {
      const rowIndex = pivot + offset
      return Math.abs(row[pivot]) > Math.abs(rows[bestRow][pivot]) ? rowIndex : bestRow
    }, pivot)
    ;[rows[pivot], rows[pivotRow]] = [rows[pivotRow], rows[pivot]]
    const divisor = rows[pivot][pivot]
    for (let column = pivot; column <= rows.length; column += 1) rows[pivot][column] /= divisor
    for (let row = 0; row < rows.length; row += 1) {
      if (row === pivot) continue
      const factor = rows[row][pivot]
      for (let column = pivot; column <= rows.length; column += 1) {
        rows[row][column] -= factor * rows[pivot][column]
      }
    }
  }
  return rows.map((row) => row[rows.length])
}

function meanRadiusForClass(dataset: (typeof DATASET_OPTIONS)[number], classValue: number): number {
  const [xValues, yValues] = dataset.featureValues
  const radii = dataset.targetValue.data.flatMap((target, index) =>
    target === classValue ? [Math.hypot(xValues.data[index], yValues.data[index])] : [],
  )
  return radii.reduce((sum, radius) => sum + radius, 0) / radii.length
}

function parabolaSeparationAccuracy(dataset: (typeof DATASET_OPTIONS)[number]): number {
  const [xValues, yValues] = dataset.featureValues
  const correct = dataset.targetValue.data.filter((target, index) => {
    const prediction = yValues.data[index] > 0.55 * xValues.data[index] ** 2 - 0.65 ? 1 : 0
    return prediction === target
  }).length
  return correct / dataset.targetValue.data.length
}

describe('scalar autodiff engine', () => {
  it('lays out the starter graph with enough vertical room for full node cards', () => {
    const graph = createStarterGraph()
    const nodeWidth = 176
    const nodeHeight = 150
    const minimumGap = 24

    for (const [index, first] of graph.nodes.entries()) {
      for (const second of graph.nodes.slice(index + 1)) {
        const overlapsHorizontally =
          first.position.x < second.position.x + nodeWidth + minimumGap &&
          first.position.x + nodeWidth + minimumGap > second.position.x
        const overlapsVertically =
          first.position.y < second.position.y + nodeHeight + minimumGap &&
          first.position.y + nodeHeight + minimumGap > second.position.y

        expect(
          overlapsHorizontally && overlapsVertically,
          `${first.label} at (${first.position.x}, ${first.position.y}) overlaps ${second.label} at (${second.position.x}, ${second.position.y})`,
        ).toBe(false)
      }
    }
  })

  it('uses a flexible-node baseline height that fits formulas and metrics', () => {
    expect(heightForInputCount(2)).toBeGreaterThanOrEqual(150)
    expect(heightForInputCount(3)).toBe(heightForInputCount(2))
    expect(heightForInputCount(4)).toBeGreaterThan(heightForInputCount(3))
  })

  it('offers one- and two-feature toy datasets for regression and binary classification', () => {
    const taskKinds = new Set(DATASET_OPTIONS.map((dataset) => dataset.task))
    const featureCounts = new Set(DATASET_OPTIONS.map((dataset) => dataset.featureLabels.length))

    expect(taskKinds).toEqual(new Set(['regression', 'binary-classification']))
    expect(featureCounts).toEqual(new Set([1, 2]))
  })

  it('keeps toy datasets near 20 aligned rows with deterministic noise', () => {
    for (const dataset of DATASET_OPTIONS) {
      expect(dataset.targetValue.shape).toEqual([20])
      for (const feature of dataset.featureValues) {
        expect(feature.shape).toEqual([20])
      }

      if (dataset.task === 'regression') {
        expect(maxLinearResidual(dataset.featureValues.map((feature) => feature.data), dataset.targetValue.data)).toBeGreaterThan(0.03)
      } else {
        expect(new Set(dataset.targetValue.data)).toEqual(new Set([0, 1]))
        expect(dataset.featureValues.some((feature) => feature.data.some((value) => !Number.isInteger(value)))).toBe(true)
      }
    }
  })

  it('includes the requested cubic, circle, and parabola toy datasets', () => {
    const cubic = DATASET_OPTIONS.find((dataset) => dataset.kind === 'cubic-1d')
    const circle = DATASET_OPTIONS.find((dataset) => dataset.kind === 'circle-center')
    const parabola = DATASET_OPTIONS.find((dataset) => dataset.kind === 'parabola-boundary')

    expect(cubic).toBeDefined()
    expect(cubic?.task).toBe('regression')
    expect(cubic?.featureValues).toHaveLength(1)
    expect(maxPolynomialResidual(cubic!.featureValues[0].data, cubic!.targetValue.data, 3)).toBeLessThan(0.25)
    expect(maxPolynomialResidual(cubic!.featureValues[0].data, cubic!.targetValue.data, 2)).toBeGreaterThan(1)

    expect(circle).toBeDefined()
    expect(circle?.task).toBe('binary-classification')
    expect(circle?.featureValues).toHaveLength(2)
    expect(meanRadiusForClass(circle!, 1)).toBeGreaterThan(1.45)
    expect(meanRadiusForClass(circle!, 0)).toBeLessThan(0.4)

    expect(parabola).toBeDefined()
    expect(parabola?.task).toBe('binary-classification')
    expect(parabola?.featureValues).toHaveLength(2)
    expect(parabolaSeparationAccuracy(parabola!)).toBeGreaterThanOrEqual(0.95)
  })

  it('routes dataset feature and target outputs by source slot', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'dataset', type: 'dataset', label: 'dataset', position: { x: 0, y: 0 }, params: { dataset: 'circle-center' } },
        { id: 'add', type: 'add', label: 'add', position: { x: 240, y: 0 }, params: {} },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 480, y: 0 }, params: {} },
      ],
      edges: [
        { id: 'x1-add', source: 'dataset', sourceSlot: 0, target: 'add', inputSlot: 0 },
        { id: 'x2-add', source: 'dataset', sourceSlot: 1, target: 'add', inputSlot: 1 },
        { id: 'add-loss', source: 'add', target: 'loss', inputSlot: 0 },
        { id: 'y-loss', source: 'dataset', sourceSlot: 2, target: 'loss', inputSlot: 1 },
      ],
    }

    const forward = forwardPass(graph)

    expect(inputArityForNode(graph.nodes[0])).toBe(0)
    expect(outputArityForNode(graph.nodes[0])).toBe(3)
    expect(forward.graph.edges.find((edge) => edge.id === 'x1-add')?.value?.shape).toEqual([20])
    expect(forward.graph.edges.find((edge) => edge.id === 'x2-add')?.value?.shape).toEqual([20])
    expect(forward.graph.edges.find((edge) => edge.id === 'y-loss')?.value?.shape).toEqual([20])
    expect(formulaForNode(graph.nodes[1], graph)).toBe('z1 = x1 + x2')
    expect(formulaForNode(graph.nodes[2], graph)).toBe('L = (1/n) * Σ_i (z1_i - y_i)^2')
  })

  it('lets input and target nodes pass through connected dataset outputs', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'dataset', type: 'dataset', label: 'dataset', position: { x: 0, y: 0 }, params: { dataset: 'line-1d' } },
        { id: 'x', type: 'input', label: 'x', position: { x: 240, y: 0 }, params: { value: scalarValue(0) } },
        { id: 'target', type: 'target', label: 'y', position: { x: 240, y: 160 }, params: { value: scalarValue(0) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 480, y: 80 }, params: {} },
      ],
      edges: [
        { id: 'dataset-x', source: 'dataset', sourceSlot: 0, target: 'x', inputSlot: 0 },
        { id: 'x-loss', source: 'x', target: 'loss', inputSlot: 0 },
        { id: 'dataset-y', source: 'dataset', sourceSlot: 1, target: 'target', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const forward = forwardPass(graph)

    expect(forward.graph.nodes.find((node) => node.id === 'x')?.value?.shape).toEqual([20])
    expect(forward.graph.nodes.find((node) => node.id === 'target')?.value?.shape).toEqual([20])
    expect(forward.graph.edges.find((edge) => edge.id === 'x-loss')?.value?.shape).toEqual([20])
    expect(forward.graph.edges.find((edge) => edge.id === 'target-loss')?.value?.shape).toEqual([20])
  })

  it('remaps dataset output slots by role when changing feature count', () => {
    const oneFeature: GraphModel['nodes'][number] = {
      id: 'dataset',
      type: 'dataset',
      label: 'dataset',
      position: { x: 0, y: 0 },
      params: { dataset: 'line-1d' },
    }
    const twoFeature: GraphModel['nodes'][number] = {
      ...oneFeature,
      params: { dataset: 'circle-center' },
    }

    expect(remapDatasetOutputSlot(oneFeature, twoFeature, 0)).toBe(0)
    expect(remapDatasetOutputSlot(oneFeature, twoFeature, 1)).toBe(2)
    expect(remapDatasetOutputSlot(twoFeature, oneFeature, 0)).toBe(0)
    expect(remapDatasetOutputSlot(twoFeature, oneFeature, 1)).toBeUndefined()
    expect(remapDatasetOutputSlot(twoFeature, oneFeature, 2)).toBe(1)
  })

  it('computes the starter graph forward values', () => {
    const graph = createStarterGraph()
    const result = forwardPass(graph)

    const multiply = result.graph.nodes.find((node) => node.id === 'mul')
    const add = result.graph.nodes.find((node) => node.id === 'add')
    const prediction = result.graph.nodes.find((node) => node.id === 'pred')
    const loss = result.graph.nodes.find((node) => node.id === 'loss')

    expect(scalarOf(multiply?.value)).toBeCloseTo(1)
    expect(scalarOf(add?.value)).toBeCloseTo(0.7)
    expect(scalarOf(prediction?.value)).toBeCloseTo(1 / (1 + Math.exp(-0.7)))
    expect(scalarOf(loss?.value)).toBeCloseTo(0.5 * (scalarOf(prediction!.value) - 1) ** 2)
    expect(result.loss).toBeCloseTo(scalarOf(loss!.value))
  })

  it('uses the standard activation node label in the starter graph', () => {
    const starterActivation = createStarterGraph().nodes.find((node) => node.id === 'pred')
    const paletteActivation = createNode('activation', 1)

    expect(starterActivation).toMatchObject({
      type: 'activation',
      label: paletteActivation.label,
    })
  })

  it('uses connected variable names in operation formulas', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'input-1', type: 'input', label: 'x1', position: { x: 0, y: 0 }, params: { value: 2 } },
        { id: 'weight-3', type: 'weight', label: 'w3', position: { x: 0, y: 160 }, params: { value: 0.5 } },
        { id: 'multiply-1', type: 'multiply', label: 'multiply', position: { x: 240, y: 160 }, params: {} },
        { id: 'add-1', type: 'add', label: 'add', position: { x: 240, y: 0 }, params: {} },
        { id: 'activation-1', type: 'activation', label: 'activation', position: { x: 480, y: 0 }, params: { activation: 'sigmoid' } },
        { id: 'target-1', type: 'target', label: 'y1', position: { x: 480, y: 160 }, params: { value: 1 } },
        { id: 'loss-1', type: 'loss', label: 'loss', position: { x: 720, y: 0 }, params: {} },
      ],
      edges: [
        { id: 'x-add', source: 'input-1', target: 'add-1', inputSlot: 0 },
        { id: 'w-add', source: 'weight-3', target: 'add-1', inputSlot: 1 },
        { id: 'x-mul', source: 'input-1', target: 'multiply-1', inputSlot: 0 },
        { id: 'w-mul', source: 'weight-3', target: 'multiply-1', inputSlot: 1 },
        { id: 'add-act', source: 'add-1', target: 'activation-1', inputSlot: 0 },
        { id: 'act-loss', source: 'activation-1', target: 'loss-1', inputSlot: 0 },
        { id: 'target-loss', source: 'target-1', target: 'loss-1', inputSlot: 1 },
      ],
    }

    expect(formulaForNode(graph.nodes[2], graph)).toBe('z2 = x1 * w3')
    expect(formulaForNode(graph.nodes[3], graph)).toBe('z1 = x1 + w3')
    expect(formulaForNode(graph.nodes[4], graph)).toBe('z3 = sigmoid(z1)')
    expect(formulaForNode(graph.nodes[6], graph)).toBe('L = 0.5 * (z3 - y1)^2')
  })

  it('numbers computed outputs so downstream formulas can reference them', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'input-1', type: 'input', label: 'x1', position: { x: 0, y: 0 }, params: { value: 1 } },
        { id: 'input-2', type: 'input', label: 'x2', position: { x: 0, y: 120 }, params: { value: 2 } },
        { id: 'input-3', type: 'input', label: 'x3', position: { x: 0, y: 240 }, params: { value: 3 } },
        { id: 'add-1', type: 'add', label: 'add', position: { x: 240, y: 0 }, params: {} },
        { id: 'add-2', type: 'add', label: 'add', position: { x: 480, y: 0 }, params: {} },
      ],
      edges: [
        { id: 'x1-add1', source: 'input-1', target: 'add-1', inputSlot: 0 },
        { id: 'x2-add1', source: 'input-2', target: 'add-1', inputSlot: 1 },
        { id: 'add1-add2', source: 'add-1', target: 'add-2', inputSlot: 0 },
        { id: 'x3-add2', source: 'input-3', target: 'add-2', inputSlot: 1 },
      ],
    }

    expect(formulaForNode(graph.nodes[3], graph)).toBe('z1 = x1 + x2')
    expect(formulaForNode(graph.nodes[4], graph)).toBe('z2 = z1 + x3')
  })

  it('uses tensor loss formulas when an incoming operation output shape is tensor', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x', type: 'input', label: 'x', position: { x: 0, y: 0 }, params: { value: tensorValue([2], [1, 2]) } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 0, y: 140 }, params: { value: scalarValue(2) } },
        { id: 'mul', type: 'multiply', label: 'multiply', position: { x: 240, y: 70 }, params: {} },
        { id: 'target', type: 'target', label: 'y', position: { x: 240, y: 220 }, params: { value: scalarValue(1) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 480, y: 140 }, params: { loss: 'mse' } },
      ],
      edges: [
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-loss', source: 'mul', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    expect(formulaForNode(graph.nodes[4], graph)).toBe('L = (1/n) * Σ_i (z1_i - y_i)^2')
  })

  it('supports add nodes with more than two inputs', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x1', type: 'input', label: 'x1', position: { x: 0, y: 0 }, params: { value: 1 } },
        { id: 'x2', type: 'input', label: 'x2', position: { x: 0, y: 120 }, params: { value: 2 } },
        { id: 'x3', type: 'input', label: 'x3', position: { x: 0, y: 240 }, params: { value: 3 } },
        { id: 'add-1', type: 'add', label: 'add', position: { x: 240, y: 80 }, params: { inputCount: 3 } },
        { id: 'target', type: 'target', label: 'y', position: { x: 480, y: 160 }, params: { value: 5 } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 720, y: 80 }, params: {} },
      ],
      edges: [
        { id: 'x1-add', source: 'x1', target: 'add-1', inputSlot: 0 },
        { id: 'x2-add', source: 'x2', target: 'add-1', inputSlot: 1 },
        { id: 'x3-add', source: 'x3', target: 'add-1', inputSlot: 2 },
        { id: 'add-loss', source: 'add-1', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const forward = forwardPass(graph)
    const backward = backwardPass(forward.graph)

    expect(inputArityForNode(graph.nodes[3])).toBe(3)
    expect(formulaForNode(graph.nodes[3], graph)).toBe('z1 = x1 + x2 + x3')
    expect(scalarOf(forward.graph.nodes.find((node) => node.id === 'add-1')?.value)).toBe(6)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x1')?.grad)).toBe(1)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x2')?.grad)).toBe(1)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x3')?.grad)).toBe(1)
  })

  it('supports multiply nodes with more than two inputs', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x1', type: 'input', label: 'x1', position: { x: 0, y: 0 }, params: { value: 2 } },
        { id: 'x2', type: 'input', label: 'x2', position: { x: 0, y: 120 }, params: { value: 3 } },
        { id: 'x3', type: 'input', label: 'x3', position: { x: 0, y: 240 }, params: { value: 4 } },
        { id: 'mul-1', type: 'multiply', label: 'multiply', position: { x: 240, y: 80 }, params: { inputCount: 3 } },
        { id: 'target', type: 'target', label: 'y', position: { x: 480, y: 160 }, params: { value: 20 } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 720, y: 80 }, params: {} },
      ],
      edges: [
        { id: 'x1-mul', source: 'x1', target: 'mul-1', inputSlot: 0 },
        { id: 'x2-mul', source: 'x2', target: 'mul-1', inputSlot: 1 },
        { id: 'x3-mul', source: 'x3', target: 'mul-1', inputSlot: 2 },
        { id: 'mul-loss', source: 'mul-1', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const forward = forwardPass(graph)
    const backward = backwardPass(forward.graph)

    expect(inputArityForNode(graph.nodes[3])).toBe(3)
    expect(formulaForNode(graph.nodes[3], graph)).toBe('z1 = x1 * x2 * x3')
    expect(scalarOf(forward.graph.nodes.find((node) => node.id === 'mul-1')?.value)).toBe(24)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x1')?.grad)).toBe(48)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x2')?.grad)).toBe(32)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'x3')?.grad)).toBe(24)
  })

  it('starts the forward trace at the first computed operation, not fixed source nodes', () => {
    const result = forwardPass(createStarterGraph())

    expect(result.steps.map((step) => step.nodeId)).toEqual(['mul', 'add', 'pred', 'loss'])
    expect(result.steps[0]).toMatchObject({
      phase: 'forward',
      nodeId: 'mul',
      title: 'Evaluate x * w',
      edgeIds: ['x-mul', 'w-mul'],
    })
  })

  it('computes squared-error prediction gradient and trainable gradients', () => {
    const forward = forwardPass(createStarterGraph())
    const backward = backwardPass(forward.graph)

    const prediction = backward.graph.nodes.find((node) => node.id === 'pred')
    const weight = backward.graph.nodes.find((node) => node.id === 'w')
    const bias = backward.graph.nodes.find((node) => node.id === 'b')
    const target = backward.graph.nodes.find((node) => node.id === 'target')

    const expectedPrediction = 1 / (1 + Math.exp(-0.7))
    const expectedPredGrad = expectedPrediction - 1
    const expectedSigmoidDerivative = expectedPrediction * (1 - expectedPrediction)

    expect(scalarOf(prediction?.grad)).toBeCloseTo(expectedPredGrad)
    expect(scalarOf(weight?.grad)).toBeCloseTo(expectedPredGrad * expectedSigmoidDerivative * 2)
    expect(scalarOf(bias?.grad)).toBeCloseTo(expectedPredGrad * expectedSigmoidDerivative)
    expect(scalarOf(target?.grad)).toBeCloseTo(0)
  })

  it('omits fixed leaf nodes from the visible backward trace after accumulating their gradients', () => {
    const forward = forwardPass(createStarterGraph())
    const backward = backwardPass(forward.graph)

    expect(backward.steps.map((step) => step.nodeId)).toEqual(['loss', 'pred', 'add', 'mul'])
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'w')?.grad)).not.toBe(0)
    expect(scalarOf(backward.graph.nodes.find((node) => node.id === 'b')?.grad)).not.toBe(0)
  })

  it('updates trainable parameters and clears gradients', () => {
    const forward = forwardPass(createStarterGraph())
    const backward = backwardPass(forward.graph)
    const updated = updateParameters(backward.graph, 0.1)

    const oldWeight = backward.graph.nodes.find((node) => node.id === 'w')!
    const oldBias = backward.graph.nodes.find((node) => node.id === 'b')!
    const weight = updated.graph.nodes.find((node) => node.id === 'w')!
    const bias = updated.graph.nodes.find((node) => node.id === 'b')!

    expect(scalarOf(weight.params.value)).toBeCloseTo(scalarOf(oldWeight.params.value) - 0.1 * scalarOf(oldWeight.grad))
    expect(scalarOf(bias.params.value)).toBeCloseTo(scalarOf(oldBias.params.value) - 0.1 * scalarOf(oldBias.grad))
    expect(weight.grad).toEqual(scalarValue(0))
    expect(bias.grad).toEqual(scalarValue(0))
  })

  it('stops upstream gradient for a negative ReLU gate', () => {
    const forward = forwardPass(createReluGateGraph())
    const backward = backwardPass(forward.graph)

    const relu = backward.graph.nodes.find((node) => node.id === 'relu')
    const weight = backward.graph.nodes.find((node) => node.id === 'w')
    const bias = backward.graph.nodes.find((node) => node.id === 'b')

    expect(scalarOf(relu?.value)).toBe(0)
    expect(scalarOf(relu?.localDerivative)).toBe(0)
    expect(scalarOf(weight?.grad)).toBeCloseTo(0)
    expect(scalarOf(bias?.grad)).toBeCloseTo(0)
  })

  it('reports invalid graph shapes with friendly messages', () => {
    const starter = createStarterGraph()
    const noLoss: GraphModel = {
      ...starter,
      nodes: starter.nodes.filter((node) => node.type !== 'loss'),
      edges: starter.edges.filter((edge) => edge.target !== 'loss'),
    }
    expect(validateGraph(noLoss).some((issue) => issue.code === 'missing-loss')).toBe(true)

    const multipleLosses: GraphModel = {
      ...starter,
      nodes: [...starter.nodes, { ...starter.nodes.find((node) => node.id === 'loss')!, id: 'loss-2' }],
    }
    expect(validateGraph(multipleLosses).some((issue) => issue.code === 'multiple-losses')).toBe(true)

    const cycle: GraphModel = {
      ...starter,
      edges: [...starter.edges, { id: 'cycle', source: 'pred', target: 'mul', inputSlot: 2 }],
    }
    expect(validateGraph(cycle).some((issue) => issue.code === 'cycle')).toBe(true)

    const badArity: GraphModel = {
      ...starter,
      edges: starter.edges.filter((edge) => edge.target !== 'mul'),
    }
    expect(validateGraph(badArity).some((issue) => issue.code === 'invalid-arity')).toBe(true)
  })

  it('computes elementwise tensor values and defaults tensor loss to mean squared error', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x', type: 'input', label: 'x', position: { x: 0, y: 0 }, params: { value: tensorValue([3], [1, 2, 3]) } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 0, y: 120 }, params: { value: tensorValue([3], [0.5, 1, 1.5]) } },
        { id: 'b', type: 'bias', label: 'b', position: { x: 240, y: 180 }, params: { value: tensorValue([3], [0, 1, -1]) } },
        { id: 'mul', type: 'multiply', label: 'multiply', position: { x: 240, y: 40 }, params: {} },
        { id: 'add', type: 'add', label: 'add', position: { x: 480, y: 80 }, params: {} },
        { id: 'pred', type: 'activation', label: 'activation', position: { x: 720, y: 80 }, params: { activation: 'identity' } },
        { id: 'target', type: 'target', label: 'y', position: { x: 720, y: 220 }, params: { value: tensorValue([3], [1, 3, 4]) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 960, y: 140 }, params: {} },
      ],
      edges: [
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
        { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
        { id: 'add-pred', source: 'add', target: 'pred', inputSlot: 0 },
        { id: 'pred-loss', source: 'pred', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const forward = forwardPass(graph)

    expect(forward.graph.nodes.find((node) => node.id === 'mul')?.value).toEqual(tensorValue([3], [0.5, 2, 4.5]))
    expect(forward.graph.nodes.find((node) => node.id === 'add')?.value).toEqual(tensorValue([3], [0.5, 3, 3.5]))
    expect(forward.graph.nodes.find((node) => node.id === 'pred')?.value).toEqual(tensorValue([3], [0.5, 3, 3.5]))
    expect(forward.graph.nodes.find((node) => node.id === 'loss')?.value).toEqual(scalarValue(1 / 6))
    expect(forward.loss).toBeCloseTo(1 / 6)
  })

  it('averages selected mean squared error over tensor entries and scales gradients', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'pred', type: 'input', label: 'pred', position: { x: 0, y: 0 }, params: { value: tensorValue([3], [2, 4, 6]) } },
        { id: 'target', type: 'target', label: 'y', position: { x: 0, y: 140 }, params: { value: tensorValue([3], [1, 1, 3]) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 260, y: 70 }, params: { loss: 'mse' } },
      ],
      edges: [
        { id: 'pred-loss', source: 'pred', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const lossNode = graph.nodes.find((node) => node.id === 'loss')!
    const forward = forwardPass(graph)
    const backward = backwardPass(forward.graph)

    expect(formulaForNode(lossNode, graph)).toBe('L = (1/n) * Σ_i (pred_i - y_i)^2')
    expect(forward.graph.nodes.find((node) => node.id === 'loss')?.value).toEqual(scalarValue(19 / 3))
    expect(forward.loss).toBeCloseTo(19 / 3)
    expect(backward.graph.nodes.find((node) => node.id === 'pred')?.grad).toEqual(tensorValue([3], [2 / 3, 2, 2]))
  })

  it('defaults tensor losses to mean squared error so dataset regression trains with the standard learning rate', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'dataset', type: 'dataset', label: 'dataset', position: { x: 0, y: 0 }, params: { dataset: 'line-1d' } },
        { id: 'x', type: 'input', label: 'x', position: { x: 240, y: 0 }, params: { value: scalarValue(0) } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 240, y: 140 }, params: { value: scalarValue(0.5) } },
        { id: 'b', type: 'bias', label: 'b', position: { x: 480, y: 200 }, params: { value: scalarValue(-0.3) } },
        { id: 'mul', type: 'multiply', label: 'x * w', position: { x: 480, y: 60 }, params: {} },
        { id: 'add', type: 'add', label: 'xw + b', position: { x: 720, y: 120 }, params: {} },
        { id: 'pred', type: 'activation', label: 'activation', position: { x: 960, y: 120 }, params: { activation: 'identity' } },
        { id: 'target', type: 'target', label: 'y', position: { x: 960, y: 280 }, params: { value: scalarValue(0) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 1200, y: 200 }, params: {} },
      ],
      edges: [
        { id: 'dataset-x', source: 'dataset', sourceSlot: 0, target: 'x', inputSlot: 0 },
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
        { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
        { id: 'add-pred', source: 'add', target: 'pred', inputSlot: 0 },
        { id: 'pred-loss', source: 'pred', target: 'loss', inputSlot: 0 },
        { id: 'dataset-target', source: 'dataset', sourceSlot: 1, target: 'target', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }
    const lossNode = graph.nodes.find((node) => node.id === 'loss')!

    expect(formulaForNode(lossNode, graph)).toBe('L = (1/n) * Σ_i (z3_i - y_i)^2')

    let trained = graph
    for (let step = 0; step < 100; step += 1) {
      trained = runTrainingStep(trained).graph
    }

    expect(scalarOf(trained.nodes.find((node) => node.id === 'w')?.params.value)).toBeCloseTo(1.9964160401, 6)
    expect(scalarOf(trained.nodes.find((node) => node.id === 'b')?.params.value)).toBeCloseTo(1.015112782, 6)
  })

  it('backpropagates tensor gradients and reduces broadcast scalar parameter gradients', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'x', type: 'input', label: 'x', position: { x: 0, y: 0 }, params: { value: tensorValue([3], [1, 2, 3]) } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 0, y: 120 }, params: { value: scalarValue(2) } },
        { id: 'b', type: 'bias', label: 'b', position: { x: 240, y: 180 }, params: { value: scalarValue(1) } },
        { id: 'mul', type: 'multiply', label: 'multiply', position: { x: 240, y: 40 }, params: {} },
        { id: 'add', type: 'add', label: 'add', position: { x: 480, y: 80 }, params: {} },
        { id: 'pred', type: 'activation', label: 'activation', position: { x: 720, y: 80 }, params: { activation: 'identity' } },
        { id: 'target', type: 'target', label: 'y', position: { x: 720, y: 220 }, params: { value: tensorValue([3], [4, 5, 8]) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 960, y: 140 }, params: {} },
      ],
      edges: [
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
        { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
        { id: 'add-pred', source: 'add', target: 'pred', inputSlot: 0 },
        { id: 'pred-loss', source: 'pred', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }

    const backward = backwardPass(forwardPass(graph).graph)

    expect(backward.graph.nodes.find((node) => node.id === 'pred')?.grad).toEqual(tensorValue([3], [-2 / 3, 0, -2 / 3]))
    expect(backward.graph.nodes.find((node) => node.id === 'x')?.grad).toEqual(tensorValue([3], [-4 / 3, 0, -4 / 3]))
    expect(backward.graph.nodes.find((node) => node.id === 'w')?.grad).toEqual(scalarValue(-8 / 3))
    expect(backward.graph.nodes.find((node) => node.id === 'b')?.grad).toEqual(scalarValue(-4 / 3))
    expect(backward.graph.nodes.find((node) => node.id === 'target')?.grad).toEqual(tensorValue([3], [0, 0, 0]))
  })
})
