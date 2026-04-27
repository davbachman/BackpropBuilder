import type { DatasetKind, DatasetTask, GraphNode, TensorValue } from './types'
import { tensorValue } from './tensor'

export interface ToyDataset {
  kind: DatasetKind
  label: string
  task: DatasetTask
  featureLabels: string[]
  targetLabel: string
  featureValues: TensorValue[]
  targetValue: TensorValue
}

export const DATASET_OPTIONS: ToyDataset[] = [
  {
    kind: 'line-1d',
    label: 'Line regression (1D)',
    task: 'regression',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.3]),
    ],
    targetValue: tensorValue([20], [-3.62, -3.32, -2.51, -2.2, -1.26, -0.85, 0.01, 0.24, 1.04, 1.77, 2.09, 2.88, 3.22, 4.12, 4.53, 5.4, 5.65, 6.45, 7.13, 7.5]),
  },
  {
    kind: 'cubic-1d',
    label: 'Cubic regression (1D)',
    task: 'regression',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.3]),
    ],
    targetValue: tensorValue([20], [-11.92, -8.97, -6.13, -4.29, -2.32, -1.39, -0.31, 0.05, 0.62, 0.67, 1.12, 1.13, 1.62, 1.76, 2.57, 3.16, 4.71, 6.11, 8.38, 10.81]),
  },
  {
    kind: 'plane-2d',
    label: 'Plane regression (2D)',
    task: 'regression',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2, -1.8, -1.1, -0.3, 0.5, 1.3, 2.1, -1.5, 0.2, 1.7]),
      tensorValue([20], [-1.5, -0.9, -0.2, 0.4, 1.1, 1.7, -1.3, -0.6, 0.1, 0.8, 1.5, 1.4, 0.7, -1.1, -1.8, -0.4, 0.3, -1.7, 1.2, -1]),
    ],
    targetValue: tensorValue([20], [-0.71, -0.97, -0.71, -0.86, -0.55, -0.76, 2.05, 1.77, 1.92, 1.71, 2.03, -2.73, -1.19, 0.71, 2.44, 2.21, 3.02, -0.17, 0, 3.07]),
  },
  {
    kind: 'threshold-1d',
    label: 'Threshold classification (1D)',
    task: 'binary-classification',
    featureLabels: ['x'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.05, -1.76, -1.42, -1.18, -0.92, -0.71, -0.53, -0.31, -0.16, -0.04, 0.08, 0.19, 0.36, 0.57, 0.79, 1.02, 1.28, 1.51, 1.74, 2.03]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
  },
  {
    kind: 'circle-center',
    label: 'Circle vs center classification (2D)',
    task: 'binary-classification',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-0.18, 0.12, -0.08, 0.22, 0.05, -0.25, 0.16, -0.04, 0.28, -0.14, 1.68, 1.23, 0.46, -0.42, -1.18, -1.72, -1.2, -0.5, 0.44, 1.18]),
      tensorValue([20], [0.06, -0.15, 0.22, 0.14, -0.26, -0.12, 0.3, -0.31, 0.02, 0.18, 0.08, 1.14, 1.58, 1.61, 1.18, 0.05, -1.09, -1.53, -1.61, -1.13]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
  },
  {
    kind: 'parabola-boundary',
    label: 'Parabola boundary classification (2D)',
    task: 'binary-classification',
    featureLabels: ['x1', 'x2'],
    targetLabel: 'y',
    featureValues: [
      tensorValue([20], [-2.3, -1.9, -1.55, -1.2, -0.85, -0.45, -0.12, 0.32, 0.78, 1.18, 1.58, 2.02, -2.15, -1.65, -1.05, -0.62, -0.18, 0.48, 1.08, 1.72]),
      tensorValue([20], [1.78, 1.07, 0.62, 0.05, -0.37, -0.64, -0.81, -0.68, -0.38, 0.03, 0.69, 1.42, 2.72, 1.55, 0.42, -0.11, -0.22, 0.32, 1.02, 2.14]),
    ],
    targetValue: tensorValue([20], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
  },
]

export function datasetForNode(node: GraphNode): ToyDataset {
  return DATASET_OPTIONS.find((dataset) => dataset.kind === node.params.dataset) ?? DATASET_OPTIONS[0]
}

export function datasetOutputCountForNode(node: GraphNode): number {
  const dataset = datasetForNode(node)
  return dataset.featureValues.length + 1
}

export function datasetOutputLabelForSlot(node: GraphNode, slot: number): string {
  const dataset = datasetForNode(node)
  return dataset.featureLabels[slot] ?? dataset.targetLabel
}

export function datasetOutputValueForSlot(node: GraphNode, slot: number): TensorValue {
  const dataset = datasetForNode(node)
  return dataset.featureValues[slot] ?? dataset.targetValue
}

export function remapDatasetOutputSlot(fromNode: GraphNode, toNode: GraphNode, fromSlot: number): number | undefined {
  const role = datasetOutputRoleForSlot(fromNode, fromSlot)
  if (!role) return undefined
  if (role === 'target') return datasetForNode(toNode).featureValues.length

  const featureIndex = Number(role.replace('feature-', ''))
  return featureIndex < datasetForNode(toNode).featureValues.length ? featureIndex : undefined
}

export function isDatasetKind(value: string): value is DatasetKind {
  return DATASET_OPTIONS.some((dataset) => dataset.kind === value)
}

function datasetOutputRoleForSlot(node: GraphNode, slot: number): string | undefined {
  const dataset = datasetForNode(node)
  if (slot < 0) return undefined
  if (slot < dataset.featureValues.length) return `feature-${slot}`
  if (slot === dataset.featureValues.length) return 'target'
  return undefined
}
