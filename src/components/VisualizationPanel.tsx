import { useMemo, useState, type ReactElement } from 'react'
import { formatNumber, forwardPass } from '../domain/engine'
import { isScalarTensor, toTensor } from '../domain/tensor'
import type { GraphEdge, GraphModel, GraphNode, TensorValue } from '../domain/types'

const PLOT_WIDTH = 300
const PLOT_HEIGHT = 220
const PLOT_PADDING = 30

type VisualizationData =
  | {
      kind: 'line'
      inputLabel: string
      points: Array<{ x: number; target: number; prediction: number }>
    }
  | {
      kind: 'heatmap'
      inputLabels: [string, string]
      target: HeatmapData
      prediction: HeatmapData
    }
  | {
      kind: 'unsupported'
      message: string
    }

interface HeatmapData {
  rows: number
  columns: number
  values: number[]
}

export function VisualizationPanel({ graph }: { graph: GraphModel }): ReactElement {
  const [heatmapMode, setHeatmapMode] = useState<'target' | 'prediction'>('target')
  const data = useMemo(() => buildVisualizationData(graph), [graph])

  return (
    <section className="inspector-card visualization-panel" aria-label="Visualization panel">
      <div className="visualization-heading">
        <div>
          <p className="eyebrow">Visualization</p>
          <h3>{titleForVisualization(data)}</h3>
        </div>
        {data.kind === 'heatmap' ? (
          <div className="segmented-control" role="group" aria-label="Heatmap layer">
            <button
              type="button"
              aria-pressed={heatmapMode === 'target'}
              onClick={() => setHeatmapMode('target')}
            >
              Target
            </button>
            <button
              type="button"
              aria-pressed={heatmapMode === 'prediction'}
              onClick={() => setHeatmapMode('prediction')}
            >
              Predictions
            </button>
          </div>
        ) : null}
      </div>

      {data.kind === 'line' ? <LineVisualization data={data} /> : null}
      {data.kind === 'heatmap' ? (
        <HeatmapVisualization data={heatmapMode === 'target' ? data.target : data.prediction} mode={heatmapMode} />
      ) : null}
      {data.kind === 'unsupported' ? <p className="visualization-empty">{data.message}</p> : null}
    </section>
  )
}

function LineVisualization({ data }: { data: Extract<VisualizationData, { kind: 'line' }> }): ReactElement {
  const bounds = boundsForLinePoints(data.points)
  const predictionPath = data.points
    .toSorted((first, second) => first.x - second.x)
    .map((point, index) => {
      const command = index === 0 ? 'M' : 'L'
      return `${command} ${plotX(point.x, bounds)} ${plotY(point.prediction, bounds)}`
    })
    .join(' ')

  return (
    <div className="visualization-content">
      <svg
        className="visualization-plot"
        role="img"
        aria-label="Input-output visualization"
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
      >
        <line
          className="visualization-axis"
          x1={PLOT_PADDING}
          y1={PLOT_HEIGHT - PLOT_PADDING}
          x2={PLOT_WIDTH - PLOT_PADDING}
          y2={PLOT_HEIGHT - PLOT_PADDING}
        />
        <line
          className="visualization-axis"
          x1={PLOT_PADDING}
          y1={PLOT_PADDING}
          x2={PLOT_PADDING}
          y2={PLOT_HEIGHT - PLOT_PADDING}
        />
        <path className="visualization-prediction-line" d={predictionPath} />
        {data.points.map((point, index) => (
          <circle
            key={`target-${index}`}
            className="visualization-target-point"
            cx={plotX(point.x, bounds)}
            cy={plotY(point.target, bounds)}
            r="4.5"
          >
            <title>{`target (${formatNumber(point.x)}, ${formatNumber(point.target)})`}</title>
          </circle>
        ))}
        {data.points.map((point, index) => (
          <circle
            key={`prediction-${index}`}
            className="visualization-prediction-point"
            cx={plotX(point.x, bounds)}
            cy={plotY(point.prediction, bounds)}
            r="3.6"
          >
            <title>{`prediction (${formatNumber(point.x)}, ${formatNumber(point.prediction)})`}</title>
          </circle>
        ))}
        <text className="visualization-axis-label" x={PLOT_WIDTH / 2} y={PLOT_HEIGHT - 5}>
          {data.inputLabel}
        </text>
        <text className="visualization-axis-label" x="4" y={PLOT_HEIGHT / 2} transform={`rotate(-90 4 ${PLOT_HEIGHT / 2})`}>
          output
        </text>
      </svg>
      <div className="visualization-meta-row">
        <span>x-axis: {data.inputLabel}</span>
        <span>y-axis: output</span>
      </div>
      <div className="visualization-legend">
        <span>
          <i className="legend-dot legend-target" />
          Target data
        </span>
        <span>
          <i className="legend-dot legend-prediction" />
          Predictions
        </span>
      </div>
    </div>
  )
}

function HeatmapVisualization({ data, mode }: { data: HeatmapData; mode: 'target' | 'prediction' }): ReactElement {
  const range = numericRange(data.values)

  return (
    <div className="visualization-content">
      <div
        className="visualization-heatmap"
        role="img"
        aria-label={mode === 'target' ? 'Target heatmap' : 'Prediction heatmap'}
        style={{ gridTemplateColumns: `repeat(${data.columns}, 1fr)` }}
      >
        {data.values.map((value, index) => (
          <div
            key={index}
            className="visualization-heatmap-cell"
            style={{ backgroundColor: heatmapColor(value, range) }}
            title={formatNumber(value)}
          >
            {formatNumber(value, 2)}
          </div>
        ))}
      </div>
      <div className="visualization-meta-row">
        <span>
          {data.rows} x {data.columns}
        </span>
        <span>
          {formatNumber(range.min)} to {formatNumber(range.max)}
        </span>
      </div>
    </div>
  )
}

function buildVisualizationData(graph: GraphModel): VisualizationData {
  let evaluatedGraph: GraphModel
  try {
    evaluatedGraph = forwardPass(graph).graph
  } catch {
    return {
      kind: 'unsupported',
      message: 'Build a valid graph with one loss node to visualize target data and predictions.',
    }
  }

  const lossNode = evaluatedGraph.nodes.find((node) => node.type === 'loss')
  if (!lossNode) {
    return { kind: 'unsupported', message: 'Add a loss node to define the prediction and target tensors.' }
  }

  const predictionEdge = incomingEdges(evaluatedGraph, lossNode.id).find((edge) => (edge.inputSlot ?? 0) === 0)
  const targetEdge = incomingEdges(evaluatedGraph, lossNode.id).find((edge) => (edge.inputSlot ?? 0) === 1)
  const predictionNode = predictionEdge ? nodeById(evaluatedGraph, predictionEdge.source) : undefined
  const targetNode = targetEdge ? nodeById(evaluatedGraph, targetEdge.source) : undefined

  if (!predictionNode || !targetNode) {
    return { kind: 'unsupported', message: 'Connect prediction and target values to the loss node.' }
  }

  const prediction = toTensor(predictionNode.value ?? predictionNode.params.value)
  const target = toTensor(targetNode.value ?? targetNode.params.value)
  const inputNodes = inputNodesUpstreamOf(evaluatedGraph, predictionNode.id)

  if (inputNodes.length === 1) {
    return lineDataFor(inputNodes[0], prediction, target)
  }

  if (inputNodes.length === 2) {
    return heatmapDataFor(inputNodes as [GraphNode, GraphNode], prediction, target)
  }

  return {
    kind: 'unsupported',
    message: 'Visualization supports networks with one or two input nodes connected to the prediction.',
  }
}

function lineDataFor(inputNode: GraphNode, prediction: TensorValue, target: TensorValue): VisualizationData {
  const input = toTensor(inputNode.value ?? inputNode.params.value)
  const pointCount = Math.max(input.data.length, prediction.data.length, target.data.length)

  if (!canExpandToSize(input, pointCount) || !canExpandToSize(prediction, pointCount) || !canExpandToSize(target, pointCount)) {
    return {
      kind: 'unsupported',
      message: 'Single-input visualization needs matching input, prediction, and target tensor lengths.',
    }
  }

  return {
    kind: 'line',
    inputLabel: inputNode.label,
    points: Array.from({ length: pointCount }, (_, index) => ({
      x: tensorEntryAt(input, index),
      prediction: tensorEntryAt(prediction, index),
      target: tensorEntryAt(target, index),
    })),
  }
}

function heatmapDataFor(inputNodes: [GraphNode, GraphNode], prediction: TensorValue, target: TensorValue): VisualizationData {
  const shape = firstTwoDimensionalShape([prediction, target, ...inputNodes.map((node) => toTensor(node.value ?? node.params.value))])
  if (!shape) {
    return {
      kind: 'unsupported',
      message: 'Two-input visualization needs 2D tensor values so entries can be drawn as heatmap cells.',
    }
  }

  const [rows, columns] = shape
  const size = rows * columns
  if (!canExpandToSize(prediction, size) || !canExpandToSize(target, size)) {
    return {
      kind: 'unsupported',
      message: 'Prediction and target tensors must match the heatmap shape.',
    }
  }

  return {
    kind: 'heatmap',
    inputLabels: [inputNodes[0].label, inputNodes[1].label],
    prediction: { rows, columns, values: expandTensor(prediction, size) },
    target: { rows, columns, values: expandTensor(target, size) },
  }
}

function incomingEdges(graph: GraphModel, nodeId: string): GraphEdge[] {
  return graph.edges
    .filter((edge) => edge.target === nodeId)
    .sort((first, second) => (first.inputSlot ?? 0) - (second.inputSlot ?? 0))
}

function nodeById(graph: GraphModel, nodeId: string): GraphNode | undefined {
  return graph.nodes.find((node) => node.id === nodeId)
}

function inputNodesUpstreamOf(graph: GraphModel, nodeId: string): GraphNode[] {
  const inputIds = new Set<string>()
  const visited = new Set<string>()

  const visit = (currentId: string) => {
    if (visited.has(currentId)) return
    visited.add(currentId)
    const current = nodeById(graph, currentId)
    if (!current) return
    if (current.type === 'input') {
      inputIds.add(current.id)
      return
    }
    for (const edge of incomingEdges(graph, currentId)) {
      visit(edge.source)
    }
  }

  visit(nodeId)
  return graph.nodes.filter((node) => inputIds.has(node.id))
}

function firstTwoDimensionalShape(values: TensorValue[]): [number, number] | undefined {
  const candidate = values.find((value) => value.shape.length === 2)
  if (!candidate) return undefined
  return [candidate.shape[0], candidate.shape[1]]
}

function canExpandToSize(value: TensorValue, size: number): boolean {
  return isScalarTensor(value) || value.data.length === size
}

function expandTensor(value: TensorValue, size: number): number[] {
  return Array.from({ length: size }, (_, index) => tensorEntryAt(value, index))
}

function tensorEntryAt(value: TensorValue, index: number): number {
  if (isScalarTensor(value)) return value.data[0] ?? 0
  return value.data[index] ?? 0
}

interface LineBounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

function boundsForLinePoints(points: Array<{ x: number; target: number; prediction: number }>): LineBounds {
  const xRange = numericRange(points.map((point) => point.x))
  const yRange = numericRange(points.flatMap((point) => [point.target, point.prediction]))
  return {
    minX: paddedMin(xRange.min, xRange.max),
    maxX: paddedMax(xRange.min, xRange.max),
    minY: paddedMin(yRange.min, yRange.max),
    maxY: paddedMax(yRange.min, yRange.max),
  }
}

function plotX(value: number, bounds: LineBounds): number {
  const span = bounds.maxX - bounds.minX || 1
  return PLOT_PADDING + ((value - bounds.minX) / span) * (PLOT_WIDTH - PLOT_PADDING * 2)
}

function plotY(value: number, bounds: LineBounds): number {
  const span = bounds.maxY - bounds.minY || 1
  return PLOT_HEIGHT - PLOT_PADDING - ((value - bounds.minY) / span) * (PLOT_HEIGHT - PLOT_PADDING * 2)
}

function paddedMin(min: number, max: number): number {
  if (min !== max) return min
  return min - 1
}

function paddedMax(min: number, max: number): number {
  if (min !== max) return max
  return max + 1
}

function numericRange(values: number[]): { min: number; max: number } {
  if (values.length === 0) return { min: 0, max: 1 }
  return {
    min: Math.min(...values),
    max: Math.max(...values),
  }
}

function heatmapColor(value: number, range: { min: number; max: number }): string {
  if (range.min === range.max) return 'hsl(180 45% 55%)'
  const normalized = (value - range.min) / (range.max - range.min)
  const hue = 210 - normalized * 190
  const lightness = 76 - Math.abs(normalized - 0.5) * 28
  return `hsl(${hue} 72% ${lightness}%)`
}

function titleForVisualization(data: VisualizationData): string {
  if (data.kind === 'line') return 'Input-output graph'
  if (data.kind === 'heatmap') return 'Two-input heatmap'
  return 'No visualization available'
}
