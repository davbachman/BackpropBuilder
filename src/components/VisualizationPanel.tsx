import { useMemo, type ReactElement } from 'react'
import { formatNumber, forwardPass } from '../domain/engine'
import { isScalarTensor, tensorSize, tensorValue, toTensor } from '../domain/tensor'
import type { GraphEdge, GraphModel, GraphNode, TensorValue } from '../domain/types'

const PLOT_WIDTH = 300
const PLOT_HEIGHT = 220
const PLOT_PADDING = 30
const PREDICTION_SAMPLE_COUNT = 80
const SURFACE_GRID_SIZE = 25

type VisualizationData =
  | {
      kind: 'single-input'
      inputLabel: string
      xRange: NumericRange
      yRange: NumericRange
      targetPoints: Point2D[]
      predictionSamples: Point2D[]
    }
  | {
      kind: 'two-input'
      inputLabels: [string, string]
      xRange: NumericRange
      yRange: NumericRange
      targetRange: NumericRange
      predictionRange: NumericRange
      targetPoints: ColoredPoint[]
      predictionSurface: PredictionSurface
    }
  | {
      kind: 'unsupported'
      message: string
    }

interface Point2D {
  x: number
  y: number
}

interface ColoredPoint extends Point2D {
  value: number
}

interface PredictionSurface {
  rows: number
  columns: number
  cells: ColoredPoint[]
}

interface NumericRange {
  min: number
  max: number
}

export function VisualizationPanel({ graph }: { graph: GraphModel }): ReactElement {
  const data = useMemo(() => buildVisualizationData(graph), [graph])

  return (
    <section className="inspector-card visualization-panel" aria-label="Visualization panel">
      <div className="visualization-heading">
        <div>
          <p className="eyebrow">Visualization</p>
          <h3>{titleForVisualization(data)}</h3>
        </div>
      </div>

      {data.kind === 'single-input' ? <SingleInputVisualization data={data} /> : null}
      {data.kind === 'two-input' ? <TwoInputVisualization data={data} /> : null}
      {data.kind === 'unsupported' ? <p className="visualization-empty">{data.message}</p> : null}
    </section>
  )
}

function SingleInputVisualization({
  data,
}: {
  data: Extract<VisualizationData, { kind: 'single-input' }>
}): ReactElement {
  const bounds = boundsForRanges(data.xRange, data.yRange)
  const predictionPath = data.predictionSamples
    .toSorted((first, second) => first.x - second.x)
    .map((point, index) => {
      const command = index === 0 ? 'M' : 'L'
      return `${command} ${plotX(point.x, bounds)} ${plotY(point.y, bounds)}`
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
        <PlotAxes xLabel={data.inputLabel} yLabel="output" />
        <path
          className="visualization-prediction-line"
          data-sample-count={data.predictionSamples.length}
          d={predictionPath}
        >
          <title>prediction curve over sampled input range</title>
        </path>
        {data.targetPoints.map((point, index) => (
          <circle
            key={`target-${index}`}
            className="visualization-target-point"
            cx={plotX(point.x, bounds)}
            cy={plotY(point.y, bounds)}
            fill="var(--loss)"
            r="4.5"
          >
            <title>{`target (${formatNumber(point.x)}, ${formatNumber(point.y)})`}</title>
          </circle>
        ))}
      </svg>
      <div className="visualization-meta-row">
        <span>x-axis: {data.inputLabel}</span>
        <span>y-axis: output</span>
      </div>
      <VisualizationLegend />
    </div>
  )
}

function TwoInputVisualization({
  data,
}: {
  data: Extract<VisualizationData, { kind: 'two-input' }>
}): ReactElement {
  const bounds = boundsForRanges(data.xRange, data.yRange)
  const cellWidth = (PLOT_WIDTH - PLOT_PADDING * 2) / data.predictionSurface.columns
  const cellHeight = (PLOT_HEIGHT - PLOT_PADDING * 2) / data.predictionSurface.rows

  return (
    <div className="visualization-content">
      <svg
        className="visualization-plot visualization-surface-plot"
        role="img"
        aria-label="Two-input prediction heatmap with target data"
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
      >
        {data.predictionSurface.cells.map((cell, index) => {
          const row = Math.floor(index / data.predictionSurface.columns)
          const column = index % data.predictionSurface.columns
          return (
            <rect
              key={`prediction-${index}`}
              className="visualization-heatmap-cell"
              x={PLOT_PADDING + column * cellWidth}
              y={PLOT_PADDING + row * cellHeight}
              width={cellWidth + 0.5}
              height={cellHeight + 0.5}
              fill={heatmapColor(cell.value, data.predictionRange)}
              fillOpacity="0.46"
            >
              <title>{`prediction ${formatNumber(cell.value)}`}</title>
            </rect>
          )
        })}
        <PlotAxes xLabel={data.inputLabels[0]} yLabel={data.inputLabels[1]} />
        {data.targetPoints.map((point, index) => (
          <circle
            key={`target-${index}`}
            className="visualization-target-point"
            cx={plotX(point.x, bounds)}
            cy={plotY(point.y, bounds)}
            r="5"
            fill={heatmapColor(point.value, data.targetRange)}
          >
            <title>{`target (${formatNumber(point.x)}, ${formatNumber(point.y)}) = ${formatNumber(point.value)}`}</title>
          </circle>
        ))}
      </svg>
      <div className="visualization-meta-row">
        <span>x-axis: {data.inputLabels[0]}</span>
        <span>y-axis: {data.inputLabels[1]}</span>
      </div>
      <VisualizationLegend />
    </div>
  )
}

function PlotAxes({ xLabel, yLabel }: { xLabel: string; yLabel: string }): ReactElement {
  return (
    <>
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
      <text className="visualization-axis-label" x={PLOT_WIDTH / 2} y={PLOT_HEIGHT - 5}>
        {xLabel}
      </text>
      <text className="visualization-axis-label" x="4" y={PLOT_HEIGHT / 2} transform={`rotate(-90 4 ${PLOT_HEIGHT / 2})`}>
        {yLabel}
      </text>
    </>
  )
}

function VisualizationLegend(): ReactElement {
  return (
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

  const inputNodes = inputNodesUpstreamOf(evaluatedGraph, predictionNode.id)

  if (inputNodes.length === 1) {
    return singleInputDataFor(evaluatedGraph, inputNodes[0], predictionNode, targetNode)
  }

  if (inputNodes.length === 2) {
    return twoInputDataFor(evaluatedGraph, inputNodes as [GraphNode, GraphNode], predictionNode, targetNode)
  }

  return {
    kind: 'unsupported',
    message: 'Visualization supports networks with one or two input nodes connected to the prediction.',
  }
}

function singleInputDataFor(
  graph: GraphModel,
  inputNode: GraphNode,
  predictionNode: GraphNode,
  targetNode: GraphNode,
): VisualizationData {
  const input = toTensor(inputNode.value ?? inputNode.params.value)
  const target = toTensor(targetNode.value ?? targetNode.params.value)
  const pointCount = Math.max(input.data.length, target.data.length)

  if (!canExpandToSize(input, pointCount) || !canExpandToSize(target, pointCount)) {
    return {
      kind: 'unsupported',
      message: 'Single-input visualization needs matching input and target tensor lengths.',
    }
  }

  const targetPoints = Array.from({ length: pointCount }, (_, index) => ({
    x: tensorEntryAt(input, index),
    y: tensorEntryAt(target, index),
  }))
  const xRange = paddedRange(targetPoints.map((point) => point.x))
  const yRange = paddedRange(targetPoints.map((point) => point.y))
  const predictionSamples = sampleSingleInputPredictions(graph, inputNode.id, predictionNode.id, targetNode.id, xRange)

  if (!predictionSamples) {
    return {
      kind: 'unsupported',
      message: 'Could not evaluate predictions across the sampled input range.',
    }
  }

  return {
    kind: 'single-input',
    inputLabel: inputNode.label,
    xRange,
    yRange,
    targetPoints,
    predictionSamples,
  }
}

function twoInputDataFor(
  graph: GraphModel,
  inputNodes: [GraphNode, GraphNode],
  predictionNode: GraphNode,
  targetNode: GraphNode,
): VisualizationData {
  const firstInput = toTensor(inputNodes[0].value ?? inputNodes[0].params.value)
  const secondInput = toTensor(inputNodes[1].value ?? inputNodes[1].params.value)
  const target = toTensor(targetNode.value ?? targetNode.params.value)
  const pointCount = Math.max(firstInput.data.length, secondInput.data.length, target.data.length)

  if (
    !canExpandToSize(firstInput, pointCount) ||
    !canExpandToSize(secondInput, pointCount) ||
    !canExpandToSize(target, pointCount)
  ) {
    return {
      kind: 'unsupported',
      message: 'Two-input visualization needs matching x1, x2, and target tensor lengths.',
    }
  }

  const targetPoints = Array.from({ length: pointCount }, (_, index) => ({
    x: tensorEntryAt(firstInput, index),
    y: tensorEntryAt(secondInput, index),
    value: tensorEntryAt(target, index),
  }))
  const xRange = paddedRange(targetPoints.map((point) => point.x))
  const yRange = paddedRange(targetPoints.map((point) => point.y))
  const predictionSurface = sampleTwoInputPredictions(
    graph,
    [inputNodes[0].id, inputNodes[1].id],
    predictionNode.id,
    targetNode.id,
    xRange,
    yRange,
  )

  if (!predictionSurface) {
    return {
      kind: 'unsupported',
      message: 'Could not evaluate predictions across the sampled input area.',
    }
  }

  return {
    kind: 'two-input',
    inputLabels: [inputNodes[0].label, inputNodes[1].label],
    xRange,
    yRange,
    targetRange: paddedRange(targetPoints.map((point) => point.value)),
    predictionRange: paddedRange(predictionSurface.cells.map((point) => point.value)),
    targetPoints,
    predictionSurface,
  }
}

function sampleSingleInputPredictions(
  graph: GraphModel,
  inputNodeId: string,
  predictionNodeId: string,
  targetNodeId: string,
  xRange: NumericRange,
): Point2D[] | undefined {
  const sampledInputs = Array.from({ length: PREDICTION_SAMPLE_COUNT }, (_, index) =>
    interpolateRange(xRange, index, PREDICTION_SAMPLE_COUNT),
  )
  const prediction = evaluatePredictionWithInputs(
    graph,
    predictionNodeId,
    targetNodeId,
    new Map([[inputNodeId, tensorValue([PREDICTION_SAMPLE_COUNT], sampledInputs)]]),
    [PREDICTION_SAMPLE_COUNT],
  )

  if (!prediction || !canExpandToSize(prediction, PREDICTION_SAMPLE_COUNT)) return undefined
  return sampledInputs.map((x, index) => ({ x, y: tensorEntryAt(prediction, index) }))
}

function sampleTwoInputPredictions(
  graph: GraphModel,
  inputNodeIds: [string, string],
  predictionNodeId: string,
  targetNodeId: string,
  xRange: NumericRange,
  yRange: NumericRange,
): PredictionSurface | undefined {
  const rows = SURFACE_GRID_SIZE
  const columns = SURFACE_GRID_SIZE
  const firstInputValues: number[] = []
  const secondInputValues: number[] = []
  const cells: ColoredPoint[] = []

  for (let row = 0; row < rows; row += 1) {
    const y = interpolateRange({ min: yRange.max, max: yRange.min }, row, rows)
    for (let column = 0; column < columns; column += 1) {
      const x = interpolateRange(xRange, column, columns)
      firstInputValues.push(x)
      secondInputValues.push(y)
      cells.push({ x, y, value: 0 })
    }
  }

  const prediction = evaluatePredictionWithInputs(
    graph,
    predictionNodeId,
    targetNodeId,
    new Map([
      [inputNodeIds[0], tensorValue([rows, columns], firstInputValues)],
      [inputNodeIds[1], tensorValue([rows, columns], secondInputValues)],
    ]),
    [rows, columns],
  )

  const size = rows * columns
  if (!prediction || !canExpandToSize(prediction, size)) return undefined

  return {
    rows,
    columns,
    cells: cells.map((cell, index) => ({ ...cell, value: tensorEntryAt(prediction, index) })),
  }
}

function evaluatePredictionWithInputs(
  graph: GraphModel,
  predictionNodeId: string,
  targetNodeId: string,
  inputValues: Map<string, TensorValue>,
  sampledShape: number[],
): TensorValue | undefined {
  const overriddenNodeIds = new Set([...inputValues.keys(), targetNodeId])
  const sampledGraph: GraphModel = {
    ...graph,
    edges: graph.edges.filter((edge) => !overriddenNodeIds.has(edge.target)),
    nodes: graph.nodes.map((node) => {
      const inputValue = inputValues.get(node.id)
      if (inputValue) {
        return { ...node, params: { ...node.params, value: inputValue }, value: inputValue }
      }
      if (node.id === targetNodeId) {
        const targetValue = tensorValue(sampledShape, Array.from({ length: tensorSize(sampledShape) }, () => 0))
        return { ...node, params: { ...node.params, value: targetValue }, value: targetValue }
      }
      return node
    }),
  }

  try {
    const evaluated = forwardPass(sampledGraph).graph
    const predictionNode = nodeById(evaluated, predictionNodeId)
    return predictionNode ? toTensor(predictionNode.value ?? predictionNode.params.value) : undefined
  } catch {
    return undefined
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

function canExpandToSize(value: TensorValue, size: number): boolean {
  return isScalarTensor(value) || value.data.length === size
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

function boundsForRanges(xRange: NumericRange, yRange: NumericRange): LineBounds {
  return {
    minX: xRange.min,
    maxX: xRange.max,
    minY: yRange.min,
    maxY: yRange.max,
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

function paddedRange(values: number[]): NumericRange {
  if (values.length === 0) return { min: 0, max: 1 }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const padding = min === max ? Math.max(1, Math.abs(min) * 0.1) : (max - min) * 0.05
  return { min: min - padding, max: max + padding }
}

function interpolateRange(range: NumericRange, index: number, count: number): number {
  if (count <= 1) return range.min
  return range.min + (index / (count - 1)) * (range.max - range.min)
}

function heatmapColor(value: number, range: NumericRange): string {
  if (range.min === range.max) return 'hsl(180 45% 55%)'
  const normalized = (value - range.min) / (range.max - range.min)
  const hue = 210 - normalized * 190
  const lightness = 76 - Math.abs(normalized - 0.5) * 28
  return `hsl(${hue} 72% ${lightness}%)`
}

function titleForVisualization(data: VisualizationData): string {
  if (data.kind === 'single-input') return 'Input-output graph'
  if (data.kind === 'two-input') return 'Two-input prediction heatmap'
  return 'No visualization available'
}
