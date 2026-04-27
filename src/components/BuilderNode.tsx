import {
  Handle,
  Position,
  useUpdateNodeInternals,
  type NodeProps,
} from '@xyflow/react'
import { Box, CircleDot, Crosshair, Database, Plus, Sigma } from 'lucide-react'
import { useEffect, useState, type ReactElement } from 'react'
import {
  FLEX_INPUT_HEIGHT_STEP,
  DATASET_OPTIONS,
  LOSS_OPTIONS,
  MAX_FLEX_INPUT_COUNT,
  datasetForNode,
  datasetOutputLabelForSlot,
  datasetOutputValueForSlot,
  heightForInputCount,
  inputArityForNode,
  isFlexibleInputNodeType,
  lossKindForNode,
  outputArityForNode,
} from '../domain/engine'
import { formatCompactTensor, formatFullTensor, formatTensorInput, parseTensorInput } from '../domain/tensor'
import type { DatasetKind, GraphNode, LossKind, NodeType, TensorValue } from '../domain/types'

export interface BuilderNodeData extends Record<string, unknown> {
  graphNode: GraphNode
  showMath: boolean
  showGradient: boolean
  formula: string
  fullFormula: string
  active: boolean
  hasIncomingValue: boolean
  onFlexibleInputAdd: (nodeId: string) => void
  onValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, value: string) => void
  onLossChange: (nodeId: string, value: LossKind) => void
  onDatasetChange: (nodeId: string, value: DatasetKind) => void
}

const ICON_BY_TYPE: Record<NodeType, typeof CircleDot> = {
  dataset: Database,
  input: CircleDot,
  weight: Sigma,
  bias: Sigma,
  multiply: Crosshair,
  add: Crosshair,
  activation: Box,
  target: CircleDot,
  loss: Sigma,
}

export function BuilderNode(props: NodeProps): ReactElement {
  const data = props.data as BuilderNodeData
  const node = data.graphNode
  const updateNodeInternals = useUpdateNodeInternals()
  const Icon = ICON_BY_TYPE[node.type]
  const inputCount = inputArityForNode(node)
  const outputCount = outputArityForNode(node)
  const isFlexibleInputNode = isFlexibleInputNodeType(node.type)
  const nodeHeight = node.dimensions?.height ?? heightForInputCount(inputCount)
  const canAddInput = isFlexibleInputNode && inputCount < MAX_FLEX_INPUT_COUNT
  const isSource = outputCount > 0
  const editableValue =
    node.type === 'weight' ||
    node.type === 'bias' ||
    ((node.type === 'input' || node.type === 'target') && !data.hasIncomingValue)
  const selectedDataset = node.type === 'dataset' ? datasetForNode(node) : undefined
  const showTypeBadge = node.label.trim().toLowerCase() !== node.type
  const valueText = formatTensorInput(node.params.value)
  const [valueDraft, setValueDraft] = useState({ source: valueText, text: valueText, valid: true })
  const draftValue = valueDraft.source === valueText ? valueDraft.text : valueText
  const valueIsValid = valueDraft.source === valueText ? valueDraft.valid : true

  useEffect(() => {
    if (!isFlexibleInputNode) return
    updateNodeInternals(node.id)
  }, [inputCount, isFlexibleInputNode, node.id, updateNodeInternals])

  return (
    <div
      className={`builder-node node-${node.type} ${isFlexibleInputNode ? 'has-flex-inputs' : ''} ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''}`}
      style={isFlexibleInputNode ? { height: nodeHeight } : undefined}
    >
      {Array.from({ length: inputCount }).map((_, index) => (
        <Handle
          key={index}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          className="node-handle"
          style={{ top: `${FLEX_INPUT_HEIGHT_STEP + index * FLEX_INPUT_HEIGHT_STEP}px` }}
        />
      ))}
      {canAddInput ? (
        <button
          type="button"
          className="node-add-input-button nodrag nopan"
          style={{ top: `${FLEX_INPUT_HEIGHT_STEP + inputCount * FLEX_INPUT_HEIGHT_STEP}px` }}
          aria-label={`Add input to ${node.label}`}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation()
            data.onFlexibleInputAdd(node.id)
          }}
        >
          <Plus size={12} strokeWidth={3} />
        </button>
      ) : null}
      <div className="node-title-row">
        <span className="node-icon" aria-hidden="true">
          <Icon size={14} />
        </span>
        <strong>{node.label}</strong>
        {showTypeBadge ? <span className="node-kind">{node.type}</span> : null}
      </div>
      {data.showMath ? (
        <div className="node-formula">
          <HoverText text={data.formula} tooltip={data.fullFormula} />
        </div>
      ) : null}
      {node.type === 'activation' ? (
        <label className="node-field">
          activation
          <select
            className="nodrag nowheel"
            value={node.params.activation ?? 'identity'}
            onChange={(event) => data.onActivationChange(node.id, event.target.value)}
          >
            <option value="identity">identity</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">sigmoid</option>
            <option value="tanh">tanh</option>
          </select>
        </label>
      ) : null}
      {node.type === 'loss' ? (
        <label className="node-field">
          loss
          <select
            aria-label="loss"
            className="nodrag nowheel"
            value={lossKindForNode(node)}
            onChange={(event) => data.onLossChange(node.id, event.target.value as LossKind)}
          >
            {LOSS_OPTIONS.map((option) => (
              <option key={option.kind} value={option.kind}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {selectedDataset ? (
        <label className="node-field">
          dataset
          <select
            aria-label="dataset"
            className="nodrag nowheel"
            value={selectedDataset.kind}
            onChange={(event) => data.onDatasetChange(node.id, event.target.value as DatasetKind)}
          >
            {DATASET_OPTIONS.map((option) => (
              <option key={option.kind} value={option.kind}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {editableValue ? (
        <label className="node-field">
          value
          <input
            className={`nodrag nowheel ${valueIsValid ? '' : 'is-invalid'}`}
            type="text"
            inputMode="decimal"
            value={draftValue}
            onChange={(event) => {
              const nextValue = event.target.value
              const parsedValue = parseTensorInput(nextValue)
              setValueDraft({ source: valueText, text: nextValue, valid: Boolean(parsedValue) })
              if (parsedValue) data.onValueChange(node.id, parsedValue)
            }}
          />
        </label>
      ) : null}
      <div className="node-metrics">
        {node.type === 'dataset' ? (
          Array.from({ length: outputCount }).map((_, index) => (
            <TensorMetric
              key={index}
              label={datasetOutputLabelForSlot(node, index)}
              value={datasetOutputValueForSlot(node, index)}
            />
          ))
        ) : (
          <TensorMetric label="out" value={node.value} />
        )}
        {node.localDerivative !== undefined ? <TensorMetric label="d local" value={node.localDerivative} /> : null}
        {data.showGradient ? <TensorMetric label="grad" value={node.grad} /> : null}
      </div>
      {isSource ? (
        Array.from({ length: outputCount }).map((_, index) => (
          <Handle
            key={index}
            id={sourceHandleId(index, outputCount)}
            type="source"
            position={Position.Right}
            className="node-handle source-handle"
            style={outputCount > 1 ? { top: `${outputHandleTop(index)}px` } : undefined}
          />
        ))
      ) : null}
      {outputCount > 1 ? (
        <div className="node-output-labels" aria-hidden="true">
          {Array.from({ length: outputCount }).map((_, index) => (
            <span key={index} style={{ top: `${outputHandleTop(index)}px` }}>
              {datasetOutputLabelForSlot(node, index)}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function sourceHandleId(index: number, outputCount: number): string {
  return outputCount === 1 ? 'out' : `out-${index}`
}

function outputHandleTop(index: number): number {
  return FLEX_INPUT_HEIGHT_STEP + index * 30
}

function TensorMetric({ label, value }: { label: string; value: TensorValue | undefined }): ReactElement {
  return (
    <HoverText
      text={`${label} ${formatCompactTensor(value)}`}
      tooltip={`${label} ${formatFullTensor(value)}`}
    />
  )
}

function HoverText({ text, tooltip }: { text: string; tooltip: string }): ReactElement {
  const hasTooltip = text !== tooltip
  return (
    <span
      className={hasTooltip ? 'tensor-hover-text' : undefined}
      data-tooltip={hasTooltip ? tooltip : undefined}
      tabIndex={hasTooltip ? 0 : undefined}
    >
      {text}
    </span>
  )
}
