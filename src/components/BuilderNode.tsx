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
  LOSS_OPTIONS,
  TENSOR_TRANSFORM_OPTIONS,
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
import { DATASET_MENU_OPTIONS, customCsvCardHeight, customCsvCardWidth, customCsvLabelWidth, customCsvOutputTop, datasetTargetSlotForNode } from '../domain/datasets'
import { formatCompactTensor, formatFullTensor, formatTensorInput, parseTensorInput } from '../domain/tensor'
import { parseArithmetic } from '../domain/arithmetic'
import type { DatasetKind, GraphNode, LossKind, NodeType, TensorTransformKind, TensorValue } from '../domain/types'

export interface BuilderNodeData extends Record<string, unknown> {
  graphNode: GraphNode
  showMath: boolean
  showGradient: boolean
  formula: string
  fullFormula: string
  lossKind?: LossKind
  lossOptions?: Array<{ kind: LossKind; label: string }>
  active: boolean
  hasIncomingValue: boolean
  onFlexibleInputAdd: (nodeId: string) => void
  onValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, value: string) => void
  onExpressionChange: (nodeId: string, expression: string) => void
  onTransformChange: (nodeId: string, transform: TensorTransformKind) => void
  onLossChange: (nodeId: string, value: LossKind) => void
  onDatasetChange: (nodeId: string, value: DatasetKind) => void
}

const ICON_BY_TYPE: Record<NodeType, typeof CircleDot> = {
  dataset: Database,
  input: CircleDot,
  weight: Sigma,
  bias: Sigma,
  multiply: Crosshair,
  matmul: Crosshair,
  add: Crosshair,
  arithmetic: Sigma,
  activation: Box,
  target: CircleDot,
  loss: Sigma,
  embedding: Database,
  transpose: Crosshair,
  slice: Crosshair,
  concat: Crosshair,
  softmax: Sigma,
  'causal-mask': Box,
  'layer-norm': Sigma,
  reshape: Box,
  'tensor-transform': Box,
  mean: Sigma,
  'cross-entropy': Sigma,
  conv2d: Crosshair,
  avgpool2d: Sigma,
}

export function BuilderNode(props: NodeProps): ReactElement {
  const data = props.data as BuilderNodeData
  const node = data.graphNode
  const updateNodeInternals = useUpdateNodeInternals()
  const Icon = ICON_BY_TYPE[node.type]
  const inputCount = inputArityForNode(node)
  const outputCount = outputArityForNode(node)
  const isFlexibleInputNode = isFlexibleInputNodeType(node.type)
  const variableInputLayout = isFlexibleInputNode || node.type === 'arithmetic'
  const nodeHeight = node.dimensions?.height ?? heightForInputCount(inputCount)
  const canAddInput = isFlexibleInputNode && inputCount < MAX_FLEX_INPUT_COUNT
  const isSource = outputCount > 0
  const editableValue =
    node.type === 'weight' ||
    node.type === 'bias' ||
    ((node.type === 'input' || node.type === 'target') && !data.hasIncomingValue)
  const selectedDataset = node.type === 'dataset' ? datasetForNode(node) : undefined
  const customCsv = node.type === 'dataset' && node.params.dataset === 'custom-csv'
  const lossKind = data.lossKind ?? lossKindForNode(node)
  const lossOptions = data.lossOptions ?? LOSS_OPTIONS
  const showTypeBadge = node.label.trim().toLowerCase() !== node.type && !(node.type === 'weight' && /^param\s+\d+$/i.test(node.label))
  const valueText = formatTensorInput(node.params.value)
  const [valueDraft, setValueDraft] = useState({ source: valueText, text: valueText, valid: true })
  const draftValue = valueDraft.source === valueText ? valueDraft.text : valueText
  const valueIsValid = valueDraft.source === valueText ? valueDraft.valid : true
  const savedExpression = node.params.expression ?? 'x1 * x2'
  const [expressionDraft, setExpressionDraft] = useState({ source: savedExpression, text: savedExpression, valid: true })
  const expressionText = expressionDraft.source === savedExpression ? expressionDraft.text : savedExpression
  const expressionValid = expressionDraft.source === savedExpression ? expressionDraft.valid : true
  const commitExpression = () => {
    if (expressionValid && expressionText !== savedExpression) data.onExpressionChange(node.id, expressionText)
  }

  useEffect(() => {
    if (!variableInputLayout) return
    updateNodeInternals(node.id)
  }, [inputCount, variableInputLayout, node.id, updateNodeInternals])

  return (
    <div
      className={`builder-node node-${node.type} ${customCsv ? 'is-custom-csv' : ''} ${isFlexibleInputNode ? 'has-flex-inputs' : ''} ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''}`}
      style={{ ...(variableInputLayout ? { height: nodeHeight } : {}), ...(customCsv ? { width: customCsvCardWidth(node), height: customCsvCardHeight(node, true) } : {}), ...(typeof data.sceneScale === 'number' ? { transform: `scale(${data.sceneScale})`, transformOrigin: 'top left' } : {}) }}
    >
      {Array.from({ length: inputCount }).map((_, index) => (
        <Handle
          key={index}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          className="node-handle"
          style={node.type === 'loss' ? { top: `${(index + 1) * 100 / (inputCount + 1)}%` } : { top: `${FLEX_INPUT_HEIGHT_STEP + index * FLEX_INPUT_HEIGHT_STEP}px` }}
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
        {showTypeBadge ? <span className="node-kind">{node.type === 'weight' || node.type === 'bias' ? 'Param' : node.type}</span> : null}
      </div>
      {data.showMath && node.type !== 'arithmetic' ? (
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
      {node.type === 'arithmetic' ? (
        <label className="node-field arithmetic-expression-field">
          <span>{data.fullFormula.split(' = ')[0]} =</span>
          <input
            aria-label="Arithmetic expression"
            className={`nodrag nowheel ${expressionValid ? '' : 'is-invalid'}`}
            type="text"
            spellCheck={false}
            value={expressionText}
            placeholder="x1 * x2"
            onChange={(event) => {
              const text = event.target.value
              let valid = true
              try { parseArithmetic(text) } catch { valid = false }
              setExpressionDraft({ source: savedExpression, text, valid })
            }}
            onBlur={commitExpression}
            onKeyDown={(event) => { if (event.key === 'Enter') { event.currentTarget.blur(); event.stopPropagation() } }}
          />
        </label>
      ) : null}
      {node.type === 'tensor-transform' ? (
        <label className="node-field">
          transform
          <select aria-label="Tensor transform" className="nodrag nowheel" value={node.params.transform ?? 'reshape'} onChange={(event) => data.onTransformChange(node.id, event.target.value as TensorTransformKind)}>
            {TENSOR_TRANSFORM_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
          </select>
        </label>
      ) : null}
      {node.type === 'loss' ? (
        <label className="node-field">
          loss
          <select
            aria-label="loss"
            className="nodrag nowheel"
            value={lossKind}
            onChange={(event) => data.onLossChange(node.id, event.target.value as LossKind)}
          >
            {lossOptions.map((option) => (
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
            {DATASET_MENU_OPTIONS.map((option) => (
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
              valueOnly={customCsv}
            />
          ))
        ) : (
          <TensorMetric label="out" value={node.value} />
        )}
        {node.localDerivative !== undefined ? <TensorMetric label="d local" value={node.localDerivative} /> : null}
        {data.showGradient && !customCsv ? <TensorMetric label="grad" value={node.grad} /> : null}
      </div>
      {isSource ? (
        Array.from({ length: outputCount }).map((_, index) => (
          <Handle
            key={index}
            id={sourceHandleId(index, outputCount)}
            type="source"
            position={Position.Right}
            className={`node-handle source-handle${node.type === 'dataset' && node.params.dataset !== 'custom-csv' && index === datasetTargetSlotForNode(node) ? ' dataset-target-handle' : ''}`}
            style={customCsv ? { top: `${customCsvOutputTop(index, true)}px` } : outputCount > 1 ? { top: `${outputHandleTop(index)}px` } : undefined}
          />
        ))
      ) : null}
      {outputCount > 1 ? (
        <div className="node-output-labels" aria-hidden="true" style={customCsv ? { width: customCsvLabelWidth(node) } : undefined}>
          {Array.from({ length: outputCount }).map((_, index) => (
            <span key={index} style={{ top: `${customCsv ? customCsvOutputTop(index, true) : outputHandleTop(index)}px` }} title={datasetOutputLabelForSlot(node, index)}>
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

function TensorMetric({ label, value, valueOnly = false }: { label: string; value: TensorValue | undefined; valueOnly?: boolean }): ReactElement {
  return (
    <HoverText
      text={valueOnly ? formatCompactTensor(value) : `${label} ${formatCompactTensor(value)}`}
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
