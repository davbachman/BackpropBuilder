import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Sigma, SlidersHorizontal, Sparkles, ArrowRight, CirclePlus, X, Grid2X2, Database } from 'lucide-react'
import { Fragment, useState, type ReactElement } from 'react'
import { parseArithmetic } from '../domain/arithmetic'
import { inputArityForNode, outputArityForNode } from '../domain/engine'
import { DATASET_MENU_OPTIONS, customCsvCardHeight, customCsvCardWidth, customCsvLabelWidth, customCsvOutputTop, datasetForNode, datasetExamples, datasetOutputLabelForSlot, datasetOutputValueForSlot, datasetTargetSlotForNode } from '../domain/datasets'
import type { DatasetKind, LossKind } from '../domain/types'
import { formatCompactTensor, formatFullTensor } from '../domain/tensor'
import type { BuilderNodeData } from './BuilderNode'

/** A quiet, legible architecture glyph; the original computation and its live
 * values remain the same objects used by the builder and backpropagation. */
export function SemanticNode(props: NodeProps): ReactElement {
  const data = props.data as BuilderNodeData
  const vertical = Boolean(props.data.vertical)
  const coordinate = Boolean(props.data.coordinate)
  const compactStage = Boolean(props.data.compactStage)
  const node = data.graphNode
  const dataset = node.type === 'dataset' ? datasetForNode(node) : undefined
  const customCsv = node.type === 'dataset' && node.params.dataset === 'custom-csv'
  const parameter = node.type === 'weight' || node.type === 'bias'
  const savedExpression = node.params.expression ?? 'x1 * x2'
  const [expressionDraft, setExpressionDraft] = useState({ source: savedExpression, text: savedExpression, valid: true })
  const expressionText = expressionDraft.source === savedExpression ? expressionDraft.text : savedExpression
  const expressionValid = expressionDraft.source === savedExpression ? expressionDraft.valid : true
  const value = node.value ?? (typeof node.params.value === 'object' ? node.params.value : undefined)
  const imageInput = node.type === 'input' && value?.shape.join(',') === '8,8,1'
  const Icon = dataset ? Database : parameter ? SlidersHorizontal : node.type === 'add' ? CirclePlus : node.type === 'multiply' ? X : node.type === 'matmul' ? Grid2X2 : node.type === 'activation' ? Sparkles : node.type === 'loss' ? Sigma : ArrowRight
  const shape = value?.shape.length ? value.shape.join(' × ') : 'scalar'
  const displayValue = formatCompactTensor(data.showGradient ? node.grad : value)
  const inputs = Math.max(inputArityForNode(node), Number(props.data.displayInputCount ?? 0))
  const outputs = outputArityForNode(node)
  return <div aria-hidden={data.accessible === false} inert={data.accessible === false} style={{ transform: `scale(${Number(data.sceneScale ?? 1)})`, transformOrigin: 'top left', ...(customCsv ? { width: customCsvCardWidth(node), height: customCsvCardHeight(node), paddingRight: customCsvLabelWidth(node) + 22 } : {}) }} className={`semantic-operation ${coordinate ? 'is-coordinate' : ''} ${vertical ? 'is-model-stage' : ''} ${compactStage ? 'is-compact-stage' : ''} ${customCsv ? 'is-custom-csv' : ''} ${imageInput ? 'is-image-input' : ''} ${parameter ? 'is-parameter' : ''} operation-${node.type} ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''}`}>
    {Array.from({ length: inputs }, (_, index) => <Handle key={`in-${index}`} id={`in-${index}`} type="target" position={vertical ? Position.Top : Position.Left} style={vertical ? { left: `${(index + 1) * 100 / (inputs + 1)}%` } : { top: `${(index + 1) * 100 / (inputs + 1)}%` }} className="node-handle" />)}
    <div className="semantic-operation-heading"><Icon size={14}/><span>{imageInput ? 'Digit image' : node.label}</span></div>
    {imageInput ? <div className="semantic-image-thumbnail" aria-label="Handwritten digit input">{value.data.map((pixel,index)=><span key={index} style={{background:`rgba(53,78,112,${.06+Math.max(0,Math.min(1,pixel))*.94})`}}/>)}</div> : null}
    {dataset ? <>
      <select aria-label={`Dataset for ${node.label}`} title={dataset.label} className="semantic-dataset-select nodrag nowheel" value={dataset.kind} onChange={event => data.onDatasetChange(node.id, event.target.value as DatasetKind)}>
        {DATASET_MENU_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
      </select>
      <span className="semantic-dataset-summary">{datasetExamples(dataset).length} examples · {dataset.featureLabels.length} {dataset.featureLabels.length === 1 ? 'feature' : 'features'}</span>
    </> : <>
      {node.type === 'loss' ? <select aria-label="loss" className="semantic-operation-select nodrag nowheel" value={data.lossKind} onChange={event => data.onLossChange(node.id, event.target.value as LossKind)}>
        {data.lossOptions?.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
      </select> : node.type === 'activation' ? <select aria-label="activation" className="semantic-operation-select nodrag nowheel" value={node.params.activation ?? 'identity'} onChange={event => data.onActivationChange(node.id, event.target.value)}>
        <option value="identity">identity</option><option value="relu">ReLU</option><option value="sigmoid">sigmoid</option><option value="tanh">tanh</option>
      </select> : node.type === 'arithmetic' ? <label className="semantic-arithmetic-field"><span>{data.fullFormula.split(' = ')[0]} =</span><input aria-label="Arithmetic expression" className={`nodrag nowheel ${expressionValid ? '' : 'is-invalid'}`} value={expressionText} spellCheck={false} onChange={event => {
        const text = event.target.value
        let valid = true
        try { parseArithmetic(text) } catch { valid = false }
        setExpressionDraft({ source: savedExpression, text, valid })
      }} onBlur={() => { if (expressionValid && expressionText !== savedExpression) data.onExpressionChange(node.id, expressionText) }} onKeyDown={event => { if (event.key === 'Enter') { event.currentTarget.blur(); event.stopPropagation() } }}/></label> : <div className="semantic-operation-formula">{data.showMath ? data.formula : parameter ? 'learned parameter' : node.type.replaceAll('-', ' ')}</div>}
      <div className="semantic-operation-value" title={formatFullTensor(data.showGradient ? node.grad : value)}>{displayValue}</div>
      <span className="semantic-shape">{data.showGradient ? '∂L / ∂x · ' : ''}{shape}</span>
    </>}
    {Array.from({ length: outputs }, (_, index) => {
      const target = dataset && node.params.dataset !== 'custom-csv' && index === datasetTargetSlotForNode(node)
      const label = dataset ? datasetOutputLabelForSlot(node, index) : undefined
      const description = dataset ? node.params.dataset === 'custom-csv' ? label : `${target ? 'Target' : 'Feature'} ${label}` : undefined
      return <Fragment key={`out-${index}`}>
        <Handle id={outputs > 1 ? `out-${index}` : 'out'} type="source" position={vertical ? Position.Bottom : Position.Right} style={vertical ? { left: `${(index + 1) * 100 / (outputs + 1)}%` } : customCsv ? { top: `${customCsvOutputTop(index)}px` } : { top: `${(index + 1) * 100 / (outputs + 1)}%` }} className={`node-handle source-handle${target ? ' dataset-target-handle' : ''}`} aria-label={description ? `${description} output` : undefined} title={description} />
        {dataset && <span className={`semantic-dataset-port${target ? ' is-target' : ''}`} style={vertical ? { left: `${(index + 1) * 100 / (outputs + 1)}%` } : customCsv ? { top: `${customCsvOutputTop(index)}px`, width: customCsvLabelWidth(node) } : { top: `${(index + 1) * 100 / (outputs + 1)}%` }} title={`${description}: ${formatFullTensor(datasetOutputValueForSlot(node, index))}`}>{label}</span>}
      </Fragment>
    })}
  </div>
}
