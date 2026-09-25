import { useState } from 'react'
import { DATASET_MENU_OPTIONS, datasetExampleIndex, datasetExamples, datasetExamplesForNode, datasetForNode, datasetMode, datasetOutputLabelForSlot, datasetOutputValueForSlot } from '../domain/datasets'
import { analyzeCustomCsv } from '../domain/customCsv'
import type { CustomCsvData, DatasetKind, GraphModel, GraphNode, NodeParams } from '../domain/types'
import './datasetWorkbench.css'

interface Props {
  graph: GraphModel
  node: GraphNode
  onParams: (id: string, params: NodeParams) => void
  onDataset: (id: string, dataset: DatasetKind) => void
  onRename?: (id: string, label: string) => void
  onChooseCustomCsv?: (nodeId: string) => void
}

export function DatasetWorkbench({ graph, node, onParams, onDataset, onRename, onChooseCustomCsv }: Props) {
  const dataset = datasetForNode(node), examples = datasetExamplesForNode(node)
  const index = datasetExampleIndex(node), mode = datasetMode(node)
  const [split, setSplit] = useState<'all' | 'train' | 'test'>('all')
  const [csvSettingsError, setCsvSettingsError] = useState('')
  const counts = { train: examples.filter(example => example.split === 'train').length, test: examples.filter(example => example.split === 'test').length }
  const original = datasetExamples(dataset)
  const originalCounts = { train: original.filter(example => example.split === 'train').length, test: original.filter(example => example.split === 'test').length }
  const update = (params: NodeParams) => onParams(node.id, { datasetValues: undefined, ...params })
  const updateCsv = (changes: Partial<CustomCsvData>) => {
    const csv = node.params.customCsv
    if (!csv) return
    const next = { ...csv, ...changes }
    try { analyzeCustomCsv(next); setCsvSettingsError(''); update({ customCsv: next }) }
    catch (error) { setCsvSettingsError(error instanceof Error ? error.message : 'Invalid CSV settings.') }
  }
  const image = dataset.kind === 'digits-8x8' ? datasetOutputValueForSlot(node, 0) : undefined
  return <section className="dataset-workbench" aria-label="Dataset configuration">
    <div className="dataset-workbench-heading"><p className="eyebrow">Dataset</p></div>
    <label className="inspector-field">Source<select aria-label="Dataset selection" value={dataset.kind} onChange={event => onDataset(node.id, event.target.value as DatasetKind)}>
      {DATASET_MENU_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
    </select></label>
    {dataset.description && <p className="coordinate-note">{dataset.description}</p>}
    <label className="inspector-field">Train / test split<select aria-label="Train/test split" value={node.params.trainPercent ?? 'default'} onChange={event => {
      const trainPercent = event.target.value === 'default' ? undefined : Number(event.target.value)
      const nextExamples = datasetExamplesForNode({ ...node, params: { ...node.params, trainPercent } })
      setSplit('train')
      update({ trainPercent, datasetSplit: 'train', datasetIndex: nextExamples.findIndex(example => example.split === 'train') })
    }}>
      <option value="default">Dataset default · {originalCounts.train} train / {originalCounts.test} test</option>
      {[60, 70, 75, 80, 90].map(percent => <option key={percent} value={percent}>{percent}% train / {100 - percent}% test</option>)}
    </select></label>
    {node.params.customCsv && <div className="csv-settings" aria-label="Custom CSV settings">
      <button type="button" className="inspector-wide" onClick={() => onChooseCustomCsv?.(node.id)}>Replace CSV file</button>
      <p className="coordinate-note">Outputs use the CSV column names. Connect a column to a Target block to use it as the target. {graph.edges.some(edge => edge.source === node.id && (edge.sourceSlot ?? 0) === node.params.customCsv!.targetColumn && graph.nodes.some(target => target.id === edge.target && (target.type === 'target' || ((target.type === 'loss' || target.type === 'cross-entropy') && edge.inputSlot === 1))))
        ? <>Connected target: <strong>{datasetOutputLabelForSlot(node, node.params.customCsv.targetColumn)}</strong>.</>
        : <>Until then, <strong>{datasetOutputLabelForSlot(node, node.params.customCsv.targetColumn)}</strong> is the preview target.</>}</p>
      <label className="inspector-field">Task<select aria-label="CSV task" value={node.params.customCsv.task} onChange={event => updateCsv({ task: event.target.value as CustomCsvData['task'] })}>
        <option value="regression">Regression</option><option value="binary-classification">Binary classification</option><option value="classification">Multiclass classification</option>
      </select></label>
      <label className="csv-header-check"><input type="checkbox" checked={node.params.customCsv.hasHeader} onChange={event => updateCsv({ hasHeader: event.target.checked })}/> First row contains headers</label>
      {csvSettingsError && <p role="alert" className="coordinate-note">{csvSettingsError}</p>}
    </div>}
    {!dataset.examples && <label className="inspector-field">Output mode<select aria-label="Dataset output mode" value={mode} onChange={event => update({ datasetMode: event.target.value as 'sample' | 'batch', datasetSplit: 'train', datasetIndex: Math.max(0, examples.findIndex(example => example.split === 'train')) })}><option value="sample">One example</option><option value="batch">Numeric batch</option></select></label>}
    <label className="inspector-field">Examples<select aria-label="Dataset split" value={mode === 'batch' ? node.params.datasetSplit ?? 'all' : split} onChange={event => {
      const next = event.target.value as typeof split; setSplit(next)
      if (mode === 'batch') update({datasetSplit:next})
      else update({datasetSplit:next,datasetIndex:examples.findIndex(example => next === 'all' || example.split === next)})
    }}><option value="all">All examples</option><option value="train">Training · {counts.train}</option><option value="test">Held out · {counts.test}</option></select></label>
    {mode === 'sample' && <label className="inspector-field">Current example<select aria-label="Dataset example" value={index} onChange={event => update({ datasetIndex: Number(event.target.value) })}>
      {examples.map((example, i) => (split === 'all' || example.split === split || i === index) && <option key={i} value={i}>{i + 1}. {example.label} · {example.split}</option>)}
    </select></label>}
    {image && <div className="dataset-image-preview"><div role="img" aria-label={`Handwritten digit ${datasetOutputValueForSlot(node, 1).data[0]}`} style={{display:'grid',gridTemplateColumns:'repeat(8, 1fr)'}}>{image.data.map((pixel, i) => <i key={i} style={{background:`rgba(77,67,128,${pixel})`}} />)}</div><span>Label<strong>{datasetOutputValueForSlot(node, 1).data[0]}</strong><small>8 × 8 × 1 · normalized pixels</small></span></div>}
    {node.params.datasetValues && <p className="coordinate-note">Showing a custom experiment. Choose an example to return to the included data.</p>}
    {onRename && <details className="dataset-name-editor"><summary>Block name</summary><label className="inspector-field">Name<input key={node.id} aria-label="Node name" defaultValue={node.label} onBlur={event => { if (event.target.value.trim() && event.target.value !== node.label) onRename(node.id, event.target.value.trim()) }}/></label></details>}
    {dataset.source && <a href={dataset.source} target="_blank" rel="noreferrer">Dataset source · UCI · CC BY 4.0 ↗</a>}
  </section>
}
