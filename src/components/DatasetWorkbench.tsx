import { useEffect, useRef, useState } from 'react'
import { datasetExampleIndex, datasetExamples, datasetForNode, datasetMode, datasetOutputLabelForSlot, datasetOutputValueForSlot } from '../domain/datasets'
import { analyzeCustomCsv } from '../domain/customCsv'
import { evaluateDataset, trainDataset, type DatasetMetrics } from '../domain/datasetTraining'
import type { CustomCsvData, GraphModel, GraphNode, NodeParams } from '../domain/types'
import './datasetWorkbench.css'

interface Props {
  graph: GraphModel
  node: GraphNode
  onParams: (id: string, params: NodeParams) => void
  onGraphChange: (graph: GraphModel, epochs?: number) => void
  onChooseCustomCsv?: (nodeId: string) => void
}

export function DatasetWorkbench({ graph, node, onParams, onGraphChange, onChooseCustomCsv }: Props) {
  const dataset = datasetForNode(node), examples = datasetExamples(dataset)
  const index = datasetExampleIndex(node), mode = datasetMode(node)
  const [split, setSplit] = useState<'all' | 'train' | 'test'>('all')
  const [status, setStatus] = useState(''), [busy, setBusy] = useState(false)
  const [csvSettingsError, setCsvSettingsError] = useState('')
  const [metrics, setMetrics] = useState<{ graph: GraphModel; train: DatasetMetrics; test: DatasetMetrics }>()
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const counts = { train: examples.filter(example => example.split === 'train').length, test: examples.filter(example => example.split === 'test').length }
  const update = (params: NodeParams) => onParams(node.id, { datasetValues: undefined, ...params })
  const updateCsv = (changes: Partial<CustomCsvData>) => {
    const csv = node.params.customCsv
    if (!csv) return
    const next = { ...csv, ...changes }
    try { analyzeCustomCsv(next); setCsvSettingsError(''); update({ customCsv: next }) }
    catch (error) { setCsvSettingsError(error instanceof Error ? error.message : 'Invalid CSV settings.') }
  }
  async function train(epochs: number) {
    const request = new AbortController()
    controller.current = request
    setBusy(true); setStatus('Preparing training examples…')
    try {
      const next = await trainDataset(graph, node.id, epochs, { signal: request.signal, progress: (done, total) => setStatus(`Training ${done} / ${total} examples`) })
      onGraphChange(next, epochs)
      setStatus(`Completed ${epochs} ${epochs === 1 ? 'epoch' : 'epochs'} on training examples.`)
    } catch (error) { setStatus(error instanceof Error ? error.message : 'Training failed.') }
    finally { setBusy(false); controller.current = null }
  }
  const image = dataset.kind === 'digits-8x8' ? datasetOutputValueForSlot(node, 0) : undefined
  return <section className="dataset-workbench" aria-label="Dataset training and testing">
    <div className="dataset-workbench-heading"><p className="eyebrow">Data → model → evidence</p><span>{counts.train} train · {counts.test} test</span></div>
    <h3>{dataset.label}</h3>
    <p className="coordinate-note">{dataset.description ?? 'Features and targets stay synchronized. Inspect one example or feed a full numeric batch through the graph.'}</p>
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
    {!dataset.examples && <label className="inspector-field">Output mode<select aria-label="Dataset output mode" value={mode} disabled={busy} onChange={event => update({ datasetMode: event.target.value as 'sample' | 'batch', datasetSplit: 'train', datasetIndex: Math.max(0, examples.findIndex(example => example.split === 'train')) })}><option value="sample">One example</option><option value="batch">Numeric batch</option></select></label>}
    <label className="inspector-field">Examples<select aria-label="Dataset split" value={mode === 'batch' ? node.params.datasetSplit ?? 'all' : split} disabled={busy} onChange={event => {
      const next = event.target.value as typeof split; setSplit(next)
      if (mode === 'batch') update({datasetSplit:next})
      else update({datasetIndex:examples.findIndex(example => next === 'all' || example.split === next)})
    }}><option value="all">All examples</option><option value="train">Training · {counts.train}</option><option value="test">Held out · {counts.test}</option></select></label>
    {mode === 'sample' && <label className="inspector-field">Current example<select aria-label="Dataset example" value={index} disabled={busy} onChange={event => update({ datasetIndex: Number(event.target.value) })}>
      {examples.map((example, i) => (split === 'all' || example.split === split || i === index) && <option key={i} value={i}>{i + 1}. {example.label} · {example.split}</option>)}
    </select></label>}
    {image && <div className="dataset-image-preview"><div role="img" aria-label={`Handwritten digit ${datasetOutputValueForSlot(node, 1).data[0]}`} style={{display:'grid',gridTemplateColumns:'repeat(8, 1fr)'}}>{image.data.map((pixel, i) => <i key={i} style={{background:`rgba(77,67,128,${pixel})`}} />)}</div><span>Label<strong>{datasetOutputValueForSlot(node, 1).data[0]}</strong><small>8 × 8 × 1 · normalized pixels</small></span></div>}
    {node.params.datasetValues && <p className="coordinate-note">Showing a custom experiment. Choose an example to return to the included data.</p>}
    <div className="dataset-training-buttons"><button disabled={busy} onClick={() => void train(1)}>Train 1 epoch</button><button disabled={busy} onClick={() => void train(5)}>Train 5 epochs</button></div>
    {busy ? <button className="inspector-wide" onClick={() => controller.current?.abort()}>Stop training</button> : <button className="inspector-wide" onClick={() => {
      try { setMetrics({ graph, train: evaluateDataset(graph, node.id, 'train'), test: evaluateDataset(graph, node.id, 'test') }); setStatus('Evaluated without updating parameters.') }
      catch (error) { setStatus(error instanceof Error ? error.message : 'Evaluation failed.') }
    }}>Evaluate training & held-out data</button>}
    {metrics?.graph === graph && <div className="dataset-metrics">{(['train','test'] as const).map(split => <div key={split}><span>{split === 'train' ? 'Training' : 'Held out'} · {metrics[split].examples} examples</span><strong>{metrics[split].loss.toFixed(4)} <small>loss</small></strong>{metrics[split].accuracy !== undefined && <span>{(100 * metrics[split].accuracy).toFixed(1)}% {dataset.task === 'sequence' ? 'token' : 'class'} accuracy</span>}</div>)}</div>}
    {status && <p className="coordinate-note" role="status">{status}</p>}
    <p className="coordinate-note">Epoch training uses only the training split. Step follows the example currently on the canvas.</p>
    {dataset.source && <a href={dataset.source} target="_blank" rel="noreferrer">Dataset source · UCI · CC BY 4.0 ↗</a>}
  </section>
}
