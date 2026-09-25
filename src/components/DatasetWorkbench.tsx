import { useEffect, useRef, useState } from 'react'
import { datasetExampleIndex, datasetExamples, datasetExamplesForNode, datasetForNode, datasetMode, datasetOutputLabelForSlot, datasetOutputValueForSlot } from '../domain/datasets'
import { analyzeCustomCsv } from '../domain/customCsv'
import { evaluateDataset, trainDataset, type DatasetMetrics } from '../domain/datasetTraining'
import { forwardPass } from '../domain/engine'
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
  const dataset = datasetForNode(node), examples = datasetExamplesForNode(node)
  const index = datasetExampleIndex(node), mode = datasetMode(node)
  const [split, setSplit] = useState<'all' | 'train' | 'test'>('all')
  const [status, setStatus] = useState(''), [busy, setBusy] = useState(false)
  const [csvSettingsError, setCsvSettingsError] = useState('')
  const [metrics, setMetrics] = useState<{ graph: GraphModel; train?: DatasetMetrics; test?: DatasetMetrics }>()
  const [predictionPage, setPredictionPage] = useState(0)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const counts = { train: examples.filter(example => example.split === 'train').length, test: examples.filter(example => example.split === 'test').length }
  const original = datasetExamples(dataset)
  const originalCounts = { train: original.filter(example => example.split === 'train').length, test: original.filter(example => example.split === 'test').length }
  const testRows = metrics?.graph === graph ? metrics.test?.rows ?? [] : []
  const pageCount = Math.max(1, Math.ceil(testRows.length / 25))
  const page = Math.min(predictionPage, pageCount - 1)
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
  function inferOnTest() {
    const firstTest = examples.findIndex(example => example.split === 'test')
    if (firstTest < 0) { setStatus('Choose a split with held-out examples first.'); return }
    const testNode = { ...node, params: { ...node.params, datasetSplit: 'test' as const, datasetIndex: firstTest, datasetValues: undefined } }
    const testGraph = { ...graph, nodes: graph.nodes.map(candidate => candidate.id === node.id ? testNode : candidate) }
    try {
      const inferred = forwardPass(testGraph).graph
      let test: DatasetMetrics | undefined
      try { test = evaluateDataset(inferred, node.id, 'test') }
      catch (error) { setStatus(`Ran test-set inference. ${error instanceof Error ? error.message : 'Connect a loss to see test metrics.'}`) }
      if (test) setStatus(`Ran inference on ${test.examples} held-out examples without updating parameters.`)
      setMetrics(test ? { graph: inferred, test } : undefined)
      setPredictionPage(0)
      setSplit('test')
      onGraphChange(inferred)
    } catch (error) { setStatus(error instanceof Error ? error.message : 'Test inference failed.') }
  }
  const image = dataset.kind === 'digits-8x8' ? datasetOutputValueForSlot(node, 0) : undefined
  return <section className="dataset-workbench" aria-label="Dataset training and testing">
    <div className="dataset-workbench-heading"><p className="eyebrow">Data → model → evidence</p><span>{counts.train} train · {counts.test} test</span></div>
    <h3>{dataset.label}</h3>
    <p className="coordinate-note">{dataset.description ?? 'Features and targets stay synchronized. Inspect one example or feed a full numeric batch through the graph.'}</p>
    <label className="inspector-field">Train / test split<select aria-label="Train/test split" value={node.params.trainPercent ?? 'default'} disabled={busy} onChange={event => {
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
    {!dataset.examples && <label className="inspector-field">Output mode<select aria-label="Dataset output mode" value={mode} disabled={busy} onChange={event => update({ datasetMode: event.target.value as 'sample' | 'batch', datasetSplit: 'train', datasetIndex: Math.max(0, examples.findIndex(example => example.split === 'train')) })}><option value="sample">One example</option><option value="batch">Numeric batch</option></select></label>}
    <label className="inspector-field">Examples<select aria-label="Dataset split" value={mode === 'batch' ? node.params.datasetSplit ?? 'all' : split} disabled={busy} onChange={event => {
      const next = event.target.value as typeof split; setSplit(next)
      if (mode === 'batch') update({datasetSplit:next})
      else update({datasetSplit:next,datasetIndex:examples.findIndex(example => next === 'all' || example.split === next)})
    }}><option value="all">All examples</option><option value="train">Training · {counts.train}</option><option value="test">Held out · {counts.test}</option></select></label>
    {mode === 'sample' && <label className="inspector-field">Current example<select aria-label="Dataset example" value={index} disabled={busy} onChange={event => update({ datasetIndex: Number(event.target.value) })}>
      {examples.map((example, i) => (split === 'all' || example.split === split || i === index) && <option key={i} value={i}>{i + 1}. {example.label} · {example.split}</option>)}
    </select></label>}
    {image && <div className="dataset-image-preview"><div role="img" aria-label={`Handwritten digit ${datasetOutputValueForSlot(node, 1).data[0]}`} style={{display:'grid',gridTemplateColumns:'repeat(8, 1fr)'}}>{image.data.map((pixel, i) => <i key={i} style={{background:`rgba(77,67,128,${pixel})`}} />)}</div><span>Label<strong>{datasetOutputValueForSlot(node, 1).data[0]}</strong><small>8 × 8 × 1 · normalized pixels</small></span></div>}
    {node.params.datasetValues && <p className="coordinate-note">Showing a custom experiment. Choose an example to return to the included data.</p>}
    <div className="dataset-training-buttons"><button disabled={busy} onClick={() => void train(1)}>Train 1 epoch</button><button disabled={busy} onClick={() => void train(5)}>Train 5 epochs</button></div>
    <button className="inspector-wide dataset-inference-button" disabled={busy || counts.test === 0} onClick={inferOnTest}>Run inference on test set</button>
    {busy ? <button className="inspector-wide" onClick={() => controller.current?.abort()}>Stop training</button> : <button className="inspector-wide" onClick={() => {
      try { setMetrics({ graph, train: evaluateDataset(graph, node.id, 'train'), test: evaluateDataset(graph, node.id, 'test') }); setPredictionPage(0); setStatus('Evaluated without updating parameters.') }
      catch (error) { setStatus(error instanceof Error ? error.message : 'Evaluation failed.') }
    }}>Evaluate training & held-out data</button>}
    {metrics?.graph === graph && <div className={`dataset-metrics${metrics.train ? '' : ' is-single'}`}>{(['train','test'] as const).filter(part => metrics[part]).map(part => <div key={part}><span>{part === 'train' ? 'Training' : 'Held out'} · {metrics[part]!.examples} examples</span><strong>{metrics[part]!.loss.toFixed(4)} <small>loss</small></strong>{metrics[part]!.accuracy !== undefined && <strong className="dataset-accuracy" aria-label={`${part === 'test' ? 'Test' : 'Training'} accuracy`}>{(100 * metrics[part]!.accuracy!).toFixed(1)}% <small>{dataset.task === 'sequence' ? 'token' : 'class'} accuracy</small></strong>}</div>)}</div>}
    {testRows.length > 0 && <section className="dataset-predictions" aria-label="Test predictions">
      <div className="dataset-predictions-heading"><h4>Test predictions</h4><span>{testRows.length} outputs</span></div>
      <div className="dataset-predictions-scroll"><table><thead><tr><th>Example</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>{testRows.slice(page * 25, (page + 1) * 25).map((row, index) => <tr key={`${page}-${index}`} className={row.correct === undefined ? '' : row.correct ? 'is-correct' : 'is-incorrect'}><td>{row.example}</td><td>{row.actual}</td><td>{row.predicted}{row.correct !== undefined && <span className="dataset-prediction-result" aria-label={row.correct ? 'Correct' : 'Incorrect'}>{row.correct ? '✓' : '×'}</span>}</td></tr>)}</tbody></table></div>
      {pageCount > 1 && <div className="dataset-predictions-pages"><button disabled={page === 0} onClick={() => setPredictionPage(page - 1)}>Previous</button><span>Page {page + 1} of {pageCount}</span><button disabled={page >= pageCount - 1} onClick={() => setPredictionPage(page + 1)}>Next</button></div>}
    </section>}
    {status && <p className="coordinate-note" role="status">{status}</p>}
    <p className="coordinate-note">Epoch training uses only the training split. Step follows the example currently on the canvas.</p>
    {dataset.source && <a href={dataset.source} target="_blank" rel="noreferrer">Dataset source · UCI · CC BY 4.0 ↗</a>}
  </section>
}
