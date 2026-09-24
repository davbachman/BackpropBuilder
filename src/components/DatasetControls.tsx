import { useMemo, useState } from 'react'
import { ArrowRight, ChartScatter, Play } from 'lucide-react'
import { datasetsForModel, graphWithDatasetSample, modelDatasetEvaluation, trainModelDataset } from '../domain/modelDatasets'
import type { ModelDataset, ModelSample } from '../domain/modelDatasets'
import type { GraphModel } from '../domain/types'
import './datasetControls.css'

export interface DatasetControlsProps {
  graph: GraphModel
  onGraphChange: (graph: GraphModel, epochs?: number, loss?: number) => void
}

const PLOT = { width: 270, height: 195, padding: 23 }
const GRID = 20
const nice = (number: number) => Number.isFinite(number) ? number.toFixed(3) : '—'
function extent(values: number[]): [number, number] { const low = Math.min(...values), high = Math.max(...values), gap = (high - low || 1) * .1; return [low - gap, high + gap] }
const lerp = (range: [number, number], fraction: number) => range[0] + fraction * (range[1] - range[0])

export function DatasetControls({ graph, onGraphChange }: DatasetControlsProps) {
  const options = useMemo(() => datasetsForModel(graph), [graph])
  const detail = graph.groups?.find(group => group.id === 'network')?.detail
  const source = graph.nodes.find(node => node.type === 'dataset')
  const dataset = options.find(candidate => candidate.id === (source?.params.dataset ?? detail?.datasetKind)) ?? options[0]
  const selected = source?.params.datasetIndex ?? (typeof detail?.sampleIndex === 'number' ? detail.sampleIndex : undefined)
  const [busy, setBusy] = useState(false), [message, setMessage] = useState('')
  const results = useMemo(() => {
    try {
      const training = modelDatasetEvaluation(graph, dataset.samples.filter(sample => sample.split === 'train'))
      const test = modelDatasetEvaluation(graph, dataset.samples.filter(sample => sample.split === 'test'))
      const oneInput = dataset.samples[0].x.length === 1
      const xRange = extent(dataset.samples.map(sample => sample.x[0]))
      const yRange = oneInput ? extent(dataset.samples.map(sample => sample.y)) : extent(dataset.samples.map(sample => sample.x[1]))
      const plotSamples: ModelSample[] = oneInput ? Array.from({ length: 50 }, (_, index) => ({ x: [lerp(xRange, index / 49)], y: 0, split: 'test' })) : Array.from({ length: GRID * GRID }, (_, index) => ({ x: [lerp(xRange, (index % GRID + .5) / GRID), lerp(yRange, 1 - (Math.floor(index / GRID) + .5) / GRID)], y: 0, split: 'test' }))
      const curve = modelDatasetEvaluation(graph, plotSamples).predictions
      return { training: training.loss, test: test.loss, curve, oneInput, xRange, yRange: oneInput ? extent([...dataset.samples.map(sample => sample.y), ...curve]) : yRange, error: '' }
    } catch (error) { return { training: NaN, test: NaN, curve: [], oneInput: true, xRange: [0, 1] as [number, number], yRange: [0, 1] as [number, number], error: error instanceof Error ? error.message : 'Complete the graph to evaluate this dataset.' } }
  }, [graph, dataset])

  function selectSample(nextDataset: ModelDataset, index: number) {
    try { onGraphChange(graphWithDatasetSample(graph, nextDataset, index)); setMessage('') }
    catch (error) { setMessage(error instanceof Error ? error.message : 'This sample cannot run yet.') }
  }

  async function train(epochs: number) {
    setBusy(true); setMessage('')
    await new Promise(resolve => window.setTimeout(resolve, 0))
    try {
      const next = trainModelDataset(graph, dataset, epochs)
      const loss = modelDatasetEvaluation(next, dataset.samples.filter(sample => sample.split === 'train')).loss
      onGraphChange(next, epochs, loss)
    } catch (error) { setMessage(error instanceof Error ? error.message : 'The update failed. Try a smaller learning rate.') }
    finally { setBusy(false) }
  }

  const plotX = (number: number) => PLOT.padding + (number - results.xRange[0]) / (results.xRange[1] - results.xRange[0]) * (PLOT.width - 2 * PLOT.padding)
  const plotY = (number: number) => PLOT.height - PLOT.padding - (number - results.yRange[0]) / (results.yRange[1] - results.yRange[0]) * (PLOT.height - 2 * PLOT.padding)
  const curvePath = results.oneInput ? results.curve.map((prediction, index) => `${index ? 'L' : 'M'} ${plotX(lerp(results.xRange, index / (results.curve.length - 1)))} ${plotY(prediction)}`).join(' ') : ''
  const trainCount = dataset.samples.filter(sample => sample.split === 'train').length
  const targetRange = extent(dataset.samples.map(sample => sample.y))
  const color = (value: number) => { const fraction = dataset.classification ? value : (value - targetRange[0]) / (targetRange[1] - targetRange[0]); return `hsl(${205 - Math.max(0, Math.min(1, fraction)) * 180} 55% 66%)` }

  return <section className="dataset-controls" aria-label="Dataset experiments">
    <div className="dataset-panel-heading"><span>LEARN FROM DATA</span><ChartScatter size={15} /></div>
    <h3>{dataset.classification ? 'Learn a decision boundary' : 'Fit a function'}</h3>
    <p className="dataset-intro">The curve and colors come from this canvas’s weights. Choose a point to follow its calculation.</p>
    <label className="dataset-select-label">Dataset<select aria-label="Training dataset" value={dataset.id} disabled={busy} onChange={event => { const next = options.find(option => option.id === event.target.value)!; selectSample(next, next.samples.findIndex(sample => sample.split === 'train')) }}>{options.map(option => <option key={option.id} value={option.id}>{option.label}</option>)}</select></label>
    <svg viewBox={`0 0 ${PLOT.width} ${PLOT.height}`} className="dataset-plot" role="group" aria-label={dataset.classification ? 'Decision boundary and dataset samples' : 'Regression predictions and dataset samples'}>
      <defs><clipPath id="dataset-plot-clip"><rect x={PLOT.padding} y={PLOT.padding} width={PLOT.width - 2 * PLOT.padding} height={PLOT.height - 2 * PLOT.padding} /></clipPath></defs>
      <rect className="dataset-plot-background" x={PLOT.padding} y={PLOT.padding} width={PLOT.width - 2 * PLOT.padding} height={PLOT.height - 2 * PLOT.padding} rx="4" />
      <g clipPath="url(#dataset-plot-clip)">{!results.oneInput && results.curve.map((prediction, index) => <rect key={index} className="dataset-boundary-cell" x={PLOT.padding + index % GRID * (PLOT.width - 2 * PLOT.padding) / GRID} y={PLOT.padding + Math.floor(index / GRID) * (PLOT.height - 2 * PLOT.padding) / GRID} width={(PLOT.width - 2 * PLOT.padding) / GRID + .2} height={(PLOT.height - 2 * PLOT.padding) / GRID + .2} fill={color(prediction)} opacity=".34" />)}
        {results.oneInput && <path className="dataset-regression-curve" d={curvePath} />}
        {dataset.samples.map((sample, index) => <circle key={index} className={`dataset-point ${sample.split === 'test' ? 'is-test' : ''} ${selected === index ? 'is-selected' : ''}`} cx={plotX(sample.x[0])} cy={plotY(results.oneInput ? sample.y : sample.x[1])} r={selected === index ? 5 : 3.6} fill={sample.split === 'test' ? 'white' : dataset.classification || !results.oneInput ? color(sample.y) : '#638e94'} stroke={dataset.classification || !results.oneInput ? color(sample.y) : '#638e94'} role="button" tabIndex={0} aria-label={`Use sample ${index + 1}, ${sample.split} set`} onClick={() => selectSample(dataset, index)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectSample(dataset, index) } }}><title>{`${sample.split === 'train' ? 'Training' : 'Test'}: x = ${sample.x.map(nice).join(', ')}, target = ${nice(sample.y)}`}</title></circle>)}</g>
      <text x={PLOT.width / 2} y={PLOT.height - 3} textAnchor="middle">{results.oneInput ? 'input x' : 'input x₁'}</text><text x="8" y={PLOT.height / 2} textAnchor="middle" transform={`rotate(-90 8 ${PLOT.height / 2})`}>{results.oneInput ? 'target / prediction' : 'input x₂'}</text>
      <text x={PLOT.padding} y={PLOT.height - 10}>{results.xRange[0].toFixed(1)}</text><text x={PLOT.width - PLOT.padding} y={PLOT.height - 10} textAnchor="end">{results.xRange[1].toFixed(1)}</text>
    </svg>
    <div className="dataset-legend"><span><i /> {trainCount} training</span><span><i className="is-test" /> {dataset.samples.length - trainCount} held out</span><span className="dataset-prediction-key">{dataset.classification ? 'color = P(class 1)' : results.oneInput ? 'line = prediction' : 'color = prediction'}</span></div>
    <label className="dataset-select-label dataset-sample-label">Inspect a point<select aria-label="Dataset sample" value={selected ?? ''} disabled={busy} onChange={event => selectSample(dataset, Number(event.target.value))}><option value="" disabled>Choose a data point</option>{dataset.samples.map((sample, index) => <option value={index} key={index}>{index + 1} · {sample.split} · ({sample.x.map(n => n.toFixed(2)).join(', ')}) → {sample.y.toFixed(2)}</option>)}</select></label>
    {selected !== undefined && dataset.samples[selected]?.split === 'test' && <p className="dataset-training-note">This is a held-out point. Inspect its gradients; use the epoch buttons below to train on the training split.</p>}
    <div className="dataset-losses"><div><span>Training loss</span><output aria-label="Training loss">{nice(results.training)}</output></div><ArrowRight size={12} /><div><span>Test loss</span><output aria-label="Test loss">{nice(results.test)}</output></div></div>
    <div className="dataset-train-actions"><button type="button" disabled={busy || !!results.error} onClick={() => void train(1)}><Play size={12} /> Train 1 epoch</button><button type="button" disabled={busy || !!results.error} onClick={() => void train(25)}>Train 25 epochs</button></div>
    <p className="dataset-training-note">{busy ? 'Updating the canvas weights…' : 'Each epoch uses all training points. Held-out points never update the weights.'}</p>
    {(message || results.error) && <p className="dataset-error" role="alert">{message || results.error}</p>}
  </section>
}
