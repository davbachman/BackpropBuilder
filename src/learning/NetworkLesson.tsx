import { useMemo, useRef, useState } from 'react'
import type { GraphModel } from '../domain/types'
import { applyNetworkGradients, createNetwork, createNetworkData, networkLoss, networkToGraph, neuronCalculation, runNetwork, trainNetwork } from './network'
import type { Example, LossPoint, NetworkCalculation, NetworkGradient, NetworkKind, NetworkModel } from './network'
import './network.css'

type Phase = 'edit' | 'forward' | 'loss' | 'backward' | 'update'
interface RecordedCalculation { model: NetworkModel; calculation: NetworkCalculation; loss: number | null; gradients: NetworkGradient[] | null }
interface LessonState {
  model: NetworkModel; example: Example; phase: Phase; recorded: RecordedCalculation | null; epoch: number; losses: LossPoint[]
  expanded: string[]; selectedLayer: number; selectedNeuron: number; detail: 'network' | 'layer' | 'neuron' | 'arithmetic'; visibleStep: number
}
interface Comparison { name: string; model: NetworkModel; epochs: number; train: number; validation: number }
const titles: Record<NetworkKind, string> = { linear: 'One line. Two learned numbers.', neuron: 'A neuron is a calculation.', 'small-network': 'Small parts, a bigger idea.', playground: 'Let a network find the boundary.' }
const questions: Record<NetworkKind, string> = {
  linear: 'With x = 2, weight = 3 and bias = 1, what will the prediction be? How far is it from 9?',
  neuron: 'What happens to a negative weighted sum when you switch from identity to ReLU?',
  'small-network': 'Which hidden neuron responds to this input, and how does it change the output?',
  playground: 'Can straight-line neurons combine to separate the two diagonal pairs of regions?',
}
const number = (value: number | undefined) => value === undefined ? '—' : Number(value.toFixed(4)).toString()
const copy = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T

function initialState(kind: NetworkKind): LessonState {
  const model = createNetwork(kind)
  const example: Example = kind === 'linear' || kind === 'neuron' ? { id: 'custom', x: [2], y: 9, split: 'train' } : kind === 'small-network' ? { id: 'custom', x: [1, 2], y: 1.5, split: 'train' } : createNetworkData(kind)[1]
  return { model, example, phase: 'edit', recorded: null, epoch: 0, losses: [], expanded: [], selectedLayer: 0, selectedNeuron: 0, detail: 'network', visibleStep: 0 }
}

function validState(value: unknown, kind: NetworkKind): value is LessonState {
  if (!value || typeof value !== 'object') return false
  const state = value as LessonState
  const model = state.model
  if (!model || model.kind !== kind || !Number.isInteger(model.seed) || !Number.isInteger(model.dataSeed) || !Number.isFinite(model.learningRate) || model.learningRate <= 0 || model.learningRate > 1 || !Array.isArray(model.layers) || !model.layers.length || model.layers.length > 5) return false
  if (!model.layers.every((layer, i) => layer && layer.id === `layer-${i}` && Number.isInteger(layer.inputs) && Number.isInteger(layer.outputs) && layer.outputs > 0 && layer.outputs <= 8 && layer.inputs === (i === 0 ? kind === 'linear' || kind === 'neuron' ? 1 : 2 : model.layers[i - 1].outputs) && Array.isArray(layer.weights) && layer.weights.length === layer.inputs * layer.outputs && Array.isArray(layer.biases) && layer.biases.length === layer.outputs && [...layer.weights, ...layer.biases].every(Number.isFinite) && ['identity', 'relu'].includes(layer.activation))) return false
  if (model.layers.at(-1)!.outputs !== (kind === 'playground' ? 2 : 1)) return false
  if (!state.example || !Array.isArray(state.example.x) || state.example.x.length !== model.layers[0].inputs || !state.example.x.every(Number.isFinite) || !Number.isFinite(state.example.y) || (kind === 'playground' && ![0, 1].includes(state.example.y))) return false
  return ['edit', 'forward', 'loss', 'backward', 'update'].includes(state.phase) && Array.isArray(state.expanded) && state.expanded.every(id => typeof id === 'string') && Number.isInteger(state.selectedLayer) && state.selectedLayer >= 0 && state.selectedLayer < model.layers.length && Number.isInteger(state.selectedNeuron) && state.selectedNeuron >= 0 && state.selectedNeuron < model.layers[state.selectedLayer].outputs && ['network', 'layer', 'neuron', 'arithmetic'].includes(state.detail) && Number.isInteger(state.epoch) && state.epoch >= 0 && Array.isArray(state.losses) && state.losses.every(p => p && Number.isFinite(p.train) && Number.isFinite(p.validation) && Number.isFinite(p.epoch))
}

export function NetworkLesson({ kind, onBuilder }: { kind: NetworkKind; onBuilder?: (graph: GraphModel) => void }) {
  const [state, setState] = useState(() => initialState(kind))
  const [undo, setUndo] = useState<LessonState[]>([])
  const [showMath, setShowMath] = useState(false)
  const [showCode, setShowCode] = useState(false)
  const [showGradients, setShowGradients] = useState(false)
  const [instructorMode, setInstructorMode] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [notice, setNotice] = useState('Choose a prediction, then run the forward calculation.')
  const [training, setTraining] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const data = useMemo(() => createNetworkData(kind, state.model.dataSeed), [kind, state.model.dataSeed])
  const selectedLayer = state.model.layers[state.selectedLayer]
  const calculation = state.recorded?.calculation
  const neuron = calculation ? neuronCalculation(state.recorded!.model, calculation, state.selectedLayer, state.selectedNeuron) : null

  const commit = (next: LessonState) => { setUndo(history => [...history.slice(-29), copy(state)]); setState(next) }
  const invalidate = (model: NetworkModel, example = state.example) => {
    commit({ ...state, model, example, phase: 'edit', recorded: null, visibleStep: 0 })
    setNotice('The experiment changed. Run forward to calculate current values.')
  }
  const runForward = () => {
    commit({ ...state, phase: 'forward', visibleStep: 0, recorded: { model: copy(state.model), calculation: runNetwork(state.model, state.example.x), loss: null, gradients: null } })
    setNotice('Forward pass recorded. Opening a layer or neuron reads this same calculation.')
  }
  const evaluateLoss = () => {
    if (!state.recorded) return
    commit({ ...state, phase: 'loss', recorded: { ...state.recorded, loss: networkLoss(state.model, [state.example]).loss } })
    setNotice(kind === 'playground' ? 'Cross-entropy measures how little probability the model gave the correct class.' : 'Squared error: subtract the target from the prediction, then square the difference.')
  }
  const backward = () => {
    if (!state.recorded || state.recorded.loss === null) return
    const result = networkLoss(state.model, [state.example], true)
    commit({ ...state, phase: 'backward', recorded: { ...state.recorded, gradients: result.gradients } })
    setShowGradients(true)
    setNotice('Gradients computed for the selected example. Parameters have not changed yet.')
  }
  const update = () => {
    if (!state.recorded?.gradients) return
    if (state.example.split === 'validation' && state.example.id !== 'custom') { setNotice('This example is validation data. Inspect its values and gradients, then select a training example to update parameters.'); return }
    try {
      const model = applyNetworkGradients(state.model, state.recorded.gradients)
      commit({ ...state, model, phase: 'update', recorded: { model: copy(model), calculation: runNetwork(model, state.example.x), loss: networkLoss(model, [state.example]).loss, gradients: null } })
      setNotice('One simultaneous parameter update completed. The displayed prediction uses the new parameters. Run forward to start another cycle.')
    } catch (error) { setNotice((error as Error).message) }
  }
  const nextStep = () => {
    if (!state.recorded || state.phase === 'update') { runForward(); return }
    if (state.phase === 'forward' && state.visibleStep < state.model.layers.length - 1) {
      setState({ ...state, visibleStep: state.visibleStep + 1, selectedLayer: state.visibleStep + 1, selectedNeuron: 0 })
      setNotice(`Forward component ${state.visibleStep + 2} of ${state.model.layers.length}: the recorded output of layer ${state.visibleStep + 1} is the input to layer ${state.visibleStep + 2}. Parameters stay unchanged.`)
    } else if (state.phase === 'forward') evaluateLoss()
    else if (state.phase === 'loss') backward()
    else if (state.phase === 'backward') update()
  }
  const train = () => {
    setTraining(true)
    setNotice('Running 20 full-batch epochs on training examples only…')
    window.setTimeout(() => {
      try {
        const result = trainNetwork(state.model, data, 20)
        commit({ ...state, model: result.model, epoch: state.epoch + 20, losses: [...state.losses, ...result.losses.map(loss => ({ ...loss, epoch: state.epoch + loss.epoch }))], phase: 'update', recorded: { model: copy(result.model), calculation: runNetwork(result.model, state.example.x), loss: networkLoss(result.model, [state.example]).loss, gradients: null } })
        setNotice('20 epochs completed: forward → loss → backward → update each epoch. Validation examples were never used for updates.')
      } catch (error) { setNotice((error as Error).message) }
      setTraining(false)
    }, 30)
  }
  const resetParameters = () => {
    const model = createNetwork(kind, state.model.seed, state.model.layers.slice(0, -1).map(layer => layer.outputs))
    if (kind === 'neuron') model.layers[0].activation = selectedLayer.activation
    model.dataSeed = state.model.dataSeed
    model.learningRate = state.model.learningRate
    commit({ ...state, model, epoch: 0, losses: [], phase: 'edit', recorded: null })
    setNotice('Parameters reset reproducibly. The dataset and selected input are unchanged.')
  }
  const save = () => {
    const blob = new Blob([JSON.stringify({ kind: 'backprop-network-experiment', version: 1, state, comparisons, view: { showMath, showCode, showGradients, instructorMode, showControls, zoom } }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob), anchor = document.createElement('a')
    anchor.href = url; anchor.download = `backprop-${kind}-experiment.json`; anchor.click(); URL.revokeObjectURL(url)
    setNotice('Saved inputs, parameters, seeds, hierarchy, comparisons, and display settings.')
  }
  const restore = async (file?: File) => {
    if (!file) return
    try {
      const document = JSON.parse(await file.text())
      if (document.kind !== 'backprop-network-experiment' || document.version !== 1 || !validState(document.state, kind)) throw new Error('This is not a valid saved experiment for this lesson.')
      const imported = document.state as LessonState
      // Recompute cached output from saved parameters instead of trusting imported caches.
      imported.recorded = imported.recorded ? { model: copy(imported.model), calculation: runNetwork(imported.model, imported.example.x), loss: ['loss', 'backward', 'update'].includes(imported.phase) ? networkLoss(imported.model, [imported.example]).loss : null, gradients: imported.phase === 'backward' ? networkLoss(imported.model, [imported.example], true).gradients : null } : null
      imported.visibleStep = Math.min(Math.max(0, imported.visibleStep || 0), imported.model.layers.length - 1)
      commit(imported)
      if (document.view) { setShowMath(!!document.view.showMath); setShowCode(!!document.view.showCode); setShowGradients(!!document.view.showGradients); setInstructorMode(!!document.view.instructorMode); setShowControls(document.view.showControls !== false); setZoom(Math.min(1.8, Math.max(.6, Number(document.view.zoom) || 1))) }
      setComparisons(Array.isArray(document.comparisons) ? document.comparisons.filter((entry: Comparison) => entry && typeof entry.name === 'string' && Number.isFinite(entry.train) && Number.isFinite(entry.validation) && validState({ ...initialState(kind), model: entry.model }, kind)).slice(0, 4) : [])
      setNotice('Experiment restored. Values were reproduced from the saved model and input.')
    } catch (error) { setNotice(`Import failed: ${(error as Error).message}`) }
    if (fileInput.current) fileInput.current.value = ''
  }
  const changeParameter = (index: number, value: number, bias = false) => {
    if (!Number.isFinite(value)) return
    const model = copy(state.model)
    if (bias) model.layers[state.selectedLayer].biases[index] = value
    else model.layers[state.selectedLayer].weights[index] = value
    invalidate(model)
  }
  const chooseNeuron = (layer: number, unit: number) => setState({ ...state, selectedLayer: layer, selectedNeuron: unit, detail: 'neuron', expanded: [...new Set([...state.expanded, state.model.layers[layer].id])] })

  return <div className="network-lesson">
    <section className="nl-intro"><div><p className="nl-eyebrow">{kind === 'playground' ? 'Learn from examples' : 'Follow the numbers'}</p><h2>{titles[kind]}</h2><p>{questions[kind]}</p></div><span className="nl-local">Local · deterministic</span></section>
    <div className="nl-guide" aria-label="Guided activity"><span><b>1 Predict</b> Say what you expect.</span><span><b>2 Run</b> Follow the calculation.</span><span><b>3 Explain</b> Open one neuron.</span><span><b>4 Try</b> Change one input or weight.</span></div>
    <div className="nl-toolbar">
      <button onClick={save}>Save experiment</button><button onClick={() => fileInput.current?.click()}>Import experiment</button><input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={event => void restore(event.target.files?.[0])} />
      <button disabled={!undo.length || training} onClick={() => { const previous = undo.at(-1); if (previous) { setState(previous); setUndo(undo.slice(0, -1)); setNotice('Previous experiment restored.'); } }}>Undo</button>
      {onBuilder && <button onClick={() => onBuilder(networkToGraph(state.model, state.example))}>Open in graph builder</button>}
      <button onClick={() => setShowControls(!showControls)}>{showControls ? 'Hide controls' : 'Show controls'}</button>
      <label className="nl-toggle"><input type="checkbox" checked={instructorMode} onChange={event => setInstructorMode(event.target.checked)} />Presentation mode</label>
    </div>
    <div className={`nl-layout ${showControls ? '' : 'nl-controls-hidden'}`}>
      <main className="nl-main">
        <section className="nl-card nl-execution">
          <div className="nl-card-heading"><h3>Follow one example</h3><span className="nl-phase">{state.phase === 'edit' ? 'Ready to calculate' : `${state.phase[0].toUpperCase()}${state.phase.slice(1)} phase`}</span></div>
          <div className="nl-example">
            {state.example.x.map((value, index) => <label key={index}>Input x{state.example.x.length > 1 ? index + 1 : ''}<input aria-label={`Input x${index + 1}`} type="number" step="0.1" value={value} disabled={training} onChange={event => { const x = [...state.example.x]; x[index] = Number(event.target.value); invalidate(state.model, { ...state.example, id: 'custom', x }); }} /></label>)}
            <label>Target<input aria-label="Target" type="number" step={kind === 'playground' ? 1 : .1} min={kind === 'playground' ? 0 : undefined} max={kind === 'playground' ? 1 : undefined} value={state.example.y} disabled={training} onChange={event => { const y = Number(event.target.value); if (kind !== 'playground' || [0, 1].includes(y)) invalidate(state.model, { ...state.example, id: 'custom', y }); }} /></label>
            <div className="nl-result"><span>{kind === 'playground' ? 'Predicted class' : 'Prediction'}</span><strong data-testid="network-prediction">{number(calculation?.prediction)}</strong></div>
            <div className="nl-result"><span>{kind === 'playground' ? 'Cross-entropy' : 'Squared error'}</span><strong data-testid="network-loss">{state.recorded?.loss === null || !state.recorded ? '—' : number(state.recorded.loss)}</strong></div>
          </div>
          <div className="nl-phases" aria-label="Execution phases">
            <button className={state.phase === 'forward' ? 'active' : ''} disabled={training} onClick={runForward}>1 Forward</button>
            <button className={state.phase === 'loss' ? 'active' : ''} disabled={training || state.phase !== 'forward'} onClick={evaluateLoss}>2 Loss</button>
            <button className={state.phase === 'backward' ? 'active' : ''} disabled={training || state.phase !== 'loss'} onClick={backward}>3 Backward</button>
            <button className={state.phase === 'update' ? 'active' : ''} disabled={training || state.phase !== 'backward' || (state.example.split === 'validation' && state.example.id !== 'custom')} onClick={update}>4 Update parameters</button>
          </div>
          <p className="nl-status" role="status">{notice}</p>
          {state.phase === 'forward' && calculation && <div className="nl-recorded-step"><span>Recorded component {state.visibleStep + 1} / {state.model.layers.length}</span><strong>Layer {state.visibleStep + 1}</strong><code>[{calculation.layers[state.visibleStep].input.map(number).join(', ')}] → [{calculation.layers[state.visibleStep].output.map(number).join(', ')}]</code></div>}
          {state.example.split === 'validation' && state.example.id !== 'custom' && <p className="nl-footnote">Selected validation example: inspect its calculation, but parameter updates are disabled to keep it held out.</p>}
          {!instructorMode && <div className="nl-step-tools"><button disabled={training} onClick={nextStep}>Next visible component</button><button disabled={!calculation} onClick={() => setState({ ...state, detail: 'arithmetic', expanded: [...new Set([...state.expanded, selectedLayer.id])] })}>Step inside selected neuron</button><button disabled={training || state.phase === 'edit' || state.phase === 'update'} onClick={() => { if (state.phase === 'forward') { setState({ ...state, visibleStep: state.model.layers.length - 1 }); setNotice(`Forward phase complete: all ${state.model.layers.length} layers recorded. Evaluate loss next.`) } else if (state.phase === 'loss') setNotice('Loss phase complete. Run backward to compute parameter gradients.'); else setNotice('Backward phase complete. Update parameters to apply the recorded gradients.'); }}>Finish current phase</button></div>}
        </section>

        <section className="nl-card">
          <div className="nl-card-heading"><nav className="nl-breadcrumbs" aria-label="Network breadcrumbs"><button onClick={() => setState({ ...state, detail: 'network' })}>Network</button>{state.detail !== 'network' && <><span>›</span><button onClick={() => setState({ ...state, detail: 'layer' })}>Layer {state.selectedLayer + 1}</button></>}{['neuron', 'arithmetic'].includes(state.detail) && <><span>›</span><button onClick={() => setState({ ...state, detail: 'neuron' })}>Neuron {state.selectedNeuron + 1}</button></>}{state.detail === 'arithmetic' && <span>› Arithmetic</span>}</nav><div className="nl-zoom"><button aria-label="Zoom out network" onClick={() => setZoom(Math.max(.6, zoom - .2))}>−</button><button onClick={() => setZoom(1)}>Fit view</button><button aria-label="Zoom in network" onClick={() => setZoom(Math.min(1.8, zoom + .2))}>+</button></div></div>
          {state.detail !== 'network' && <button className="nl-up" onClick={() => setState({ ...state, detail: state.detail === 'arithmetic' ? 'neuron' : state.detail === 'neuron' ? 'layer' : 'network' })}>↑ Up one level</button>}
          <p className="nl-legend"><span className="nl-dot nl-purple" />Learned weights & biases <span className="nl-dot nl-teal" />Input-dependent activations · Tap a neuron to inspect</p>
          <NetworkDiagram model={state.model} calculation={calculation} input={state.example.x} selected={[state.selectedLayer, state.selectedNeuron]} onSelect={chooseNeuron} zoom={zoom} />
          <div className="nl-modules">{state.model.layers.map((layer, layerIndex) => {
            const isOpen = state.expanded.includes(layer.id)
            return <section className={`nl-module ${state.selectedLayer === layerIndex && state.detail !== 'network' ? 'selected' : ''}`} key={layer.id}>
              <button className="nl-module-title" aria-expanded={isOpen} onClick={() => setState({ ...state, selectedLayer: layerIndex, selectedNeuron: 0, detail: isOpen ? 'network' : 'layer', expanded: isOpen ? state.expanded.filter(id => id !== layer.id) : [...state.expanded, layer.id] })}><span>{isOpen ? '▾' : '▸'} {layerIndex === state.model.layers.length - 1 ? 'Output' : 'Hidden'} layer {layerIndex + 1}</span><small>{layer.inputs} → {layer.outputs} · {layer.activation}</small></button>
              <p>Activations [{calculation ? calculation.layers[layerIndex].output.map(number).join(', ') : 'run forward'}]</p>
              {isOpen && <div className="nl-units">{layer.biases.map((_, unit) => <button key={unit} className={state.selectedLayer === layerIndex && state.selectedNeuron === unit ? 'selected' : ''} onClick={() => chooseNeuron(layerIndex, unit)}>Neuron {unit + 1}<strong>{number(calculation?.layers[layerIndex].output[unit])}</strong></button>)}</div>}
            </section>
          })}</div>
          <p className="nl-footnote">Opening and closing modules preserves parameters and recorded values. Drag the background or scroll to pan; touch gestures scroll normally.</p>
        </section>

        {state.detail !== 'network' && <section className="nl-card nl-inspector" aria-label="Neuron calculation">
          <div className="nl-card-heading"><h3>Layer {state.selectedLayer + 1} · Neuron {state.selectedNeuron + 1}</h3><span className="nl-id">{selectedLayer.id}/neuron-{state.selectedNeuron}</span></div>
          <p>Multiply each incoming value by its weight, add the bias, then apply {selectedLayer.activation === 'identity' ? 'identity (keep the value)' : 'ReLU (keep positive values; replace negative values with zero)'}.</p>
          <div className="nl-calculation-flow"><span>Weighted sum <strong>{number(neuron?.weightedSum)}</strong></span><b>→</b><span>{selectedLayer.activation}<strong>{number(neuron?.output)}</strong></span><button onClick={() => setState({ ...state, detail: state.detail === 'arithmetic' ? 'neuron' : 'arithmetic' })}>{state.detail === 'arithmetic' ? 'Close arithmetic' : 'Open arithmetic'}</button></div>
          <div className="nl-table-scroll"><table><thead><tr><th>Input value</th><th>Learned weight</th><th>Contribution</th>{showGradients && <th>Loss gradient</th>}</tr></thead><tbody>{Array.from({ length: selectedLayer.inputs }, (_, i) => <tr key={i}><td>x{i + 1} = {number(neuron?.input[i])}</td><td><input aria-label={`Layer ${state.selectedLayer + 1} neuron ${state.selectedNeuron + 1} weight ${i + 1}`} type="number" step=".1" value={selectedLayer.weights[i * selectedLayer.outputs + state.selectedNeuron]} disabled={training || instructorMode} onChange={event => changeParameter(i * selectedLayer.outputs + state.selectedNeuron, Number(event.target.value))} /></td><td>{state.detail === 'arithmetic' && neuron ? `${number(neuron.input[i])} × ${number(neuron.weights[i])} = ` : ''}{number(neuron?.products[i])}</td>{showGradients && <td>{number(state.recorded?.gradients?.[state.selectedLayer].weights[i * selectedLayer.outputs + state.selectedNeuron])}</td>}</tr>)}<tr><td>Bias</td><td><input aria-label={`Layer ${state.selectedLayer + 1} neuron ${state.selectedNeuron + 1} bias`} type="number" step=".1" value={selectedLayer.biases[state.selectedNeuron]} disabled={training || instructorMode} onChange={event => changeParameter(state.selectedNeuron, Number(event.target.value), true)} /></td><td>{number(neuron?.bias)}</td>{showGradients && <td>{number(state.recorded?.gradients?.[state.selectedLayer].biases[state.selectedNeuron])}</td>}</tr></tbody></table></div>
          {state.detail === 'arithmetic' && neuron && <p className="nl-equation">{neuron.products.map(number).join(' + ')} + ({number(neuron.bias)}) = {number(neuron.weightedSum)} → {neuron.activation} → <b>{number(neuron.output)}</b></p>}
          {kind === 'playground' && state.selectedLayer === state.model.layers.length - 1 && calculation && <p>These two outputs are scores. Softmax converts them to class probabilities: [{calculation.probabilities.map(number).join(', ')}].</p>}
          {showMath && <p className="nl-equation">z = Σᵢ xᵢwᵢ + b; a = {selectedLayer.activation === 'relu' ? 'max(0, z)' : 'z'}. {kind !== 'playground' && 'L = (prediction − target)².'} Update: w ← w − learning rate × ∂L/∂w.</p>}
          {showCode && <pre>{`z = sum(x * w for x, w in zip(inputs, weights)) + bias\na = ${selectedLayer.activation === 'relu' ? 'max(0, z)' : 'z'}\n# All updates use gradients from the same forward pass\nweight = weight - learning_rate * gradient`}</pre>}
          <ResponsePlot model={state.recorded?.model ?? null} layer={state.selectedLayer} unit={state.selectedNeuron} input={state.example.x} />
        </section>}

        <section className="nl-card"><div className="nl-card-heading"><h3>{kind === 'playground' ? 'Data & decision surface' : 'Data & predictions'}</h3><span className="nl-id">Data seed {state.model.dataSeed}</span></div><p>Tap a data point to trace that example. Filled circles: training. Open circles: validation.</p><DataPlot model={state.recorded?.model ?? null} kind={kind} data={data} selected={state.example.id} onSelect={example => invalidate(state.model, example)} /><p className="nl-footnote">{kind === 'playground' ? 'Verified rule: class 1 when x₁ × x₂ > 0; class 0 otherwise. Background shows P(class 1), from 0 (gold) to 1 (purple).' : kind === 'small-network' ? 'Target surface: max(0, x₁ + x₂). Background shows the network output; point labels give the target.' : 'Targets follow y = 4x + 1. Purple is the current prediction curve.'} {state.example.id !== 'custom' && `Selected ${state.example.id}: ${state.example.split}, target ${number(state.example.y)}.`}</p></section>
      </main>

      {showControls && <aside className="nl-sidebar">
        <section className="nl-card"><h3>Experiment controls</h3>
          <label className="nl-control">Learning rate<input type="number" min=".001" max="1" step=".01" value={state.model.learningRate} disabled={training || instructorMode} onChange={event => { const learningRate = Number(event.target.value); if (learningRate > 0 && learningRate <= 1) invalidate({ ...state.model, learningRate }); }} /></label>
          {(kind === 'linear' || kind === 'neuron') && <><label className="nl-control">Weight w<input aria-label="Weight w" type="number" step=".1" value={state.model.layers[0].weights[0]} disabled={training || instructorMode} onChange={event => { const model = copy(state.model); model.layers[0].weights[0] = Number(event.target.value); invalidate(model) }} /></label><label className="nl-control">Bias b<input aria-label="Bias b" type="number" step=".1" value={state.model.layers[0].biases[0]} disabled={training || instructorMode} onChange={event => { const model = copy(state.model); model.layers[0].biases[0] = Number(event.target.value); invalidate(model) }} /></label></>}
          {kind === 'neuron' && <label className="nl-control">Activation<select value={state.model.layers[0].activation} disabled={training || instructorMode} onChange={event => { const model = copy(state.model); model.layers[0].activation = event.target.value as 'identity' | 'relu'; invalidate(model) }}><option value="identity">Identity</option><option value="relu">ReLU</option></select></label>}
          {kind === 'playground' && !instructorMode && <fieldset className="nl-architecture"><legend>Hidden layers</legend>{state.model.layers.slice(0, -1).map((layer, i) => <label className="nl-control" key={layer.id}>Layer {i + 1} neurons<select aria-label={`Hidden layer ${i + 1} width`} value={layer.outputs} disabled={training} onChange={event => { const widths = state.model.layers.slice(0, -1).map(l => l.outputs); widths[i] = Number(event.target.value); const model = createNetwork(kind, state.model.seed, widths); model.dataSeed = state.model.dataSeed; model.learningRate = state.model.learningRate; commit({ ...initialState(kind), example: state.example, model }); setNotice('Architecture changed; parameters reset. Data seed is unchanged.'); }}>{[1, 2, 3, 4, 6, 8].map(n => <option key={n}>{n}</option>)}</select></label>)}<div className="nl-button-row"><button disabled={state.model.layers.length >= 4 || training} onClick={() => { const model = createNetwork(kind, state.model.seed, [...state.model.layers.slice(0, -1).map(layer => layer.outputs), 3]); model.dataSeed = state.model.dataSeed; model.learningRate = state.model.learningRate; commit({ ...initialState(kind), example: state.example, model }); }}>+ Layer</button><button disabled={state.model.layers.length <= 2 || training} onClick={() => { const model = createNetwork(kind, state.model.seed, state.model.layers.slice(0, -2).map(layer => layer.outputs)); model.dataSeed = state.model.dataSeed; model.learningRate = state.model.learningRate; commit({ ...initialState(kind), example: state.example, model }); }}>− Layer</button></div></fieldset>}
          <button className="nl-primary nl-wide" disabled={training || instructorMode} onClick={train}>{training ? 'Training…' : 'Train 20 epochs'}</button><p className="nl-footnote">Each epoch runs all four phases using {data.filter(p => p.split === 'train').length} training examples. {data.filter(p => p.split === 'validation').length} validation examples are held out. Full batch; no shuffling.</p>
          <div className="nl-training-stat"><strong>{state.epoch}</strong><span>training epochs</span></div>
          {!instructorMode && <><button className="nl-wide" disabled={training} onClick={resetParameters}>Reset parameters</button><label className="nl-control">Parameter seed<input type="number" step="1" value={state.model.seed} disabled={training} onChange={event => { const seed = Number(event.target.value); if (Number.isInteger(seed)) invalidate({ ...state.model, seed }); }} /></label><p className="nl-footnote">Seed is applied on reset. Small-number lessons use fixed initial values.</p><button className="nl-wide" disabled={training || kind === 'linear' || kind === 'neuron'} onClick={() => { const model = { ...state.model, dataSeed: state.model.dataSeed + 1 }; commit({ ...state, model, losses: [], epoch: 0, phase: 'edit', recorded: null }); setNotice('Data regenerated using the next seed. Parameters were preserved; comparison budget restarted.') }}>Regenerate data</button></>}
        </section>
        <section className="nl-card"><h3>Training & validation loss</h3><LossPlot losses={state.losses} /><div className="nl-loss-values"><span>Training <b>{number(state.losses.at(-1)?.train)}</b></span><span>Validation <b>{number(state.losses.at(-1)?.validation)}</b></span></div><p className="nl-footnote">{kind === 'playground' ? 'Mean cross-entropy' : 'Mean squared error'} across each split. Lower is better.</p></section>
        {!instructorMode && <section className="nl-card"><h3>Controlled comparison</h3><p>Keep the data seed and epoch budget fixed. Save a result, reset parameters, then change one setting.</p><button className="nl-wide" disabled={training || comparisons.length >= 4} onClick={() => { const item: Comparison = { name: `Run ${comparisons.length + 1}`, model: copy(state.model), epochs: state.epoch, train: networkLoss(state.model, data.filter(p => p.split === 'train')).loss, validation: networkLoss(state.model, data.filter(p => p.split === 'validation')).loss }; setComparisons([...comparisons, item]); setNotice('Comparison saved with its parameters, data seed, and epoch budget.'); }}>Save comparison</button>{comparisons.map((comparison, i) => <div className="nl-comparison" key={i}><strong>{comparison.name}</strong><span>{comparison.epochs} epochs · data {comparison.model.dataSeed} · rate {comparison.model.learningRate}</span><span>Train {number(comparison.train)} · validation {number(comparison.validation)}</span><button onClick={() => { const model = copy(comparison.model); const example = model.layers[0].inputs === state.example.x.length ? state.example : initialState(kind).example; commit({ ...initialState(kind), model, example, epoch: comparison.epochs }); setNotice('Saved parameters restored. Run forward to inspect this comparison.'); }}>Restore run</button></div>)}</section>}
        <section className="nl-card"><h3>Show more detail</h3><label className="nl-check"><input type="checkbox" checked={showMath} onChange={event => setShowMath(event.target.checked)} />Formulas</label><label className="nl-check"><input type="checkbox" checked={showCode} onChange={event => setShowCode(event.target.checked)} />Python reference</label><label className="nl-check"><input type="checkbox" checked={showGradients} onChange={event => setShowGradients(event.target.checked)} />Gradients</label><p className="nl-footnote">Open a neuron to see these details. Gradients appear only after backward.</p></section>
      </aside>}
    </div>
  </div>
}

function NetworkDiagram({ model, calculation, input, selected, onSelect, zoom }: { model: NetworkModel; calculation?: NetworkCalculation; input: number[]; selected: number[]; onSelect: (layer: number, unit: number) => void; zoom: number }) {
  const pan = useRef<{ pointer: number; x: number; y: number; left: number; top: number } | null>(null)
  const [panning, setPanning] = useState(false)
  const width = Math.max(630, (model.layers.length + 1) * 190)
  const height = Math.max(220, Math.max(...model.layers.map(layer => layer.outputs)) * 63 + 65)
  const x = (column: number) => 70 + column * (width - 140) / model.layers.length
  const y = (unit: number, count: number) => height / 2 + (unit - (count - 1) / 2) * 62
  return <div className={`nl-diagram-scroll${panning ? ' nl-panning' : ''}`}
    onPointerDown={event => {
      if (event.pointerType !== 'mouse' || event.button !== 0 || (event.target as Element).closest('[role="button"], button, input, select, textarea')) return
      const surface = event.currentTarget
      pan.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY, left: surface.scrollLeft, top: surface.scrollTop }
      surface.setPointerCapture(event.pointerId)
      setPanning(true)
      event.preventDefault()
    }}
    onPointerMove={event => {
      const origin = pan.current
      if (!origin || origin.pointer !== event.pointerId) return
      event.currentTarget.scrollLeft = origin.left + origin.x - event.clientX
      event.currentTarget.scrollTop = origin.top + origin.y - event.clientY
    }}
    onPointerUp={event => {
      if (pan.current?.pointer !== event.pointerId) return
      pan.current = null
      setPanning(false)
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    }}
    onPointerCancel={() => { pan.current = null; setPanning(false) }}
    onLostPointerCapture={() => { pan.current = null; setPanning(false) }}
  ><svg className="nl-network-diagram" style={{ width: `${zoom * 100}%`, minWidth: width * zoom }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Network with weighted connections and selectable neurons">
    {model.layers.flatMap((layer, l) => layer.weights.map((weight, index) => { const from = Math.floor(index / layer.outputs), to = index % layer.outputs; return <g key={`${layer.id}-${index}`}><line x1={x(l) + 25} y1={y(from, layer.inputs)} x2={x(l + 1) - 25} y2={y(to, layer.outputs)} stroke={weight >= 0 ? '#9370c2' : '#c8934a'} strokeWidth={Math.min(4, Math.abs(weight) + .7)} opacity={selected[0] === l && selected[1] === to ? .95 : .3} />{layer.weights.length <= 6 && <text className="nl-edge-number" x={(x(l) + x(l + 1)) / 2} y={(y(from, layer.inputs) + y(to, layer.outputs)) / 2 - 6}>{number(weight)}</text>}</g> }))}
    {input.map((value, i) => <g key={i}><circle cx={x(0)} cy={y(i, input.length)} r={24} fill="#e8f6f1" stroke="#77b6a4" /><text x={x(0)} y={y(i, input.length) + 4} textAnchor="middle">{number(value)}</text><text x={x(0)} y={height - 14} textAnchor="middle">{i === 0 ? 'Inputs' : ''}</text></g>)}
    {model.layers.map((layer, l) => <g key={layer.id}><text x={x(l + 1)} y={22} textAnchor="middle" className="nl-diagram-label">{l === model.layers.length - 1 ? 'Output scores' : `Layer ${l + 1}`} · {layer.activation}</text>{layer.biases.map((bias, unit) => <g key={unit} role="button" tabIndex={0} aria-label={`Inspect layer ${l + 1} neuron ${unit + 1}`} onClick={() => onSelect(l, unit)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(l, unit) } }} className="nl-neuron"><circle cx={x(l + 1)} cy={y(unit, layer.outputs)} r={25} fill={selected[0] === l && selected[1] === unit ? '#eee5fa' : '#faf8fd'} stroke={selected[0] === l && selected[1] === unit ? '#7549a8' : '#c6b4dc'} strokeWidth={2} /><text x={x(l + 1)} y={y(unit, layer.outputs) + 4} textAnchor="middle">{number(calculation?.layers[l].output[unit])}</text><text x={x(l + 1)} y={y(unit, layer.outputs) + 42} textAnchor="middle" className="nl-bias-number">b {number(bias)}</text></g>)}</g>)}
  </svg></div>
}

function DataPlot({ model, kind, data, selected, onSelect }: { model: NetworkModel | null; kind: NetworkKind; data: Example[]; selected: string; onSelect: (example: Example) => void }) {
  const oneD = kind === 'linear' || kind === 'neuron'
  const width = 620, height = 290, left = 46, top = 18, plotWidth = width - 70, plotHeight = height - 55
  const xScale = (v: number) => left + (v + (oneD ? 2.3 : 1.1)) / (oneD ? 4.6 : 2.2) * plotWidth
  const yScale = (v: number) => top + plotHeight - (v + (oneD ? 9 : 1.1)) / (oneD ? 20 : 2.2) * plotHeight
  const curve = model && oneD ? Array.from({ length: 55 }, (_, i) => { const x = -2.2 + i * 4.4 / 54; return `${xScale(x)},${Math.max(top, Math.min(top + plotHeight, yScale(runNetwork(model, [x]).prediction)))}` }).join(' ') : ''
  const cells = model && !oneD ? Array.from({ length: 18 * 18 }, (_, index) => { const col = index % 18, row = Math.floor(index / 18), x = -1.1 + (col + .5) * 2.2 / 18, y = 1.1 - (row + .5) * 2.2 / 18, result = runNetwork(model, [x, y]); const value = kind === 'playground' ? result.probabilities[1] : Math.max(0, Math.min(1, result.prediction / 2)); return <rect key={index} x={left + col * plotWidth / 18} y={top + row * plotHeight / 18} width={plotWidth / 18 + .4} height={plotHeight / 18 + .4} fill={`rgb(${Math.round(242 - 95 * value)},${Math.round(211 - 100 * value)},${Math.round(151 + 52 * value)})`} opacity=".62"><title>{`x₁ ${number(x)}, x₂ ${number(y)}: ${kind === 'playground' ? 'P(1)' : 'output'} ${number(kind === 'playground' ? result.probabilities[1] : result.prediction)}`}</title></rect> }) : []
  return <svg className="nl-data-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={oneD ? 'Regression chart: input x versus target and prediction' : 'Two-dimensional data with decision surface'}>
    <rect x={left} y={top} width={plotWidth} height={plotHeight} rx="4" fill="#faf9fc" />{cells}
    {(oneD ? [-2, -1, 0, 1, 2] : [-1, 0, 1]).map(v => <g key={`x${v}`}><line x1={xScale(v)} y1={top} x2={xScale(v)} y2={top + plotHeight} stroke="#ddd7e5" strokeDasharray="3 4" /><text x={xScale(v)} y={height - 19} textAnchor="middle">{v}</text></g>)}
    {(oneD ? [-8, -4, 0, 4, 8] : [-1, 0, 1]).map(v => <g key={`y${v}`}><line x1={left} y1={yScale(v)} x2={left + plotWidth} y2={yScale(v)} stroke="#ddd7e5" strokeDasharray="3 4" /><text x={left - 9} y={yScale(v) + 4} textAnchor="end">{v}</text></g>)}
    {curve && <polyline points={curve} fill="none" stroke="#794ca8" strokeWidth="3" />}
    {data.map(point => { const cy = yScale(oneD ? point.y : point.x[1]), color = kind === 'playground' ? point.y ? '#694197' : '#9a6014' : '#417e71'; return <circle key={point.id} cx={xScale(point.x[0])} cy={cy} r={selected === point.id ? 7 : 4.5} fill={point.split === 'validation' ? '#fff' : color} stroke={selected === point.id ? '#222' : color} strokeWidth={selected === point.id ? 2.5 : 1.8} role="button" tabIndex={0} aria-label={`${point.id}, ${point.split}, inputs ${point.x.map(number).join(', ')}, target ${number(point.y)}`} onClick={() => onSelect(point)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(point) } }}><title>{`${point.id}: (${point.x.map(number).join(', ')}) → ${number(point.y)} · ${point.split}`}</title></circle> })}
    <text x={width / 2} y={height - 2} textAnchor="middle">{oneD ? 'Input x' : 'Input x₁'}</text><text transform={`translate(13 ${height / 2}) rotate(-90)`} textAnchor="middle">{oneD ? 'Target / prediction' : 'Input x₂'}</text>
    {!model && <text x={width / 2} y={34} textAnchor="middle" className="nl-plot-prompt">Run forward to show current predictions</text>}
  </svg>
}

function ResponsePlot({ model, layer, unit, input }: { model: NetworkModel | null; layer: number; unit: number; input: number[] }) {
  const width = 580, height = 130
  const points = model ? Array.from({ length: 41 }, (_, i) => { const x = -2 + i / 10, next = [...input]; next[0] = x; return { x, y: runNetwork(model, next).layers[layer].output[unit] } }) : []
  const max = Math.max(1, ...points.map(p => Math.abs(p.y)))
  const responseMap = model && input.length === 2 ? Array.from({ length: 196 }, (_, i) => { const x = -1 + (i % 14 + .5) / 7, y = 1 - (Math.floor(i / 14) + .5) / 7; return { x, y, output: runNetwork(model, [x, y]).layers[layer].output[unit] } }) : []
  const mapMin = Math.min(0, ...responseMap.map(point => point.output)), mapMax = Math.max(.001, ...responseMap.map(point => point.output))
  return <div className="nl-response"><h4>Neuron response · vary x₁, hold other inputs fixed</h4><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Selected neuron activation response graph"><line x1="42" y1="58" x2="550" y2="58" stroke="#c9c0d3" /><line x1="296" y1="12" x2="296" y2="104" stroke="#c9c0d3" />{points.length > 0 && <polyline points={points.map(p => `${42 + (p.x + 2) / 4 * 508},${58 - p.y / max * 44}`).join(' ')} stroke="#438d7a" fill="none" strokeWidth="3" />}<text x="35" y="18">{number(max)}</text><text x="35" y="104">{number(-max)}</text><text x="42" y="124">−2</text><text x="296" y="124" textAnchor="middle">Input x₁</text><text x="544" y="124">2</text>{!model && <text x="300" y="40" textAnchor="middle">Run forward to see the response</text>}</svg>{responseMap.length > 0 && <div className="nl-response-map"><svg viewBox="0 0 250 200" role="img" aria-label="Selected neuron two-dimensional response map">{responseMap.map((point, i) => <rect key={i} x={36 + i % 14 * 11} y={12 + Math.floor(i / 14) * 11} width="11.2" height="11.2" fill={`rgb(${Math.round(238 - (point.output - mapMin) / (mapMax - mapMin) * 145)},${Math.round(246 - (point.output - mapMin) / (mapMax - mapMin) * 80)},${Math.round(241 - (point.output - mapMin) / (mapMax - mapMin) * 100)})`}><title>{`x₁ ${number(point.x)}, x₂ ${number(point.y)}: activation ${number(point.output)}`}</title></rect>)}<circle cx={36 + (Math.max(-1, Math.min(1, input[0])) + 1) * 77} cy={12 + (1 - Math.max(-1, Math.min(1, input[1]))) * 77} r="4" fill="none" stroke="#42245d" strokeWidth="2" /><text x="36" y="179">−1</text><text x="187" y="179">1</text><text x="105" y="194">Input x₁</text><text x="20" y="20">1</text><text x="16" y="166">−1</text><text x="8" y="93" transform="rotate(-90 8 93)">Input x₂</text></svg><p>This neuron's response across both inputs. Pale → dark: {number(mapMin)} → {number(mapMax)}. The ring marks the selected input (clipped to the plotted range).</p></div>}</div>
}

function LossPlot({ losses }: { losses: LossPoint[] }) {
  const width = 270, height = 150, max = Math.max(.01, ...losses.flatMap(point => [point.train, point.validation]))
  return <svg className="nl-loss-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss versus epoch"><line x1="36" y1="12" x2="36" y2="122" stroke="#d5ccdf" /><line x1="36" y1="122" x2="258" y2="122" stroke="#d5ccdf" /><text x="30" y="20" textAnchor="end">{number(max)}</text><text x="30" y="124" textAnchor="end">0</text>{(['train', 'validation'] as const).map((split, index) => <polyline key={split} points={losses.map((point, i) => `${36 + i / Math.max(1, losses.length - 1) * 222},${122 - point[split] / max * 105}`).join(' ')} fill="none" stroke={index === 0 ? '#7d51ac' : '#328574'} strokeWidth="2.5" strokeDasharray={index === 0 ? undefined : '4 3'} />)}{!losses.length && <text x="148" y="72" textAnchor="middle">Train to record loss</text>}<text x="36" y="145">{losses[0]?.epoch ?? 0}</text><text x="147" y="145" textAnchor="middle">Epoch</text><text x="258" y="145" textAnchor="end">{losses.at(-1)?.epoch ?? 0}</text></svg>
}
