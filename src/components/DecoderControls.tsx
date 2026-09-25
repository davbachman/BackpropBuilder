import { useMemo, useState } from 'react'
import { ArrowRight, RotateCcw, Sparkles } from 'lucide-react'
import { datasetForNode, datasetOutputValueForSlot } from '../domain/datasets'
import { predictionNode } from '../domain/datasetTraining'
import { forwardPass } from '../domain/engine'
import type { GraphModel } from '../domain/types'
import { samplingDistribution } from '../learning/decoder'
import './decoderControls.css'

export interface DecoderControlsProps {
  graph: GraphModel
  onGraphChange: (graph: GraphModel) => void
}

/** Prompt edits and generation execute the editable canvas itself, including
 * parameter changes made at any depth of the architecture. */
export function DecoderControls({ graph, onGraphChange }: DecoderControlsProps) {
  const dataNode = graph.nodes.find(node => node.type === 'dataset' && datasetForNode(node).task === 'sequence')
  const dataset = dataNode ? datasetForNode(dataNode) : undefined
  const vocabulary = dataset?.vocabulary ?? []
  const maxLength = dataset?.maxLength ?? 12
  const ids = dataNode ? datasetOutputValueForSlot(dataNode,0).data : []
  const prompt = ids.map(id => vocabulary[id] ?? `?${id}`).join(' ')
  const [draft, setDraft] = useState<string | null>(null)
  const [mode, setMode] = useState<'greedy' | 'sample'>('greedy')
  const [temperature, setTemperature] = useState(1)
  const [topK, setTopK] = useState(vocabulary.length)
  const [message, setMessage] = useState('')
  const inference = useMemo(() => {
    try {
      const next = forwardPass(graph).graph
      const logits = predictionNode(next)?.value
      return { graph: next, logits: logits?.data.slice(-logits.shape.at(-1)!) ?? [], error: '' }
    } catch (error) {
      return { graph, logits: [], error: error instanceof Error ? error.message : 'This graph cannot run yet.' }
    }
  }, [graph])
  const distribution = inference.logits.length ? samplingDistribution(inference.logits, mode === 'sample' ? temperature : 1, mode === 'sample' ? topK : vocabulary.length) : []
  const ended = vocabulary[ids.at(-1)!] === '<eos>'
  const full = ids.length >= maxLength
  const hasOutput = Boolean(predictionNode(graph))

  function applyPrompt(nextIds: number[]) {
    if (!dataNode || !vocabulary.length) return
    const period = vocabulary.length - 2
    const values = [nextIds, nextIds.map((_,index)=>index), nextIds.map(id => id === vocabulary.length - 1 ? id : id % period + 1)].map(data => ({shape:[nextIds.length],data}))
    const next: GraphModel = { ...graph, nodes: graph.nodes.map(node => node.id === dataNode.id ? {...node,params:{...node.params,datasetValues:values}} : node) }
    try {
      onGraphChange(forwardPass(next).graph)
      setDraft(null)
      setMessage('')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'This graph cannot run yet.')
    }
  }

  function submitPrompt() {
    const tokens = (draft ?? prompt).trim().split(/\s+/).filter(Boolean)
    const unknown = tokens.filter(token => !vocabulary.includes(token))
    if (unknown.length) { setMessage(`Unknown token: ${unknown.join(', ')}. Choose a token from the vocabulary below.`); return }
    if (!tokens.length || tokens.length > maxLength) { setMessage(`Use between 1 and ${maxLength} tokens.`); return }
    applyPrompt(tokens.map(token => vocabulary.indexOf(token)))
  }

  function generate() {
    if (!distribution.length || ended || full || inference.error) return
    let token = distribution.indexOf(Math.max(...distribution))
    if (mode === 'sample') {
      let remaining = Math.random()
      token = distribution.length - 1
      for (let index = 0; index < distribution.length; index++) {
        remaining -= distribution[index]
        if (remaining < 0) { token = index; break }
      }
    }
    applyPrompt([...ids, token])
  }

  return <section className="decoder-controls" aria-label="Model prompt and generation">
    <div className="decoder-controls-heading"><span className="decoder-eyebrow">LIVE MODEL</span><span className="decoder-context-count">{ids.length} / {maxLength} tokens</span></div>
    <h3>{hasOutput ? 'Continue the sequence' : 'Give tokens coordinates'}</h3>
    <p className="decoder-explanation">{dataset?.description} Every result comes from the graph on this canvas.</p>
    <form onSubmit={event => { event.preventDefault(); submitPrompt() }}>
      <label className="decoder-field-label" htmlFor="decoder-prompt">Prompt</label>
      <textarea id="decoder-prompt" value={draft ?? prompt} onChange={event => { setDraft(event.target.value); setMessage('') }} spellCheck={false} rows={2} />
      <div className="decoder-vocabulary" aria-label="Vocabulary">{vocabulary.map(token => <button type="button" key={token} data-token={token} title={`Append ${token} to prompt`} onClick={() => setDraft(`${draft ?? prompt} ${token}`.trim())}>{token}</button>)}</div>
      <button type="submit" className="decoder-apply"><RotateCcw size={13} /> Apply prompt</button>
      {hasOutput && <p className="decoder-status">Prompt edits also refresh the training targets using this cycle.</p>}
    </form>
    {hasOutput && <>
      <div className="decoder-sampling-controls">
        <label>Choose next token<select aria-label="Generation mode" value={mode} onChange={event => setMode(event.target.value as 'greedy' | 'sample')}><option value="greedy">Most likely</option><option value="sample">Sample</option></select></label>
        {mode === 'sample' && <div className="decoder-sampling-fields"><label>Temperature<input aria-label="Sampling temperature" type="number" min="0.1" max="3" step="0.1" value={temperature} onChange={event => { const next = Number(event.target.value); if (Number.isFinite(next) && next >= .1 && next <= 3) setTemperature(next) }} /></label><label>Top-k<select aria-label="Sampling top-k" value={topK} onChange={event => setTopK(Number(event.target.value))}>{vocabulary.map((_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}</select></label></div>}
      </div>
      <div className="decoder-distribution-heading"><span>{mode === 'sample' ? 'Sampling probabilities' : 'Next-token probabilities'}</span><span>after {vocabulary[ids.at(-1)!]}</span></div>
      <div className="decoder-distribution" role="img" aria-label="Next-token probability distribution">{vocabulary.map((token, index) => <div className="decoder-probability-row" key={token} data-token={token}><span>{token}</span><div className="decoder-probability-track"><div style={{ width: `${(distribution[index] ?? 0) * 100}%` }} /></div><output>{((distribution[index] ?? 0) * 100).toFixed(1)}%</output></div>)}</div>
      <button type="button" className="decoder-generate" onClick={generate} disabled={!distribution.length || ended || full || !!inference.error || draft !== null}><Sparkles size={15} /> Generate next token<ArrowRight size={15} /></button>
      {(ended || full || draft !== null) && <p className="decoder-status">{ended ? 'End token reached. Edit the prompt to continue.' : full ? `The ${maxLength}-token context is full. Shorten the prompt to continue.` : 'Apply your prompt before generating.'}</p>}
    </>}
    {(message || inference.error) && <p className="decoder-error" role="alert">{message || inference.error}</p>}
  </section>
}
