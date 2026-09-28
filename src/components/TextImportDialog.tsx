import { useEffect, useRef, useState } from 'react'
import { importText, isTextDatasetData } from '../domain/textData'
import type { TextDatasetData } from '../domain/types'

export function TextImportDialog({ onImport, onCancel }: { onImport: (data: TextDatasetData) => void; onCancel: () => void }) {
  const [task, setTask] = useState<'sentiment' | 'classification' | 'language'>('sentiment')
  const [tokenizer, setTokenizer] = useState<'word' | 'character'>('word')
  const [representation, setRepresentation] = useState<'counts' | 'tokens' | 'facts'>('counts')
  const [size, setSize] = useState(1000), [length, setLength] = useState(64), [stride, setStride] = useState(16)
  const [lowercase, setLowercase] = useState(true)
  const [file, setFile] = useState<File>(), [error, setError] = useState(''), [busy, setBusy] = useState(false)
  const requestVersion = useRef(0)
  useEffect(() => () => { requestVersion.current++ }, [])
  const cancel = () => {
    requestVersion.current++
    setBusy(false)
    onCancel()
  }
  async function load() {
    if (!file) return
    const request = ++requestVersion.current
    const isCurrent = () => request === requestVersion.current
    setBusy(true)
    try {
      const text = await file.text()
      if (!isCurrent()) return
      const data = file.name.endsWith('.json') ? JSON.parse(text) as unknown : importText(text, file.name, {task, tokenizer, representation, vocabularySize: size, maxLength: length, stride: Math.min(length, stride), lowercase, fixedLength: task==='classification' && representation==='tokens', factWords: representation==='facts'?10:undefined, maxFacts: representation==='facts'?64:undefined})
      if (!isTextDatasetData(data)) throw new Error('Invalid prepared text dataset.')
      onImport(data)
    } catch (cause) { if (isCurrent()) setError(cause instanceof Error ? cause.message : 'Could not read text data.') }
    finally { if (isCurrent()) setBusy(false) }
  }
  return <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-labelledby="text-import-title" className="csv-picker-dialog" onKeyDown={event => { if (event.key === 'Escape') cancel() }}>
    <h2 id="text-import-title">Import text data</h2>
    <p>Reviews: CSV headers text,label,split with positive/negative labels and train/test splits. Without split, every fifth review per class is held out. Language: a plain-text corpus, split by passage before creating windows. Multiclass: text,label,split; for fact inputs, text contains newline-separated facts and a question column supplies the question (up to 64 facts, 10 tokens each). Prepared dataset JSON preserves its vocabulary and settings.</p>
    <label className="inspector-field">Task<select aria-label="Text task" value={task} onChange={event => {const next = event.target.value as typeof task; setTask(next); setTokenizer(next === 'language' ? 'character' : 'word'); setRepresentation(next === 'language' ? 'tokens' : 'counts'); setLength(next === 'language' ? 16 : 64)}}><option value="sentiment">Review sentiment</option><option value="classification">Multiclass text / question answering</option><option value="language">Next-token prediction</option></select></label>
    <label className="inspector-field">Tokenizer<select aria-label="Tokenizer" value={tokenizer} onChange={event => setTokenizer(event.target.value as typeof tokenizer)}><option value="word">Words and punctuation</option><option value="character">Characters</option></select></label>
    {task !== 'language' && <label className="inspector-field">Representation<select aria-label="Text representation" value={representation} onChange={event => setRepresentation(event.target.value as typeof representation)}><option value="counts">Word counts</option><option value="tokens">Token IDs</option>{task==='classification' && <option value="facts">Facts and question</option>}</select></label>}
    <label className="inspector-field">Maximum vocabulary (includes unknown)<input aria-label="Vocabulary limit" type="number" min="2" max="8192" value={size} onChange={event => setSize(Number(event.target.value))}/></label>
    <label className="inspector-field">{task === 'language' ? 'Context length' : 'Maximum review tokens'}<input aria-label="Text context length" type="number" min="1" max="256" value={length} onChange={event => setLength(Number(event.target.value))}/></label>
    {task === 'language' && <label className="inspector-field">Window stride<input aria-label="Window stride" type="number" min="1" max={length} value={stride} onChange={event => setStride(Number(event.target.value))}/></label>}
    <label><input type="checkbox" checked={lowercase} onChange={event => setLowercase(event.target.checked)}/> Lowercase text</label>
    <p>Vocabulary is fitted on training documents only. Reviews are truncated to the selected length; individual sequences have no padding. Tensor training pads batches internally and masks padded positions.</p>
    <input type="file" accept=".csv,.txt,.json,text/plain,text/csv,application/json" aria-label="Choose text data file" onChange={event => {requestVersion.current++; setBusy(false); setFile(event.target.files?.[0]); setError('')}}/>
    {error && <p role="alert">{error}</p>}
    <button type="button" disabled={!file || busy} onClick={() => void load()}>{busy ? 'Preparing…' : 'Import text dataset'}</button>
    <button type="button" onClick={cancel}>Cancel</button>
  </section></div>
}
