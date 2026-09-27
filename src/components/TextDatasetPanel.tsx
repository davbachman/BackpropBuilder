import { useState } from 'react'
import { datasetExampleIndex, datasetExamplesForNode, datasetOutputValueForSlot } from '../domain/datasets'
import { decodeTokens, encodeText, tokenize } from '../domain/textData'
import type { GraphNode, NodeParams } from '../domain/types'

export function TextDatasetPanel({ node, onParams, onReplace }: { node: GraphNode; onParams: (id: string, params: NodeParams) => void; onReplace: () => void }) {
  const [search, setSearch] = useState(''), [draft, setDraft] = useState(''), [error, setError] = useState('')
  const data = node.params.textData!
  const example = datasetExamplesForNode(node)[datasetExampleIndex(node)]
  const document = data.documents[example.documentIndex ?? 0]
  const ids = data.representation === 'tokens' ? datasetOutputValueForSlot(node, 0).data : encodeText(document.text, data).slice(0, data.maxLength)
  const tokens = data.vocabulary.map((token, id) => ({token, id})).filter(({token, id}) => token.includes(search) || String(id) === search).slice(0, 40)
  const apply = (reverse = false) => {
    let next = encodeText(draft || document.text, data).slice(0, data.maxLength)
    if (reverse) next = [...ids].reverse()
    if (!next.length) { setError('Enter at least one token.'); return }
    const counts = Array(data.vocabulary.length).fill(0) as number[]
    next.forEach(id => counts[id]++)
    onParams(node.id, {datasetValues: [data.representation === 'counts' ? {shape: [1, counts.length], data: counts} : {shape: [next.length], data: next}, {shape: [next.length], data: next.map((_, i) => i)}, example.target]})
    setError('')
  }
  return <section className="text-dataset-panel" aria-label="Text data inspection">
    <button type="button" onClick={onReplace}>Replace text dataset</button>
    {data.task === 'sentiment' && <label className="inspector-field">Representation<select aria-label="Review representation" value={data.representation} onChange={event => onParams(node.id, {textData: {...data, representation: event.target.value as 'tokens' | 'counts'}, datasetValues: undefined})}><option value="counts">Word counts</option><option value="tokens">Token IDs</option></select></label>}
    <p>Context limit: {data.maxLength}. Vocabulary and document splits stay fixed when switching representation. Tokens are processed without padding.</p>
    <details><summary>Original text · document {(example.documentIndex ?? 0) + 1}</summary><p className="text-original">{document.text}</p><p>{tokenize(document.text, data.tokenizer, data.lowercase).length} original tokens; {ids.length} displayed input tokens.</p></details>
    <details open><summary>{data.representation === 'counts' ? 'Original review tokens and IDs' : 'Input tokens and IDs'}</summary><div className="text-token-list">{ids.map((id, position) => <span key={position} title={'Position ' + position + ', token ID ' + id}>{position}: {JSON.stringify(data.vocabulary[id])} <small>#{id}</small></span>)}</div></details>
    {data.task === 'language' && <p>Targets: <code>{decodeTokens(example.target.data, data)}</code></p>}
    {data.task === 'sentiment' && <details><summary>Word-order experiment</summary><textarea aria-label="Review experiment" placeholder="Enter a review or reverse the current token order" value={draft} onChange={event => setDraft(event.target.value)}/><button type="button" onClick={() => apply()}>Apply review</button><button type="button" onClick={() => apply(true)}>Reverse token order</button><p>Experiment keeps the selected example’s label for comparison. Choose a dataset example to restore its original input.</p>{error && <p role="alert">{error}</p>}</details>}
    <details><summary>Vocabulary · {data.vocabulary.length} entries</summary><input aria-label="Search vocabulary" placeholder="Find a word or ID" value={search} onChange={event => setSearch(event.target.value)}/><div className="text-token-list">{tokens.map(({token, id}) => <span key={id}>{id}: {JSON.stringify(token)}</span>)}</div><p>Showing up to 40 matches. ID 0 means an unknown token. Words and punctuation tokenize separately; internal apostrophes stay within words.</p></details>
  </section>
}
