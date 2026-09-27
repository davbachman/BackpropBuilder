import { parseRows } from './customCsv'
import type { DatasetExample, ToyDataset } from './datasets'
import type { TextDatasetData, TextDocument, TensorValue } from './types'

export const TEXT_LIMITS = { characters: 80_000_000, documents: 60_000, vocabulary: 8192, length: 256, examples: 60_000 }
const tensor = (shape: number[], data: number[]): TensorValue => ({ shape, data })

export function tokenize(text: string, mode: 'word' | 'character', lowercase = true): string[] {
  const clean = text.replace(/<br\s*\/?\s*>/gi, ' ').normalize('NFC')
  const normalized = lowercase ? clean.toLowerCase() : clean
  return mode === 'character' ? Array.from(normalized) : normalized.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu) ?? []
}

export function encodeText(text: string, data: TextDatasetData): number[] {
  const index = new Map(data.vocabulary.map((token, id) => [token, id]))
  // The visible unknown marker must round-trip when continuing generated text.
  return text.split(/(<unk>)/g).flatMap(part => part === '<unk>' ? [0] : tokenize(part, data.tokenizer, data.lowercase).map(token => index.get(token) ?? 0))
}

export function decodeTokens(ids: number[], data: TextDatasetData): string {
  return ids.map(id => data.vocabulary[id] ?? '<unk>').join(data.tokenizer === 'character' ? '' : ' ')
}

export interface TextPreparation {
  task: TextDatasetData['task']
  tokenizer?: TextDatasetData['tokenizer']
  representation?: TextDatasetData['representation']
  lowercase?: boolean
  vocabularySize?: number
  maxLength?: number
  stride?: number
  fixedLength?: boolean
  factWords?: number
  maxFacts?: number
  targetMode?: 'last'
}

export function prepareTextDocuments(documents: TextDocument[], fileName: string, options: TextPreparation): TextDatasetData {
  const tokenizer = options.tokenizer ?? (options.task === 'language' ? 'character' : 'word')
  const lowercase = options.lowercase ?? true
  const vocabularySize = options.vocabularySize ?? 1000
  const maxLength = options.maxLength ?? (options.task === 'language' ? 16 : 64)
  const stride = options.stride ?? maxLength
  if (!Number.isInteger(vocabularySize) || vocabularySize < 2 || vocabularySize > TEXT_LIMITS.vocabulary) throw new Error('Vocabulary size must be 2–8192, including <unk>.')
  if (!Number.isInteger(maxLength) || maxLength < 1 || maxLength > TEXT_LIMITS.length || !Number.isInteger(stride) || stride < 1 || stride > maxLength) throw new Error('Use a context length of 1–256 and a stride from 1 to the context length.')
  validateDocuments(documents, options.task)
  const counts = new Map<string, number>()
  for (const doc of documents.filter(doc => doc.split === 'train')) {
    for (const token of tokenize(doc.text, tokenizer, lowercase)) counts.set(token, (counts.get(token) ?? 0) + 1)
  }
  const vocabulary = ['<unk>', ...(options.fixedLength || options.representation === 'facts' ? ['<pad>'] : []), ...[...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).filter(([token]) => token !== '<unk>' && token !== '<pad>').slice(0, vocabularySize - (options.fixedLength || options.representation === 'facts' ? 2 : 1)).map(([token]) => token)]
  if (vocabulary.length < 2) throw new Error('Training text must contain tokens.')
  const data: TextDatasetData = { version: 1, fileName, task: options.task, tokenizer, representation: options.task === 'language' ? 'tokens' : options.representation ?? 'counts', lowercase, vocabulary, maxLength, stride, documents, fixedLength: options.fixedLength, factWords: options.factWords, maxFacts: options.maxFacts, targetMode: options.targetMode, classLabels: options.task === 'classification' ? [...new Set(documents.filter(d=>d.split==='train').map(d=>d.label!))].sort() : undefined }
  textDataset(data) // Check that both splits produce examples before accepting the import.
  return data
}

function validateDocuments(documents: TextDocument[], task: TextDatasetData['task']) {
  if (!Array.isArray(documents) || documents.length < 2 || documents.length > TEXT_LIMITS.documents) throw new Error('Use 2–60,000 text documents or passages.')
  if (documents.some(doc => !doc || typeof doc.text !== 'string' || !doc.text.trim() || !['train', 'test'].includes(doc.split) || (task === 'sentiment' && !['negative', 'positive'].includes(doc.label ?? '')) || (task === 'classification' && (typeof doc.label !== 'string' || !doc.label.trim())))) throw new Error('Each text row needs nonempty text, a train/test split, and (for sentiment) a positive/negative label.')
  if (documents.reduce((total, doc) => total + doc.text.length, 0) > TEXT_LIMITS.characters) throw new Error('Text datasets support at most 80 million characters.')
  if (!documents.some(doc => doc.split === 'train') || !documents.some(doc => doc.split === 'test')) throw new Error('Include both training and held-out documents.')
}

/** CSV headers: text,label[,split]. Explicit splits are preserved. Otherwise
 * every fifth review within each label is held out BEFORE vocabulary fitting. */
export function importText(text: string, fileName: string, options: TextPreparation): TextDatasetData {
  if (text.length > TEXT_LIMITS.characters) throw new Error('Text file is too large (5 million characters maximum).')
  let documents: TextDocument[]
  if (options.task !== 'language') {
    const rows = parseRows(text.replace(/^\uFEFF/, '')).filter(row => row.some(cell => cell.trim()))
    const headers = rows.shift()?.map(cell => cell.toLowerCase().trim()) ?? []
    const textColumn = headers.indexOf('text') >= 0 ? headers.indexOf('text') : headers.indexOf('review')
    const labelColumn = headers.indexOf('label') >= 0 ? headers.indexOf('label') : headers.indexOf('sentiment')
    const splitColumn = headers.indexOf('split')
    if (textColumn < 0 || labelColumn < 0) throw new Error('Review CSV needs text (or review) and label (or sentiment) headers; split is optional.')
    const counts = new Map<string, number>()
    documents = rows.map(row => {
      if (row.length !== headers.length) throw new Error('Each review row must have the same number of columns as the header.')
      const raw = row[labelColumn].toLowerCase().trim()
      const label = options.task === 'classification' ? row[labelColumn].trim() : ['1', 'pos', 'positive'].includes(raw) ? 'positive' : ['0', 'neg', 'negative'].includes(raw) ? 'negative' : raw
      const count = counts.get(label) ?? 0
      counts.set(label, count + 1)
      const split = splitColumn < 0 ? (count % 5 === 0 ? 'test' : 'train') : row[splitColumn].toLowerCase().trim()
      if (split !== 'train' && split !== 'test') throw new Error('Split values must be train or test.')
      const questionColumn = headers.indexOf('question')
      const facts = options.representation === 'facts' ? row[textColumn].split(/\n+/).filter(x=>x.trim()) : undefined
      const question = questionColumn >= 0 ? row[questionColumn] : undefined
      return { text: facts ? facts.join(' ') + ' ' + (question ?? '') : row[textColumn], facts, question, label, split }
    })
  } else {
    // Contiguous passage split before any window creation: no overlapping windows across splits.
    const paragraphs = text.trim().split(/\n\s*\n/).filter(paragraph => paragraph.trim())
    const passages = paragraphs.length >= 2 ? paragraphs : [text.slice(0, Math.floor(text.length * .8)), text.slice(Math.floor(text.length * .8))]
    const boundary = Math.max(1, Math.min(passages.length - 1, Math.floor(passages.length * .8)))
    documents = passages.map((passage, index) => ({ text: passage, split: index < boundary ? 'train' : 'test' }))
  }
  return prepareTextDocuments(documents, fileName, options)
}

export function isTextDatasetData(value: unknown): value is TextDatasetData {
  if (!value || typeof value !== 'object') return false
  const data = value as TextDatasetData
  try {
    if (data.version !== 1 || typeof data.fileName !== 'string' || data.fileName.length > 256 || !['sentiment', 'classification', 'language'].includes(data.task) || !['word', 'character'].includes(data.tokenizer) || !['counts', 'tokens', 'facts'].includes(data.representation) || (data.task === 'language' && data.representation !== 'tokens') || typeof data.lowercase !== 'boolean') return false
    if (!Array.isArray(data.vocabulary) || data.vocabulary[0] !== '<unk>' || data.vocabulary.length < 2 || data.vocabulary.length > TEXT_LIMITS.vocabulary || data.vocabulary.some(token => typeof token !== 'string' || token.length > 1024) || new Set(data.vocabulary).size !== data.vocabulary.length) return false
    if (!Number.isInteger(data.maxLength) || data.maxLength < 1 || data.maxLength > TEXT_LIMITS.length || !Number.isInteger(data.stride) || data.stride < 1 || data.stride > data.maxLength) return false
    if (data.fixedLength !== undefined && typeof data.fixedLength !== 'boolean') return false
    if (data.targetMode !== undefined && (data.targetMode !== 'last' || data.task !== 'language')) return false
    validateDocuments(data.documents, data.task)
    textDataset(data)
    return true
  } catch { return false }
}

const cache = new WeakMap<TextDatasetData, ToyDataset>()
export function textDataset(data: TextDatasetData): ToyDataset {
  const cached = cache.get(data)
  if (cached) return cached
  const examples: DatasetExample[] = []
  if (data.task === 'classification' && (!Array.isArray(data.classLabels) || data.classLabels.length < 2 || new Set(data.classLabels).size !== data.classLabels.length || data.classLabels.some(x=>typeof x!=='string'||!x) || data.documents.some(d=>!data.classLabels!.includes(d.label!)))) throw Error('Every answer must belong to the training-fitted class labels.')
  if ((data.fixedLength || data.representation === 'facts') && data.vocabulary[1] !== '<pad>') throw Error('Fixed inputs require a distinct <pad> token at ID 1.')
  if (data.representation === 'facts' && (data.task !== 'classification' || !Number.isInteger(data.factWords) || data.factWords!<1 || data.factWords!>32 || !Number.isInteger(data.maxFacts) || data.maxFacts!<1 || data.maxFacts!>64)) throw Error('Fact inputs require classification, 1–32 words per fact and 1–64 facts.')
  const index = new Map(data.vocabulary.map((token, id) => [token, id]))
  data.documents.forEach((doc, documentIndex) => {
    const ids = tokenize(doc.text, data.tokenizer, data.lowercase).map(token => index.get(token) ?? 0)
    const add = (tokens: number[], target: TensorValue, offset: number) => {
      if (examples.length >= TEXT_LIMITS.examples) throw new Error('Too many text windows (60,000 maximum). Increase stride or use a smaller corpus.')
      const rawLength = tokens.length
      if (data.fixedLength) tokens = [...tokens,...Array(data.maxLength-tokens.length).fill(1)]
      const features = [tensor([tokens.length], tokens), tensor([tokens.length], tokens.map((_, i) => i))]
      if (data.representation === 'counts') {
        // Keep token IDs in the cached dataset; materialize only requested count
        // vectors. Caching a dense corpus would require documents × vocabulary cells.
        Object.defineProperty(features, 0, {enumerable: true, get: () => {
          const counts = Array(data.vocabulary.length).fill(0) as number[]
          for (const id of tokens.slice(0,rawLength)) counts[id]++
          return tensor([1, counts.length], counts)
        }})
      }
      if (data.representation === 'facts') {
        const words=data.factWords!, n=data.maxFacts!
        if (!Array.isArray(doc.facts) || !doc.facts.length || doc.facts.length>n || typeof doc.question!=='string' || !doc.question.trim()) throw Error('Provide nonempty facts and question within the configured fact limit.')
        const encoded = [...doc.facts,doc.question].map(text=>tokenize(text,data.tokenizer,data.lowercase).map(t=>index.get(t)??0))
        if (encoded.some(ids=>!ids.length || ids.length>words)) throw Error('A fact or question exceeds the word limit; increase it instead of truncating.')
        const padded=(ids:number[])=>[...ids,...Array(words-ids.length).fill(1)]
        const f=[...encoded.slice(0,-1).map(padded),...Array.from({length:n-doc.facts.length},()=>Array(words).fill(1))].flat()
        const q=padded(encoded.at(-1)!)
        features.splice(0,features.length,tensor([n,words],f),tensor([1,words],q),tensor([n],Array.from({length:n},(_,i)=>i)),tensor([1,n],Array.from({length:n},(_,i)=>Number(i<doc.facts!.length))),tensor([n,words,1],f.map(id=>Number(id!==1))),tensor([1,words,1],q.map(id=>Number(id!==1))))
      }
      examples.push({ label: 'Document ' + (documentIndex + 1) + ' · offset ' + offset + ' · ' + (data.task !== 'language' ? doc.label : decodeTokens(tokens, data).slice(0, 45)), split: doc.split, features, target, text: decodeTokens(tokens, data), documentIndex, offset })
    }
    if (data.task !== 'language') {
      if (!ids.length) throw new Error('A review has no tokens after tokenization.')
      if (data.task === 'classification' && ids.length>data.maxLength && data.representation!=='facts') throw Error('Classification text exceeds the context limit; increase it instead of discarding facts.')
      add(ids.slice(0, data.maxLength), data.task==='sentiment' ? tensor([1, 1], [Number(doc.label === 'positive')]) : tensor([1],[data.classLabels!.indexOf(doc.label!)]), 0)
    } else {
      for (let start = 0; start + data.maxLength < ids.length; start += data.stride) {
        add(ids.slice(start, start + data.maxLength), data.targetMode==='last' ? tensor([1],[ids[start+data.maxLength]]) : tensor([data.maxLength], ids.slice(start + 1, start + data.maxLength + 1)), start)
      }
    }
  })
  if (!examples.some(example => example.split === 'train') || !examples.some(example => example.split === 'test')) throw new Error('Both splits need usable examples. Shorten the context or provide longer passages.')
  const dataset: ToyDataset = { kind: 'custom-text', label: data.fileName, task: data.task === 'sentiment' ? 'binary-classification' : data.task==='classification' ? 'classification' : 'sequence', featureLabels: data.representation==='facts' ? ['fact token IDs','question token IDs','fact position IDs','fact mask','fact word mask','question word mask'] : [data.representation === 'counts' ? 'word counts' : 'token IDs', 'position IDs'], targetLabel: data.task === 'sentiment' ? 'sentiment' : data.task==='classification' ? 'answer' : 'next-token IDs', featureValues: examples[0].features, targetValue: examples[0].target, examples, vocabulary: data.vocabulary, classLabels: data.task === 'sentiment' ? ['negative', 'positive'] : data.classLabels, maxLength: data.maxLength, description: 'Training-only vocabulary · ' + data.vocabulary.length + ' tokens including <unk> · ' + data.tokenizer + ' tokenization · ' + data.representation + (data.fixedLength || data.representation==='facts' ? ' · fixed padded inputs. ' : ' · unpadded canvas examples. ') + 'Held-out documents retain their split.' }
  cache.set(data, dataset)
  return dataset
}
