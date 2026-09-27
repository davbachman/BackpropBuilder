import { expect, it } from 'vitest'
import { tensorAxisLabels } from './tensorLabels'
import { prepareTextDocuments } from './textData'
import { buildTextModel } from '../test/textModels'

it('labels vocabulary and token axes through graph connections, even after renaming', () => {
  const data = prepareTextDocuments([{text:'good good film',label:'positive',split:'train'}, {text:'bad film',label:'negative',split:'train'}, {text:'new film',label:'negative',split:'test'}], 'reviews.csv', {task:'sentiment'})
  const counts = buildTextModel(data,'counts-linear',4)
  counts.nodes.forEach(node => {node.label = 'renamed'})
  expect(tensorAxisLabels(counts,'sentiment-weights')[0]).toEqual(data.vocabulary)
  const tokens = buildTextModel(data,'position-attention',4)
  expect(tensorAxisLabels(tokens,'word-embeddings')[0]).toEqual(data.vocabulary)
  expect(tensorAxisLabels(tokens,'position-embeddings')[0]?.[1]).toBe('position 1')
  const scores = tensorAxisLabels(tokens,'attention-scores')
  expect(scores[0]).toEqual(scores[1])
  expect(scores[0]?.[0]).toContain('good')
  expect(tensorAxisLabels(tokens,'mean-review')[0]).toBeUndefined()
})
