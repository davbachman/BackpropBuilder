import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TextImportDialog } from './TextImportDialog'
import { TextGenerationControls } from './TextGenerationControls'
import { TextDatasetPanel } from './TextDatasetPanel'
import { prepareTextDocuments } from '../domain/textData'
import { buildTextModel } from '../test/textModels'

const csv = 'text,label,split\n"good, fun",positive,train\nvery bad,negative,train\ngood,positive,test\n'
function selectFile(text: string, name: string) {
  const file = new File([text], name)
  Object.defineProperty(file, 'text', {value: async () => text})
  fireEvent.change(screen.getByLabelText('Choose text data file'), {target: {files: [file]}})
  fireEvent.click(screen.getByRole('button', {name: 'Import text dataset'}))
}
const reviews = () => prepareTextDocuments([{text:'good fun',label:'positive',split:'train'}, {text:'bad boring',label:'negative',split:'train'}, {text:'new good',label:'positive',split:'test'}], 'reviews.csv', {task:'sentiment'})

describe('text workflow controls', () => {
  it('imports quoted review CSV with frozen vocabulary and explicit splits', async () => {
    const onImport = vi.fn()
    render(<TextImportDialog onImport={onImport} onCancel={vi.fn()}/>)
    selectFile(csv, 'reviews.csv')
    await waitFor(() => expect(onImport).toHaveBeenCalledOnce())
    expect(onImport.mock.calls[0][0]).toMatchObject({task:'sentiment',representation:'counts',documents:[{split:'train'}, {split:'train'}, {split:'test'}]})
  })
  it('imports plain text as next-token windows', async () => {
    const onImport = vi.fn()
    render(<TextImportDialog onImport={onImport} onCancel={vi.fn()}/>)
    fireEvent.change(screen.getByLabelText('Text task'), {target:{value:'language'}})
    selectFile('Alice was beginning to get tired of sitting beside her sister.\n\nThe rabbit was running across the field with a watch.', 'alice.txt')
    await waitFor(() => expect(onImport).toHaveBeenCalledOnce())
    expect(onImport.mock.calls[0][0]).toMatchObject({task:'language',tokenizer:'character',representation:'tokens',maxLength:16})
  })
  it('preserves prepared data settings rather than the form defaults', async () => {
    const onImport = vi.fn(), data = {...reviews(), representation:'tokens',maxLength:6,stride:6}
    render(<TextImportDialog onImport={onImport} onCancel={vi.fn()}/>)
    selectFile(JSON.stringify(data), 'prepared.json')
    await waitFor(() => expect(onImport).toHaveBeenCalledWith(data))
  })
  it('shows an actionable invalid-file error', async () => {
    const onImport = vi.fn()
    render(<TextImportDialog onImport={onImport} onCancel={vi.fn()}/>)
    selectFile('wrong,headers\na,b', 'bad.csv')
    expect(await screen.findByRole('alert')).toHaveTextContent('Review CSV needs')
    expect(onImport).not.toHaveBeenCalled()
  })
  it('switches counts to IDs without changing the vocabulary or split', () => {
    const data = reviews(), node = buildTextModel(data, 'counts-linear', 4).nodes[0], onParams = vi.fn()
    render(<TextDatasetPanel node={node} onParams={onParams} onReplace={vi.fn()}/>)
    fireEvent.change(screen.getByLabelText('Review representation'), {target:{value:'tokens'}})
    expect(onParams).toHaveBeenCalledWith(node.id, {textData:{...data,representation:'tokens'},datasetValues:undefined})
  })
  it('inspects and generates text from current weights without training', async () => {
    const data = prepareTextDocuments([{text:'alice was beside her sister on the bank',split:'train'}, {text:'the rabbit ran across the field',split:'test'}], 'alice.txt', {task:'language',maxLength:4})
    const graph = buildTextModel(data,'alice-transformer',4), before = JSON.stringify(graph)
    render(<TextGenerationControls graph={graph}/>)
    fireEvent.click(screen.getByRole('button',{name:'Inspect next-token probabilities'}))
    await waitFor(() => expect(screen.getByLabelText('Text next-token probabilities')).toHaveTextContent('%'))
    fireEvent.click(screen.getByRole('button',{name:'Generate 40 tokens'}))
    await waitFor(() => expect(screen.getByRole('button',{name:'Generate 40 tokens'})).toBeEnabled())
    expect((screen.getByLabelText('Text generation prompt') as HTMLTextAreaElement).value.length).toBeGreaterThanOrEqual(45)
    expect(JSON.stringify(graph)).toBe(before)
    fireEvent.change(screen.getByLabelText('Text generation prompt'),{target:{value:''}})
    fireEvent.click(screen.getByRole('button',{name:'Generate one token'}))
    expect(await screen.findByRole('alert')).toHaveTextContent('nonempty prompt')
  })
})
