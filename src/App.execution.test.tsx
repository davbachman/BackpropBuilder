import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as datasetTraining from './domain/datasetTraining'
import { formatNumber, runTrainingStep } from './domain/engine'
import { createStarterGraph } from './domain/examples'
import { createModelPreset } from './domain/modelPresets'
import { createProjectStateFile } from './domain/session'
import type { GraphModel } from './domain/types'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

function startTraining() {
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '1' } })
  fireEvent.click(screen.getByRole('button', { name: /Run 1 epoch/ }))
}

function newWorkspace() {
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'New' }))
}

function savedProject(graph: GraphModel, epoch: number): string {
  return JSON.stringify(createProjectStateFile({
    graph, visualizationGraph: graph, initialParameterValues: {}, selectedNodeIds: [],
    phase: 'edit', traceSteps: [], traceIndex: 0, epoch, currentLoss: null,
    display: { showMath: true, showGradient: true, showCode: false, showVisualization: false },
  }))
}

afterEach(() => vi.restoreAllMocks())

describe('execution ownership and reports', () => {
  it('adds single updates to the same dataset loss curves as epoch runs', async () => {
    const graph = createModelPreset('linear')
    const updated = runTrainingStep(graph).graph
    const dataset = updated.nodes.find(node => node.type === 'dataset')!
    const expected = datasetTraining.evaluateDataset(updated, dataset.id, 'train').loss
    render(<App initialGraph={graph} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
    fireEvent.click(screen.getByRole('button', { name: /Run one full training step/ }))
    fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
    await waitFor(() => expect(within(screen.getByRole('list', { name: 'Reported losses' })).getAllByRole('listitem')).toHaveLength(2))
    let history = screen.getByRole('list', { name: 'Reported losses' })
    expect(within(history).getAllByRole('listitem')).toHaveLength(2)
    expect(within(history).getAllByRole('listitem')[1]).toHaveTextContent('Train ' + formatNumber(expected))
    expect(within(history).getAllByRole('listitem')[1]).toHaveTextContent('Held-out')
    startTraining()
    await waitFor(() => expect(screen.getByText('Completed 1 epoch.')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /Run one full training step/ }))
    fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
    await waitFor(() => expect(within(screen.getByRole('list', { name: 'Reported losses' })).getAllByRole('listitem')).toHaveLength(4))
    history = screen.getByRole('list', { name: 'Reported losses' })
    expect(within(history).getAllByRole('listitem').map(item => item.querySelector('span')?.textContent)).toEqual(['Epoch 0', 'Epoch 1', 'Epoch 2', 'Epoch 3'])
  })

  it('records a manually stepped loop only after its parameter update', async () => {
    render(<App initialGraph={createStarterGraph(true)} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
    const step = screen.getByRole('button', { name: /^Step$/i })
    fireEvent.click(step)
    expect(screen.queryByRole('list', { name: 'Reported losses' })).not.toBeInTheDocument()
    for (let i = 0; i < 100 && !screen.queryByText('Epoch 1'); i++) fireEvent.click(step)
    await waitFor(() => expect(within(screen.getByRole('list', { name: 'Reported losses' })).getAllByRole('listitem')).toHaveLength(2))
    expect(within(screen.getByRole('list', { name: 'Reported losses' })).getAllByRole('listitem')).toHaveLength(2)
  })

  it('discards single-step loss evaluation after opening a new workspace', async () => {
    const pending = deferred<ReturnType<typeof datasetTraining.evaluateDataset>>()
    vi.spyOn(datasetTraining, 'evaluateDatasetAsync').mockReturnValueOnce(pending.promise)
    render(<App initialGraph={createModelPreset('linear')} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
    fireEvent.click(screen.getByRole('button', { name: /Run one full training step/ }))
    await waitFor(() => expect(datasetTraining.evaluateDatasetAsync).toHaveBeenCalled())
    newWorkspace()
    await act(async () => { pending.resolve({ loss: 123, examples: 1, predictions: 0, rows: [] }); await pending.promise })
    fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
    expect(screen.queryByRole('list', { name: 'Reported losses' })).not.toBeInTheDocument()
  })

  it.each(['Cancel', 'Escape'])('discards a pending CSV read after %s', async action => {
    const pending = deferred<string>()
    const { container } = render(<App initialGraph={createStarterGraph(true)} />)
    const dataset = container.querySelector<HTMLSelectElement>('.node-dataset select')!
    const original = dataset.value
    fireEvent.change(dataset, { target: { value: 'custom-csv' } })
    const file = new File([], 'delayed.csv', { type: 'text/csv' })
    vi.spyOn(file, 'text').mockReturnValue(pending.promise)
    fireEvent.change(screen.getByLabelText('Choose custom CSV file'), { target: { files: [file] } })
    if (action === 'Cancel') fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    else fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await act(async () => { pending.resolve('x,target\n1,2\n2,4\n3,6\n'); await pending.promise })
    expect(dataset).toHaveValue(original)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('discards an older project read after starting training', async () => {
    const graph = createStarterGraph(true), pendingFile = deferred<string>(), pendingTraining = deferred<GraphModel>()
    const train = vi.spyOn(datasetTraining, 'trainDataset').mockReturnValueOnce(pendingTraining.promise)
    render(<App initialGraph={graph} />)
    const file = new File([], 'delayed.json', { type: 'application/json' })
    vi.spyOn(file, 'text').mockReturnValue(pendingFile.promise)
    fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [file] } })
    startTraining()
    await waitFor(() => expect(train).toHaveBeenCalledTimes(1))
    const replacement = createProjectStateFile({
      graph, visualizationGraph: graph, initialParameterValues: {}, selectedNodeIds: [],
      phase: 'edit', traceSteps: [], traceIndex: 0, epoch: 99, currentLoss: null,
      display: { showMath: true, showGradient: true, showCode: false, showVisualization: false },
    })
    await act(async () => { pendingFile.resolve(JSON.stringify(replacement)); await pendingFile.promise })
    expect(screen.getByText('Epoch 0')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop training' })).toBeInTheDocument()
    expect(train.mock.calls[0][3]?.signal?.aborted).toBe(false)
    newWorkspace()
    await act(async () => { pendingTraining.resolve(graph); await pendingTraining.promise })
  })

  it('does not let an older text read cancel a newer project import', async () => {
    const graph = createStarterGraph(true), pendingText = deferred<string>(), pendingProject = deferred<string>()
    const { container } = render(<App initialGraph={graph} />)
    fireEvent.change(container.querySelector('.node-dataset select')!, { target: { value: 'custom-text' } })
    const textFile = new File([], 'older.csv', { type: 'text/csv' })
    vi.spyOn(textFile, 'text').mockReturnValue(pendingText.promise)
    fireEvent.change(screen.getByLabelText('Choose text data file'), { target: { files: [textFile] } })
    fireEvent.click(screen.getByRole('button', { name: 'Import text dataset' }))
    expect(screen.getByRole('button', { name: 'Preparing…' })).toBeDisabled()

    const projectFile = new File([], 'newer.json', { type: 'application/json' })
    vi.spyOn(projectFile, 'text').mockReturnValue(pendingProject.promise)
    fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [projectFile] } })
    await act(async () => {
      pendingText.resolve('text,label,split\ngood film,positive,train\nbad film,negative,train\ngood movie,positive,test\n')
      await pendingText.promise
    })
    await act(async () => { pendingProject.resolve(savedProject(graph, 99)); await pendingProject.promise })
    expect(screen.getByText('Epoch 99')).toBeInTheDocument()
    expect(container.querySelector('.node-dataset select')).not.toHaveValue('custom-text')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('supersedes an older project read when the text import dialog opens', async () => {
    const graph = createStarterGraph(true), pendingProject = deferred<string>()
    const { container } = render(<App initialGraph={graph} />)
    const projectFile = new File([], 'older.json', { type: 'application/json' })
    vi.spyOn(projectFile, 'text').mockReturnValue(pendingProject.promise)
    fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [projectFile] } })

    fireEvent.change(container.querySelector('.node-dataset select')!, { target: { value: 'custom-text' } })
    expect(screen.getByRole('dialog', { name: 'Import text data' })).toBeInTheDocument()
    await act(async () => { pendingProject.resolve(savedProject(graph, 99)); await pendingProject.promise })
    expect(screen.getByText('Epoch 0')).toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: 'Import text data' })).toBeInTheDocument()
    expect(screen.getByLabelText('Choose text data file')).toBeInTheDocument()
  })

  it('discards a training result after File New, even if the worker ignores abort', async () => {
    const graph = createStarterGraph(true), pending = deferred<GraphModel>()
    const train = vi.spyOn(datasetTraining, 'trainDataset').mockReturnValueOnce(pending.promise)
    const { container } = render(<App initialGraph={graph} />)
    startTraining()
    await waitFor(() => expect(train).toHaveBeenCalledTimes(1))
    const signal = train.mock.calls[0][3]?.signal
    newWorkspace()
    expect(signal?.aborted).toBe(true)
    await act(async () => { pending.resolve(graph); await pending.promise })
    expect(container.querySelectorAll('.react-flow__node')).toHaveLength(0)
    expect(screen.queryByText('Completed 1 epoch.')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop training' })).not.toBeInTheDocument()
  })

  it('cancels a run when its parameter is edited and preserves the edit', async () => {
    const graph = createStarterGraph(true), pending = deferred<GraphModel>()
    const train = vi.spyOn(datasetTraining, 'trainDataset').mockReturnValueOnce(pending.promise)
    const { container } = render(<App initialGraph={graph} />)
    startTraining()
    await waitFor(() => expect(train).toHaveBeenCalledTimes(1))
    const weight = container.querySelector<HTMLInputElement>('[data-id="w"] input')!
    fireEvent.change(weight, { target: { value: '7' } })
    await act(async () => { pending.resolve(graph); await pending.promise })
    expect(weight).toHaveValue('7')
    expect(screen.getByText('Epoch 0')).toBeInTheDocument()
  })

  it('does not let an older run clear the busy state of a newer run', async () => {
    const graph = createStarterGraph(true), first = deferred<GraphModel>(), second = deferred<GraphModel>()
    const train = vi.spyOn(datasetTraining, 'trainDataset').mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    const { container } = render(<App initialGraph={graph} />)
    startTraining()
    await waitFor(() => expect(train).toHaveBeenCalledTimes(1))
    fireEvent.change(container.querySelector('[data-id="w"] input')!, { target: { value: '7' } })
    startTraining()
    await waitFor(() => expect(train).toHaveBeenCalledTimes(2))
    await act(async () => { first.resolve(graph); await first.promise })
    expect(screen.getByRole('button', { name: 'Stop training' })).toBeInTheDocument()
    newWorkspace()
    await act(async () => { second.resolve(graph); await second.promise })
    expect(container.querySelectorAll('.react-flow__node')).toHaveLength(0)
  })

  it('discards inference results after workspace replacement', async () => {
    const graph = createStarterGraph(true), pending = deferred<datasetTraining.DatasetMetrics>()
    const evaluate = vi.spyOn(datasetTraining, 'evaluateDatasetAsync').mockReturnValueOnce(pending.promise)
    const { container } = render(<App initialGraph={graph} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Test' }))
    fireEvent.click(screen.getByRole('button', { name: 'Run inference' }))
    expect(evaluate).toHaveBeenCalledTimes(1)
    newWorkspace()
    await act(async () => {
      pending.resolve(datasetTraining.evaluateDataset(graph, graph.nodes.find(node => node.type === 'dataset')!.id, 'test'))
      await pending.promise
    })
    expect(container.querySelectorAll('.react-flow__node')).toHaveLength(0)
    expect(screen.queryByRole('region', { name: 'Inference report' })).not.toBeInTheDocument()
  })

  it.each(['value', 'activation', 'loss', 'dataset'])('clears inference reports after a %s edit', async field => {
    const { container } = render(<App initialGraph={createStarterGraph(true)} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Test' }))
    fireEvent.click(screen.getByRole('button', { name: 'Run inference' }))
    await screen.findByRole('region', { name: 'Inference report' })
    if (field === 'value') fireEvent.change(container.querySelector('[data-id="w"] input')!, { target: { value: '-10' } })
    if (field === 'activation') fireEvent.change(container.querySelector('.node-activation select')!, { target: { value: 'tanh' } })
    if (field === 'loss') fireEvent.change(container.querySelector('.node-loss select')!, { target: { value: 'mse' } })
    if (field === 'dataset') fireEvent.change(container.querySelector('.node-dataset select')!, { target: { value: 'line-1d' } })
    expect(screen.queryByRole('region', { name: 'Inference report' })).not.toBeInTheDocument()
  })

  it('updates the canonical tensor coordinate from a projected inline weight editor', () => {
    const graph = createModelPreset('decoder')
    graph.view = { ...graph.view!, inspectedNeuron: { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 } }
    const { container } = render(<App initialGraph={graph} />)
    const projected = container.querySelector('[data-id="inspect:blocks.0.ff1.layer:0:w0"]')!
    fireEvent.click(projected)
    fireEvent.change(projected.querySelector('input')!, { target: { value: '42' } })
    expect(screen.getByLabelText('Edit tensor coordinate')).toHaveValue(42)
    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    expect(screen.getByLabelText('Edit tensor coordinate')).toHaveValue(0.12621368371146552)
  })
})
