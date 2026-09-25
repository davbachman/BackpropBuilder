import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App'
import { DATASET_MENU_OPTIONS, DATASET_OPTIONS } from './domain/datasets'
import { createNode, createSingleNeuronGraph } from './domain/examples'
import { formatFullTensor } from './domain/tensor'

function datasetModel(semanticZoom: boolean) {
  const graph = createSingleNeuronGraph('identity', { x: 1, w: 2, b: 0, target: 3 })
  graph.nodes.push(createNode('dataset', 1))
  graph.edges.push(
    { id: 'dataset-feature', source: 'dataset-1', sourceSlot: 0, target: 'x', inputSlot: 0 },
    { id: 'dataset-target', source: 'dataset-1', sourceSlot: 1, target: 'target', inputSlot: 0 },
  )
  graph.view = { semanticZoom, expandedGroupIds: [] }
  return graph
}

describe.each([true, false])('Dataset controls with continuous zoom %s', semanticZoom => {
  it('opens a CSV picker from the Dataset block and installs editable feature and target ports', async () => {
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    fireEvent.click(card)
    const selector = within(card).getByRole('combobox', { name: 'Dataset for dataset' })
    fireEvent.change(selector, { target: { value: 'custom-csv' } })
    const chooser = screen.getByRole('dialog', { name: 'Choose a CSV file' })
    const input = within(chooser).getByLabelText('Choose custom CSV file') as HTMLInputElement
    expect(input).toBeVisible()
    const text = 'height,width,y\n1,2,3\n2,3,5\n3,4,7\n4,5,9\n'
    const file = Object.assign(new File([text], 'measurements.csv', { type: 'text/csv' }), { text: async () => text })
    fireEvent.change(input, { target: { files: [file] } })
    await waitFor(() => expect(selector).toHaveValue('custom-csv'))
    expect(screen.queryByRole('dialog', { name: 'Choose a CSV file' })).not.toBeInTheDocument()
    expect(card.querySelector('.semantic-operation, .builder-node')).toHaveStyle({ width: '280px' })
    if (semanticZoom) expect(card).toHaveStyle({ width: '280px' })
    expect(within(card).getByLabelText('height output')).toHaveAttribute('data-handleid', 'out-0')
    expect(within(card).getByLabelText('width output')).toHaveAttribute('data-handleid', 'out-1')
    expect(within(card).getByLabelText('y output')).toHaveAttribute('data-handleid', 'out-2')
    expect(screen.queryByRole('combobox', { name: 'CSV target column' })).not.toBeInTheDocument()
    expect(screen.getAllByText(/Connect a column to a Target block/).length).toBeGreaterThan(0)
    expect(screen.getByRole('combobox', { name: 'CSV task' })).toHaveValue('regression')
    expect(screen.getByRole('button', { name: 'Replace CSV file' })).toBeInTheDocument()
  })

  it('restores every dataset choice and labels the actual feature and target handles', () => {
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    const selector = within(card).getByRole('combobox', { name: 'Dataset for dataset' })
    expect(selector).toHaveClass('nodrag', 'nowheel')
    expect(within(selector).getAllByRole('option').map(option => option.textContent)).toEqual(DATASET_MENU_OPTIONS.map(option => option.label))
    for (const dataset of DATASET_OPTIONS) {
      fireEvent.change(selector, { target: { value: dataset.kind } })
      expect(selector).toHaveValue(dataset.kind)
      expect(card.querySelectorAll('.source-handle')).toHaveLength(dataset.featureLabels.length + 1)
      expect(Array.from(card.querySelectorAll('.semantic-dataset-port')).map(label => label.textContent)).toEqual([...dataset.featureLabels, dataset.targetLabel])
      dataset.featureLabels.forEach((label, slot) => {
        expect(within(card).getByLabelText(`Feature ${label} output`)).toHaveAttribute('data-handleid', `out-${slot}`)
      })
      expect(within(card).getByLabelText(`Target ${dataset.targetLabel} output`)).toHaveAttribute('data-handleid', `out-${dataset.featureLabels.length}`)
    }
  })

  it('keeps feature and target connections correct when either selector changes feature count, and supports undo', async () => {
    const user = userEvent.setup()
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    fireEvent.click(card)
    const nodeSelector = within(card).getByRole('combobox', { name: 'Dataset for dataset' })
    const inspectorSelector = screen.getByRole('combobox', { name: 'Dataset selection' })
    await user.selectOptions(nodeSelector, 'circle-center')
    expect(inspectorSelector).toHaveValue('circle-center')
    expect(screen.getByLabelText('Dataset outputs')).toHaveTextContent('Feature x1')
    expect(screen.getByLabelText('Dataset outputs')).toHaveTextContent('Feature x2')
    expect(screen.getByLabelText('Dataset outputs')).toHaveTextContent('Target y')
    const expectResults = (kind: string) => {
      const dataset = DATASET_OPTIONS.find(option => option.kind === kind)!
      expect(container.querySelector('[data-id="x"] .semantic-operation-value')).toHaveAttribute('title', formatFullTensor(dataset.featureValues[0]))
      expect(container.querySelector('[data-id="target"] .semantic-operation-value')).toHaveAttribute('title', formatFullTensor(dataset.targetValue))
    }
    await user.click(screen.getByRole('tab', { name: 'Train' }))
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('circle-center')
    fireEvent.click(card)
    await user.selectOptions(screen.getByRole('combobox', { name: 'Dataset selection' }), 'cubic-1d')
    expect(nodeSelector).toHaveValue('cubic-1d')
    expect(within(card).getByLabelText('Target y output')).toHaveAttribute('data-handleid', 'out-1')
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('cubic-1d')
    // Undo the forward pass, then the dataset change, restoring the old ports.
    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    expect(nodeSelector).toHaveValue('circle-center')
    expect(within(card).getByLabelText('Target y output')).toHaveAttribute('data-handleid', 'out-2')
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('circle-center')
  })
})
