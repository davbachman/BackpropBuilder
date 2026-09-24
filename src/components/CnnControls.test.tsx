import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { toTensor } from '../domain/tensor'
import type { GraphModel } from '../domain/types'
import { CNN_DIGITS, createCnnPreset, digitTensor } from '../learning/cnn'
import { CnnControls } from './CnnControls'

function Harness({ initial, changed }: { initial: GraphModel; changed: (graph: GraphModel) => void }) {
  const [graph, setGraph] = useState(initial)
  return <CnnControls graph={graph} onGraphChange={next => { setGraph(next); changed(next) }} />
}

describe('CNN controls run the connected graph', () => {
  it('changes the handwritten image, target, and predictions while retaining learned filters', () => {
    const graph = createCnnPreset(), changed = vi.fn()
    render(<Harness initial={graph} changed={changed} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show handwritten digit 8' }))
    const sample = CNN_DIGITS.find(example => example.label === 8 && example.split === 'test')!
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'image-input')!.value).toEqual(digitTensor(sample))
    expect(next.nodes.find(node => node.id === 'digit-target')!.value?.data).toEqual([8])
    expect(next.nodes.find(node => node.id === 'conv-weights')!.params).toEqual(graph.nodes.find(node => node.id === 'conv-weights')!.params)
    expect(next.nodes.find(node => node.id === 'cnn-probabilities')!.value).not.toEqual(graph.nodes.find(node => node.id === 'cnn-probabilities')!.value)
    expect(screen.getByLabelText('8 by 8 handwritten digit 8').children).toHaveLength(64)
    fireEvent.click(screen.getByRole('button', { name: 'Next digit image' }))
    expect(changed.mock.calls[1][0].nodes.find((node: { id: string }) => node.id === 'image-input').value).not.toEqual(next.nodes.find(node => node.id === 'image-input')!.value)
  })

  it('edits one shared convolution weight and updates only that raw feature channel', () => {
    const graph = createCnnPreset(), changed = vi.fn()
    render(<Harness initial={graph} changed={changed} />)
    fireEvent.click(screen.getByRole('button', { name: 'Inspect convolution filter 2' }))
    fireEvent.click(screen.getByRole('button', { name: 'Select filter weight row 1 column 3' }))
    fireEvent.change(screen.getByLabelText('Convolution filter weight'), { target: { value: '0.75' } })
    const next: GraphModel = changed.mock.calls[0][0]
    const beforeWeights = toTensor(graph.nodes.find(node => node.id === 'conv-weights')!.params.value)
    const weights = toTensor(next.nodes.find(node => node.id === 'conv-weights')!.params.value)
    expect(weights.data[11]).toBe(.75)
    weights.data.forEach((weight, index) => { if (index !== 11) expect(weight).toBe(beforeWeights.data[index]) })
    const beforeMap = graph.nodes.find(node => node.id === 'conv-output')!.value!.data
    const afterMap = next.nodes.find(node => node.id === 'conv-output')!.value!.data
    expect(afterMap.some((entry, index) => index % 4 === 1 && entry !== beforeMap[index])).toBe(true)
    afterMap.forEach((entry, index) => { if (index % 4 !== 1) expect(entry).toBe(beforeMap[index]) })
    expect(next.nodes.find(node => node.id === 'cnn-probabilities')!.value).not.toEqual(graph.nodes.find(node => node.id === 'cnn-probabilities')!.value)
  })

  it('traces a selected feature-map cell to exactly nine input pixels and its real calculation', () => {
    const graph = createCnnPreset()
    const { container } = render(<CnnControls graph={graph} onGraphChange={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Feature 3 row 5 column 2' }))
    expect(container.querySelectorAll('.cnn-image-grid .in-patch')).toHaveLength(9)
    expect(screen.getByText('Filter 3 · patch (5, 2)')).toBeInTheDocument()
    const expected = graph.nodes.find(node => node.id === 'conv-output')!.value!.data[(4 * 6 + 1) * 4 + 2]
    expect(container.querySelector('.cnn-patch-readout')).toHaveTextContent(`= ${expected.toFixed(4)}`)
    const raw = graph.nodes.find(node => node.id === 'conv-output')!.value!.data
    const negative = raw.findIndex(number => number < 0)
    expect(negative).toBeGreaterThanOrEqual(0)
    const cell = Math.floor(negative / 4), channel = negative % 4
    const button = screen.getByRole('button', { name: `Feature ${channel + 1} row ${Math.floor(cell / 6) + 1} column ${cell % 6 + 1}` })
    expect(button).toHaveAttribute('title', '0.00000')
    fireEvent.change(screen.getByLabelText('Feature map activation'), { target: { value: 'conv' } })
    expect(button).toHaveAttribute('title', raw[negative].toFixed(5))
  })
})
