import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { forwardPass } from '../domain/engine'
import { createModelPreset } from '../domain/modelPresets'
import type { GraphModel } from '../domain/types'
import { DecoderControls } from './DecoderControls'

function Harness({ initial, changed }: { initial: GraphModel; changed: (graph: GraphModel) => void }) {
  const [graph, setGraph] = useState(initial)
  return <DecoderControls graph={graph} onGraphChange={next => { setGraph(next); changed(next) }} />
}

describe('decoder controls on the actual graph', () => {
  it('generates from an evaluated graph and updates token, position and training rows together', () => {
    const changed = vi.fn()
    render(<Harness initial={forwardPass(createModelPreset('decoder')).graph} changed={changed} />)
    fireEvent.click(screen.getByRole('button', { name: 'Generate next token' }))
    expect(changed).toHaveBeenCalledTimes(1)
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'token-ids')?.value).toEqual({ shape: [4], data: [0, 1, 2, 3] })
    expect(next.nodes.find(node => node.id === 'position-ids')?.value).toEqual({ shape: [4], data: [0, 1, 2, 3] })
    expect(next.nodes.find(node => node.id === 'decoder-targets')?.value).toEqual({ shape: [4], data: [1, 2, 3, 1] })
    expect(next.nodes.find(node => node.id === 'decoder-logits')?.value?.shape).toEqual([4, 5])
    expect(screen.getByLabelText('Prompt')).toHaveValue('<bos> red green blue')
    fireEvent.click(screen.getByRole('button', { name: 'Generate next token' }))
    expect(screen.getByLabelText('Prompt')).toHaveValue('<bos> red green blue red')
  })

  it('samples from edited canvas parameters, preserving them through generation', () => {
    const changed = vi.fn(), graph = createModelPreset('decoder')
    graph.nodes.find(node => node.id === 'outputBias')!.params.value = { shape: [5], data: [0, 100, 0, 0, 0] }
    render(<Harness initial={forwardPass(graph).graph} changed={changed} />)
    fireEvent.change(screen.getByLabelText('Generation mode'), { target: { value: 'sample' } })
    fireEvent.change(screen.getByLabelText('Sampling top-k'), { target: { value: '1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Generate next token' }))
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'token-ids')?.value?.data).toEqual([0, 1, 2, 1])
    expect(next.nodes.find(node => node.id === 'outputBias')?.params.value).toEqual({ shape: [5], data: [0, 100, 0, 0, 0] })
  })

  it('validates typed tokens and applies a new prompt without stale cached values', () => {
    const changed = vi.fn()
    render(<Harness initial={forwardPass(createModelPreset('decoder')).graph} changed={changed} />)
    fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'red chartreuse' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply prompt' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Unknown token: chartreuse')
    expect(changed).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'red blue' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply prompt' }))
    const next: GraphModel = changed.mock.calls[0][0]
    expect(next.nodes.find(node => node.id === 'token-ids')?.value?.data).toEqual([1, 3])
    expect(next.nodes.find(node => node.id === 'decoder-probabilities')?.value?.shape).toEqual([2, 5])
    expect(next.nodes.find(node => node.id === 'decoder-targets')?.value?.data).toEqual([2, 1])
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('stops generation at end of sequence and at the context limit', () => {
    const changed = vi.fn()
    render(<Harness initial={createModelPreset('decoder')} changed={changed} />)
    fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: '<eos>' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply prompt' }))
    expect(screen.getByRole('button', { name: 'Generate next token' })).toBeDisabled()
    expect(screen.getByText(/End token reached/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: Array(12).fill('red').join(' ') } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply prompt' }))
    expect(screen.getByRole('button', { name: 'Generate next token' })).toBeDisabled()
    expect(screen.getByText(/context is full/)).toBeInTheDocument()
  })
})
