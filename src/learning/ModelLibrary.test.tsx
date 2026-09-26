/// <reference types="node" />
import { readFileSync } from 'node:fs'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { forwardPass, validateGraph } from '../domain/engine'
import { createStarterGraph } from '../domain/examples'
import { mergeNodesIntoVisualGroup } from '../domain/grouping'
import { parseProjectStateFile } from '../domain/session'
import { LESSONS } from './presets'

const files = [...LESSONS.map(lesson => lesson.id), 'block-untrained', 'decoder-untrained', 'starter-neuron']
const modelText = (name: string) => readFileSync(`${process.cwd()}/public/models/${name}.json`, 'utf8')

describe('downloadable model library', () => {
  it.each(files)('imports %s as an editable dataset-backed project', name => {
    const result = parseProjectStateFile(modelText(name))
    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.graph.nodes.some(node => node.type === 'dataset')).toBe(true)
    expect(validateGraph(result.file.state.graph).filter(issue => issue.code !== 'disconnected')).toEqual([])
    expect(forwardPass(result.file.state.graph).graph.nodes.length).toBeGreaterThan(0)
  })

  it('opens blank and imports an example through the same File menu used for saved work', async () => {
    const { container } = render(<App />)
    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Preset gallery' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Architecture cards' })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    expect(screen.getByRole('menuitem', { name: 'Import' })).toBeInTheDocument()
    const modelFile = Object.assign(new File([modelText('linear')], 'linear.json', { type: 'application/json' }), {
      text: async () => modelText('linear'),
    })
    fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [modelFile] } })
    await waitFor(() => expect(container.querySelector('.canvas-header-actions p')?.textContent).toMatch(/^9 nodes, 9 edges, \d+ groups?$/))
    expect(screen.queryByRole('button', { name: 'Builder cards' })).not.toBeInTheDocument()
    expect(container.querySelector('.react-flow__node[data-id="loss"] .builder-node.node-loss')).toBeInTheDocument()
    expect(container.querySelector('.react-flow__node[data-id^="visual-group:"] .continuous-card-cover')).toBeInTheDocument()
    expect(container.querySelector('.react-flow__node[data-id="loss"] .builder-node.node-loss')).toBeInTheDocument()
  })

  it('lets a student name a custom group from its context menu and shows the name in code', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(true), ['mul', 'add'])
    expect(grouped.group).toBeDefined()
    const { container } = render(<App initialGraph={grouped.graph} />)
    const group = container.querySelector('.react-flow__node[data-id="visual-group:group-1"]')
    expect(group).toBeInTheDocument()
    expect(group!.querySelector('.node-icon')).not.toBeInTheDocument()
    fireEvent.contextMenu(group!, { clientX: 320, clientY: 220 })
    expect(screen.getByRole('dialog', { name: 'Name block group' })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Group name'), { target: { value: 'func' } })
    fireEvent.click(screen.getByRole('button', { name: 'Rename' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Code' }))
    expect(screen.getByRole('button', { name: /func\(/ })).toBeInTheDocument()
  })
})
