import { describe, expect, it } from 'vitest'
import { createStarterGraph } from './examples'
import { issueNodeIds, problemNodeIds } from './validationPresentation'

describe('validation issue locations', () => {
  it('attributes a multiple-loss error to both loss blocks and leaves a cycle as model-wide', () => {
    const graph = createStarterGraph()
    graph.nodes.push({ ...graph.nodes.find(node => node.id === 'loss')!, id: 'other-loss' })
    expect(issueNodeIds(graph, { code: 'multiple-losses', message: 'Use one loss.' })).toEqual(['loss', 'other-loss'])
    expect(problemNodeIds(graph, [{ code: 'multiple-losses', message: 'Use one loss.' }])).toEqual(new Set(['loss', 'other-loss']))
    expect(issueNodeIds(graph, { code: 'cycle', message: 'Cycle.' })).toEqual([])
  })
})
