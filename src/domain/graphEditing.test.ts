import { describe, expect, it } from 'vitest'
import { connectGraphNodes } from './graphEditing'
import { createStarterGraph } from './examples'
import type { GraphModel } from './types'

describe('graph edge editing', () => {
  it('replaces an existing edge when connecting to an occupied input slot', () => {
    const graph = createStarterGraph()

    const result = connectGraphNodes(
      graph,
      { source: 'x', target: 'add', targetHandle: 'in-1' },
      () => 'x-add-replacement',
    )

    expect(result).toBeDefined()
    expect(result?.edges).not.toEqual(expect.arrayContaining([expect.objectContaining({ id: 'b-add' })]))
    expect(result?.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'x-add-replacement',
          source: 'x',
          target: 'add',
          inputSlot: 1,
        }),
      ]),
    )
    expect(result?.edges).toHaveLength(graph.edges.length)
  })

  it('rejects replacement connections that would create a cycle', () => {
    const graph = createStarterGraph()

    const result = connectGraphNodes(
      graph,
      { source: 'pred', target: 'mul', targetHandle: 'in-0' },
      () => 'pred-mul-cycle',
    )

    expect(result).toBeUndefined()
  })

  it('records the selected dataset output slot from source handles', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'dataset', type: 'dataset', label: 'dataset', position: { x: 0, y: 0 }, params: { dataset: 'circle-center' } },
        { id: 'add', type: 'add', label: 'add', position: { x: 240, y: 0 }, params: {} },
      ],
      edges: [],
    }

    const result = connectGraphNodes(
      graph,
      { source: 'dataset', sourceHandle: 'out-1', target: 'add', targetHandle: 'in-0' },
      () => 'dataset-add',
    )

    expect(result?.edges).toEqual([
      expect.objectContaining({
        id: 'dataset-add',
        source: 'dataset',
        sourceSlot: 1,
        target: 'add',
        inputSlot: 0,
      }),
    ])
  })

  it('replaces an existing source edge when reconnecting from that output handle', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'source', type: 'input', label: 'x', position: { x: 0, y: 0 }, params: { value: 1 } },
        { id: 'old-target', type: 'activation', label: 'old', position: { x: 240, y: 0 }, params: { activation: 'identity' } },
        { id: 'new-target', type: 'activation', label: 'new', position: { x: 240, y: 160 }, params: { activation: 'identity' } },
      ],
      edges: [{ id: 'source-old', source: 'source', target: 'old-target', inputSlot: 0 }],
    }
    const connection = {
      source: 'source',
      target: 'new-target',
      targetHandle: 'in-0',
      replaceEdgeId: 'source-old',
    }

    const result = connectGraphNodes(graph, connection, () => 'source-new')

    expect(result?.edges.map((edge) => edge.id)).toEqual(['source-new'])
    expect(result?.edges[0]).toEqual(
      expect.objectContaining({
        source: 'source',
        target: 'new-target',
        inputSlot: 0,
      }),
    )
  })

  it('allows dataset outputs to connect into input and target nodes', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'dataset', type: 'dataset', label: 'dataset', position: { x: 0, y: 0 }, params: { dataset: 'line-1d' } },
        { id: 'x', type: 'input', label: 'x', position: { x: 240, y: 0 }, params: { value: 0 } },
        { id: 'target', type: 'target', label: 'y', position: { x: 240, y: 160 }, params: { value: 0 } },
      ],
      edges: [],
    }

    const withInput = connectGraphNodes(
      graph,
      { source: 'dataset', sourceHandle: 'out-0', target: 'x', targetHandle: 'in-0' },
      () => 'dataset-x',
    )
    const withTarget = connectGraphNodes(
      withInput!,
      { source: 'dataset', sourceHandle: 'out-1', target: 'target', targetHandle: 'in-0' },
      () => 'dataset-target',
    )

    expect(withTarget?.edges).toEqual([
      expect.objectContaining({ id: 'dataset-x', source: 'dataset', target: 'x', inputSlot: 0 }),
      expect.objectContaining({ id: 'dataset-target', source: 'dataset', sourceSlot: 1, target: 'target', inputSlot: 0 }),
    ])
  })
})
