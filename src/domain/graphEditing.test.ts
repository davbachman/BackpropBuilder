import { describe, expect, it } from 'vitest'
import { connectGraphNodes } from './graphEditing'
import { createStarterGraph } from './examples'

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
})
