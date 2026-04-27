import { describe, expect, it } from 'vitest'
import { MIN_NODE_HEIGHT, NODE_WIDTH } from './engine'
import { createStarterGraph } from './examples'
import {
  deleteVisualGroup,
  explodeVisualGroup,
  mergeNodesIntoVisualGroup,
  moveVisualGroup,
  resolveVisualGroupInputHandle,
  resolveVisualGroupOutputHandle,
  visualGroupInterface,
} from './grouping'
import type { GraphModel } from './types'

describe('visual graph grouping', () => {
  it('merges selected nodes into editor metadata without changing computation nodes or edges', () => {
    const graph = createStarterGraph()

    const result = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul'])

    expect(result.group).toEqual(
      expect.objectContaining({
        id: 'group-1',
        label: 'Group 1',
        nodeIds: ['x', 'w', 'mul'],
        dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
      }),
    )
    expect(result.graph.nodes).toEqual(graph.nodes)
    expect(result.graph.edges).toEqual(graph.edges)
    expect(result.graph.groups).toHaveLength(1)
  })

  it('moves a merged group by translating the hidden member nodes with it', () => {
    const graph = mergeNodesIntoVisualGroup(createStarterGraph(), ['x', 'w', 'mul']).graph
    const group = graph.groups?.[0]
    expect(group).toBeDefined()

    const moved = moveVisualGroup(graph, group!.id, {
      x: group!.position.x + 75,
      y: group!.position.y + 40,
    })

    expect(moved.groups?.[0].position).toEqual({ x: group!.position.x + 75, y: group!.position.y + 40 })
    expect(moved.nodes.find((node) => node.id === 'x')?.position).toEqual({ x: 115, y: 100 })
    expect(moved.nodes.find((node) => node.id === 'w')?.position).toEqual({ x: 115, y: 300 })
    expect(moved.nodes.find((node) => node.id === 'mul')?.position).toEqual({ x: 395, y: 200 })
    expect(moved.edges).toEqual(graph.edges)
  })

  it('explodes a merged group back into the original visible computation nodes', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['x', 'w', 'mul']).graph

    const exploded = explodeVisualGroup(grouped, 'group-1')

    expect(exploded.groups).toEqual([])
    expect(exploded.nodes.map((node) => node.id)).toEqual(grouped.nodes.map((node) => node.id))
    expect(exploded.edges).toEqual(grouped.edges)
  })

  it('deletes a merged group by removing its hidden computation nodes and incident edges', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['x', 'w', 'mul']).graph

    const deleted = deleteVisualGroup(grouped, 'group-1')

    expect(deleted.groups).toEqual([])
    expect(deleted.nodes.map((node) => node.id)).toEqual(['b', 'add', 'pred', 'target', 'loss'])
    expect(deleted.edges.map((edge) => edge.id)).toEqual(['b-add', 'add-act', 'act-loss', 'target-loss'])
  })

  it('derives collapsed group inputs and outputs from boundary edges', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph
    const group = grouped.groups?.[0]
    expect(group).toBeDefined()

    const groupInterface = visualGroupInterface(grouped, group!)

    expect(groupInterface.inputs).toEqual([
      { edgeId: 'x-mul', handleId: 'in-0', target: 'mul', inputSlot: 0 },
      { edgeId: 'w-mul', handleId: 'in-1', target: 'mul', inputSlot: 1 },
      { edgeId: 'b-add', handleId: 'in-2', target: 'add', inputSlot: 1 },
    ])
    expect(groupInterface.outputs).toEqual([{ edgeId: 'add-act', handleId: 'out-0', source: 'add' }])
  })

  it('keeps a collapsed group input exposed after its boundary edge is deleted', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph
    const group = grouped.groups?.[0]
    expect(group).toBeDefined()

    const disconnected = {
      ...grouped,
      edges: grouped.edges.filter((edge) => edge.id !== 'x-mul'),
    }

    const groupInterface = visualGroupInterface(disconnected, group!)

    expect(groupInterface.inputs).toEqual([
      { handleId: 'in-0', target: 'mul', inputSlot: 0 },
      { edgeId: 'w-mul', handleId: 'in-1', target: 'mul', inputSlot: 1 },
      { edgeId: 'b-add', handleId: 'in-2', target: 'add', inputSlot: 1 },
    ])
  })

  it('resolves a group input handle to its internal node input slot', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    expect(resolveVisualGroupInputHandle(grouped, 'group-1', 'in-1')).toEqual({
      target: 'mul',
      targetHandle: 'in-1',
    })
  })

  it('resolves a group output handle to its internal source and represented boundary edge', () => {
    const grouped = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    expect(resolveVisualGroupOutputHandle(grouped, 'group-1', 'out-0')).toEqual({
      source: 'add',
      edgeId: 'add-act',
    })
  })

  it('uses one output handle for multiple boundary edges from the same internal output', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      groups: [
        {
          id: 'group-1',
          label: 'Group 1',
          nodeIds: ['source', 'add'],
          position: { x: 80, y: 80 },
          dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
        },
      ],
      nodes: [
        { id: 'source', type: 'input', label: 'x', position: { x: 0, y: 0 }, params: { value: 1 } },
        { id: 'add', type: 'add', label: 'add', position: { x: 240, y: 0 }, params: {} },
        { id: 'first-target', type: 'activation', label: 'a1', position: { x: 520, y: 0 }, params: { activation: 'identity' } },
        { id: 'second-target', type: 'activation', label: 'a2', position: { x: 520, y: 160 }, params: { activation: 'identity' } },
      ],
      edges: [
        { id: 'source-add', source: 'source', target: 'add', inputSlot: 0 },
        { id: 'add-first', source: 'add', target: 'first-target', inputSlot: 0 },
        { id: 'add-second', source: 'add', target: 'second-target', inputSlot: 0 },
      ],
    }

    const groupInterface = visualGroupInterface(graph, graph.groups![0])

    expect(groupInterface.outputs).toEqual([
      {
        edgeIds: ['add-first', 'add-second'],
        handleId: 'out-0',
        source: 'add',
      },
    ])
    expect(resolveVisualGroupOutputHandle(graph, 'group-1', 'out-0')).toEqual({ source: 'add' })
  })

  it('exposes and resolves group outputs for internal nodes with no outgoing edge', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'input-1', type: 'input', label: 'x1', position: { x: 40, y: 60 }, params: { value: 1 } },
        { id: 'weight-1', type: 'weight', label: 'w1', position: { x: 40, y: 220 }, params: { value: 0.5 } },
      ],
      edges: [],
    }
    const grouped = mergeNodesIntoVisualGroup(graph, ['input-1', 'weight-1']).graph

    const groupInterface = visualGroupInterface(grouped, grouped.groups![0])

    expect(groupInterface.outputs).toEqual([
      { handleId: 'out-0', source: 'input-1' },
      { handleId: 'out-1', source: 'weight-1' },
    ])
    expect(resolveVisualGroupOutputHandle(grouped, 'group-1', 'out-1')).toEqual({ source: 'weight-1' })
  })
})
