import { describe, expect, it } from 'vitest'
import { cloneGraph, forwardPass } from './engine'
import { createStarterGraph } from './examples'
import { createModelPreset } from './modelPresets'
import {
  createProjectStateFile,
  parseProjectStateFile,
} from './session'
import { scalarValue } from './tensor'
import type { ProjectStateSnapshot } from './types'

function projectSnapshot(): ProjectStateSnapshot {
  const forward = forwardPass(createStarterGraph())
  return {
    graph: forward.graph,
    visualizationGraph: forward.graph,
    initialParameterValues: {
      w: scalarValue(0.5),
      b: scalarValue(-0.3),
    },
    selectedNodeIds: ['pred'],
    selectedGroupId: 'group-1',
    phase: 'forward',
    traceSteps: forward.steps,
    traceIndex: 1,
    epoch: 3,
    currentLoss: forward.loss ?? null,
    display: {
      showMath: true,
      showGradient: false,
      showCode: true,
      showVisualization: true,
    },
  }
}

describe('project state files', () => {
  it('creates a versioned project state file that preserves graph and workspace state', () => {
    const snapshot = projectSnapshot()

    const file = createProjectStateFile(snapshot)

    expect(file.kind).toBe('backprop-builder-state')
    expect(file.version).toBe(1)
    expect(file.savedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
    expect(file.state.graph.learningRate).toBe(0.1)
    expect(file.state.graph.nodes.find((node) => node.id === 'w')?.params.value).toEqual(scalarValue(0.5))
    expect(file.state.initialParameterValues.w).toEqual(scalarValue(0.5))
    expect(file.state.selectedNodeIds).toEqual(['pred'])
    expect(file.state.selectedGroupId).toBe('group-1')
    expect(file.state.phase).toBe('forward')
    expect(file.state.traceSteps).toHaveLength(snapshot.traceSteps.length)
    expect(file.state.traceIndex).toBe(1)
    expect(file.state.epoch).toBe(3)
    expect(file.state.currentLoss).toBeCloseTo(snapshot.currentLoss ?? 0)
    expect(file.state.display).toEqual(snapshot.display)
  })

  it('parses a valid project state file into cloned graph and tensor values', () => {
    const snapshot = projectSnapshot()
    const file = createProjectStateFile(snapshot)

    const result = parseProjectStateFile(JSON.stringify(file))

    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.graph).toEqual(file.state.graph)
    expect(result.file.state.graph).not.toBe(file.state.graph)
    expect(result.file.state.graph.nodes[0]).not.toBe(file.state.graph.nodes[0])
    expect(result.file.state.initialParameterValues.w).toEqual(scalarValue(0.5))
    expect(result.file.state.initialParameterValues.w).not.toBe(file.state.initialParameterValues.w)
  })

  it('normalizes a null selected group from JSON into an undefined selection', () => {
    const file = createProjectStateFile(projectSnapshot())
    const serialized = JSON.stringify({
      ...file,
      state: {
        ...file.state,
        selectedGroupId: null,
      },
    })

    const result = parseProjectStateFile(serialized)

    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.selectedGroupId).toBeUndefined()
  })

  it('round-trips transformer tensor operations, exact masks and semantic inspection state', () => {
    const snapshot = projectSnapshot()
    snapshot.graph = forwardPass(createModelPreset('decoder')).graph
    snapshot.graph.view = { expandedGroupIds: ['blocks.0', 'blocks.0.mlp'], semanticZoom: true, inspectedNeuron: { groupId: 'blocks.0.ff1.layer', unitIndex: 2, row: 1 } }
    const file = createProjectStateFile(snapshot)
    const result = parseProjectStateFile(JSON.stringify(file))
    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.graph).toEqual(file.state.graph)
    expect(result.file.state.graph.nodes.find(node => node.type === 'causal-mask')!.value!.excluded).toContain(true)
    expect(result.file.state.graph.groups!.find(group => group.id === 'blocks.0.ff1.layer')!.detail).toEqual(snapshot.graph.groups!.find(group => group.id === 'blocks.0.ff1.layer')!.detail)
    file.state.graph.view!.inspectedNeuron!.unitIndex = 7
    expect(snapshot.graph.view.inspectedNeuron!.unitIndex).toBe(2)
  })

  it('saves manual block and projected-node placement with independent cloned coordinates', () => {
    const snapshot = projectSnapshot()
    snapshot.graph.view = { expandedGroupIds: [], manualNodePlacements: {
      pred: { parentId: 'group-1', offset: { x: 12, y: 34 } },
      w: { offset: { x: -25, y: 40 }, scale: .025 },
    }, layoutOffsets: {
      'visual-group:blocks.0': { x: 120, y: -35 },
      'inspect:blocks.0.ff1.layer:0:w0': { x: 24, y: 12 },
    } }
    const cloned = cloneGraph(snapshot.graph)
    cloned.view!.layoutOffsets!['visual-group:blocks.0'].x = 999
    expect(snapshot.graph.view.layoutOffsets!['visual-group:blocks.0'].x).toBe(120)
    cloned.view!.manualNodePlacements!.pred.offset.x = 999
    expect(snapshot.graph.view.manualNodePlacements!.pred.offset.x).toBe(12)

    const file = createProjectStateFile(snapshot)
    const result = parseProjectStateFile(JSON.stringify(file))
    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.graph.view!.layoutOffsets).toEqual(snapshot.graph.view.layoutOffsets)
    expect(result.file.state.graph.view!.manualNodePlacements).toEqual(snapshot.graph.view.manualNodePlacements)
    result.file.state.graph.view!.layoutOffsets!['inspect:blocks.0.ff1.layer:0:w0'].y = 999
    expect(file.state.graph.view!.layoutOffsets!['inspect:blocks.0.ff1.layer:0:w0'].y).toBe(12)
    expect(snapshot.graph.view.layoutOffsets!['inspect:blocks.0.ff1.layer:0:w0'].y).toBe(12)
  })

  it('round-trips the wiring-independent layout without sharing mutable edge references', () => {
    const snapshot = projectSnapshot()
    snapshot.graph.view = { expandedGroupIds: [], semanticZoom: true, layoutEdges: [{ id: 'layout-edge', source: 'x', target: 'mul', inputSlot: 0 }] }
    const file = createProjectStateFile(snapshot)
    const result = parseProjectStateFile(JSON.stringify(file))
    expect(result.ok).toBe(true)
    if (!result.ok) throw new Error(result.error)
    expect(result.file.state.graph.view!.layoutEdges).toEqual(snapshot.graph.view.layoutEdges)
    result.file.state.graph.view!.layoutEdges![0].source = 'changed'
    expect(file.state.graph.view!.layoutEdges![0].source).toBe('x')
    expect(snapshot.graph.view.layoutEdges![0].source).toBe('x')
    const malformed = { ...file, state: { ...file.state, graph: { ...file.state.graph, view: { expandedGroupIds: [], layoutEdges: [{ source: 'x' }] } } } }
    expect(parseProjectStateFile(JSON.stringify(malformed)).ok).toBe(false)
  })

  it('rejects malformed semantic layout coordinates when loading a project', () => {
    for (const offset of [{ x: '10', y: 0 }, { x: null, y: 0 }, { x: 4 }, []]) {
      const file = createProjectStateFile(projectSnapshot())
      const graph = { ...file.state.graph, view: { expandedGroupIds: [], layoutOffsets: { block: offset } } }
      expect(parseProjectStateFile(JSON.stringify({ ...file, state: { ...file.state, graph } })).ok).toBe(false)
    }
  })

  it('rejects malformed project state files with a helpful error', () => {
    expect(parseProjectStateFile('{').ok).toBe(false)
    expect(parseProjectStateFile(JSON.stringify({ kind: 'session-summary', version: 1 })).ok).toBe(false)
    expect(parseProjectStateFile(JSON.stringify({ kind: 'backprop-builder-state', version: 99 })).ok).toBe(false)
    expect(
      parseProjectStateFile(
        JSON.stringify({
          kind: 'backprop-builder-state',
          version: 1,
          savedAt: new Date().toISOString(),
          state: { graph: { nodes: [], learningRate: 0.1 } },
        }),
    ).ok,
    ).toBe(false)
  })
})
