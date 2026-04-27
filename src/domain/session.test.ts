import { describe, expect, it } from 'vitest'
import { forwardPass } from './engine'
import { createStarterGraph } from './examples'
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
