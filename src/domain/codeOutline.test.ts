import { describe, expect, it } from 'vitest'
import { buildCodeOutline, codeForGroup, type CodeOutlineLine } from './codeOutline'
import { compactVisualHierarchy } from './continuousScene'
import { formulaForNode } from './engine'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'

function flatten(lines: CodeOutlineLine[]): CodeOutlineLine[] {
  return lines.flatMap(line => [line, ...flatten(line.children)])
}

describe('code outline', () => {
  it.each(LESSONS)('lists every calculation and visible block exactly once for $id', ({ id }) => {
    const graph = compactVisualHierarchy(createModelPreset(id))
    const lines = flatten(buildCodeOutline(graph))
    expect(lines.filter(line => line.kind === 'node').map(line => line.id).sort()).toEqual(graph.nodes.map(node => node.id).sort())
    expect(lines.filter(line => line.kind === 'group').map(line => line.id).sort()).toEqual((graph.groups ?? []).map(group => group.id).sort())
    for (const node of graph.nodes) {
      const line = lines.find(candidate => candidate.kind === 'node' && candidate.id === node.id)!
      expect(line.code).toBe(formulaForNode(node, graph))
    }
    for (const group of graph.groups ?? []) {
      const line = lines.find(candidate => candidate.kind === 'group' && candidate.id === group.id)!
      expect(line.code).toBe(codeForGroup(graph, group))
      expect(line.parentId).toBe(group.parentId)
    }
  })

  it('places the neuron calculations under its foldable call and leaves data and loss at model level', () => {
    const lines = buildCodeOutline(createModelPreset('linear'))
    const neuron = lines.find(line => line.kind === 'group' && line.label === 'Linear neuron 1')!
    expect(neuron.code).toMatch(/^linear_neuron_1 = neuron\(/)
    expect(neuron.children.map(line => line.label)).toEqual(expect.arrayContaining(['layer-0/weight-0-0', 'layer-0/bias-0', 'Weighted sum']))
    expect(lines.map(line => line.label)).toEqual(expect.arrayContaining(['Dataset', 'Squared error (full square)']))
    expect(lines.indexOf(neuron)).toBeLessThan(lines.findIndex(line => line.label === 'Squared error (full square)'))
  })

  it('gives repeated module names distinct code variables', () => {
    const graph = createModelPreset('decoder')
    const normalizations = graph.groups!.filter(group => group.label === 'Layer normalization')
    const names = normalizations.map(group => codeForGroup(graph, group).split(' = ')[0])
    expect(new Set(names).size).toBe(normalizations.length)
  })
})
