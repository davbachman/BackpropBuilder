import { describe, expect, it } from 'vitest'
import { buildCodeOutline, codeForGroup, type CodeOutlineLine } from './codeOutline'
import { compactVisualHierarchy } from './continuousScene'
import { formulaForNode } from './engine'
import { createModelPreset } from './modelPresets'
import { createNode } from './examples'
import type { GraphModel } from './types'
import { LESSONS } from '../learning/presets'

function flatten(lines: CodeOutlineLine[]): CodeOutlineLine[] {
  return lines.flatMap(line => [line, ...flatten(line.children)])
}

describe('code outline', () => {
  it('uses an explicit output variable in assignments and downstream references, and restores automatic naming', () => {
    const input = { ...createNode('input', 0), label: 'mean_radius' }
    const standard = createNode('standardize', 1)
    const activation = createNode('activation', 2)
    activation.params.activation = 'sigmoid'
    const graph: GraphModel = { nodes: [input, standard, activation], edges: [
      { id: 'in', source: input.id, target: standard.id, inputSlot: 0 },
      { id: 'out', source: standard.id, target: activation.id, inputSlot: 0 },
    ], learningRate: 0.1 }
    standard.params.outputName = 'u'
    let lines = buildCodeOutline(graph)
    expect(lines.find(line => line.id === standard.id)?.code).toBe('u = (mean_radius − μ_train) / s_train')
    expect(lines.find(line => line.id === activation.id)?.code).toBe('z1 = sigmoid(u)')
    expect(standard.label).toBe('Standardize features')
    const group = { id: 'normalization', label: 'Normalize', nodeIds: [standard.id], position: { x: 0, y: 0 }, dimensions: { width: 200, height: 100 } }
    expect(codeForGroup(graph, group)).toBe('u = normalize(mean_radius)')
    expect(codeForGroup(graph, { ...group, nodeIds: [activation.id] })).toMatch(/ = normalize\(u\)$/)
    standard.params.outputName = undefined
    lines = buildCodeOutline(graph)
    expect(lines.find(line => line.id === standard.id)?.code).toBe('z1 = (mean_radius − μ_train) / s_train')
    expect(lines.find(line => line.id === activation.id)?.code).toBe('z2 = sigmoid(z1)')
    standard.params.outputName = 'z1'
    expect(buildCodeOutline(graph).find(line => line.id === activation.id)?.code).toBe('z2 = sigmoid(z1)')
    standard.params.outputName = 'mean_radius'
    activation.params.outputName = 'mean_radius'
    lines = buildCodeOutline(graph)
    expect(lines.find(line => line.id === standard.id)?.code).toBe('mean_radius_2 = (mean_radius − μ_train) / s_train')
    expect(lines.find(line => line.id === activation.id)?.code).toBe('mean_radius_3 = sigmoid(mean_radius_2)')
  })

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
    expect(neuron.code).toMatch(/= linear_neuron_1\(/)
    expect(neuron.children.map(line => line.label)).toEqual(expect.arrayContaining(['layer-0/weight-0-0', 'layer-0/bias-0', 'Weighted sum']))
    expect(lines.map(line => line.label)).toEqual(expect.arrayContaining(['Dataset', 'Squared error (full square)']))
    expect(lines.indexOf(neuron)).toBeLessThan(lines.findIndex(line => line.label === 'Squared error (full square)'))
  })

  it('uses the group name as the function and a real boundary output as the assignment', () => {
    const graph = {
      nodes: [
        { id: 'input', type: 'input' as const, label: 'x', position: { x: 0, y: 0 }, params: { value: { shape: [], data: [1] } } },
        { id: 'product', type: 'multiply' as const, label: 'product', position: { x: 1, y: 0 }, params: {} },
        { id: 'result', type: 'activation' as const, label: 'y', position: { x: 2, y: 0 }, params: { activation: 'relu' as const } },
        { id: 'outside', type: 'target' as const, label: 'outside', position: { x: 3, y: 0 }, params: { value: { shape: [], data: [0] } } },
      ],
      edges: [
        { id: 'a', source: 'input', target: 'product', inputSlot: 0 },
        { id: 'b', source: 'product', target: 'result', inputSlot: 0 },
        { id: 'c', source: 'result', target: 'outside', inputSlot: 0 },
      ],
      groups: [{ id: 'custom', label: 'func', kind: 'module', nodeIds: ['product', 'result'], position: { x: 1, y: 0 }, dimensions: { width: 220, height: 150 } }],
      learningRate: 0.1,
    }
    expect(codeForGroup(graph, graph.groups[0])).toBe('y = func(x)')
    expect(buildCodeOutline(graph).find(line => line.id === 'custom')?.code).toBe('y = func(x)')
  })

  it('keeps repeated group calls distinguishable by their real outputs', () => {
    const graph = createModelPreset('decoder')
    const normalizations = graph.groups!.filter(group => group.label === 'Layer normalization')
    const calls = normalizations.map(group => codeForGroup(graph, group))
    expect(new Set(calls).size).toBe(normalizations.length)
  })
})
