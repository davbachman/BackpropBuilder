import { describe, expect, it } from 'vitest'
import { createModelPreset } from './modelPresets'
import { layoutSemanticGraph, type SemanticRect } from './semanticLayout'
import { setVisualGroupExpanded, visibleGroups, collapsedGroupForNode, groupAncestors } from './grouping'
import { projectDenseNeurons } from './neuronProjection'
import { LESSONS } from '../learning/presets'

const overlap = (a: SemanticRect, b: SemanticRect) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y

describe('one continuous architecture layout', () => {
  it.each(LESSONS)('keeps every level of $id free of overlapping blocks', ({ id }) => {
    let graph = createModelPreset(id)
    for (const group of graph.groups ?? []) {
      graph = setVisualGroupExpanded(graph, group.id, true)
      const layout = layoutSemanticGraph(graph)
      for (const parent of [undefined, ...(graph.groups ?? [])]) {
        if (parent && !graph.view?.expandedGroupIds.includes(parent.id)) continue
        const childGroups = visibleGroups(graph).filter(candidate => candidate.parentId === parent?.id)
        const covered = new Set(childGroups.flatMap(child => child.nodeIds))
        const nodes = graph.nodes.filter(node => (!parent || parent.nodeIds.includes(node.id)) && !collapsedGroupForNode(graph, node.id) && !covered.has(node.id))
        const rects = [...childGroups.map(child => layout.groups.get(child.id)), ...nodes.map(node => layout.nodes.get(node.id))].filter((rect): rect is SemanticRect => Boolean(rect))
        for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) expect(overlap(rects[i], rects[j])).toBe(false)
        if (parent) {
          const bounds = layout.groups.get(parent.id)
          if (!bounds) continue
          for (const rect of rects) {
            expect(rect.x).toBeGreaterThanOrEqual(bounds.x)
            expect(rect.y).toBeGreaterThanOrEqual(bounds.y + 50)
            expect(rect.x + rect.width).toBeLessThanOrEqual(bounds.x + bounds.width)
            expect(rect.y + rect.height).toBeLessThanOrEqual(bounds.y + bounds.height)
          }
        }
      }
    }
  })

  it('retains each original node, parameter and residual wire while changing view geometry', () => {
    const graph = setVisualGroupExpanded(createModelPreset('decoder'), 'blocks.0', true)
    const snapshot = JSON.stringify(graph)
    const collapsed = layoutSemanticGraph(graph)
    const group = graph.groups!.find(candidate => candidate.kind === 'attention')!
    const expanded = layoutSemanticGraph(setVisualGroupExpanded(graph, group.id, true))
    expect(expanded.groups.get(group.id)!.width).toBeGreaterThan(collapsed.groups.get(group.id)!.width)
    expect(JSON.stringify(graph)).toBe(snapshot)
    expect(graph.edges.some(edge => edge.target.includes('residual'))).toBe(true)
  })

  it('gives each projected input and weight its own clear path into their product', () => {
    const layerId = 'blocks.0.ff1.layer'
    const source = setVisualGroupExpanded(createModelPreset('decoder'), layerId, true)
    const projected = projectDenseNeurons(source, { groupId: layerId, unitIndex: 0, row: 0 }).graph
    const layout = layoutSemanticGraph(projected)
    const prefix = `inspect:${layerId}:0`
    for (let input = 0; input < 8; input++) {
      const x = layout.nodes.get(`${prefix}:x${input}`)!
      const w = layout.nodes.get(`${prefix}:w${input}`)!
      const product = layout.nodes.get(`${prefix}:product${input}`)!
      expect(x.x + x.width).toBeLessThan(w.x)
      expect(x.y + x.height / 2).toBeLessThan(w.y)
      expect(product.y).toBe(x.y)
      expect(x.x + x.width).toBeLessThan(product.x)
      expect(x.height).toBe(64)
    }
    expect(layout.groups.get(prefix)!.height).toBeLessThan(1100)
  })

  it('arranges the linear neuron like a readable calculation with a separate target lane', () => {
    const graph = createModelPreset('linear'), layout = layoutSemanticGraph(graph)
    const multiply = graph.nodes.find(node => node.type === 'multiply')!
    const product = layout.nodes.get(multiply.id)!
    const bias = layout.nodes.get('layer-0/bias-0')!, weight = layout.nodes.get('layer-0/weight-0-0')!
    const input = layout.nodes.get('input-0')!, target = layout.nodes.get('target')!, network = layout.groups.get('network')!
    expect(bias.x).toBe(product.x)
    expect(bias.y).toBeGreaterThan(product.y + product.height)
    expect(weight.x).toBeLessThan(product.x)
    expect(input.y + input.height / 2).toBeCloseTo(product.y + product.height / 3)
    expect(weight.y).toBeGreaterThan(input.y + input.height / 2)
    expect(target.y + target.height / 2).toBeGreaterThan(network.y + network.height)
    expect(target.y).toBeLessThan(network.y + network.height)
  })

  it('moves a block and every visible descendant together without moving other blocks', () => {
    const graph = setVisualGroupExpanded(createModelPreset('decoder'), 'blocks.0.ff1.layer', true)
    const baseline = layoutSemanticGraph(graph)
    const moved = { ...graph, view: { ...graph.view!, layoutOffsets: { 'visual-group:blocks.0': { x: 150, y: -80 } } } }
    const layout = layoutSemanticGraph(moved)
    const block = graph.groups!.find(group => group.id === 'blocks.0')!

    for (const [id, before] of baseline.groups) {
      const descendant = groupAncestors(graph, id).some(group => group.id === block.id)
      expect(layout.groups.get(id)).toEqual({ ...before, x: before.x + (descendant ? 150 : 0), y: before.y + (descendant ? -80 : 0) })
    }
    for (const [id, before] of baseline.nodes) {
      const descendant = block.nodeIds.includes(id)
      const after = layout.nodes.get(id)!
      expect(after.width).toBe(before.width)
      expect(after.height).toBe(before.height)
      expect(after.x).toBeCloseTo(before.x + (descendant ? 150 : 0))
      expect(after.y).toBeCloseTo(before.y + (descendant ? -80 : 0))
    }
    expect(moved.nodes).toBe(graph.nodes)
    expect(moved.edges).toBe(graph.edges)
  })

  it('retains independent nested offsets across collapsing and reopening a block', () => {
    const layerId = 'blocks.0.ff1.layer'
    const graph = setVisualGroupExpanded(createModelPreset('decoder'), layerId, true)
    const nodeId = graph.groups!.find(group => group.id === layerId)!.nodeIds[0]
    const baseline = layoutSemanticGraph(graph)
    const moved = { ...graph, view: { ...graph.view!, layoutOffsets: {
      'visual-group:blocks.0': { x: 100, y: 40 },
      [`visual-group:${layerId}`]: { x: -30, y: 60 },
      [nodeId]: { x: 12, y: -16 },
    } } }
    const before = baseline.nodes.get(nodeId)!
    const original = layoutSemanticGraph(moved)
    expect(original.nodes.get(nodeId)).toEqual({ ...before, x: before.x + 82, y: before.y + 84 })

    const collapsed = setVisualGroupExpanded(moved, 'blocks.0', false)
    expect(layoutSemanticGraph(collapsed).nodes.has(nodeId)).toBe(false)
    const collapsedBaseline = layoutSemanticGraph(setVisualGroupExpanded(graph, 'blocks.0', false)).groups.get('blocks.0')!
    expect(layoutSemanticGraph(collapsed).groups.get('blocks.0')).toEqual({ ...collapsedBaseline, x: collapsedBaseline.x + 100, y: collapsedBaseline.y + 40 })
    const reopened = layoutSemanticGraph(setVisualGroupExpanded(collapsed, 'blocks.0', true))
    expect(reopened.nodes).toEqual(original.nodes)
    expect(reopened.groups).toEqual(original.groups)
  })

  it('keeps projected neuron and scalar layout adjustments outside the executable graph', () => {
    const layerId = 'blocks.0.ff1.layer'
    const prefix = `inspect:${layerId}:0`
    const source = setVisualGroupExpanded(createModelPreset('decoder'), layerId, true)
    const focus = { groupId: layerId, unitIndex: 0, row: 0 }
    const baseline = layoutSemanticGraph(projectDenseNeurons(source, focus).graph)
    const moved = { ...source, view: { ...source.view!, layoutOffsets: {
      [`visual-group:${prefix}`]: { x: 80, y: 40 },
      [`${prefix}:w0`]: { x: -20, y: 15 },
    } } }
    const projected = projectDenseNeurons(moved, focus).graph
    const layout = layoutSemanticGraph(projected)
    const before = baseline.nodes.get(`${prefix}:w0`)!
    expect(layout.nodes.get(`${prefix}:w0`)).toEqual({ ...before, x: before.x + 60, y: before.y + 55 })
    expect(moved.nodes.some(node => node.id.startsWith('inspect:'))).toBe(false)
    expect(moved.groups!.some(group => group.id.startsWith('inspect:'))).toBe(false)
    expect(moved.nodes).toBe(source.nodes)
  })
})
