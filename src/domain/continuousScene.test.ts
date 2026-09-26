import { describe, expect, it } from 'vitest'
import { cardReveal, compactVisualHierarchy, continuousSceneMaxZoom, layoutContinuousScene, routeContinuousScene, sceneContentBounds, sceneGroupId } from './continuousScene'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'
import { segmentCrossesRect } from './wireRouting'
import { projectDenseNeurons } from './neuronProjection'
import { parseCustomCsv } from './customCsv'
import { customCsvCardHeight, customCsvOutputTop } from './datasets'
import { builderCardHeight, builderInputPortY } from './builderGeometry'
import { createNode } from './examples'
import { mergeNodesIntoVisualGroup, visualGroupInterface } from './grouping'
import type { GraphModel } from './types'

const close = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y) < 1e-7

describe('one continuous nested scene', () => {
  it.each(['linear', 'neuron'] as const)('opens %s directly from one neuron card onto its calculations', id => {
    const model = createModelPreset(id)
    const original = structuredClone(model)
    const graph = compactVisualHierarchy(model)
    expect(graph.groups).toHaveLength(1)
    const neuron = graph.groups![0]
    expect(neuron.kind).toBe('neuron')
    expect(neuron.parentId).toBeUndefined()
    const scene = layoutContinuousScene(graph)
    const contents = scene.levels.find(level => level.parentId === neuron.id)!
    expect(new Set(contents.ids)).toEqual(new Set(neuron.nodeIds))
    expect(contents.ids.every(nodeId => scene.nodes.has(nodeId))).toBe(true)
    expect(model).toEqual(original)
    const zoom = continuousSceneMaxZoom(scene)
    expect(zoom).toBeLessThan(30)
    expect(cardReveal(scene.groups.get(neuron.id)!, zoom, 900, 600)).toBe(1)
    for (const nodeId of neuron.nodeIds) {
      expect(scene.scales.get(nodeId)! * zoom).toBeCloseTo(2.5)
    }
  })

  it('redirects focus and carries movement through skipped wrappers', () => {
    const model = createModelPreset('linear')
    model.view = { ...model.view!, focusedGroupId: 'network', layoutOffsets: { [sceneGroupId('network')]: { x: 25, y: 30 } } }
    const graph = compactVisualHierarchy(model)
    expect(graph.view?.focusedGroupId).toBe('layer-0/neuron-0')
    expect(graph.view?.expandedGroupIds).toEqual(['layer-0/neuron-0'])
    expect(graph.view?.layoutOffsets?.[sceneGroupId('layer-0/neuron-0')]).toEqual({ x: 25, y: 30 })
  })

  it.each(['linear', 'neuron'] as const)('keeps the %s prediction and loss together in a compact overview', id => {
    const graph = compactVisualHierarchy(createModelPreset(id))
    const scene = layoutContinuousScene(graph)
    const input = scene.nodes.get('input-0')!, target = scene.nodes.get('target')!, loss = scene.nodes.get('loss')!
    const neuron = scene.groups.get(graph.groups![0].id)!
    const gap = target.y - input.y - input.height
    expect(gap).toBeGreaterThanOrEqual(24)
    expect(gap).toBeLessThanOrEqual(40)
    expect(target.y).toBeLessThan(neuron.y + neuron.height + 40)
    expect(target.y + target.height / 2).toBeGreaterThan(neuron.y + neuron.height)
    expect(loss.y + loss.height * 2 / 3).toBeCloseTo(target.y + target.height / 2)
    const overview = scene.levels.find(level => !level.parentId)!.ids.map(id => scene.nodes.get(id) ?? scene.groups.get(id.slice(13))!)
    const height = Math.max(...overview.map(rect => rect.y + rect.height)) - Math.min(...overview.map(rect => rect.y))
    expect(height).toBeLessThan(500)
    const targetWire = routeContinuousScene(graph, scene).find(wire => graph.edges.find(edge => edge.id === wire.edgeId)?.source === 'target')!
    expect(targetWire.route).toHaveLength(2)
  })

  it('brings misplaced weights, bias, and products back inside their neuron without altering the computation', () => {
    const graph = compactVisualHierarchy(createModelPreset('linear'))
    const neuron = graph.groups![0]
    const before = layoutContinuousScene(graph)
    graph.view = { ...graph.view!, layoutOffsets: {
      'layer-0/weight-0-0': { x: -300, y: 100 },
      'layer-0/bias-0': { x: 500, y: 100 },
      'layer-0/neuron-0/product-0': { x: 0, y: -200 },
      'input-0': { x: -20, y: 5 },
    } }
    const original = structuredClone(graph)
    const scene = layoutContinuousScene(graph)
    expect(scene.repairedOffsetIds).toEqual(new Set(['layer-0/weight-0-0', 'layer-0/bias-0', 'layer-0/neuron-0/product-0']))
    const bounds = sceneContentBounds(scene, neuron.id)
    for (const id of neuron.nodeIds) {
      const rect = scene.nodes.get(id)!
      expect(rect).toEqual(before.nodes.get(id))
      expect(rect.x).toBeGreaterThanOrEqual(bounds.x)
      expect(rect.y).toBeGreaterThanOrEqual(bounds.y)
      expect(rect.x + rect.width).toBeLessThanOrEqual(bounds.x + bounds.width)
      expect(rect.y + rect.height).toBeLessThanOrEqual(bounds.y + bounds.height)
    }
    expect(scene.nodes.get('input-0')!.x).toBe(before.nodes.get('input-0')!.x - 20)
    expect(graph).toEqual(original)
    const internal = routeContinuousScene(graph, scene).filter(wire => {
      const edge = graph.edges.find(edge => edge.id === wire.edgeId)!
      return neuron.nodeIds.includes(edge.source) && neuron.nodeIds.includes(edge.target)
    })
    expect(internal).toHaveLength(4)
    for (const wire of internal) {
      const a = wire.route[0], b = wire.route.at(-1)!
      const length = wire.route.reduce((sum, point, i) => i ? sum + Math.abs(point.x - wire.route[i - 1].x) + Math.abs(point.y - wire.route[i - 1].y) : 0, 0)
      expect(length).toBeLessThanOrEqual((Math.abs(a.x - b.x) + Math.abs(a.y - b.y)) * 1.01)
    }
  })

  it('fades smoothly in either direction and clears the cover before a card fills the viewport', () => {
    const rect = { x: 0, y: 0, width: 400, height: 200 }
    expect(cardReveal(rect, .5, 800, 600)).toBe(0)
    expect(cardReveal(rect, 1.2, 800, 600)).toBeGreaterThan(0)
    expect(cardReveal(rect, 1.2, 800, 600)).toBeLessThan(1)
    expect(cardReveal(rect, 1.8, 800, 600)).toBe(1)
    for (let zoom = .5; zoom < 2; zoom += .01) {
      const delta = cardReveal(rect, zoom + .01, 800, 600) - cardReveal(rect, zoom, 800, 600)
      expect(delta).toBeGreaterThanOrEqual(0)
      expect(delta).toBeLessThan(.02)
    }
  })

  it.each(LESSONS)('keeps $id geometry fixed at every zoom and connects signals through every card boundary', ({ id }) => {
    const graph = compactVisualHierarchy(createModelPreset(id))
    const scene = layoutContinuousScene(graph)
    const changedView = layoutContinuousScene({ ...graph, view: { expandedGroupIds: graph.groups!.map(group => group.id), viewport: { x: -100, y: 240, zoom: 250 } } })
    expect(changedView.nodes).toEqual(scene.nodes)
    expect(changedView.groups).toEqual(scene.groups)
    expect(scene.nodes.size).toBe(graph.nodes.length)
    for (const level of scene.levels) {
      const parent = level.parentId && scene.groups.get(level.parentId)
      if (!parent) continue
      for (const child of level.ids) {
        const rect = scene.nodes.get(child) ?? scene.groups.get(child.slice(13))!
        expect(rect.x).toBeGreaterThan(parent.x)
        expect(rect.y).toBeGreaterThan(parent.y)
        expect(rect.x + rect.width).toBeLessThan(parent.x + parent.width)
        expect(rect.y + rect.height).toBeLessThan(parent.y + parent.height)
      }
    }
    const wires = routeContinuousScene(graph, scene)
    expect(scene.repairedOffsetIds.size).toBe(0)
    for (const edge of graph.edges) {
      const segments = wires.filter(wire => wire.edgeId === edge.id)
      expect(segments.length).toBeGreaterThan(0)
      const endpoints = segments.flatMap(wire => [wire.route[0], wire.route.at(-1)!])
      // Each card boundary is shared by the inner and outer portion. Only the
      // true source and target are loose ends, regardless of hierarchy depth.
      expect(endpoints.filter(point => endpoints.filter(other => close(point, other)).length === 1)).toHaveLength(2)
    }
    for (const wire of wires) {
      const level = scene.levels.find(item => item.parentId === wire.parentId)!
      const frame = wire.parentId && scene.groups.get(wire.parentId)
      if (frame) for (const point of wire.route) {
        expect(point.x).toBeGreaterThanOrEqual(frame.x - 1e-7)
        expect(point.y).toBeGreaterThanOrEqual(frame.y - 1e-7)
        expect(point.x).toBeLessThanOrEqual(frame.x + frame.width + 1e-7)
        expect(point.y).toBeLessThanOrEqual(frame.y + frame.height + 1e-7)
      }
      for (const child of level.ids) {
        const rect = scene.nodes.get(child) ?? scene.groups.get(child.slice(13))!
        // Normalize to local units so tiny nested rectangles use the same
        // clearance tolerance as a full-size operation.
        const local = { id: child, x: rect.x / level.scale, y: rect.y / level.scale, width: rect.width / level.scale, height: rect.height / level.scale }
        for (let i = 1; i < wire.route.length; i++) {
          const a = wire.route[i - 1], b = wire.route[i]
          expect(segmentCrossesRect({ x: a.x / level.scale, y: a.y / level.scale }, { x: b.x / level.scale, y: b.y / level.scale }, local), `${id}: ${wire.id} crosses ${child}`).toBe(false)
        }
      }
    }
  })

  it.each(LESSONS)('keeps separate signal runs distinguishable at every level of $id', ({ id }) => {
    const graph = compactVisualHierarchy(createModelPreset(id))
    const wires = routeContinuousScene(graph, layoutContinuousScene(graph)).filter(wire => !wire.hidden)
    const collisions: string[] = []
    for (let i = 0; i < wires.length; i++) for (let j = 0; j < i; j++) {
      const a = wires[i], b = wires[j]
      if (a.parentId !== b.parentId) continue
      const first = a.route.map(point => ({ x: point.x / a.scale, y: point.y / a.scale }))
      const second = b.route.map(point => ({ x: point.x / b.scale, y: point.y / b.scale }))
      for (let x = 1; x < first.length; x++) for (let y = 1; y < second.length; y++) {
        // A real fan-out can share its short departure from the same port.
        if (x === 1 && y === 1 && close(first[0], second[0])) continue
        const [p, q, r, s] = [first[x - 1], first[x], second[y - 1], second[y]]
        const horizontal = Math.abs(p.y - q.y) < .01
        if (horizontal !== (Math.abs(r.y - s.y) < .01)) continue
        const [along, across] = horizontal ? ['x', 'y'] as const : ['y', 'x'] as const
        const overlap = Math.min(Math.max(p[along], q[along]), Math.max(r[along], s[along])) - Math.max(Math.min(p[along], q[along]), Math.min(r[along], s[along]))
        if (overlap > 30 && Math.abs(p[across] - r[across]) < 8) collisions.push(`${a.id} / ${b.id}`)
      }
    }
    expect(collisions).toEqual([])
  })

  it('routes distinct custom CSV columns into the correct ports of user-created groups', () => {
    const dataset = createNode('dataset', 1)
    dataset.params = { dataset: 'custom-csv', customCsv: parseCustomCsv('a,b,c,d,target\n1,2,3,4,0\n2,3,4,5,1\n', 'columns.csv') }
    const first = [createNode('input', 1), createNode('weight', 1), createNode('arithmetic', 1)]
    const second = [createNode('input', 2), createNode('weight', 2), createNode('arithmetic', 2)]
    const graph: GraphModel = {
      nodes: [dataset, ...first, ...second],
      edges: [
        { id: 'column-b', source: dataset.id, sourceSlot: 1, target: first[0].id, inputSlot: 0 },
        { id: 'column-d', source: dataset.id, sourceSlot: 3, target: second[0].id, inputSlot: 0 },
        { id: 'first-value', source: first[0].id, target: first[2].id, inputSlot: 0 },
        { id: 'first-weight', source: first[1].id, target: first[2].id, inputSlot: 1 },
        { id: 'second-value', source: second[0].id, target: second[2].id, inputSlot: 0 },
        { id: 'second-weight', source: second[1].id, target: second[2].id, inputSlot: 1 },
      ],
      groups: [first, second].map((members, index) => ({ id: `group-${index + 1}`, label: `Group ${index + 1}`, kind: 'module', nodeIds: members.map(node => node.id), position: { x: 0, y: 0 }, dimensions: { width: 176, height: 112 } })),
      learningRate: .1,
    }
    const scene = layoutContinuousScene(graph)
    const wires = routeContinuousScene(graph, scene)
    const datasetRect = scene.nodes.get(dataset.id)!
    for (const [edgeId, slot, groupId] of [['column-b', 1, 'group-1'], ['column-d', 3, 'group-2']] as const) {
      const outside = wires.find(wire => wire.edgeId === edgeId && !wire.parentId)!
      const inside = wires.find(wire => wire.edgeId === edgeId && wire.parentId === groupId)!
      expect(outside.route[0].x).toBeCloseTo(datasetRect.x + datasetRect.width)
      expect(outside.route[0].y).toBeCloseTo(datasetRect.y + datasetRect.height * customCsvOutputTop(slot, true) / customCsvCardHeight(dataset, true))
      expect(close(outside.route.at(-1)!, inside.route[0])).toBe(true)
    }
  })

  it('moves a group and all of its nested contents by the same world-space displacement', () => {
    const graph = createModelPreset('linear')
    const group = graph.groups!.find(group => !group.parentId)!
    const before = layoutContinuousScene(graph)
    const after = layoutContinuousScene({ ...graph, view: { ...graph.view!, layoutOffsets: { [sceneGroupId(group.id)]: { x: 28, y: -13 } } } })
    for (const id of group.nodeIds) {
      expect(after.nodes.get(id)!.x).toBeCloseTo(before.nodes.get(id)!.x + 28)
      expect(after.nodes.get(id)!.y).toBeCloseTo(before.nodes.get(id)!.y - 13)
    }
  })

  it('routes three wires to the visible ports of a manually placed Concat card', () => {
    const inputs = [1, 2, 3].map(index => createNode('input', index))
    const concat = createNode('concat', 1)
    concat.params = { axis: 1, inputCount: 3 }
    const graph: GraphModel = {
      nodes: [...inputs, concat],
      edges: inputs.map((node, index) => ({ id: `input-${index}`, source: node.id, target: concat.id, inputSlot: index })),
      learningRate: .1,
      view: { expandedGroupIds: [], manualNodePlacements: { [concat.id]: { offset: { x: 600, y: 80 } } } },
    }
    const scene = layoutContinuousScene(graph)
    const rect = scene.nodes.get(concat.id)!
    expect(rect.height).toBe(builderCardHeight(concat))

    const wires = routeContinuousScene(graph, scene)
    for (let index = 0; index < inputs.length; index++) {
      const endpoint = wires.find(wire => wire.edgeId === `input-${index}`)!.route.at(-1)!
      expect(endpoint.x).toBeCloseTo(rect.x)
      expect(endpoint.y).toBeCloseTo(rect.y + builderInputPortY(concat, index))
    }
  })

  it('places a selected Concat inside its new group despite an older root placement', () => {
    const features = [0, 1, 2].map(index => ({ ...createNode('input', index + 1), id: `feature-${index}` }))
    const branches = [0, 1, 2].map(index => ({ ...createNode('activation', index + 1), id: `branch-${index}` }))
    const concat = { ...createNode('concat', 1), id: 'concat' }
    concat.params = { axis: 1, inputCount: 3 }
    const graph: GraphModel = {
      nodes: [...features, ...branches, concat],
      edges: [
        ...features.map((node, index) => ({ id: `feature-${index}-branch`, source: node.id, target: branches[index].id, inputSlot: 0 })),
        ...branches.map((node, index) => ({ id: `branch-${index}-concat`, source: node.id, target: concat.id, inputSlot: index })),
      ],
      learningRate: .1,
      view: { expandedGroupIds: [], manualNodePlacements: { [concat.id]: { offset: { x: 600, y: 80 } } } },
    }
    const merged = mergeNodesIntoVisualGroup(graph, [...branches.map(node => node.id), concat.id])
    const group = merged.group!
    const scene = layoutContinuousScene(merged.graph)

    expect(group.nodeIds).toContain(concat.id)
    expect(scene.parents.get(concat.id)).toBe(group.id)
    expect(scene.levels.find(level => level.parentId === group.id)?.ids).toContain(concat.id)
    expect(scene.levels.find(level => !level.parentId)?.ids).not.toContain(concat.id)
    const frame = scene.groups.get(group.id)!, card = scene.nodes.get(concat.id)!
    expect(card.x).toBeGreaterThan(frame.x)
    expect(card.x + card.width).toBeLessThan(frame.x + frame.width)
    expect(visualGroupInterface(merged.graph, group).outputs.map(port => port.source)).toEqual([concat.id])
  })

  it('uses the scalar card dimensions and staggered input/weight rows inside a projected neuron', () => {
    const graph = projectDenseNeurons(createModelPreset('decoder'), { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 }).graph
    const scene = layoutContinuousScene(graph)
    const prefix = 'inspect:blocks.0.ff1.layer:0:'
    const input = scene.nodes.get(`${prefix}x0`)!, weight = scene.nodes.get(`${prefix}w0`)!
    const scale = scene.scales.get(`${prefix}w0`)!
    expect(weight.width / scale).toBeCloseTo(176)
    expect(weight.height / scale).toBeCloseTo(205)
    expect((weight.x - input.x) / scale).toBeCloseTo(200)
    expect((weight.y - input.y) / scale).toBeCloseTo(110)
  })
})
