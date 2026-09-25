import { act, fireEvent, render } from '@testing-library/react'
import type { Node, ReactFlowProps } from '@xyflow/react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { GraphCanvas } from './GraphCanvas'
import { createModelPreset } from '../domain/modelPresets'
import { setVisualGroupExpanded } from '../domain/grouping'
import { projectDenseNeurons } from '../domain/neuronProjection'
import { compactVisualHierarchy, continuousSceneMaxZoom, layoutContinuousScene } from '../domain/continuousScene'
import type { GraphModel } from '../domain/types'
import { LESSONS } from '../learning/presets'
import { segmentCrossesRect } from '../domain/wireRouting'
import { createNode } from '../domain/examples'
import { placeCanvasNode } from '../domain/nodePlacement'

// Keep React Flow's real state hooks. Capture its boundary so these tests can
// exercise our drag lifecycle without relying on jsdom's missing geometry.
const flow = vi.hoisted(() => ({
  props: undefined as ReactFlowProps | undefined,
  fitView: vi.fn(),
  setViewport: vi.fn(),
  getViewport: vi.fn(() => ({ x: 30, y: 50, zoom: 0.8 })),
  screenToFlowPosition: vi.fn((position: { x: number; y: number }) => position),
}))

vi.mock('@xyflow/react', async (importOriginal) => ({
  ...await importOriginal<typeof import('@xyflow/react')>(),
  ReactFlow: (props: ReactFlowProps) => { flow.props = props; return null },
  useReactFlow: () => ({
    fitView: flow.fitView,
    setViewport: flow.setViewport,
    getViewport: flow.getViewport,
    screenToFlowPosition: flow.screenToFlowPosition,
  }),
}))

function mountCanvas(graph: GraphModel, displayGraph?: GraphModel, problemNodeIds?: ReadonlySet<string>) {
  const onGraphChange = vi.fn()
  const onViewChange = vi.fn()
  const onSelectionChange = vi.fn()
  const onCreateNode = vi.fn()
  const canvas = (current: GraphModel) => <GraphCanvas
    graph={current} displayGraph={displayGraph} problemNodeIds={problemNodeIds} showMath showGradient={false} phase="edit"
    onGraphChange={onGraphChange} onViewChange={onViewChange} onSelectionChange={onSelectionChange}
    onCreateNode={onCreateNode} onCancelPendingPlacement={vi.fn()} onNodeValueChange={vi.fn()}
    onActivationChange={vi.fn()} onGroupCreate={vi.fn()} onGroupExplode={vi.fn()} onGroupMove={vi.fn()}
  />
  const result = render(canvas(graph))
  return { ...result, rerenderGraph: (current: GraphModel) => result.rerender(canvas(current)), onGraphChange, onViewChange, onSelectionChange, onCreateNode }
}

function nodeById(id: string): Node {
  const node = flow.props!.nodes!.find(candidate => candidate.id === id)
  if (!node) throw new Error(`Missing canvas node ${id}`)
  return node
}

function displaced(node: Node, x: number, y: number): Node {
  return { ...node, position: { x: node.position.x + x, y: node.position.y + y } }
}

// XYFlow passes the drag library's native sourceEvent at runtime despite the
// public callback type using React's event type. Our handlers ignore this arg.
const dragEvent = (type: string) => new MouseEvent(type) as unknown as ReactMouseEvent

function dragStart(nodes: Node[]) {
  act(() => flow.props!.onNodeDragStart!(dragEvent('mousedown'), nodes[0], nodes))
}

function dragMove(nodes: Node[]) {
  act(() => {
    flow.props!.onNodesChange!(nodes.map(node => ({ type: 'position', id: node.id, position: node.position, dragging: true })))
    flow.props!.onNodeDrag!(dragEvent('mousemove'), nodes[0], nodes)
  })
}

function dragStop(nodes: Node[]) {
  act(() => flow.props!.onNodeDragStop!(dragEvent('mouseup'), nodes[0], nodes))
}

describe('canvas movement gestures', () => {
  beforeEach(() => vi.clearAllMocks())

  it('opens the block picker on a blank-canvas double-click and places the chosen block there', () => {
    flow.screenToFlowPosition.mockImplementationOnce(() => ({ x: 417, y: -83 }))
    const { container, getByRole, queryByRole, onCreateNode, onSelectionChange } = mountCanvas(createModelPreset('blank'))
    const pane = document.createElement('div')
    pane.className = 'react-flow__pane'
    container.querySelector('.flow-shell')!.append(pane)
    act(() => flow.props!.onPaneClick!(new MouseEvent('click', { clientX: 523, clientY: 186 }) as unknown as ReactMouseEvent))
    expect(queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
    expect(onSelectionChange).toHaveBeenCalledWith({ nodeIds: [] })
    fireEvent.doubleClick(pane, { clientX: 523, clientY: 186 })
    expect(getByRole('dialog', { name: 'Add a block' })).toBeInTheDocument()
    expect(getByRole('searchbox', { name: 'Search blocks' })).toHaveFocus()
    expect(flow.props!.zoomOnDoubleClick).toBe(false)
    fireEvent.change(getByRole('searchbox', { name: 'Search blocks' }), { target: { value: 'convol' } })
    expect(getByRole('option', { name: /Convolution/ })).toBeInTheDocument()
    fireEvent.keyDown(getByRole('searchbox', { name: 'Search blocks' }), { key: 'Enter' })
    expect(onCreateNode).toHaveBeenCalledExactlyOnceWith('conv2d', { x: 417, y: -83 })
    expect(queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
  })

  it('dismisses the blank-canvas block picker with Escape', () => {
    const { container, getByRole, queryByRole, onCreateNode } = mountCanvas(createModelPreset('blank'))
    const pane = document.createElement('div')
    pane.className = 'react-flow__pane'
    container.querySelector('.flow-shell')!.append(pane)
    fireEvent.doubleClick(pane, { clientX: 100, clientY: 100 })
    fireEvent.keyDown(getByRole('searchbox', { name: 'Search blocks' }), { key: 'Escape' })
    expect(queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
    expect(onCreateNode).not.toHaveBeenCalled()
  })

  it('adds a block to the visible transformer region at that region’s scale', () => {
    const graph = createModelPreset('block')
    const scene = layoutContinuousScene(compactVisualHierarchy(graph))
    const rect = scene.groups.get('blocks.0.norm1')!
    const position = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
    graph.view = { ...graph.view!, viewport: { x: 0, y: 0, zoom: continuousSceneMaxZoom(scene) } }
    flow.screenToFlowPosition.mockImplementationOnce(() => position)
    const { container, getByRole, onCreateNode } = mountCanvas(graph)
    const pane = document.createElement('div')
    pane.className = 'react-flow__pane'
    container.querySelector('.flow-shell')!.append(pane)
    fireEvent.doubleClick(pane, { clientX: 450, clientY: 300 })
    fireEvent.change(getByRole('searchbox', { name: 'Search blocks' }), { target: { value: 'arithmetic' } })
    fireEvent.keyDown(getByRole('searchbox', { name: 'Search blocks' }), { key: 'Enter' })
    expect(onCreateNode).toHaveBeenCalledExactlyOnceWith('arithmetic', position, 'blocks.0.norm1')
  })

  it('uses a close-up scale when adding to empty canvas outside a module', () => {
    const graph = createModelPreset('block')
    const scene = layoutContinuousScene(compactVisualHierarchy(graph))
    graph.view = { ...graph.view!, viewport: { x: 0, y: 0, zoom: continuousSceneMaxZoom(scene) } }
    const position = { x: -1000, y: -1000 }
    flow.screenToFlowPosition.mockImplementationOnce(() => position)
    const { container, getByRole, onCreateNode } = mountCanvas(graph)
    const pane = document.createElement('div')
    pane.className = 'react-flow__pane'
    container.querySelector('.flow-shell')!.append(pane)
    fireEvent.doubleClick(pane, { clientX: 450, clientY: 300 })
    fireEvent.change(getByRole('searchbox', { name: 'Search blocks' }), { target: { value: 'tensor transform' } })
    fireEvent.keyDown(getByRole('searchbox', { name: 'Search blocks' }), { key: 'Enter' })
    expect(onCreateNode).toHaveBeenCalledOnce()
    expect(onCreateNode.mock.calls[0][0]).toBe('tensor-transform')
    expect(onCreateNode.mock.calls[0][1]).toEqual(position)
    expect(onCreateNode.mock.calls[0][2]).toBeUndefined()
    expect(onCreateNode.mock.calls[0][3]).toBeGreaterThan(0)
    expect(onCreateNode.mock.calls[0][3]).toBeLessThan(1)
  })

  it('keeps blocks and the camera still through connection edits until Compact layout is requested', () => {
    let graph = placeCanvasNode(createModelPreset('linear'), { ...createNode('dataset', 1), position: { x: -210, y: 300 } })
    const { onGraphChange, onViewChange, rerenderGraph, getByRole } = mountCanvas(graph)
    const geometry = () => flow.props!.nodes!.map(({ id, position, width, height }) => ({ id, position, width, height }))
    const before = geometry()
    flow.fitView.mockClear()
    flow.setViewport.mockClear()
    for (const [target, sourceHandle] of [['input-0', 'out-0'], ['target', 'out-1']]) {
      act(() => flow.props!.onConnect!({ source: 'dataset-1', sourceHandle, target, targetHandle: 'in-0' }))
      graph = onGraphChange.mock.lastCall![0]
      rerenderGraph(graph)
      expect(geometry()).toEqual(before)
      expect(flow.props!.edges!.some(edge => edge.source === 'dataset-1' && edge.target === target)).toBe(true)
    }
    const wire = flow.props!.edges!.find(edge => edge.source === 'dataset-1')!
    act(() => flow.props!.onEdgesChange!([{ type: 'remove', id: wire.id }]))
    graph = onGraphChange.mock.lastCall![0]
    rerenderGraph(graph)
    expect(geometry()).toEqual(before)
    expect(flow.props!.edges!.some(edge => edge.id === wire.id)).toBe(false)
    expect(flow.fitView).not.toHaveBeenCalled()
    expect(flow.setViewport).not.toHaveBeenCalled()
    fireEvent.click(getByRole('button', { name: 'Compact layout' }))
    const view = onViewChange.mock.lastCall![0]
    expect(view.layoutEdges).toBeUndefined()
    expect(view.layoutOffsets).toBeUndefined()
    rerenderGraph({ ...graph, view })
    expect(geometry()).not.toEqual(before)
    // Undo restores both the wiring and its saved arrangement.
    rerenderGraph(graph)
    expect(geometry()).toEqual(before)
  })

  it('reveals nested contents during zoom without rebuilding geometry or refitting the camera', () => {
    const graph = createModelPreset('small-network')
    const root = graph.groups!.find(group => !group.parentId)!
    const child = graph.groups!.find(group => group.parentId === root.id)!
    const { onViewChange } = mountCanvas(graph)
    const positions = flow.props!.nodes!.map(node => [node.id, node.position])
    const initialEdges = flow.props!.edges!.map(edge => [edge.id, edge.data?.route])
    const rootNode = nodeById(`visual-group:${root.id}`)
    const occupancy = Math.max(rootNode.width! / 900, rootNode.height! / 600)
    const overview = .25 / occupancy, halfway = .6 / occupancy, closeup = .9 / occupancy
    act(() => flow.props!.onMove!(null, { x: 0, y: 0, zoom: overview }))
    flow.fitView.mockClear()
    expect(nodeById(`visual-group:${root.id}`).data.reveal).toBe(0)
    act(() => flow.props!.onMove!(null, { x: 0, y: 0, zoom: halfway }))
    const intermediate = nodeById(`visual-group:${root.id}`).data.reveal as number
    expect(intermediate).toBeGreaterThan(0)
    expect(intermediate).toBeLessThan(1)
    expect(nodeById(`visual-group:${child.id}`).selectable).toBe(false)
    act(() => flow.props!.onMove!(null, { x: -500, y: -100, zoom: closeup }))
    expect(nodeById(`visual-group:${root.id}`).data.reveal).toBe(1)
    expect(nodeById(`visual-group:${root.id}`).selectable).toBe(false)
    expect(nodeById(`visual-group:${child.id}`).selectable).toBe(true)
    act(() => flow.props!.onMoveEnd!(null, { x: -500, y: -100, zoom: closeup }))
    expect(onViewChange).toHaveBeenLastCalledWith(expect.objectContaining({ expandedGroupIds: graph.view!.expandedGroupIds, viewport: { x: -500, y: -100, zoom: closeup } }))
    act(() => flow.props!.onMove!(null, { x: 0, y: 0, zoom: overview }))
    expect(nodeById(`visual-group:${root.id}`).data.reveal).toBe(0)
    expect(flow.props!.nodes!.map(node => [node.id, node.position])).toEqual(positions)
    expect(flow.props!.edges!.map(edge => [edge.id, edge.data?.route])).toEqual(initialEdges)
    expect(flow.fitView).not.toHaveBeenCalled()
  })

  it('marks a collapsed parent card when an inner calculation has an error', () => {
    const graph = createModelPreset('linear')
    const root = compactVisualHierarchy(graph).groups!.find(group => !group.parentId)!
    const memberId = root.nodeIds[0]
    mountCanvas(graph, undefined, new Set([memberId]))
    expect(nodeById(memberId).data.validationError).toBe(true)
    expect(nodeById(`visual-group:${root.id}`).data.validationError).toBe(true)
  })

  it('moves a revealed continuous card and its actual descendants together', () => {
    const graph = createModelPreset('linear')
    const root = compactVisualHierarchy(graph).groups!.find(group => !group.parentId)!
    const { onViewChange } = mountCanvas(graph)
    act(() => flow.props!.onMove!(null, { x: 0, y: 0, zoom: 3.2 }))
    const block = nodeById(`visual-group:${root.id}`)
    const member = nodeById(root.nodeIds[0])
    expect(block.dragHandle).toBe('.visual-group-title-row')
    dragStart([block])
    const moved = displaced(block, 20, 30)
    dragMove([moved])
    expect(nodeById(member.id).position.x).toBeCloseTo(member.position.x + 20)
    expect(nodeById(member.id).position.y).toBeCloseTo(member.position.y + 30)
    dragStop([moved])
    expect(onViewChange).toHaveBeenLastCalledWith(expect.objectContaining({ layoutOffsets: { [block.id]: { x: 20, y: 30 } } }))
  })

  it('reveals the linear arithmetic without duplicate wrappers and keeps it accessible at maximum zoom', () => {
    const graph = createModelPreset('linear')
    mountCanvas(graph)
    const groups = flow.props!.nodes!.filter(node => node.type === 'groupNode')
    expect(groups.map(node => node.id)).toEqual(['visual-group:layer-0/neuron-0'])
    const zoom = flow.props!.maxZoom!
    act(() => flow.props!.onMove!(null, { x: 0, y: 0, zoom }))
    expect(nodeById(groups[0].id).data.reveal).toBe(1)
    for (const node of flow.props!.nodes!.filter(node => node.type === 'semanticNode')) {
      expect(node.data.accessible).toBe(true)
      expect(node.selectable).toBe(true)
    }
    expect(flow.props!.edges!.every(edge => edge.data?.accessible)).toBe(true)
  })

  it('opens a neuron by centering its calculation bounds without waiting for DOM measurement', () => {
    const graph = createModelPreset('linear')
    mountCanvas(graph)
    const group = nodeById('visual-group:layer-0/neuron-0')
    act(() => (group.data.onToggle as (id: string) => void)('layer-0/neuron-0'))
    const viewport = flow.setViewport.mock.lastCall![0]
    const operations = flow.props!.nodes!.filter(node => group.data.group && (group.data.group as { nodeIds: string[] }).nodeIds.includes(node.id))
    for (const operation of operations) {
      expect(operation.position.x * viewport.zoom + viewport.x).toBeGreaterThan(0)
      expect(operation.position.y * viewport.zoom + viewport.y).toBeGreaterThan(0)
      expect((operation.position.x + operation.width!) * viewport.zoom + viewport.x).toBeLessThan(900)
      expect((operation.position.y + operation.height!) * viewport.zoom + viewport.y).toBeLessThan(600)
    }
    expect(viewport.zoom).toBeGreaterThan(3)
    expect(flow.fitView).not.toHaveBeenCalled()
  })

  it('does not replay a persisted camera event over an ongoing zoom, but restores external view changes', () => {
    const graph = createModelPreset('linear')
    const { rerenderGraph } = mountCanvas(graph)
    const viewport = { x: -400, y: -200, zoom: 2 }
    act(() => flow.props!.onMoveEnd!(null, viewport))
    rerenderGraph({ ...graph, view: { ...graph.view!, viewport } })
    expect(flow.setViewport).not.toHaveBeenCalled()
    const restored = { x: 100, y: 100, zoom: .5 }
    rerenderGraph({ ...graph, view: { ...graph.view!, viewport: restored } })
    expect(flow.setViewport).toHaveBeenLastCalledWith(restored)
  })

  it('repairs a scattered neuron once and constrains its calculations to the enclosing frame', () => {
    const graph = createModelPreset('linear')
    graph.view = { ...graph.view!, layoutOffsets: {
      'layer-0/weight-0-0': { x: -300, y: 100 },
      'layer-0/bias-0': { x: 500, y: 100 },
      'input-0': { x: -20, y: 5 },
    } }
    const { onViewChange, rerenderGraph } = mountCanvas(graph)
    expect(flow.props!.autoPanOnNodeDrag).toBe(false)
    expect(onViewChange).toHaveBeenCalledTimes(1)
    const view = onViewChange.mock.lastCall![0]
    expect(view.layoutOffsets).toEqual({ 'input-0': { x: -20, y: 5 } })
    rerenderGraph({ ...graph, view })
    expect(onViewChange).toHaveBeenCalledTimes(1)
    for (const id of ['layer-0/weight-0-0', 'layer-0/bias-0', 'layer-0/neuron-0/product-0', 'layer-0/neuron-0/sum', 'layer-0/neuron-0/activation']) {
      const node = nodeById(id)
      const [[left, top], [right, bottom]] = node.extent as [[number, number], [number, number]]
      expect(node.position.x).toBeGreaterThanOrEqual(left)
      expect(node.position.y).toBeGreaterThanOrEqual(top)
      expect(node.position.x + node.width!).toBeLessThanOrEqual(right)
      expect(node.position.y + node.height!).toBeLessThanOrEqual(bottom)
    }
    expect(nodeById('input-0').extent).toBeUndefined()
  })

  it('compacts an existing layout without changing its trained weights or calculation graph', () => {
    const graph = createModelPreset('linear')
    graph.nodes.find(node => node.type === 'weight')!.params.value = 7
    graph.view = { ...graph.view!, layoutOffsets: { target: { x: 0, y: 500 }, loss: { x: 0, y: 500 } } }
    const before = structuredClone(graph)
    const { getByRole, onViewChange, onGraphChange, rerenderGraph } = mountCanvas(graph)
    fireEvent.click(getByRole('button', { name: 'Compact layout' }))
    const view = onViewChange.mock.lastCall![0]
    expect(view.layoutOffsets).toBeUndefined()
    expect(view.viewport.zoom).toBeGreaterThan(0)
    expect(onGraphChange).not.toHaveBeenCalled()
    expect(graph).toEqual(before)
    rerenderGraph({ ...graph, view })
    const input = nodeById('input-0'), target = nodeById('target')
    expect(target.position.y - input.position.y - input.height!).toBeLessThanOrEqual(40)
    expect(nodeById('layer-0/weight-0-0').data.graphNode).toMatchObject({ params: { value: 7 } })
  })

  it('keeps the camera still when a calculation is moved inside an already focused neuron', () => {
    vi.useFakeTimers()
    try {
      const graph = createModelPreset('linear')
      graph.view = { ...graph.view!, focusedGroupId: 'layer-0/neuron-0' }
      const { rerenderGraph, unmount } = mountCanvas(graph)
      act(() => vi.advanceTimersByTime(150))
      expect(flow.setViewport).toHaveBeenCalledTimes(1)
      flow.setViewport.mockClear()
      rerenderGraph({ ...graph, view: { ...graph.view!, layoutOffsets: { 'layer-0/weight-0-0': { x: 1, y: 1 } } } })
      act(() => vi.advanceTimersByTime(150))
      expect(flow.setViewport).not.toHaveBeenCalled()
      unmount()
    } finally { vi.useRealTimers() }
  })

  it('supplies precise geometry and ports even below one world-space pixel', () => {
    const graph = createModelPreset('decoder')
    mountCanvas(graph, projectDenseNeurons(graph, { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 }).graph)
    const tiny = flow.props!.nodes!.filter(node => node.height! < 1)
    expect(tiny.length).toBeGreaterThan(0)
    for (const node of tiny) {
      expect(node.measured).toEqual({ width: node.width, height: node.height })
      expect(node.handles?.length).toBeGreaterThan(0)
      expect(node.handles!.every(handle => Number.isFinite(handle.x) && Number.isFinite(handle.y))).toBe(true)
    }
  })

  it('keeps placement coordinates in the blank free-form builder', () => {
    const graph = createModelPreset('blank')
    graph.nodes.push({ id: 'placed', type: 'input', label: 'Placed input', params: { value: 1 }, position: { x: 315, y: 207 } })
    mountCanvas(graph)
    expect(nodeById('placed').position).toEqual({ x: 315, y: 207 })
    expect(nodeById('placed').type).toBe('builderNode')
  })

  it.each(LESSONS)('routes the wires of $id around other visible blocks at every hierarchy level', ({ id }) => {
    const base = createModelPreset(id)
    base.view = { ...base.view!, expandedGroupIds: [], semanticZoom: false }
    const views = [base, ...(base.groups ?? []).map(group => setVisualGroupExpanded(base, group.id, true)), { ...base, view: { ...base.view!, expandedGroupIds: base.groups!.map(group => group.id) } }]
    for (const graph of views) {
      const { unmount } = mountCanvas(graph)
      const obstacles = flow.props!.nodes!.filter(node => !node.data.expanded).map(node => ({ id: node.id, ...node.position, width: node.width!, height: node.height! }))
      for (const edge of flow.props!.edges!) {
        const route = edge.data?.route as { x: number; y: number }[]
        expect(route?.length).toBeGreaterThan(1)
        for (let i = 1; i < route.length; i++) for (const obstacle of obstacles) {
          expect(segmentCrossesRect(route[i - 1], route[i], obstacle), `${id} ${graph.view?.focusedGroupId ?? 'overview'}: ${edge.id} crosses ${obstacle.id}`).toBe(false)
        }
      }
      unmount()
    }
  })

  it('moves an expanded block and its contents live and commits only one relative offset', () => {
    const graph = setVisualGroupExpanded(createModelPreset('decoder'), 'blocks.0.ff1.layer', true)
    graph.view!.semanticZoom = false
    graph.view!.layoutOffsets = { 'visual-group:blocks.0': { x: 20, y: -10 } }
    const snapshot = JSON.stringify(graph)
    const { onViewChange, onGraphChange, onSelectionChange } = mountCanvas(graph)
    const block = nodeById('visual-group:blocks.0')
    const child = nodeById('visual-group:blocks.0.ff1.layer')
    const otherBlock = nodeById('visual-group:blocks.1')
    const scalarId = graph.groups!.find(group => group.id === 'blocks.0.ff1.layer')!.nodeIds[0]
    const scalar = nodeById(scalarId)
    expect(flow.props!.nodes!.every(node => node.draggable)).toBe(true)
    expect(block.dragHandle).toBe('.visual-group-title-row')
    // The large expanded frame must not join a small marquee drawn around
    // its contents, but dragging its header still selects the group.
    expect(block.selectable).toBe(false)
    expect(otherBlock.selectable).toBe(true)
    expect(flow.props!.selectionOnDrag).toBe(true)
    expect(flow.props!.panOnDrag).toEqual([1, 2])

    dragStart([block, child])
    expect(onSelectionChange).toHaveBeenLastCalledWith({ nodeIds: [], groupId: 'blocks.0' })
    const moved = [displaced(block, 80, -25), displaced(child, 80, -25)]
    dragMove(moved)
    expect(nodeById(child.id).position).toEqual(moved[1].position)
    expect(nodeById(scalar.id).position.x).toBeCloseTo(displaced(scalar, 80, -25).position.x, 8)
    expect(nodeById(scalar.id).position.y).toBeCloseTo(displaced(scalar, 80, -25).position.y, 8)
    expect(nodeById(otherBlock.id).position).toEqual(otherBlock.position)
    dragStop(moved)

    expect(onViewChange).toHaveBeenLastCalledWith(expect.objectContaining({
      layoutOffsets: { 'visual-group:blocks.0': { x: 100, y: -35 } },
    }))
    expect(onGraphChange).not.toHaveBeenCalled()
    expect(JSON.stringify(graph)).toBe(snapshot)
  })

  it('commits every independent block in a dragged selection', () => {
    const graph = createModelPreset('decoder')
    const { onViewChange } = mountCanvas(graph)
    const blocks = [nodeById('visual-group:blocks.0'), nodeById('visual-group:blocks.1')]
    dragStart(blocks)
    const moved = blocks.map(node => displaced(node, 45, 70))
    dragMove(moved)
    dragStop(moved)
    expect(onViewChange).toHaveBeenLastCalledWith(expect.objectContaining({ layoutOffsets: {
      'visual-group:blocks.0': { x: 45, y: 70 },
      'visual-group:blocks.1': { x: 45, y: 70 },
    } }))
  })

  it('moves projected arithmetic without copying synthetic nodes into the model', () => {
    const graph = setVisualGroupExpanded(createModelPreset('decoder'), 'blocks.0.ff1.layer', true)
    const focus = { groupId: 'blocks.0.ff1.layer', unitIndex: 0, row: 0 }
    const id = 'inspect:blocks.0.ff1.layer:0:w0'
    graph.view!.layoutOffsets = { [id]: { x: 5, y: 10 } }
    const { onViewChange, onGraphChange } = mountCanvas(graph, projectDenseNeurons(graph, focus).graph)
    const node = nodeById(id)
    for (const edge of flow.props!.edges!) {
      const route = edge.data?.route as { x: number; y: number }[]
      for (const obstacle of flow.props!.nodes!.filter(candidate => !candidate.data.expanded)) {
        const rect = { id: obstacle.id, ...obstacle.position, width: obstacle.width!, height: obstacle.height! }
        for (let i = 1; i < route.length; i++) expect(segmentCrossesRect(route[i - 1], route[i], rect), `${edge.id} crosses ${obstacle.id}`).toBe(false)
      }
    }
    dragStart([node])
    const moved = [displaced(node, -30, 40)]
    dragMove(moved)
    dragStop(moved)
    expect(onViewChange.mock.lastCall![0].layoutOffsets[id].x).toBeCloseTo(-25)
    expect(onViewChange.mock.lastCall![0].layoutOffsets[id].y).toBeCloseTo(50)
    expect(onGraphChange).not.toHaveBeenCalled()
    expect(graph.nodes.some(candidate => candidate.id === id)).toBe(false)
  })

  it('uses a secondary pointer drag to pan without changing block selection or graph positions', () => {
    const graph = createModelPreset('decoder')
    const { container, onSelectionChange, onGraphChange } = mountCanvas(graph)
    const shell = container.querySelector<HTMLDivElement>('.flow-shell')!
    shell.setPointerCapture = vi.fn()
    shell.releasePointerCapture = vi.fn()
    // Pointer events extend mouse events in browsers. The native MouseEvent
    // carries the same coordinates here, with a pointer identifier added.
    const pointer = (type: string, x: number, y: number, button = 2) => {
      const event = new MouseEvent(type, { bubbles: true, cancelable: true, button, clientX: x, clientY: y })
      Object.defineProperty(event, 'pointerId', { value: 7 })
      return event
    }
    fireEvent(shell, pointer('pointerdown', 100, 200))
    fireEvent(shell, pointer('pointermove', 145, 175))
    expect(flow.setViewport).toHaveBeenLastCalledWith({ x: 75, y: 25, zoom: 0.8 })
    expect(shell.setPointerCapture).toHaveBeenCalledWith(7)
    fireEvent(shell, pointer('pointerup', 145, 175))
    expect(shell.releasePointerCapture).toHaveBeenCalledWith(7)
    flow.setViewport.mockClear()
    fireEvent(shell, pointer('pointermove', 180, 210, 0))
    expect(flow.setViewport).not.toHaveBeenCalled()
    expect(onSelectionChange).not.toHaveBeenCalled()
    expect(onGraphChange).not.toHaveBeenCalled()
  })
})
