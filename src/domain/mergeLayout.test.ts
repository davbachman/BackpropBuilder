import { describe, expect, it } from 'vitest'
import { mergePreservingLayout, ungroupPreservingLayout } from './mergeLayout'
import { createStarterGraph } from './examples'
import { layoutContinuousScene, sceneGroupId } from './continuousScene'
import { builderCardWidth, builderCardHeight } from './builderGeometry'
import { cloneGraph, forwardPass } from './engine'

const relative = (rect: {x:number;y:number;width:number;height:number}, anchor: {x:number;y:number;width:number}) => ({
  x: (rect.x-anchor.x)/anchor.width, y:(rect.y-anchor.y)/anchor.width,
  width: rect.width/anchor.width, height:rect.height/anchor.width,
})
function sameShape(a: ReturnType<typeof relative>, b: ReturnType<typeof relative>) {
  for (const key of ['x','y','width','height'] as const) expect(a[key]).toBeCloseTo(b[key], 8)
}

describe('merge layout preservation', () => {
  it('shrinks a deliberately hand-arranged selection about its center without moving other blocks', () => {
    const graph = createStarterGraph()
    graph.nodes.find(n => n.id === 'mul')!.position = {x:700,y:100}
    graph.nodes.find(n => n.id === 'add')!.position = {x:380,y:440}
    const ids = ['w','mul','add']
    const rects = new Map(graph.nodes.map(n => [n.id, {...n.position,width:builderCardWidth(n),height:builderCardHeight(n)}]))
    const original = structuredClone(graph)
    const merged = mergePreservingLayout(graph, ids)
    const scene = layoutContinuousScene(merged.graph)
    for (const id of ids) sameShape(relative(scene.nodes.get(id)!,scene.nodes.get('w')!),relative(rects.get(id)!,rects.get('w')!))
    for (const n of graph.nodes.filter(n => !ids.includes(n.id))) expect(scene.nodes.get(n.id)).toMatchObject(rects.get(n.id)!)
    const selected = ids.map(id => rects.get(id)!)
    const frame = scene.groups.get(merged.group!.id)!
    expect(frame.x+frame.width/2).toBeCloseTo((Math.min(...selected.map(r=>r.x))+Math.max(...selected.map(r=>r.x+r.width)))/2)
    expect(frame.y+frame.height/2).toBeCloseTo((Math.min(...selected.map(r=>r.y))+Math.max(...selected.map(r=>r.y+r.height)))/2)
    expect(forwardPass(merged.graph).graph.nodes.find(n=>n.id==='loss')?.value).toEqual(forwardPass(graph).graph.nodes.find(n=>n.id==='loss')?.value)
    expect(graph).toEqual(original)
  })

  it('preserves a nested selection including dragged offsets and clones saved layouts independently', () => {
    const graph = mergePreservingLayout(createStarterGraph(), ['x','w','mul','b','add','pred']).graph
    graph.view!.layoutOffsets = {mul:{x:2,y:1}}
    const before = layoutContinuousScene(graph)
    const merged = mergePreservingLayout(graph,['mul','add'])
    const after = layoutContinuousScene(merged.graph)
    sameShape(relative(after.nodes.get('add')!,after.nodes.get('mul')!),relative(before.nodes.get('add')!,before.nodes.get('mul')!))
    expect(after.parents.get('mul')).toBe(merged.group!.id)
    for (const key of ['x','y','width','height'] as const) expect(after.nodes.get('x')![key]).toBeCloseTo(before.nodes.get('x')![key], 8)
    const ungrouped = layoutContinuousScene(ungroupPreservingLayout(merged.graph, merged.group!.id))
    for (const id of ['mul','add']) for (const key of ['x','y','width','height'] as const) expect(ungrouped.nodes.get(id)![key]).toBeCloseTo(after.nodes.get(id)![key], 8)
    const cloned = cloneGraph(merged.graph)
    cloned.view!.preservedLayouts![''][sceneGroupId('group-1')].x += 99
    expect(cloned.view!.preservedLayouts).not.toEqual(merged.graph.view!.preservedLayouts)
    const reopened = JSON.parse(JSON.stringify(merged.graph))
    expect(layoutContinuousScene(reopened).nodes).toEqual(after.nodes)
  })
})
