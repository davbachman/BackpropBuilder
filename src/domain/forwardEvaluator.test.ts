import {describe,expect,it} from 'vitest'
import {createForwardEvaluator,forwardPass,runTrainingStepFast} from './engine'
import {createModelPreset} from './modelPresets'
import {withDatasetExample} from './datasetTraining'
import {LESSONS} from '../learning/presets'
import type {GraphModel} from './types'

function expectForwardParity(graph:GraphModel) {
  const snapshot=JSON.stringify(graph)
  const expected=forwardPass(graph,false),actual=createForwardEvaluator(graph)(graph)
  expect(actual.loss).toEqual(expected.loss)
  for(const node of expected.graph.nodes) expect(actual.graph.nodes.find(value=>value.id===node.id)?.value,node.id).toEqual(node.value)
  expect(actual.steps).toEqual([])
  expect(actual.graph.nodes.every(node=>node.grad===undefined&&node.cache===undefined&&node.localDerivative===undefined)).toBe(true)
  expect(actual.graph.edges.every(edge=>edge.value===undefined&&edge.grad===undefined)).toBe(true)
  expect(JSON.stringify(graph)).toBe(snapshot)
}

describe('compiled inference-only forward evaluator',()=>{
  it.each(LESSONS.map(lesson=>lesson.id))('matches every forward value for %s without changing the input graph',kind=>{
    expectForwardParity(forwardPass(createModelPreset(kind),false).graph)
  })
  it('reuses compiled operations across dataset examples and parameter updates',()=>{
    const graph=createModelPreset('linear'),source=graph.nodes.find(node=>node.type==='dataset')!
    const evaluate=createForwardEvaluator(graph)
    for(const index of [0,1,2]) {
      const example=withDatasetExample(graph,source.id,index)
      const updated=runTrainingStepFast(example)
      const expected=forwardPass(updated,false),actual=evaluate(updated)
      expect(actual.loss).toBe(expected.loss)
      expect(actual.graph.nodes.map(node=>node.value)).toEqual(expected.graph.nodes.map(node=>node.value))
    }
  })
  it.each(['l1','l2'] as const)('includes %s penalties and disables dropout during inference',regularization=>{
    const graph=createModelPreset('linear'),loss=graph.nodes.find(node=>node.type==='loss')!
    loss.params={...loss.params,regularization,regularizationStrength:.3}
    const edge=graph.edges.find(edge=>edge.target===loss.id&&edge.inputSlot===0)!
    graph.nodes.push({id:'dropout',type:'dropout',label:'Dropout',position:{x:0,y:0},params:{dropoutRate:.75}})
    graph.edges.push({id:'dropout-input',source:edge.source,target:'dropout',inputSlot:0})
    edge.source='dropout'
    expectForwardParity(graph)
    const evaluate=createForwardEvaluator(graph)
    expect(evaluate().loss).toBe(evaluate().loss)
  })
  it('requires recompilation when graph topology changes',()=>{
    const graph=createModelPreset('linear'),evaluate=createForwardEvaluator(graph)
    expect(()=>evaluate({...graph,edges:graph.edges.slice(1)})).toThrow(/topology changed/)
    expect(()=>evaluate({...graph,nodes:graph.nodes.slice().reverse()})).toThrow(/topology changed/)
    graph.edges[0].source='changed'
    expect(()=>evaluate()).toThrow(/topology changed/)
  })
})
