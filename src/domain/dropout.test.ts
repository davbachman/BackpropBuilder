import {describe,it,expect,vi,afterEach} from 'vitest'
import {forwardPass,backwardPass,runTrainingStep,runTrainingStepFast,validateGraph} from './engine'
import {createNode} from './examples'
import {createProjectStateFile,parseProjectStateFile} from './session'
import {prepareTextDocuments} from './textData'
import {addTransformerDropout,buildTextModel} from '../test/textModels'
import {generatePyTorchExport} from './pytorchExport'
import type {GraphModel} from './types'
const graph=(p=.5):GraphModel=>({learningRate:.1,nodes:[
 {...createNode('weight',0),id:'w',params:{value:{shape:[4],data:[1,2,3,4]}}},
 {...createNode('dropout',1),id:'d',params:{dropoutRate:p}},
 {...createNode('target',2),id:'y',params:{value:{shape:[4],data:[0,0,0,0]}}},
 {...createNode('loss',3),id:'loss',params:{loss:'mse'}},
],edges:[{id:'wd',source:'w',target:'d',inputSlot:0},{id:'dl',source:'d',target:'loss',inputSlot:0},{id:'yl',source:'y',target:'loss',inputSlot:1}]})
afterEach(()=>vi.restoreAllMocks())
describe('dropout operation',()=>{
 it('uses inverted dropout in training and reuses the exact mask in backward',()=>{
  const random=vi.fn().mockReturnValueOnce(.1).mockReturnValueOnce(.6).mockReturnValueOnce(.2).mockReturnValueOnce(.9)
  const result=forwardPass(graph(),true,{training:true,random})
  expect(result.graph.nodes.find(n=>n.id==='d')!.value!.data).toEqual([0,4,0,8])
  expect(result.loss).toBe(20)
  expect(result.steps.find(s=>s.nodeId==='d')!.calculation).toContain('saved scale')
  const backward=backwardPass(result.graph)
  expect(backward.graph.nodes.find(n=>n.id==='w')!.grad!.data).toEqual([0,4,0,8])
  expect(random).toHaveBeenCalledTimes(4)
 })
 it('evaluation is identity and consumes no randomness; zero dropout also bypasses masks',()=>{
  const random=()=>{throw Error('Should not draw a mask')}
  expect(forwardPass(graph(),false,{random}).loss).toBe(7.5)
  expect(forwardPass(graph(0),false,{training:true,random}).loss).toBe(7.5)
 })
 it('both SGD paths train with dropout and show an unmasked post-update prediction',()=>{
  vi.spyOn(Math,'random').mockReturnValue(.75)
  const fast=runTrainingStepFast(graph()),traced=runTrainingStep(graph())
  expect(traced.graph.nodes.find(n=>n.id==='w')!.params.value).toEqual(fast.nodes.find(n=>n.id==='w')!.params.value)
  expect(traced.graph.nodes.find(n=>n.id==='d')!.value!.data).toEqual([.8,1.6,2.4,3.2])
 })
 it.each([-0.1,1,NaN,Infinity])('rejects invalid probability %s',p=>expect(validateGraph(graph(p)).some(i=>i.code==='invalid-value')).toBe(true))
 it('preserves rate and cached mask through project save/import and exports train/eval-aware PyTorch',()=>{
  const g=forwardPass(graph(),true,{training:true,random:()=>.75}).graph
  const file=createProjectStateFile({graph:g,visualizationGraph:g,initialParameterValues:{},selectedNodeIds:[],phase:'forward',traceSteps:[],traceIndex:0,epoch:0,currentLoss:30,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
  const parsed=parseProjectStateFile(JSON.stringify(file));expect(parsed.ok).toBe(true)
  if(parsed.ok){expect(parsed.file.state.graph.nodes[1].params.dropoutRate).toBe(.5);expect(backwardPass(parsed.file.state.graph).graph.nodes[0].grad!.data).toEqual([2,4,6,8])}
  const text=prepareTextDocuments([{text:'good film',label:'positive',split:'train'},{text:'bad film',label:'negative',split:'test'}],'reviews.csv',{task:'sentiment',maxLength:4})
  expect(generatePyTorchExport(addTransformerDropout(buildTextModel(text,'transformer',4),.5)).script).toContain('p=0.5, training=self.training')
  file.state.graph.nodes[1].params.dropoutRate=1
  expect(parseProjectStateFile(JSON.stringify(file)).ok).toBe(false)
 })
})
