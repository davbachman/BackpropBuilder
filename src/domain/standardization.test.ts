import { afterEach, beforeAll, expect, it } from 'vitest'
import { builder } from '../test/curriculumModels'
import {datasetExamplesForNode} from './datasets'
import { parseCustomCsv } from './customCsv'
import {fitStandardizer} from './standardizationFit'
import { isStandardizationStats, standardize } from './standardization'
import { backwardPass, forwardPass, parameterValues, validateGraph } from './engine'
import { TensorGraph, tf } from './tensorTraining'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { generatePyTorchExport } from './pytorchExport'

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready() })
afterEach(() => expect(tf.memory().numTensors).toBe(0))
function fixture() {
  const {graph,add,reshape,linear}=builder()
  const csv=parseCustomCsv('a,b,y,split\n1,7,2,train\n3,7,4,train\n1000,900,5,test\n','features.csv');csv.task='regression'
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
  const a=reshape('a',add('a-input','input',{},[['data',0]]),[1,1])
  const b=reshape('b',add('b-input','input',{},[['data',1]]),[1,1])
  add('joined','concat',{axis:1},[a,b]);add('standard','standardize',{},['joined'])
  const output=linear('output','standard',2,1)
  add('loss','loss',{loss:'mse'},[output,['data',2]])
  return graph
}
it('fits training rows only, handles a constant feature, and reuses saved statistics',async()=>{
 const graph=fixture(),node=graph.nodes.find(n=>n.id==='standard')!
 expect(validateGraph(graph).some(i=>i.message.includes('Fit Standardize'))).toBe(true)
 node.params.standardization=await fitStandardizer(graph,node.id)
 expect(node.params.standardization).toEqual({mean:[2,7],scale:[1,1],count:2})
 expect(standardize({shape:[1,2],data:[1000,900]},node.params.standardization).value.data).toEqual([998,893])
 expect(forwardPass(graph).graph.nodes.find(n=>n.id===node.id)!.value!.data).toEqual([-1,0])
 const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:parameterValues(graph),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:0,currentLoss:null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
 const result=parseProjectStateFile(JSON.stringify(file));expect(result.ok).toBe(true)
 if(result.ok)expect(result.file.state.graph.nodes.find(n=>n.id===node.id)!.params.standardization).toEqual(node.params.standardization)
 expect(generatePyTorchExport({...graph,training:undefined}).script).toContain('[2,7]')
})
it('matches traced and accelerated values and all parameter gradients',async()=>{
 const graph=fixture();graph.nodes.find(n=>n.id==='standard')!.params.standardization=await fitStandardizer(graph,'standard')
 const traced=backwardPass(forwardPass(graph).graph),model=new TensorGraph(graph)
 try{
  const result=model.gradients(model.examples.slice(0,1))
  try{expect((await result.loss.data())[0]).toBeCloseTo(traced.loss!,5)
   for(const [id,v] of model.variables){const values=await result.grads[v.name].data();values.forEach((x,i)=>expect(x).toBeCloseTo(traced.graph.nodes.find(n=>n.id===id)!.grad!.data[i],5))}
  }finally{tf.dispose([result.loss,...Object.values(result.grads)])}
  const fit=graph.nodes.find(n=>n.id==='standard')!.params.standardization
  await model.inference(model.examples,2)
  expect(graph.nodes.find(n=>n.id==='standard')!.params.standardization).toEqual(fit)
 }finally{model.dispose()}
})
it('uses population standard deviation, supports scalar/batch columns, and rejects invalid stats',()=>{
 const stats={mean:[2],scale:[2],count:4}
 expect(standardize({shape:[],data:[4]},stats)).toEqual({value:{shape:[],data:[1]},derivative:{shape:[],data:[.5]}})
 expect(standardize({shape:[3],data:[0,2,4]},stats).value.data).toEqual([-1,0,1])
 expect(()=>standardize({shape:[3],data:[1,2,3]},{mean:[0,0],scale:[1,1],count:2})).toThrow(/count changed/)
 for(const bad of [{mean:[0],scale:[0],count:2},{mean:[NaN],scale:[1],count:2},{mean:[0,1],scale:[1],count:2},{mean:[0],scale:[1],count:0}])expect(isStandardizationStats(bad)).toBe(false)
})
it('rejects targets and trainable preprocessing and fits a scalar column',async()=>{
 const graph=fixture(),edge=graph.edges.find(e=>e.target==='standard')!
 edge.source='data';edge.sourceSlot=2
 await expect(fitStandardizer(graph,'standard')).rejects.toThrow(/target/)
 edge.source='output_w';edge.sourceSlot=0
 await expect(fitStandardizer(graph,'standard')).rejects.toThrow()
 edge.source='data';edge.sourceSlot=0
 expect(await fitStandardizer(graph,'standard')).toEqual({mean:[2],scale:[1],count:2})
})
it('accepts a full housing-sized numeric CSV',()=>{
 const csv=parseCustomCsv('a,y,split\n'+Array.from({length:20640},(_,i)=>`${i},${i/10},${i<14000?'train':'test'}`).join('\n'),'housing.csv')
 expect(csv.rows).toHaveLength(20641)
})

it('reuses numeric examples and invalidates cached rows when the CSV is replaced',()=>{
 const graph=fixture(),node=graph.nodes.find(n=>n.id==='data')!
 const original=datasetExamplesForNode(node)
 expect(datasetExamplesForNode(node)).toBe(original)
 const csv=node.params.customCsv!
 node.params.customCsv={...csv,splits:['test','train','test']}
 const changed=datasetExamplesForNode(node)
 expect(changed).not.toBe(original)
 expect(changed[0].split).toBe('test')
 expect(original[0].split).toBe('train')
})
