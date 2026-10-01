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
 node.params.outputName='u'
 expect(validateGraph(graph).some(i=>i.message.includes('Fit Standardize'))).toBe(true)
 node.params.standardization=await fitStandardizer(graph,node.id)
 expect(node.params.standardization).toEqual({mean:[2,7],scale:[1,1],count:2})
 expect(standardize({shape:[1,2],data:[1000,900]},node.params.standardization).value.data).toEqual([998,893])
 expect(forwardPass(graph).graph.nodes.find(n=>n.id===node.id)!.value!.data).toEqual([-1,0])
 const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:parameterValues(graph),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:0,currentLoss:null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
 const result=parseProjectStateFile(JSON.stringify(file));expect(result.ok).toBe(true)
 if(result.ok){
  expect(result.file.state.graph.nodes.find(n=>n.id===node.id)!.params.standardization).toEqual(node.params.standardization)
  expect(result.file.state.graph.nodes.find(n=>n.id===node.id)!.params.outputName).toBe('u')
 }
 expect(generatePyTorchExport({...graph,training:undefined}).script).toContain('[2,7]')
})
it.each(['batch', undefined] as const)('fits directly concatenated columns in %s mode without reshapes or held-out leakage', async datasetMode => {
 const {graph,add}=builder()
 const csv=parseCustomCsv('radius,width,y,split\n1,10,0,train\n3,30,1,train\n1000,9000,2,test\n','direct.csv')
 add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode})
 add('joined','concat',{axis:1},[['data',0],['data',1]])
 add('standard','standardize',{},['joined'])
 const snapshot=structuredClone(graph)
 const stats=await fitStandardizer(graph,'standard')
 expect(stats).toEqual({mean:[2,20],scale:[1,10],count:2})
 expect(graph).toEqual(snapshot)
 graph.nodes.find(node=>node.id==='standard')!.params.standardization=stats
 const result=forwardPass(graph,false).graph.nodes.find(node=>node.id==='standard')!.value!
 expect(result.shape).toEqual([3,2])
 expect(result.data).toEqual([-1,-1,1,1,998,898])
 graph.nodes[0].params.datasetSplit='test'
 expect(await fitStandardizer(graph,'standard')).toEqual(stats)
})

it('matches traced and accelerated values and all parameter gradients',async()=>{
 const graph=fixture()
 const source=graph.nodes.find(n=>n.id==='data')!
 source.params.customCsv=parseCustomCsv('a,b,y,split\n1,7,2,train\n3,17,4,train\n1000,900,5,test\n','features.csv')
 graph.nodes.find(n=>n.id==='standard')!.params.standardization=await fitStandardizer(graph,'standard')
 expect(graph.nodes.find(n=>n.id==='standard')!.params.standardization?.scale).toEqual([1,5])
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
it('fits a standalone matrix column by column, preserving shape and fixed statistics', async () => {
 const {graph,add}=builder()
 add('matrix','input',{value:{shape:[3,3],data:[1,10,7,3,20,7,5,30,7]}})
 add('standard','standardize',{},['matrix'])
 const stats=await fitStandardizer(graph,'standard')
 expect(stats.mean).toEqual([3,20,7])
 expect(stats.scale).toEqual([Math.sqrt(8/3),Math.sqrt(200/3),1])
 expect(stats.count).toBe(3)
 graph.nodes[1].params.standardization=stats
 const result=forwardPass(graph,false).graph.nodes[1].value!
 expect(result.shape).toEqual([3,3])
 for(let column=0;column<3;column++){
  const values=[0,1,2].map(row=>result.data[row*3+column])
  expect(values.reduce((a,b)=>a+b,0)/3).toBeCloseTo(0)
  expect(values.reduce((a,b)=>a+b*b,0)/3).toBeCloseTo(column===2?0:1)
 }
 const later=standardize({shape:[1,3],data:[5,30,8]},stats)
 expect(later.value.data).toEqual([2/stats.scale[0],10/stats.scale[1],1])
 expect(later.derivative.data).toEqual(stats.scale.map(scale=>1/scale))
 expect(stats.mean).toEqual([3,20,7])
})
it('requires one fitted statistic per matrix column, including single-column matrices', async () => {
 const {graph,add}=builder()
 add('matrix','input',{value:{shape:[2,1],data:[2,6]}})
 add('standard','standardize',{},['matrix'])
 const stats=await fitStandardizer(graph,'standard')
 expect(stats).toEqual({mean:[4],scale:[2],count:2})
 expect(standardize({shape:[2,1],data:[2,6]},stats).value).toEqual({shape:[2,1],data:[-1,1]})
 expect(()=>standardize({shape:[2,2],data:[2,6,4,8]},stats)).toThrow(/Refit/)
 graph.nodes[0].params.value={shape:[2,2],data:[2,6,4,8]}
 graph.nodes[1].params.standardization=stats
 expect(validateGraph(graph,{requireLoss:false})).toContainEqual(expect.objectContaining({code:'shape-mismatch',nodeId:'standard'}))
 expect(await fitStandardizer(graph,'standard')).toEqual({mean:[3,7],scale:[1,1],count:2})
})
it('fits matrices wider than the CSV column limit', async () => {
 const {graph,add}=builder()
 add('matrix','input',{value:{shape:[2,130],data:[...Array(130).fill(1),...Array(130).fill(3)]}})
 add('standard','standardize',{},['matrix'])
 const stats=await fitStandardizer(graph,'standard')
 expect(stats.mean).toEqual(Array(130).fill(2))
 expect(stats.scale).toEqual(Array(130).fill(1))
 expect(stats.count).toBe(2)
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
