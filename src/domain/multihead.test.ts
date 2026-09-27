import type {TensorValue} from './types'
import {beforeAll,expect,it} from 'vitest'
import {buildTextModel,withTwoAttentionHeads} from '../test/textModels'
import {prepareTextDocuments} from './textData'
import {forwardPass,backwardPass,parameterValues} from './engine'
import {withDatasetExample} from './datasetTraining'
import {TensorGraph,tf} from './tensorTraining'
const data=prepareTextDocuments([{text:'good funny good film',label:'positive',split:'train'},{text:'bad film',label:'negative',split:'train'},{text:'good film',label:'positive',split:'test'}],'test.csv',{task:'sentiment',maxLength:4})
beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
it('splits projections without adding parameters or changing unrelated initial weights',()=>{
 const original=buildTextModel(data,'transformer',4),before=JSON.stringify(original),graph=withTwoAttentionHeads(original)
 const count=(g:typeof graph)=>Object.values(parameterValues(g)).reduce((n,v)=>n+v.data.length,0)
 expect(count(graph)).toBe(count(original));expect(JSON.stringify(original)).toBe(before)
 for(const name of ['query','key','value']){
  const full=original.nodes.find(n=>n.id===name+'-weights')!.params.value as TensorValue
  const halves=[1,2].map(i=>graph.nodes.find(n=>n.id===`attention-head-${i}-${name}-weights`)!.params.value as TensorValue)
  expect(halves.map(v=>v.shape)).toEqual([[4,2],[4,2]])
  for(let row=0;row<4;row++)expect([...halves[0].data.slice(row*2,row*2+2),...halves[1].data.slice(row*2,row*2+2)]).toEqual(full.data.slice(row*4,row*4+4))
 }
 for(const node of graph.nodes.filter(n=>n.params.value&&!n.id.startsWith('attention-head-')))expect(node.params.value).toEqual(original.nodes.find(n=>n.id===node.id)!.params.value)
 expect(()=>withTwoAttentionHeads(buildTextModel(data,'transformer',3))).toThrow(/even/)
})
it('matches trace loss and gradients with two heads and unequal review lengths',async()=>{
 const graph=withTwoAttentionHeads(buildTextModel(data,'transformer',4)),model=new TensorGraph(graph)
 try{
  const rows=model.examples.slice(0,2),expected=rows.map((_,i)=>forwardPass(withDatasetExample(graph,'text-data',i),false))
  const gradients=expected.map(r=>backwardPass(r.graph).graph),actual=model.gradients(rows)
  try{
   expect((await actual.loss.data())[0]).toBeCloseTo((expected[0].loss!+expected[1].loss!)/2,5)
   for(const [id,v] of model.variables){
    const values=await actual.grads[v.name].data(),a=gradients[0].nodes.find(n=>n.id===id)!.grad!.data,b=gradients[1].nodes.find(n=>n.id===id)!.grad!.data
    values.forEach((value,i)=>expect(value,id).toBeCloseTo((a[i]+b[i])/2,4))
   }
  }finally{tf.dispose([actual.loss,...Object.values(actual.grads)])}
  const single=await model.inspect([rows[1]]),batch=await model.inspect(rows)
  expect(single.prediction[0]).toBeCloseTo(batch.prediction[1],6)
 }finally{model.dispose()}
 expect(tf.memory().numTensors).toBe(0)
})
