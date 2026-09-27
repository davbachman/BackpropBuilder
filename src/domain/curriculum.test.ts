import {beforeAll,afterEach,it,expect} from 'vitest'
import {prepareTextDocuments,importText,isTextDatasetData} from './textData'
import {buildQaModel,buildAliceModel,builder} from '../test/curriculumModels'
import {forwardPass,backwardPass,runTrainingStepFast,updateParameters,parameterValues} from './engine'
import {TensorGraph,tf} from './tensorTraining'
import {parameterPenalty} from './regularization'
import {predictText} from './textGeneration'
import {evaluateDataset} from './datasetTraining'
import {parseCustomCsv} from './customCsv'
import {buildTextModel} from '../test/textModels'
import {generatePyTorchExport} from './pytorchExport'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
const docs=[{text:'Mary moved to the kitchen. Mary moved to the garden. Where is Mary?',facts:['Mary moved to the kitchen.','Mary moved to the garden.'],question:'Where is Mary?',label:'garden',split:'train' as const},{text:'John moved to the garden. John moved to the kitchen. Where is John?',facts:['John moved to the garden.','John moved to the kitchen.'],question:'Where is John?',label:'kitchen',split:'train' as const},{text:'Mary moved to the garden. Where is Mary?',facts:['Mary moved to the garden.'],question:'Where is Mary?',label:'garden',split:'test' as const}]
it.each(['counts-mlp','mean','ordered-mlp','sentence-mlp','memory-1'] as const)('matches trace forward and backward for QA %s',async kind=>{
 const structured=kind==='sentence-mlp'||kind==='memory-1'
 const data=prepareTextDocuments(docs,'qa.csv',{task:'classification',representation:structured?'facts':kind==='counts-mlp'?'counts':'tokens',maxLength:32,fixedLength:kind==='ordered-mlp',factWords:10,maxFacts:4})
 expect(isTextDatasetData(data)).toBe(true)
 const graph=buildQaModel(data,kind,137,4),trace=backwardPass(forwardPass(graph,false).graph),model=new TensorGraph(graph)
 try{const result=model.gradients(model.examples.slice(0,1));try{
 expect((await result.loss.data())[0]).toBeCloseTo(trace.loss!,5)
 for(const [id,v] of model.variables){const expected=trace.graph.nodes.find(n=>n.id===id)!.grad!.data;const actual=await result.grads[v.name].data();actual.forEach((g,i)=>expect(g,id).toBeCloseTo(expected[i],4))}
 }finally{tf.dispose([result.loss,...Object.values(result.grads)])}
 expect(Number.isFinite((await model.evaluate(model.examples,2)).loss)).toBe(true)
 const decoded = await model.inference(model.examples.filter(row=>row.split==='test'),2)
 const expected = evaluateDataset(graph, 'text-data', 'test')
 expect(decoded.loss).toBeCloseTo(expected.loss,5)
 expect(decoded.rows).toEqual(expected.rows)
 expect(decoded.accuracy).toBe(expected.accuracy)
 }finally{model.dispose()}
})
it.each(['bigram','mlp','attention','transformer'] as const)('Alice last-target graph agrees with trace and can generate: %s',async kind=>{
 const data=prepareTextDocuments([{text:'alice was here. alice was there.',split:'train'},{text:'alice is here.',split:'test'}],'alice.txt',{task:'language',maxLength:4,stride:1,fixedLength:true,targetMode:'last'})
 const graph=buildAliceModel(data,kind,137,4),model=new TensorGraph(graph)
 try{expect((await model.inspect(model.examples.slice(0,1))).loss).toBeCloseTo(forwardPass(graph,false).loss!,5);expect(predictText(graph,'a').logits).toHaveLength(data.vocabulary.length)}finally{model.dispose()}
})
it('rejects unseen classes, overlong facts, and keeps explicit splits',()=>{
 const d=prepareTextDocuments(docs,'qa.csv',{task:'classification',maxLength:32})
 expect(isTextDatasetData({...d,documents:[...docs,{text:'new',label:'unknown',split:'test'}]})).toBe(false)
 expect(()=>prepareTextDocuments(docs,'qa.csv',{task:'classification',representation:'facts',factWords:2,maxFacts:4})).toThrow(/word limit/)
 const csv='text,label,split\na,a,train\nb,b,train\na,a,test';expect(importText(csv,'qa.csv',{task:'classification'}).documents[2].split).toBe('test')
})
it.each(['l1','l2'] as const)('regularization has correct objective, gradients, updates, evaluation and export: %s',async kind=>{
 const data=prepareTextDocuments(docs,'qa.csv',{task:'classification',representation:'counts',maxLength:32})
 const graph=buildQaModel(data,'counts-mlp',137,4),loss=graph.nodes.find(n=>n.id==='loss')!
 const base=backwardPass(forwardPass(graph,false).graph)
 loss.params={...loss.params,regularization:kind,regularizationStrength:.2,regularizationParameterIds:['output_w']}
 const out=graph.nodes.find(n=>n.id==='output_w')!;const original=(out.params.value as {data:number[]}).data
 const trace=backwardPass(forwardPass(graph,false).graph)
 expect(trace.loss!-base.loss!).toBeCloseTo(.2*original.reduce((s,w)=>s+(kind==='l1'?Math.abs(w):w*w/2),0),10)
 const grads=trace.graph.nodes.find(n=>n.id==='output_w')!.grad!.data,old=base.graph.nodes.find(n=>n.id==='output_w')!.grad!.data
 grads.forEach((g,i)=>expect(g-old[i]).toBeCloseTo(.2*(kind==='l1'?Math.sign(original[i]):original[i]),10))
 const fast=parameterValues(runTrainingStepFast(graph)),slow=parameterValues(updateParameters(trace.graph).graph)
 expect(fast).toEqual(slow)
 const model=new TensorGraph(graph)
 try{const g=model.gradients(model.examples.slice(0,1));try{expect((await g.loss.data())[0]).toBeCloseTo(trace.loss!,5);const values=await g.grads[model.variables.get('output_w')!.name].data();values.forEach((x,i)=>expect(x).toBeCloseTo(grads[i],5))}finally{tf.dispose([g.loss,...Object.values(g.grads)])}
 const report=await model.evaluate(model.examples.slice(0,1),1);expect(report.objective!-report.loss).toBeCloseTo(parameterPenalty(graph),5)
 }finally{model.dispose()}
 expect(generatePyTorchExport({...graph,training:undefined}).script).toContain(kind==='l1'?'.abs()':'.square() * 0.5')
})
it('L1 chooses zero subgradient at zero and excludes unselected biases',()=>{
 const {graph,add}=builder();add('w','weight',{value:0});add('b','bias',{value:2});add('x','input',{value:1});add('y','target',{value:0});add('p','multiply',{},['w','x']);add('prediction','add',{},['p','b']);add('loss','loss',{loss:'squared-error',regularization:'l1',regularizationStrength:10},['prediction','y'])
 const g=backwardPass(forwardPass(graph).graph).graph
 expect(g.nodes.find(n=>n.id==='w')!.grad!.data[0]).toBe(2);expect(g.nodes.find(n=>n.id==='b')!.grad!.data[0]).toBe(2)
})

it.each(['regression','binary-classification'] as const)('minibatch inference decodes %s and excludes penalty',async task=>{
 const {graph,add,reshape,linear}=builder()
 const csv=parseCustomCsv('x,target,split\n1,0,train\n2,1,train\n3,1,test\n4,0,test\n','numeric.csv');csv.task=task
 add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
 let prediction=linear('output',reshape('x',add('input','input',{},[['data',0]]),[1,1]),1,1)
 if(task==='binary-classification')prediction=add('probability','activation',{activation:'sigmoid'},[prediction])
 add('loss','loss',{loss:task==='regression'?'mse':'binary-cross-entropy',regularization:'l2',regularizationStrength:2,regularizationParameterIds:['output_w']},[prediction,['data',1]])
 const model=new TensorGraph(graph)
 try {
  const actual=await model.inference(model.examples.filter(row=>row.split==='test'),2),expected=evaluateDataset(graph,'data','test')
  expect(actual.loss).toBeCloseTo(expected.loss,5)
  expect(actual.accuracy).toBe(expected.accuracy)
  expect(actual.rows).toEqual(expected.rows)
 }finally{model.dispose()}
})
it('all-position language inference preserves decoded targets and row counts',async()=>{
 const data=prepareTextDocuments([{text:'alice was here. alice was there.',split:'train'},{text:'alice is here.',split:'test'}],'alice.txt',{task:'language',tokenizer:'character',maxLength:4,stride:1})
 const graph=buildTextModel(data,'alice-transformer',4),model=new TensorGraph(graph)
 try {
  const actual=await model.inference(model.examples.filter(row=>row.split==='test'),3),expected=evaluateDataset(graph,'text-data','test')
  expect(actual.loss).toBeCloseTo(expected.loss,5)
  expect(actual.rows).toEqual(expected.rows)
  expect(actual.predictions).toBe(expected.predictions)
  expect(actual.accuracy).toBe(expected.accuracy)
 }finally{model.dispose()}
})
