/** Instructor recipes built exclusively from editable palette operations. */
import {createEmptyGraph,createNode} from '../domain/examples'
import {initializeTensor} from '../domain/authoring'
import {organizeTextModel,buildTextModel} from './textModels'
import type {GraphModel,NodeParams,NodeType,TextDatasetData} from '../domain/types'

export function builder(seed=137) {
 const graph=createEmptyGraph();graph.learningRate=.001
 graph.training={engine:'tensor',backend:'auto',optimizer:'adamw',weightDecay:.01,clipNorm:1,patience:5,minDelta:.001}
 const add=(id:string,type:NodeType,params:NodeParams={},inputs:Array<string|[string,number]>=[])=>{
  graph.nodes.push({...createNode(type,graph.nodes.length),id,label:id.replaceAll('_',' '),params,position:{x:Math.floor(graph.nodes.length/6)*250,y:(graph.nodes.length%6)*160}})
  inputs.forEach((x,inputSlot)=>graph.edges.push({id:id+':'+inputSlot,source:typeof x==='string'?x:x[0],sourceSlot:typeof x==='string'?0:x[1],target:id,inputSlot}))
  return id
 }
 const weight=(id:string,shape:number[],mode:'xavier'|'zeros'|'uniform'='xavier')=>add(id,'weight',{value:initializeTensor(shape,mode,seed++)})
 const linear=(id:string,x:string,from:number,to:number)=>add(id,'arithmetic',{expression:'x1 + x2'},[add(id+'_product','matmul',{},[x,weight(id+'_w',[from,to])]),weight(id+'_b',[to],'zeros')])
 const reshape=(id:string,x:string,shape:number[])=>add(id,'tensor-transform',{transform:'reshape',shape},[x])
 const relu=(id:string,x:string)=>add(id,'activation',{activation:'relu'},[x])
 return {graph,add,weight,linear,reshape,relu}
}

export function buildQaModel(data:TextDatasetData,kind:'counts-mlp'|'mean'|'ordered-mlp'|'sentence-mlp'|'memory-1',seed=137,width=32) {
 const b=builder(seed),{graph,add,weight,linear,reshape,relu}=b
 add('text-data','dataset',{dataset:'custom-text',textData:data,datasetMode:'sample',datasetIndex:0})
 let x:string,features:number
 const structured=data.representation==='facts'
 if(data.representation==='counts'){x=add('counts','input',{},[['text-data',0]]);features=data.vocabulary.length}
 else {
  const e=weight('embedding',[data.vocabulary.length,width],'uniform')
  if(structured){
   const n=data.maxFacts!,words=data.factWords!
   const f=add('fact_embeddings','arithmetic',{expression:'x1 * x2'},[add('fact_lookup','embedding',{},[e,['text-data',0]]),['text-data',4]])
   const q=add('question_embeddings','arithmetic',{expression:'x1 * x2'},[add('question_lookup','embedding',{},[e,['text-data',1]]),['text-data',5]])
   const facts=add('positioned_facts','arithmetic',{expression:'x1 + x2'},[relu('fact_relu',linear('fact',reshape('fact_slots',f,[n,words*width]),words*width,width)),add('fact_positions','embedding',{},[weight('time',[n,width],'uniform'),['text-data',2]])])
   const question=relu('question_relu',linear('question',reshape('question_slots',q,[1,words*width]),words*width,width))
   if(kind==='sentence-mlp'){
    const mask=add('fact_mask_transpose','tensor-transform',{transform:'transpose',axes:[1,0]},[['text-data',3]])
    const masked=add('masked_facts','arithmetic',{expression:'x1 * x2'},[facts,mask])
    x=add('all_slots','concat',{axis:1},[reshape('flat_facts',masked,[1,n*width]),question]);features=(n+1)*width
   }else{
    const query=add('query','matmul',{},[question,weight('h0_q',[width,width])])
    const key=add('key','matmul',{},[facts,weight('h0_k',[width,width])])
    const value=add('value','matmul',{},[facts,weight('h0_v',[width,width])])
    const kt=add('key_transpose','tensor-transform',{transform:'transpose',axes:[1,0]},[key])
    const scores=add('scores','matmul',{},[query,kt])
    const masked=add('masked_scores','arithmetic',{expression:`x1 / ${Math.sqrt(width)} + (x2 - 1) * 1000000000`},[scores,['text-data',3]])
    const probs=add('attention','softmax',{},[masked])
    x=add('retrieved_question','arithmetic',{expression:'x1 + x2'},[question,add('read_facts','matmul',{},[probs,value])]);features=width
   }
  }else{
   x=add('token_embeddings','embedding',{},[e,['text-data',0]])
   if(kind==='ordered-mlp'){x=reshape('ordered_slots',x,[1,data.maxLength*width]);features=data.maxLength*width}
   else{x=add('pooled','tensor-transform',{transform:'mean',axis:0,keepDims:true},[x]);features=width}
  }
 }
 x=relu('hidden_relu',linear('hidden',x,features,64))
 x=linear('output',x,64,data.classLabels!.length)
 add('loss','loss',{loss:'cross-entropy'},[x,['text-data',structured?6:2]])
 return graph
}

export function buildAliceModel(data:TextDatasetData,kind:'bigram'|'mlp'|'attention'|'transformer',seed=137,width=32):GraphModel {
 if(kind==='attention'||kind==='transformer'){
  const graph=buildTextModel(data,kind==='transformer'?'alice-transformer':'position-attention',width,seed)
  const source=graph.edges.find(e=>e.target==='loss'&&e.inputSlot===0)!
  graph.nodes.push({...createNode('tensor-transform',graph.nodes.length),id:'last-logits',label:'Last position logits',params:{transform:'slice',axis:0,start:data.maxLength-1,end:data.maxLength}})
  graph.edges.push({id:'last-logits:0',source:source.source,target:'last-logits',inputSlot:0});source.source='last-logits'
  graph.learningRate=.001;graph.training=builder().graph.training
  return organizeTextModel(graph)
 }
 const {graph,add,weight,linear,reshape,relu}=builder(seed)
 add('text-data','dataset',{dataset:'custom-text',textData:data,datasetMode:'sample',datasetIndex:0})
 let ids=add('ids','input',{},[['text-data',0]])
 if(kind==='bigram')ids=add('last-token','tensor-transform',{transform:'slice',axis:0,start:data.maxLength-1,end:data.maxLength},[ids])
 let x=add('embeddings','embedding',{},[weight('embedding',[data.vocabulary.length,width],'uniform'),ids])
 if(kind==='mlp')x=relu('hidden_relu',linear('hidden',reshape('context_slots',x,[1,data.maxLength*width]),data.maxLength*width,64))
 const logits=linear('vocabulary',x,kind==='mlp'?64:width,data.vocabulary.length)
 add('loss','loss',{loss:'cross-entropy'},[logits,['text-data',2]])
 return graph
}
