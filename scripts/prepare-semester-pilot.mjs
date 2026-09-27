import {existsSync} from 'node:fs'
if(existsSync('output/semester-pilot/frozen.json'))throw Error('Pilot is frozen. Use a separate checkout/output directory for a new study.')
import {mkdir,readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot'
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const load=p=>server.ssrLoadModule('/src/'+p+'.ts')
 const {prepareTextDocuments}=await load('domain/textData'),{builder,buildQaModel,buildAliceModel}=await load('test/curriculumModels')
 const {parseCustomCsv}=await load('domain/customCsv'),{datasetExamplesForNode}=await load('domain/datasets')
 const {forwardPass,validateGraph}=await load('domain/engine'),{parseArithmetic}=await load('domain/arithmetic')
 for(const dir of ['templates','numeric','models','runs'])await mkdir(root+'/'+dir,{recursive:true})
 const manifest=[]
 async function save(name,graph,epochs,batch=64){
  const issues=validateGraph(graph).filter(x=>x.code!=='disconnected');if(issues.length)throw Error(name+JSON.stringify(issues))
  const source=graph.nodes.find(n=>n.type==='dataset'),rows=datasetExamplesForNode(source)
  const first=forwardPass(graph,false).loss
  const numeric=rows.map(r=>({features:r.features,target:r.target,split:r.split==='train'?'train':'validation'}))
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  await writeFile(root+'/numeric/'+name+'.json',JSON.stringify(numeric))
  await writeFile(root+'/templates/'+name+'.json',JSON.stringify({name,graph,expressions,expectedLoss:first,epochs,batch}))
  manifest.push({name,epochs,batch,train:rows.filter(r=>r.split==='train').length,validation:rows.filter(r=>r.split==='test').length})
  console.log('Prepared',name,rows.length,first)
 }
 for(const name of ['mpg','cancer','iris','concrete','digits']){
  const raw=JSON.parse(await readFile(root+'/data/'+name+'.json','utf8')),csv=parseCustomCsv(await readFile(root+'/data/'+name+'.csv','utf8'),name+'.csv')
  for(const model of (name==='mpg'?['linear','quadratic']:name==='concrete'||name==='digits'?['linear','mlp']:['linear'])){
   for(const seed of [137,211,307]){
    const {graph,add,linear,reshape,relu}=builder(seed)
    graph.training.weightDecay=0;graph.training.patience=15;graph.training.minDelta=.0001
    add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample',datasetIndex:0})
    const xs=Array.from({length:raw.features},(_,i)=>reshape('column'+i,add('x'+i,'input',{},[['data',i]]),[1,1]))
    if(model==='quadratic')xs.push(...xs.map((x,i)=>add('square'+i,'arithmetic',{expression:'x1 * x1'},[x])))
    let x=xs.reduce((a,b,i)=>i===0?b:add('features'+i,'concat',{axis:1},[a,b]),''),width=xs.length
    if(model==='mlp'){x=relu('relu',linear('hidden',x,width,64));width=64}
    x=linear('output',x,width,raw.task==='classification'?raw.classLabels.length:1)
    if(raw.task==='binary-classification')x=add('probability','activation',{activation:'sigmoid'},[x])
    add('loss','loss',{loss:raw.task==='regression'?'mse':raw.task==='classification'?'cross-entropy':'binary-cross-entropy'},[x,['data',raw.features]])
    await save(`${name}-${model}-${seed}`,graph,120,32)
    if(name==='mpg'&&model==='quadratic'&&seed===137)for(const kind of ['l1','l2'])for(const strength of [.001,.01,.1,1]){
     const copy=structuredClone(graph);copy.nodes.find(n=>n.id==='loss').params={loss:'mse',regularization:kind,regularizationStrength:strength,regularizationParameterIds:['output_w']}
     await save(`mpg-${kind}-${strength}`,copy,120,32)
    }
   }
  }
 }
 for(const task of [0,1]){
  const raw=JSON.parse(await readFile(`output/babi-pilot/data/qa${task}.json`,'utf8'))
  const docs=raw.rows.map(r=>({text:r.facts.join(' ')+' '+r.question,facts:r.facts,question:r.question,label:r.answer,split:r.split==='train'?'train':'test'}))
  for(const kind of task===0?['counts-mlp','mean','ordered-mlp']:['ordered-mlp','sentence-mlp','memory-1']){
   const structured=kind==='sentence-mlp'||kind==='memory-1'
   const data=prepareTextDocuments(docs,`qa${task}.csv`,{task:'classification',representation:structured?'facts':kind==='counts-mlp'?'counts':'tokens',maxLength:task===0?40:80,fixedLength:kind==='ordered-mlp',maxFacts:64,factWords:10})
   // Export raw, importable CSV and prepared JSON for student data loading.
   await writeFile(root+`/data/qa${task}-${kind}.json`,JSON.stringify(data))
   const quote=s=>'"'+s.replaceAll('"','""')+'"'
   await writeFile(root+`/data/qa${task}-${kind}.csv`,['text,question,label,split',...docs.map(d=>[quote(structured?d.facts.join('\n'):d.text),quote(d.question),d.label,d.split].join(','))].join('\n'))
   await save(`qa${task}-${kind}-137`,buildQaModel(data,kind),30)
  }
 }
 const raw=await readFile('output/text-curriculum/data/alice-original.txt','utf8')
 const body=raw.split(/\*\*\* START OF[^\n]*\n/)[1]?.split(/\*\*\* END OF/)[0]??raw
 const paragraphs=body.split(/\n\s*\n/).map(p=>p.replace(/\s+/g,' ').trim()).filter(p=>p.length>=80)
 const boundary=Math.floor(paragraphs.length*.8),final=Math.floor(paragraphs.length*.9)
 const documents=[...paragraphs.slice(0,boundary).map(text=>({text,split:'train'})),...paragraphs.slice(boundary,final).map(text=>({text,split:'test'}))]
 // Character corpus, same next-character examples for every architecture; last 10% withheld.
 const alice=prepareTextDocuments(documents,'alice.txt',{task:'language',tokenizer:'character',maxLength:16,stride:8,fixedLength:true,targetMode:'last',vocabularySize:128})
 await writeFile(root+'/data/alice.json',JSON.stringify(alice));await writeFile(root+'/data/alice-final-passages.json',JSON.stringify(paragraphs.slice(final)))
 for(const seed of [137,211,307])for(const kind of ['bigram','mlp','attention','transformer'])await save(`alice-${kind}-${seed}`,buildAliceModel(alice,kind,seed),35,64)
 await writeFile(root+'/manifest.json',JSON.stringify(manifest,null,2))
}finally{await server.close()}
