import {mkdir,readFile,writeFile,copyFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createHash} from 'node:crypto'
import assert from 'node:assert/strict'
import {createServer} from 'vite'
const root=resolve(process.env.FINAL_IMDB_OUTPUT||'output/imdb-final'),source=resolve('output/imdb-two-head'),key='n20000-v4000'
const read=path=>readFile(path,'utf8').then(JSON.parse)
const data=await read(join(source,'data',key+'-prepared.json'));data.maxLength=256;data.stride=256
const contextRoot=join(root,'context256');await mkdir(join(contextRoot,'data'),{recursive:true});await mkdir(join(contextRoot,'templates'),{recursive:true})
const protocol=await read(join(source,'protocol.json'));protocol.context=256;protocol.selection='Choose context 128 or 256 by three-seed mean validation loss, then freeze all model configurations before fresh held-out evaluation.'
await writeFile(join(contextRoot,'protocol.json'),JSON.stringify(protocol,null,2))
await writeFile(join(root,'plan.json'),JSON.stringify({seeds:[137,211,307],contexts:[128,256],contextSelection:'three-seed mean validation loss; use 128 on a tie',models:['counts-linear','counts-mlp','mean','transformer-two-head'],training:20000,validation:1000,vocabulary:4000,holdout:5000,holdoutSelectionSeed:20260927,excludePriorOfficialTest:true,learningRate:.001,dropout:0,batchSize:32,maxEpochs:30,patience:3,minDelta:.001,selectionUsesTest:false},null,2))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {encodeText}=await server.ssrLoadModule('/src/domain/textData.ts')
 const {initializeTensor}=await server.ssrLoadModule('/src/domain/authoring.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const rows=data.documents.map(d=>({ids:encodeText(d.text,data).slice(0,256),target:Number(d.label==='positive'),split:d.split==='train'?'train':'validation'}))
 const old=await read(join(source,'data',key+'-numeric.json'))
 assert.deepEqual(rows.map(r=>({...r,ids:r.ids.slice(0,128)})),old.rows)
 const numeric=JSON.stringify({vocabulary:data.vocabulary,maxLength:256,rows}),stats={training:20000,validation:1000,vocabulary:4000,numericSha256:createHash('sha256').update(numeric).digest('hex')}
 await writeFile(join(contextRoot,'data',key+'-numeric.json'),numeric)
 await writeFile(join(contextRoot,'data',key+'-stats.json'),JSON.stringify(stats,null,2))
 await writeFile(join(contextRoot,'data',key+'-prepared.json'),JSON.stringify(data))
 await copyFile(join(source,'data','manifest-n20000.json'),join(contextRoot,'data','manifest-n20000.json'))
 for(const seed of protocol.seeds){
  const template=await read(join(source,'templates',`${key}-seed${seed}.json`)),graph=template.graph
  const positions=graph.nodes.find(n=>n.id==='position-embeddings'),value=initializeTensor([256,32],'uniform',seed+1);value.data=value.data.map(x=>x*5)
  assert.deepEqual(value.data.slice(0,128*32),positions.params.value.data);positions.params.value=value
  graph.nodes.find(n=>n.type==='dataset').params.textData=data
  template.expectedLoss=forwardPass(graph,false).loss
  graph.nodes.find(n=>n.type==='dataset').params.textData={...data,documents:[]}
  await writeFile(join(contextRoot,'templates',`${key}-seed${seed}.json`),JSON.stringify(template))
 }
 console.log('Prepared context256: same token prefixes and shared initial weights; only additional position rows introduced.')
}finally{await server.close()}
