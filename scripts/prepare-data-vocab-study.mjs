import {mkdir,readFile,readdir,writeFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {createServer} from 'vite'
const root=resolve(process.env.DATA_VOCAB_OUTPUT||'output/imdb-data-vocab')
const source=resolve(process.env.IMDB_STUDY_OUTPUT||'output/accelerated-imdb')
const imdb=process.env.IMDB_ROOT||'/tmp/backprop-imdb/aclImdb'
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
 const {prepareTextDocuments,encodeText}=await server.ssrLoadModule('/src/domain/textData.ts')
 const {buildTextModel}=await server.ssrLoadModule('/src/test/textModels.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 await mkdir(join(root,'templates'),{recursive:true});await mkdir(join(root,'data'),{recursive:true})
 const protocol={sizes:[5000,10000,20000],vocabularies:[2000,4000,8000],seeds:[137,211,307],validation:1000,context:128,width:32,learningRate:.001,dropout:0,optimizer:'AdamW',batchSize:32,weightDecay:.01,clipNorm:1,maxEpochs:30,patience:3,minDelta:.001,selection:'Compare all nine combinations by mean validation loss across three seeds; report accuracy separately.',finalTestUsed:false}
 await writeFile(join(root,'protocol.json'),JSON.stringify(protocol,null,2))
 const oldManifest=JSON.parse(await readFile(join(source,'manifest.json'),'utf8'))
 const oldData=JSON.parse(await readFile(join(source,'numeric-data.json'),'utf8'))
 const oldPrepared=JSON.parse(await readFile(join(source,'prepared.json'),'utf8'))
 const shuffle=(array,seed)=>{const a=[...array];for(let i=a.length-1;i>0;i--){seed=(1664525*seed+1013904223)>>>0;const j=Math.floor(seed/4294967296*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
 const classes={}
 for(const label of ['neg','pos']){
  const names=shuffle((await readdir(join(imdb,'train',label))).filter(n=>n.endsWith('.txt')).sort(),731)
  const validation=names.slice(2500,3000),train=[...names.slice(0,2500),...names.slice(3000)].slice(0,10000)
  const records=await Promise.all([...train,...validation].map(async name=>({file:'train/'+label+'/'+name,text:await readFile(join(imdb,'train',label,name),'utf8'),label:label==='pos'?'positive':'negative'})))
  classes[label]={train:records.slice(0,10000),validation:records.slice(10000)}
 }
 let previous=new Set()
 for(const size of protocol.sizes){
  const selected=['neg','pos'].flatMap(label=>[...classes[label].train.slice(0,size/2).map(d=>({...d,split:'train'})),...classes[label].validation.map(d=>({...d,split:'test'}))])
  const trainNames=new Set(selected.filter(d=>d.split==='train').map(d=>d.file)),valNames=selected.filter(d=>d.split==='test').map(d=>d.file)
  assert.equal(trainNames.size,size);assert.equal(valNames.length,1000);assert(valNames.every(n=>!trainNames.has(n)));assert([...previous].every(n=>trainNames.has(n)));previous=trainNames
  assert.deepEqual([...valNames].sort(),oldManifest.provenance.filter(d=>d.split==='validation').map(d=>d.file).sort())
  await writeFile(join(root,'data',`manifest-n${size}.json`),JSON.stringify(selected.map(({file,split})=>({file,split}))))
  for(const vocabularySize of protocol.vocabularies){
   const key=`n${size}-v${vocabularySize}`
   const data=prepareTextDocuments(selected.map(({text,label,split})=>({text,label,split})),key+'.csv',{task:'sentiment',representation:'tokens',vocabularySize,maxLength:128})
   assert.equal(data.vocabulary.length,vocabularySize)
   const rows=data.documents.map(d=>({ids:encodeText(d.text,data).slice(0,data.maxLength),target:Number(d.label==='positive'),split:d.split==='train'?'train':'validation'}))
   if(size===5000&&vocabularySize===2000){assert.deepEqual(data.vocabulary,oldPrepared.vocabulary);assert.deepEqual(rows,oldData.rows.filter(r=>r.split!=='final-test'))}
   const stats={training:size,validation:1000,vocabulary:vocabularySize}
   for(const split of ['train','validation']){const ids=rows.filter(r=>r.split===split).flatMap(r=>r.ids);stats[split+'UnknownRate']=ids.filter(id=>id===0).length/ids.length}
   await writeFile(join(root,'data',key+'-prepared.json'),JSON.stringify(data))
   const numeric=JSON.stringify({vocabulary:data.vocabulary,maxLength:128,rows})
   await writeFile(join(root,'data',key+'-numeric.json'),numeric)
   stats.numericSha256=createHash('sha256').update(numeric).digest('hex')
   await writeFile(join(root,'data',key+'-stats.json'),JSON.stringify(stats,null,2))
   for(const seed of protocol.seeds){
    const graph=buildTextModel(data,'transformer',32,seed);graph.learningRate=.001
    graph.training={engine:'tensor',backend:'auto',optimizer:'adamw',weightDecay:.01,clipNorm:1,patience:3,minDelta:.001}
    const expectedLoss=forwardPass(graph,false).loss
    const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
    graph.nodes[0].params.textData={...graph.nodes[0].params.textData,documents:[]}
    await writeFile(join(root,'templates',`${key}-seed${seed}.json`),JSON.stringify({graph,expressions,expectedLoss,seed,key}))
   }
   console.log(key,stats)
  }
 }
}finally{await server.close()}
