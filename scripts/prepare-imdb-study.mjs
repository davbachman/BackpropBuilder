import {mkdir,readFile,readdir,writeFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createServer} from 'vite'
const root=resolve(process.env.IMDB_STUDY_OUTPUT||'output/accelerated-imdb')
const imdb=process.env.IMDB_ROOT||'/tmp/backprop-imdb/aclImdb'
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
 const {prepareTextDocuments,encodeText}=await server.ssrLoadModule('/src/domain/textData.ts')
 const {buildTextModel}=await server.ssrLoadModule('/src/test/textModels.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 await mkdir(join(root,'templates'),{recursive:true})
 const documents=[],test=[],provenance=[]
 const shuffle=(array,seed)=>{const a=[...array];for(let i=a.length-1;i>0;i--){seed=(1664525*seed+1013904223)>>>0;const j=Math.floor(seed/4294967296*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
 for(const label of ['neg','pos']) {
  const dir=join(imdb,'train',label),names=shuffle((await readdir(dir)).filter(n=>n.endsWith('.txt')).sort(),731)
  for(let i=0;i<3000;i++){documents.push({text:await readFile(join(dir,names[i]),'utf8'),label:label==='pos'?'positive':'negative',split:i<2500?'train':'test'});provenance.push({file:'train/'+label+'/'+names[i],split:i<2500?'train':'validation'})}
  const td=join(imdb,'test',label),sorted=(await readdir(td)).filter(n=>n.endsWith('.txt')).sort((a,b)=>Number(a.split('_')[0])-Number(b.split('_')[0]))
  const prior=new Set(Array.from({length:100},(_,i)=>sorted[Math.floor(i*sorted.length/100)]))
  const fresh=shuffle(sorted.filter(n=>!prior.has(n)),982)
  for(const name of fresh.slice(0,500)){test.push({text:await readFile(join(td,name),'utf8'),label:label==='pos'?'positive':'negative',split:'test'});provenance.push({file:'test/'+label+'/'+name,split:'final-test'})}
 }
 const data=prepareTextDocuments(documents,'imdb-study.csv',{task:'sentiment',representation:'tokens',vocabularySize:2000,maxLength:128})
 const rows=[...documents,...test].map((d,i)=>({ids:encodeText(d.text,data).slice(0,data.maxLength),target:Number(d.label==='positive'),split:i>=documents.length?'final-test':d.split==='train'?'train':'validation'}))
 await writeFile(join(root,'prepared.json'),JSON.stringify(data))
 await writeFile(join(root,'numeric-data.json'),JSON.stringify({vocabulary:data.vocabulary,maxLength:data.maxLength,rows}))
 await writeFile(join(root,'manifest.json'),JSON.stringify({source:'https://ai.stanford.edu/~amaas/data/sentiment/',seed:731,training:5000,validation:1000,finalTest:1000,priorPilotTestExcluded:true,provenance},null,2))
 const csv=['text,label,split',...documents.map(d=>'"'+d.text.replaceAll('"','""')+'",'+d.label+','+d.split)].join('\n')
 await writeFile(join(root,'imdb-study.csv'),csv)
 for(const seed of [137,211,307]) for(const kind of ['counts-linear','mean','attention','position-attention','transformer']) {
  const graph=buildTextModel(data,kind,32,seed);graph.learningRate=.0003
  graph.training={engine:'tensor',backend:'auto',optimizer:'adamw',weightDecay:.01,clipNorm:1,patience:3,minDelta:.001}
  const expectedLoss=forwardPass(graph,false).loss
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  // Shared raw data stays in prepared.json; these templates are inputs to the study runner, not importable projects.
  graph.nodes[0].params.textData={...graph.nodes[0].params.textData,documents:[]}
  await writeFile(join(root,'templates',kind+'-'+seed+'.json'),JSON.stringify({kind,seed,graph,expressions,expectedLoss}))
 }
 console.log('Prepared 5000 training / 1000 validation / 1000 fresh final-test reviews; vocabulary '+data.vocabulary.length)
}finally{await server.close()}
