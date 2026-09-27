import {readFile,writeFile,readdir,access} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createHash} from 'node:crypto'
import assert from 'node:assert/strict'
import {createServer} from 'vite'
const root=resolve(process.env.FINAL_IMDB_OUTPUT||'output/imdb-final'),imdb=process.env.IMDB_ROOT||'/tmp/backprop-imdb/aclImdb'
const read=path=>readFile(path,'utf8').then(JSON.parse),sha=s=>createHash('sha256').update(s).digest('hex')
try{await access(join(root,'frozen.json'));throw Error('Study already frozen; refusing to replace the recorded selection.')}catch(error){if(error.code!=='ENOENT')throw error}
const plan=await read(join(root,'plan.json')),protocol=await read(join(root,'protocol.json')),selection=await read(join(root,'context-selection.json'))
const numeric=await readFile(join(root,'data/n20000-v4000-numeric.json'))
const data=JSON.parse(numeric),jobs=[]
assert(data.rows.every(r=>['train','validation'].includes(r.split)))
for(const kind of plan.models)for(const seed of plan.seeds){
 const name=`${kind}-seed${seed}`,result=await read(join(root,'runs',name+'-results.json'))
 assert.equal(result.finalTestUsed,false);assert.equal(result.numericSha256,sha(numeric))
 jobs.push({name,kind,seed,bestEpoch:result.bestEpoch,validation:result.validation,weightsSha256:sha(await readFile(join(root,'runs',name+'-weights.json'))),templateSha256:sha(await readFile(join(root,'templates',name+'.json')))})
}
const frozen={createdAt:new Date().toISOString(),plan,protocol,selection,numericSha256:sha(numeric),jobs,rule:'All four model families and all three seeds are fixed before fresh test data is loaded; no decisions will use test results.'}
const serialized=JSON.stringify(frozen,null,2);await writeFile(join(root,'frozen.json'),serialized,{flag:'wx'})
// Select a fresh balanced official-test subset, excluding every earlier recorded test/pilot review.
const old=await read(resolve('output/accelerated-imdb/manifest.json')),excluded=new Set(old.provenance.filter(d=>d.split==='final-test').map(d=>d.file))
const shuffle=(array,seed)=>{const a=[...array];for(let i=a.length-1;i>0;i--){seed=(1664525*seed+1013904223)>>>0;const j=Math.floor(seed/4294967296*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
const documents=[],files=[]
for(const label of ['neg','pos']){
 const dir=join(imdb,'test',label),sorted=(await readdir(dir)).filter(n=>n.endsWith('.txt')).sort((a,b)=>Number(a.split('_')[0])-Number(b.split('_')[0]))
 for(let i=0;i<100;i++)excluded.add('test/'+label+'/'+sorted[Math.floor(i*sorted.length/100)])
 const chosen=shuffle(sorted.filter(n=>!excluded.has('test/'+label+'/'+n)),plan.holdoutSelectionSeed).slice(0,plan.holdout/2)
 for(const name of chosen){files.push('test/'+label+'/'+name);documents.push({text:await readFile(join(dir,name),'utf8'),label:label==='pos'?'positive':'negative'})}
}
assert.equal(new Set(files).size,plan.holdout);assert(files.every(f=>!excluded.has(f)))
const prepared=await read(join(root,'data/n20000-v4000-prepared.json'))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {encodeText}=await server.ssrLoadModule('/src/domain/textData.ts')
 const rows=documents.map(d=>({ids:encodeText(d.text,prepared).slice(0,protocol.context),target:Number(d.label==='positive')})),raw=JSON.stringify({rows,maxLength:protocol.context})
 await writeFile(join(root,'holdout-numeric.json'),raw)
 await writeFile(join(root,'holdout-manifest.json'),JSON.stringify({source:'Official IMDb test split',frozenSha256:sha(serialized),numericSha256:sha(raw),examples:rows.length,seed:plan.holdoutSelectionSeed,excludedPriorTestFiles:[...excluded].sort(),files},null,2))
 console.log('Frozen',jobs.length,'checkpoints before preparing',rows.length,'fresh held-out reviews; excluded',excluded.size,'previous test reviews.')
}finally{await server.close()}
