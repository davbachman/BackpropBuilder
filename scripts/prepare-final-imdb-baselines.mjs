import {mkdir,readFile,writeFile,copyFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createServer} from 'vite'
const root=resolve(process.env.FINAL_IMDB_OUTPUT||'output/imdb-final'),key='n20000-v4000'
const read=path=>readFile(path,'utf8').then(JSON.parse)
const plan=await read(join(root,'plan.json')),comparisons=[]
for(const context of [128,256]){
 const folder=context===128?resolve('output/imdb-two-head'):join(root,'context256')
 const results=await Promise.all(plan.seeds.map(seed=>read(join(folder,'runs',`${key}-seed${seed}-results.json`))))
 comparisons.push({context,folder,meanValidationLoss:results.reduce((s,r)=>s+r.validation.loss,0)/3,meanValidationAccuracy:results.reduce((s,r)=>s+r.validation.accuracy,0)/3})
}
const chosen=[...comparisons].sort((a,b)=>a.meanValidationLoss-b.meanValidationLoss||a.context-b.context)[0]
await writeFile(join(root,'context-selection.json'),JSON.stringify({comparisons,chosen,rule:plan.contextSelection,usedTest:false},null,2))
await mkdir(join(root,'data'),{recursive:true});await mkdir(join(root,'templates'),{recursive:true});await mkdir(join(root,'runs'),{recursive:true})
for(const suffix of ['prepared','numeric','stats'])await copyFile(join(chosen.folder,'data',`${key}-${suffix}.json`),join(root,'data',`${key}-${suffix}.json`))
await copyFile(join(chosen.folder,'data','manifest-n20000.json'),join(root,'data','manifest-n20000.json'))
const protocol=await read(join(chosen.folder,'protocol.json'));protocol.models=plan.models
await writeFile(join(root,'protocol.json'),JSON.stringify(protocol,null,2))
for(const seed of plan.seeds){
 await copyFile(join(chosen.folder,'templates',`${key}-seed${seed}.json`),join(root,'templates',`transformer-two-head-seed${seed}.json`))
 for(const suffix of ['results','weights'])await copyFile(join(chosen.folder,'runs',`${key}-seed${seed}-${suffix}.json`),join(root,'runs',`transformer-two-head-seed${seed}-${suffix}.json`))
}
const data=await read(join(root,'data',key+'-prepared.json'))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {buildTextModel}=await server.ssrLoadModule('/src/test/textModels.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 for(const kind of plan.models.filter(k=>k!=='transformer-two-head'))for(const seed of plan.seeds){
  const graph=buildTextModel(data,kind,32,seed);graph.learningRate=protocol.learningRate
  graph.training={engine:'tensor',backend:'auto',optimizer:'adamw',weightDecay:protocol.weightDecay,clipNorm:protocol.clipNorm,patience:protocol.patience,minDelta:protocol.minDelta}
  const expectedLoss=forwardPass(graph,false).loss
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  graph.nodes.find(n=>n.type==='dataset').params.textData={...graph.nodes.find(n=>n.type==='dataset').params.textData,documents:[]}
  await writeFile(join(root,'templates',`${kind}-seed${seed}.json`),JSON.stringify({graph,expressions,expectedLoss,seed,kind}))
 }
 console.log(JSON.stringify({comparisons,chosen},null,2))
}finally{await server.close()}
