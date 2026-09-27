import {mkdir,readFile,writeFile,copyFile} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createServer} from 'vite'
const source=resolve(process.env.DATA_VOCAB_OUTPUT||'output/imdb-data-vocab'),root=resolve(process.env.TWO_HEAD_OUTPUT||'output/imdb-two-head')
const read=path=>readFile(path,'utf8').then(JSON.parse)
const key='n20000-v4000',protocol=await read(join(source,'protocol.json'))
await mkdir(join(root,'data'),{recursive:true});await mkdir(join(root,'templates'),{recursive:true})
await writeFile(join(root,'protocol.json'),JSON.stringify({...protocol,sizes:[20000],vocabularies:[4000],attentionHeads:2,headWidth:16,baselineRoot:source,selection:'Compare two heads against the existing one-head control, using the same three seeds and mean validation loss; report accuracy separately.'},null,2))
for(const suffix of ['prepared','numeric','stats'])await copyFile(join(source,'data',`${key}-${suffix}.json`),join(root,'data',`${key}-${suffix}.json`))
await copyFile(join(source,'data','manifest-n20000.json'),join(root,'data','manifest-n20000.json'))
const data=await read(join(root,'data',key+'-prepared.json'))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {withTwoAttentionHeads}=await server.ssrLoadModule('/src/test/textModels.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 for(const seed of protocol.seeds){
  const template=await read(join(source,'templates',`${key}-seed${seed}.json`)),graph=withTwoAttentionHeads(template.graph)
  graph.nodes.find(n=>n.type==='dataset').params.textData=data
  const expectedLoss=forwardPass(graph,false).loss
  if(!Number.isFinite(expectedLoss))throw Error('Invalid two-head graph')
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  graph.nodes.find(n=>n.type==='dataset').params.textData={...data,documents:[]}
  await writeFile(join(root,'templates',`${key}-seed${seed}.json`),JSON.stringify({graph,expressions,expectedLoss,seed,key}))
  console.log('Prepared two heads, seed',seed,'trace loss',expectedLoss)
 }
}finally{await server.close()}
