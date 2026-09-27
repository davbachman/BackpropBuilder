import {existsSync} from 'node:fs'
if(existsSync('output/semester-pilot/frozen.json'))throw Error('Pilot is frozen. Use a separate checkout/output directory for a new study.')
import {readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {buildQaModel}=await server.ssrLoadModule('/src/test/curriculumModels.ts'),{forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts'),{parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 const manifest=JSON.parse(await readFile(root+'/manifest.json','utf8'))
 for(const row of manifest.filter(r=>r.name.startsWith('qa')&&r.name.endsWith('-137')))for(const seed of [211,307]){
  const original=JSON.parse(await readFile(root+'/templates/'+row.name+'.json','utf8')),kind=row.name.slice(4,-4),graph=buildQaModel(original.graph.nodes[0].params.textData,kind,seed),name=row.name.replace(/137$/,String(seed))
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  await writeFile(root+'/templates/'+name+'.json',JSON.stringify({...original,name,graph,expressions,expectedLoss:forwardPass(graph,false).loss}));await writeFile(root+'/numeric/'+name+'.json',await readFile(root+'/numeric/'+row.name+'.json'))
  manifest.push({...row,name})
 }
 await writeFile(root+'/manifest.json',JSON.stringify(manifest,null,2))
}finally{await server.close()}
