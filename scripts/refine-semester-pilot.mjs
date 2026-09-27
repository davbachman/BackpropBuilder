import {existsSync} from 'node:fs'
if(existsSync('output/semester-pilot/frozen.json'))throw Error('Pilot is frozen. Use a separate checkout/output directory for a new study.')
import {readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
const manifest=JSON.parse(await readFile(root+'/manifest.json','utf8'))
const {buildTextModel}=await server.ssrLoadModule('/src/test/textModels.ts'),{datasetExamplesForNode}=await server.ssrLoadModule('/src/domain/datasets.ts'),{forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts'),{parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
for(const seed of [137,211,307]){
 const iris=JSON.parse(await readFile(root+`/templates/iris-linear-${seed}.json`,'utf8'));iris.name=`iris-faster-${seed}`;iris.graph.learningRate=.01;iris.epochs=200
 await writeFile(root+'/templates/'+iris.name+'.json',JSON.stringify(iris));await writeFile(root+'/numeric/'+iris.name+'.json',await readFile(root+`/numeric/iris-linear-${seed}.json`));manifest.push({name:iris.name,epochs:200,batch:32})
 const original=JSON.parse(await readFile(root+`/templates/alice-transformer-${seed}.json`,'utf8')),data={...original.graph.nodes.find(n=>n.type==='dataset').params.textData,targetMode:undefined}
 const graph=buildTextModel(data,'alice-transformer',32,seed);graph.learningRate=.001;graph.training=original.graph.training
 const name=`alice-alltokens-${seed}`,rows=datasetExamplesForNode(graph.nodes.find(n=>n.type==='dataset'))
 const numeric=rows.map(r=>({features:r.features,target:r.target,split:r.split==='train'?'train':'validation'}))
 const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
 await writeFile(root+'/templates/'+name+'.json',JSON.stringify({name,graph,expressions,expectedLoss:forwardPass(graph,false).loss,epochs:35,batch:64}))
 await writeFile(root+'/numeric/'+name+'.json',JSON.stringify(numeric));manifest.push({name,epochs:35,batch:64})
}
await writeFile(root+'/manifest.json',JSON.stringify(manifest,null,2))
}finally{await server.close()}
