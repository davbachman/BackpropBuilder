import {readFile,writeFile} from 'node:fs/promises'
import {existsSync} from 'node:fs'
import {createServer} from 'vite'
const root='output/housing-pilot'
if(existsSync(root+'/frozen.json'))throw Error('Study is frozen.')
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {builder}=await server.ssrLoadModule('/src/test/curriculumModels.ts'),{fitStandardizer}=await server.ssrLoadModule('/src/domain/standardizationFit.ts'),{forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const manifest=JSON.parse(await readFile(root+'/manifest.json','utf8'))
 // Chosen using validation only: visibly shrinks nuisance coefficients without excessive loss.
 const strength=.05;const selections=[]
 for(const seed of [137,211,307]){
  const original=JSON.parse(await readFile(root+`/templates/all-l1-${strength}-${seed}.json`,'utf8')),run=JSON.parse(await readFile(root+`/runs/all-l1-${strength}-${seed}.json`,'utf8'))
  const ranked=run.weights.map((w,i)=>({i,magnitude:Math.abs(w)})).sort((a,b)=>b.magnitude-a.magnitude).map(x=>x.i)
  for(const count of [2,3]){
   const features=ranked.slice(0,count),{graph,add,reshape,linear}=builder(seed);graph.learningRate=original.graph.learningRate;graph.training=original.graph.training
   add('data','dataset',original.graph.nodes.find(n=>n.type==='dataset').params)
   const xs=features.map(i=>reshape('column'+i,add('x'+i,'input',{},[['data',i]]),[1,1]))
   const x=xs.reduce((a,b,i)=>i===0?b:add('features'+i,'concat',{axis:1},[a,b]),'')
   add('standard','standardize',{},[x]);graph.nodes.find(n=>n.id==='standard').label='Standardize features';graph.nodes.find(n=>n.id==='standard').params.standardization=await fitStandardizer(graph,'standard')
   const pred=linear('output','standard',count,1);add('loss','loss',{loss:'mse'},[pred,['data',8]])
   const name=`top${count}-${seed}`,template={name,graph,features,epochs:1500,batch:original.batch,expectedLoss:forwardPass(graph,false).loss}
   await writeFile(root+'/templates/'+name+'.json',JSON.stringify(template));manifest.push({name,seed,strength:0});selections.push({seed,count,features})
  }
 }
 await writeFile(root+'/selection.json',JSON.stringify({strength,selections},null,2));await writeFile(root+'/manifest.json',JSON.stringify(manifest));console.log(selections)
}finally{await server.close()}
