import {existsSync} from 'node:fs'
if(existsSync('output/semester-pilot/frozen.json'))throw Error('Pilot is frozen. Use a separate checkout/output directory for a new study.')
import {readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {builder}=await server.ssrLoadModule('/src/test/curriculumModels.ts'),{parseCustomCsv}=await server.ssrLoadModule('/src/domain/customCsv.ts'),{datasetExamplesForNode}=await server.ssrLoadModule('/src/domain/datasets.ts'),{forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts'),{parseArithmetic}=await server.ssrLoadModule('/src/domain/arithmetic.ts')
 const raw=JSON.parse(await readFile(root+'/data/mpg.json','utf8')),train=raw.rows.filter(r=>r.split==='train').slice(0,40),validation=raw.rows.filter(r=>r.split==='validation')
 const mean=Array.from({length:6},(_,i)=>train.reduce((s,r)=>s+r.x[i],0)/train.length),scale=mean.map((m,i)=>Math.sqrt(train.reduce((s,r)=>s+(r.x[i]-m)**2,0)/train.length))
 const ym=train.reduce((s,r)=>s+r.y,0)/train.length,ys=Math.sqrt(train.reduce((s,r)=>s+(r.y-ym)**2,0)/train.length)
 const csvtext=['cylinders,displacement,horsepower,weight,acceleration,model_year,target,split',...[...train,...validation].map(r=>[...r.x.map((x,i)=>(x-mean[i])/scale[i]),(r.y-ym)/ys,r.split==='train'?'train':'test'].join(','))].join('\n')
 await writeFile(root+'/data/mpg-small.csv',csvtext)
 await writeFile(root+'/data/mpg-small-scaling.json',JSON.stringify({mean,scale,targetMean:ym,targetScale:ys,sourceTrainingIndices:train.map(r=>r.sourceIndex)}))
 const csv=parseCustomCsv(csvtext,'mpg-small.csv'),manifest=JSON.parse(await readFile(root+'/manifest.json','utf8'))
 for(const seed of [137,211,307])for(const strength of [0,.001,.01,.1,1]){
  const {graph,add,reshape,linear}=builder(seed);graph.learningRate=.003;graph.training={...graph.training,weightDecay:0,patience:30,minDelta:.0001}
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
  const xs=Array.from({length:6},(_,i)=>reshape('col'+i,add('x'+i,'input',{},[['data',i]]),[1,1])),terms=[...xs]
  for(let i=0;i<6;i++)for(let j=i;j<6;j++)terms.push(add(`interaction-${i}-${j}`,'arithmetic',{expression:'x1 * x2'},[xs[i],xs[j]]))
  const features=terms.reduce((a,b,i)=>i===0?b:add('features'+i,'concat',{axis:1},[a,b]),''),output=linear('output',features,terms.length,1)
  add('loss','loss',{loss:'mse',regularization:'l2',regularizationStrength:strength,regularizationParameterIds:['output_w']},[output,['data',6]])
  const name=`mpg-small-l2-${strength}-${seed}`,rows=datasetExamplesForNode(graph.nodes[0]),numeric=rows.map(r=>({features:r.features,target:r.target,split:r.split==='train'?'train':'validation'})),expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression).expression]))
  await writeFile(root+'/templates/'+name+'.json',JSON.stringify({name,graph,expressions,expectedLoss:forwardPass(graph,false).loss,epochs:400,batch:32}));await writeFile(root+'/numeric/'+name+'.json',JSON.stringify(numeric));manifest.push({name,epochs:400,batch:32})
 }
 await writeFile(root+'/manifest.json',JSON.stringify(manifest,null,2))
}finally{await server.close()}
