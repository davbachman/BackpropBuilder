import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {existsSync} from 'node:fs'
import {createServer} from 'vite'
const root='output/housing-pilot'
if(existsSync(root+'/frozen.json'))throw Error('Study is frozen.')
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {builder}=await server.ssrLoadModule('/src/test/curriculumModels.ts'),{parseCustomCsv}=await server.ssrLoadModule('/src/domain/customCsv.ts'),{fitStandardizer}=await server.ssrLoadModule('/src/domain/standardizationFit.ts'),{forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts'),{createProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const csv=parseCustomCsv(await readFile(root+'/housing.csv','utf8'),'california-housing.csv');csv.task='regression'
 for(const dir of ['templates','models','runs'])await mkdir(root+'/'+dir,{recursive:true})
 const manifest=[];let stats
 for(const seed of [137,211,307])for(const strength of [0,.01,.05,.1,.2]){
  const {graph,add,reshape,linear}=builder(seed);graph.learningRate=.03;graph.training={engine:'tensor',backend:'auto',optimizer:'adamw',weightDecay:0,clipNorm:0,patience:50,minDelta:.00001}
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
  const xs=Array.from({length:8},(_,i)=>reshape('column'+i,add('x'+i,'input',{},[['data',i]]),[1,1]))
  const features=xs.reduce((a,b,i)=>i===0?b:add('features'+i,'concat',{axis:1},[a,b]),'')
  add('standard','standardize',{},[features]);graph.nodes.find(n=>n.id==='standard').label='Standardize features'
  stats??=await fitStandardizer(graph,'standard');graph.nodes.find(n=>n.id==='standard').params.standardization=stats
  const pred=linear('output','standard',8,1);add('loss','loss',{loss:'mse',regularization:strength?'l1':'none',regularizationStrength:strength,regularizationParameterIds:['output_w']},[pred,['data',8]])
  const name=`all-l1-${strength}-${seed}`,template={name,graph,features:[0,1,2,3,4,5,6,7],epochs:1500,batch:stats.count,expectedLoss:forwardPass(graph,false).loss}
  await writeFile(root+'/templates/'+name+'.json',JSON.stringify(template));manifest.push({name,seed,strength})
  if(seed===137&&strength===.1){const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:parameterValues(graph),selectedNodeIds:['standard'],phase:'edit',traceSteps:[],traceIndex:0,epoch:0,currentLoss:null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}});await writeFile(root+'/models/initial-all-l1-0.1.json',JSON.stringify(file))}
 }
 await writeFile(root+'/stats.json',JSON.stringify(stats));await writeFile(root+'/manifest.json',JSON.stringify(manifest))
 console.log('Prepared',manifest.length,'models; training observations',stats.count)
}finally{await server.close()}
