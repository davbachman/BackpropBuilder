import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/assignment-01-mpg'
await mkdir(root+'/models',{recursive:true})
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {builder}=await server.ssrLoadModule('/src/test/curriculumModels.ts')
 const {parseCustomCsv}=await server.ssrLoadModule('/src/domain/customCsv.ts')
 const {parameterValues,forwardPass,runTrainingStepFast}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {withDatasetIndices,withDatasetExample,evaluateDataset}=await server.ssrLoadModule('/src/domain/datasetTraining.ts')
 const {datasetExamplesForNode}=await server.ssrLoadModule('/src/domain/datasets.ts')
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const csv=parseCustomCsv(await readFile(root+'/mpg.csv','utf8'),'mpg.csv')
 const report=[]
 for(const name of ['one_feature','two_features','quadratic'])for(const seed of [137,211,307]){
  const {graph,add,weight}=builder(seed)
  graph.learningRate=.05;graph.training={engine:'trace',backend:'auto',optimizer:'sgd',weightDecay:0,clipNorm:0,patience:0,minDelta:0}
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'batch',datasetSplit:'train'})
  const x=add('scaled-weight','arithmetic',{expression:'(x1 - 3000) / 1000'},[['data',0]])
  const z=name==='one_feature'?null:add('scaled-year','arithmetic',{expression:'(x1 - 1976) / 10'},[['data',1]])
  const square=name==='quadratic'?add('weight-squared','arithmetic',{expression:'x1 ^ 2'},[x]):null
  const inputs=[x,...(z?[z]:[]),...(square?[square]:[])]
  const terms=inputs.map((input,i)=>add('term'+i,'arithmetic',{expression:'x1 * x2'},[input,weight('w'+i,[],'uniform')]))
  const prediction=add('prediction','arithmetic',{expression:[...terms,'bias'].map((_,i)=>'x'+(i+1)).join(' + ')},[...terms,weight('bias',[],'zeros')])
  add('loss','loss',{loss:'mse'},[prediction,['data',2]])
  const initial=parameterValues(graph)
  const indices=datasetExamplesForNode(graph.nodes[0]).flatMap((r,i)=>r.split==='train'?[i]:[])
  let trained=withDatasetIndices(graph,'data',indices)
  const start=Date.now()
  for(let epoch=0;epoch<1000;epoch++)trained=runTrainingStepFast(trained,.05)
  trained=withDatasetExample(trained,'data',indices[0])
  const metrics={name,seed,seconds:(Date.now()-start)/1000,train:evaluateDataset(trained,'data','train').loss,validation:evaluateDataset(trained,'data','test').loss,parameters:parameterValues(trained)}
  report.push(metrics);console.log(name,seed,metrics.train,metrics.validation,metrics.seconds)
  if(seed===137)for(const [prefix,g]of [['',trained],['initial-',withDatasetExample(graph,'data',indices[0])]]){
   const file=createProjectStateFile({graph:g,visualizationGraph:g,initialParameterValues:initial,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:prefix?0:1000,currentLoss:forwardPass(g,false).loss??null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
   const json=JSON.stringify(file);if(!parseProjectStateFile(json).ok)throw Error('Invalid export')
   await writeFile(root+'/models/'+prefix+name+'.json',json)
  }
 }
 await writeFile(root+'/app-pilot.json',JSON.stringify(report,null,2))
}finally{await server.close()}
