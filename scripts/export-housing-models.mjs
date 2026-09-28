import {readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/housing-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts'),{forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 for(const {name} of JSON.parse(await readFile(root+'/manifest.json','utf8'))){
  const template=JSON.parse(await readFile(root+'/templates/'+name+'.json','utf8')),run=JSON.parse(await readFile(root+'/runs/'+name+'.json','utf8'))
  for(const initial of [false,true]){
   const graph=initial?template.graph:{...template.graph,nodes:template.graph.nodes.map(n=>n.id==='output_w'||n.id==='output_b'?{...n,params:{...n.params,value:{shape:n.id==='output_w'?[template.features.length,1]:[1],data:n.id==='output_w'?run.weights:run.bias}}}:n)}
   const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:parameterValues(template.graph),selectedNodeIds:['standard'],phase:'edit',traceSteps:[],traceIndex:0,epoch:initial?0:run.bestEpoch,currentLoss:forwardPass(graph,false).loss??null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
   const json=JSON.stringify(file);if(!parseProjectStateFile(json).ok)throw Error('Invalid model '+name)
   await writeFile(root+'/models/'+(initial?'initial-':'')+name+'.json',json)
  }
 }
 console.log('Exported initial and trained projects with saved normalization statistics.')
}finally{await server.close()}
