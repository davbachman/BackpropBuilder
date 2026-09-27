import {readFile,writeFile} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts'),{forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 for(const item of JSON.parse(await readFile(root+'/manifest.json','utf8'))){
  const template=JSON.parse(await readFile(root+'/templates/'+item.name+'.json','utf8'))
  let native;try{native=JSON.parse(await readFile(root+'/runs/'+item.name+'.json','utf8'))}catch{continue}
  const graph={...template.graph,nodes:template.graph.nodes.map(n=>native.weights[n.id]?{...n,params:{...n.params,value:native.weights[n.id]}}:n)}
  const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:parameterValues(template.graph),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:native.bestEpoch,currentLoss:forwardPass(graph,false).loss??null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
  const json=JSON.stringify(file);if(!parseProjectStateFile(json).ok)throw Error('Import check failed '+item.name)
  await writeFile(root+'/models/'+item.name+'.json',json)
 }
}finally{await server.close()}
