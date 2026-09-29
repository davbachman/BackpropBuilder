import {createRoot} from 'react-dom/client'
import {useState} from 'react'
import {tf,TensorGraph,TensorOptimizer,selectTensorBackend,trainTensorGraph} from '../../src/domain/tensorTraining'
import {forwardPass,parameterValues} from '../../src/domain/engine'
import {createProjectStateFile,parseProjectStateFile} from '../../src/domain/session'
import {generateText} from '../../src/domain/textGeneration'
import type {GraphModel} from '../../src/domain/types'
const base='/NeuralCanvas/output/semester-pilot/'
async function read(path:string){const r=await fetch(base+path);if(!r.ok)throw Error('Missing '+path);return r.json()}
async function save(name:string,result:unknown){const response=await fetch('/pilot-result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,result})});if(!response.ok)throw Error('Could not save browser result')}
const maxError=(a:ArrayLike<number>,b:ArrayLike<number>)=>Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0)
export default function App(){
 const [name,setName]=useState('mpg-l2-0.01'),[status,setStatus]=useState('Ready'),[busy,setBusy]=useState(false),[results,setResults]=useState<unknown[]>([])
 const [epochs,setEpochs]=useState(30)
 async function check(name:string){
  const template=await read('templates/'+name+'.json'),graph:GraphModel=template.graph,native=await read('runs/'+name+'.json'),parity=await read('runs/'+name+'-parity.json')
  await selectTensorBackend(graph,'webgl',graph.training)
  const model=new TensorGraph(graph),optimizer=new TensorOptimizer(graph.training!,graph.learningRate)
  let gradientError=0,updateError=0
  const updateDetails:unknown[]=[]
  try{
   const rows=model.examples.filter(r=>r.split==='train').slice(0,1),g=model.gradients(rows)
   const initial=(await g.loss.data())[0]
   for(const [id,v] of model.variables)gradientError=Math.max(gradientError,maxError(await g.grads[v.name].data(),parity.gradients[id]))
   tf.dispose([g.loss,...Object.values(g.grads)])
   for(let i=0;i<3;i++){const g=model.gradients(rows);optimizer.update(model.variables,g.grads);tf.dispose([g.loss,...Object.values(g.grads)])}
   for(const [id,v] of model.variables){const a=await v.data(),e=maxError(a,parity.weights[id].data);updateError=Math.max(updateError,e);if(e>3e-4){let index=0;for(let i=0;i<a.length;i++)if(Math.abs(a[i]-parity.weights[id].data[i])>Math.abs(a[index]-parity.weights[id].data[index]))index=i;updateDetails.push({id,error:e,index,firstNativeGradient:parity.gradients[id][index],app:a[index],native:parity.weights[id].data[index]})}}
   if(updateError>3e-4){await save(name+'-update-diagnostic',{updateDetails,after:await model.inspect(rows)});}
   const stepReference=await read('runs/'+name+'-steps.json'),after=await model.inspect(rows),predictionUpdateError=maxError(after.prediction,stepReference.steps[3].prediction)
   if(Math.abs(initial-template.expectedLoss)>1e-4||gradientError>2e-4||predictionUpdateError>2e-4)throw Error(JSON.stringify({initial,expected:template.expectedLoss,gradientError,updateError}))
   const trained={...graph,nodes:graph.nodes.map(n=>native.weights[n.id]?{...n,params:{...n.params,value:native.weights[n.id]}}:n)}
   const resultModel=new TensorGraph(trained)
   try{
    const start=performance.now(),validation=await resultModel.evaluate(resultModel.examples.filter(r=>r.split==='test'),64)
    if(Math.abs(validation.loss-native.validation.loss)>2e-4)throw Error('Validation mismatch '+validation.loss+' vs '+native.validation.loss)
    const file=createProjectStateFile({graph:trained,visualizationGraph:trained,initialParameterValues:parameterValues(graph),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:native.bestEpoch,currentLoss:forwardPass(trained,false).loss??null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
    if(!parseProjectStateFile(JSON.stringify(file)).ok)throw Error('Project import failed')
    const sample=name.startsWith('alice')?generateText(trained,'alice was ',80,{sample:true,temperature:.8,topK:8,random:()=>.43}):undefined
    const result={name,backend:tf.getBackend(),initial,gradientError,updateError,predictionUpdateError,updateDetails,validation,seconds:(performance.now()-start)/1000,projectRoundTrip:true,sample}
    await save(name+'-check',result);setResults(r=>[...r,result])
   }finally{resultModel.dispose()}
  }finally{optimizer.dispose();model.dispose()}
 }
 async function run(all=false,training=false){setBusy(true);try{
  if(training){const {graph,batch}=await read('templates/'+name+'.json');const start=performance.now();const result=await trainTensorGraph(graph,{epochs,batchSize:batch,settings:{...graph.training,backend:'webgl'},onReport:r=>setStatus(`${name}: epoch ${r.epoch}, validation ${r.validation.loss.toFixed(4)}, accuracy ${(r.validation.accuracy*100).toFixed(1)}%`)});await save(name+'-train',{...result,seconds:(performance.now()-start)/1000});setResults(r=>[...r,{name,seconds:(performance.now()-start)/1000,completed:result.completed,bestEpoch:result.bestEpoch,last:result.reports.at(-1)}])}
  else if(all){for(const item of await read('manifest.json')){if(item.name.endsWith('-137')||item.name.startsWith('mpg-l')){setStatus('Checking '+item.name);await check(item.name)}}}
  else await check(name)
  setStatus('Completed')
 }catch(e){setStatus('ERROR: '+String(e));await save('error',{name,error:String(e)})}finally{setBusy(false)}}
 return <main style={{fontFamily:'system-ui',padding:32,maxWidth:1100}}><h1>Semester app pilot</h1><p>Uses the app’s WebGL engine, project importer and text generator. Native loss, gradient and three-update parity; full validation of trained checkpoints.</p><label>Recipe <input aria-label="Recipe" value={name} onChange={e=>setName(e.target.value)}/></label><label>Epochs <input aria-label="Epochs" type="number" value={epochs} onChange={e=>setEpochs(Number(e.target.value))}/></label><p><button disabled={busy} onClick={()=>void run()}>Check recipe</button> <button disabled={busy} onClick={()=>void run(true)}>Check all recipes</button> <button disabled={busy} onClick={()=>void run(false,true)}>Train from initialization</button></p><p role="status">{status}</p><pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(results,null,2)}</pre></main>
}
createRoot(document.getElementById('root')!).render(<App/>);
