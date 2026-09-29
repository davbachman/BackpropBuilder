import {createRoot} from 'react-dom/client'
import {useState} from 'react'
import {TensorGraph,selectTensorBackend,trainTensorGraph} from '../../src/domain/tensorTraining'
import {fitStandardizer} from '../../src/domain/standardizationFit'
import type {GraphModel} from '../../src/domain/types'
const root='/NeuralCanvas/output/housing-pilot/'
async function read(path:string){const r=await fetch(root+path);if(!r.ok)throw Error(path);return r.json()}
async function save(name:string,result:unknown){const r=await fetch('/pilot-result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,result})});if(!r.ok)throw Error('Save failed')}
export default function App(){
 const [status,setStatus]=useState('Ready'),[busy,setBusy]=useState(false),[name,setName]=useState('all-l1-0.05-137')
 async function run(train:boolean){setBusy(true);try{
  const template=await read('templates/'+name+'.json'),graph:GraphModel=template.graph
  const fitted=await fitStandardizer(graph,'standard'),expected=graph.nodes.find(n=>n.id==='standard')!.params.standardization!
  if(JSON.stringify(fitted)!==JSON.stringify(expected))throw Error('Fitted statistics mismatch')
  if(train){const start=performance.now();const result=await trainTensorGraph(graph,{epochs:template.epochs,batchSize:template.batch,settings:{...graph.training!,backend:'webgl'},onReport:r=>setStatus(`Epoch ${r.epoch}: train MSE ${r.train.loss.toFixed(5)}, validation MSE ${r.validation.loss.toFixed(5)}`)});await save(name+'-train',{...result,seconds:(performance.now()-start)/1000});setStatus('Training completed; best epoch '+result.bestEpoch)}
  else{
   const native=await read('runs/'+name+'.json');await selectTensorBackend(graph,'webgl',graph.training)
   const initial=new TensorGraph(graph);let initialLoss
   try{initialLoss=(await initial.inspect(initial.examples.filter(x=>x.split==='train').slice(0,1))).loss;if(Math.abs(initialLoss-template.expectedLoss)>1e-4)throw Error('Initial loss mismatch')}finally{initial.dispose()}
   const trained={...graph,nodes:graph.nodes.map(n=>n.id==='output_w'||n.id==='output_b'?{...n,params:{...n.params,value:{shape:n.id==='output_w'?[template.features.length,1]:[1],data:n.id==='output_w'?native.weights:native.bias}}}:n)}
   const model=new TensorGraph(trained)
   try{const validation=await model.inference(model.examples.filter(x=>x.split==='test'),512);if(Math.abs(validation.loss-native.validation)>2e-4)throw Error('Validation mismatch');await save(name+'-check',{initialLoss,validation:{loss:validation.loss,examples:validation.examples},stats:fitted});setStatus('Check passed: validation MSE '+validation.loss.toFixed(5))}finally{model.dispose()}
  }
 }catch(e){setStatus('ERROR '+String(e));await save('error',{name,error:String(e)})}finally{setBusy(false)}}
 return <main style={{fontFamily:'system-ui',padding:32}}><h1>Housing pilot</h1><label>Recipe <input aria-label="Recipe" value={name} onChange={e=>setName(e.target.value)}/></label><p><button disabled={busy} onClick={()=>void run(false)}>Check WebGL</button> <button disabled={busy} onClick={()=>void run(true)}>Train WebGL</button></p><p role="status">{status}</p></main>
}
createRoot(document.getElementById('root')!).render(<App/>);
