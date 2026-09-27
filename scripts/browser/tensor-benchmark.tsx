import {createRoot} from 'react-dom/client'
import {useState} from 'react'
import {tf,TensorGraph,TensorOptimizer,selectTensorBackend} from '../../src/domain/tensorTraining'
import {DEFAULT_TRAINING} from '../../src/domain/trainingSettings'
import {prepareTextDocuments} from '../../src/domain/textData'
import {buildTextModel} from '../../src/test/textModels'
import prepared from '../../output/text-curriculum/data/imdb-prepared.json'

export default function Check(){
 const [status,setStatus]=useState('Ready'),[results,setResults]=useState<unknown[]>([]),[busy,setBusy]=useState(false)
 async function run(){
  setBusy(true);setResults([])
  const documents=prepared.documents.slice(0,32).map((d,i)=>({...d,text:i%3===0?d.text.split(' ').slice(0,9).join(' '):d.text,split:i<28?'train':'test'}))
  const data=prepareTextDocuments(documents as Parameters<typeof prepareTextDocuments>[0],'gpu-check.csv',{task:'sentiment',maxLength:64,vocabularySize:1000,representation:'tokens'})
  const graph=buildTextModel(data,'transformer',16)
  let reference:number[]=[];let referenceGradients:number[]=[]
  for(const backend of ['cpu','webgl','webgpu'] as const){
   setStatus('Checking '+backend);await new Promise(resolve=>setTimeout(resolve,0))
   let model:TensorGraph|undefined,optimizer:TensorOptimizer|undefined
   try{
    await selectTensorBackend(graph,backend)
    model=new TensorGraph(graph);const rows=model.examples.slice(0,16)
    const before=await model.inspect(rows),grad=model.gradients(rows)
    const values=Array.from((await Promise.all([...model.variables.values()].map(v=>grad.grads[v.name].data()))).flatMap(a=>Array.from(a)))
    tf.dispose([grad.loss,...Object.values(grad.grads)])
    if(backend==='cpu'){reference=before.prediction;referenceGradients=values}
    const error=Math.max(...before.prediction.map((n,i)=>Math.abs(n-reference[i])))
    const gradientError=Math.max(...values.map((n,i)=>Math.abs(n-referenceGradients[i])))
    if(error>1e-4||gradientError>2e-4)throw Error('CPU/GPU mismatch '+error+' / '+gradientError)
    optimizer=new TensorOptimizer({...DEFAULT_TRAINING,engine:'tensor',backend},.001)
    for(let i=0;i<3;i++){const g=model.gradients(rows);optimizer.update(model.variables,g.grads);tf.dispose([g.loss,...Object.values(g.grads)])}
    await model.inspect(rows)
    const start=performance.now()
    for(let i=0;i<20;i++){
     const g=model.gradients(rows);optimizer.update(model.variables,g.grads);tf.dispose([g.loss,...Object.values(g.grads)])
     if(i%4===0)await new Promise(resolve=>setTimeout(resolve,0))
    }
    const after=await model.inspect(rows)
    if(!Number.isFinite(after.loss)) throw Error("Nonfinite loss after optimizer updates")
    setResults(current=>[...current,{backend,forwardMaxError:error,gradientMaxError:gradientError,lossBefore:before.loss,lossAfter:after.loss,milliseconds:performance.now()-start,updates:20,batch:16}])
   }catch(error){setResults(current=>[...current,{backend,error:String(error)}])}
   finally{optimizer?.dispose();model?.dispose()}
  }
  setStatus('Completed');setBusy(false)
 }
 return <main style={{fontFamily:'system-ui',padding:32,maxWidth:1000}}><h1>GPU training verification</h1><p>Same editable IMDb transformer graph, batch 16, mixed review lengths, AdamW, 20 measured updates after warmup.</p><button disabled={busy} onClick={()=>void run()}>Run GPU training checks</button><p role="status">{status}</p><pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(results,null,2)}</pre></main>
}
createRoot(document.getElementById('root')!).render(<Check/>);
