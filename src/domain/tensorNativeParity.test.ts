import {expect,it,beforeAll,afterEach} from 'vitest'
import {spawnSync} from 'node:child_process'
import {resolve} from 'node:path'
import {tf,TensorGraph,TensorOptimizer} from './tensorTraining'
import {prepareTextDocuments} from './textData'
import {parseArithmetic} from './arithmetic'
import {buildTextModel,withTwoAttentionHeads,type TextModelKind} from '../test/textModels'
import {DEFAULT_TRAINING} from './trainingSettings'
const python=process.env.PYTORCH_TEST_PYTHON
beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
it.skipIf(!python).each(['counts-linear','mean','position-attention','transformer','two-head-transformer'])('matches native batched PyTorch loss, gradients and three AdamW updates: %s',async kind=>{
 const data=prepareTextDocuments([{text:'good funny good film',label:'positive',split:'train'},{text:'bad film',label:'negative',split:'train'},{text:'good film',label:'positive',split:'test'}],'test.csv',{task:'sentiment',maxLength:4})
 const graph=kind==='two-head-transformer'?withTwoAttentionHeads(buildTextModel(data,'transformer',4)):buildTextModel(data,kind as TextModelKind,4),model=new TensorGraph(graph),options={...DEFAULT_TRAINING,weightDecay:.01,clipNorm:1}
 const optimizer=new TensorOptimizer(options,.001)
 try {
  const rows=model.examples.slice(0,2),ids=rows.map(row=>data.representation==='tokens'?row.features[0].data:[])
  // Counts graphs still use the same tokenized documents in the independent interpreter.
  const {encodeText}=await import('./textData')
  ids.splice(0,ids.length,...data.documents.slice(0,2).map(doc=>encodeText(doc.text,data)))
  const expressions=Object.fromEntries(graph.nodes.filter(n=>n.type==='arithmetic').map(n=>[n.id,parseArithmetic(n.params.expression!).expression]))
  const input={template:{graph,expressions},ids}
  const program=`import sys,json,torch\nsys.path.insert(0,${JSON.stringify(resolve('scripts/lib'))})\nfrom text_tensor_model import TextTensorModel\ndata=json.load(sys.stdin)\nmodel=TextTensorModel(data['template'])\nlengths=torch.tensor([len(row) for row in data['ids']])\nids=torch.tensor([row+[0]*(int(lengths.max())-len(row)) for row in data['ids']])\ntarget=torch.tensor([1.,0.])\n_,loss=model(ids,lengths,target)\nloss.backward()\ninitial=float(loss.detach())\ngradients={key:p.grad.reshape(-1).tolist() for key,p in model.parameter_map.items()}\noptimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,eps=1e-8)\nfor _ in range(3):\n optimizer.zero_grad()\n _,loss=model(ids,lengths,target)\n loss.backward()\n torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)\n optimizer.step()\nprint(json.dumps({'loss':initial,'gradients':gradients,'weights':model.graph_weights()}))\n`
  const run=spawnSync(python!,['-c',program],{input:JSON.stringify(input),encoding:'utf8',timeout:30000})
  expect(run.status,run.stderr).toBe(0)
  const native=JSON.parse(run.stdout),first=model.gradients(rows)
  try {
   expect((await first.loss.data())[0]).toBeCloseTo(native.loss,5)
   for(const [id,v] of model.variables){const g=await first.grads[v.name].data();g.forEach((value,j)=>expect(value,id).toBeCloseTo(native.gradients[id][j],5))}
  }finally{tf.dispose([first.loss,...Object.values(first.grads)])}
  for(let i=0;i<3;i++){const g=model.gradients(rows);optimizer.update(model.variables,g.grads);tf.dispose([g.loss,...Object.values(g.grads)])}
  for(const [id,v] of model.variables){const values=await v.data();values.forEach((value,i)=>expect(value,id).toBeCloseTo(native.weights[id].data[i],4))}
 }finally{optimizer.dispose();model.dispose()}
},30000)
