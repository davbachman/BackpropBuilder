import {afterEach,beforeAll,describe,expect,it} from 'vitest'
import {spawnSync} from 'node:child_process'
import {TensorGraph,tf,trainTensorGraph} from './tensorTraining'
import {createModelPreset} from './modelPresets'
import {backwardPass,forwardPass,validateGraph} from './engine'
import {evaluateDataset,withDatasetExample} from './datasetTraining'
import {DEFAULT_TRAINING} from './trainingSettings'
import {toTensor} from './tensor'
import type {GraphModel} from './types'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))

function broadcastGraph(operation:'multiply'|'add'|'arithmetic'='multiply',shape=[2],loss:'mse'|'mae'='mse') {
  const graph=createModelPreset('linear')
  graph.nodes.find(node=>node.type==='weight')!.params.value={shape,data:Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>i+1)}
  const product=graph.nodes.find(node=>node.type==='multiply')!
  product.type=operation
  if(operation==='arithmetic') product.params.expression='x1 * x2 + x1 / 2'
  graph.nodes.find(node=>node.type==='loss')!.params.loss=loss
  return graph
}

async function expectTraceParity(graph:GraphModel,batchSize:number) {
  const source=graph.nodes.find(node=>node.type==='dataset')!
  expect(validateGraph(graph).filter(issue=>issue.code!=='disconnected')).toEqual([])
  const model=new TensorGraph(graph)
  try {
    const rows=model.examples.slice(0,batchSize)
    const traced=rows.map((_,i)=>backwardPass(forwardPass(withDatasetExample(graph,source.id,i),false).graph))
    const actual=model.gradients(rows)
    try {
      expect((await actual.loss.data())[0]).toBeCloseTo(traced.reduce((sum,result)=>sum+result.loss!,0)/batchSize,4)
      for(const [id,variable] of model.variables) {
        const expected=traced.map(result=>result.graph.nodes.find(node=>node.id===id)!.grad!.data)
        Array.from(await actual.grads[variable.name].data()).forEach((value,i)=>
          expect(value,id).toBeCloseTo(expected.reduce((sum,gradient)=>sum+gradient[i],0)/batchSize,4))
      }
    } finally {tf.dispose([actual.loss,...Object.values(actual.grads)])}
    const loss=graph.nodes.find(node=>node.type==='loss')!
    const predictionId=graph.edges.find(edge=>edge.target===loss.id&&edge.inputSlot===0)!.source
    const expected=traced.flatMap(result=>result.graph.nodes.find(node=>node.id===predictionId)!.value!.data)
    const inspected=await model.inspect(rows)
    expect(inspected.prediction).toHaveLength(expected.length)
    inspected.prediction.forEach((value,i)=>expect(value).toBeCloseTo(expected[i],5))
  } finally {model.dispose()}
}

describe('per-example tensor broadcasting',()=>{
  for(const batchSize of [1,2,3]) for(const operation of ['multiply','add','arithmetic'] as const) for(const loss of ['mse','mae'] as const) {
    it(`matches trace values and gradients for ${operation}, ${loss}, batch ${batchSize}`,async()=>{
      await expectTraceParity(broadcastGraph(operation,[2],loss),batchSize)
    })
  }
  it.each([1,2,3])('broadcasts matrix parameters and scalar targets within each of %i examples',async batchSize=>{
    await expectTraceParity(broadcastGraph('multiply',[2,2]),batchSize)
  })
  it.each([1,2,3])('repeats a shared constant prediction for each of %i examples',async batchSize=>{
    const graph=broadcastGraph(),source=graph.nodes.find(node=>node.type==='dataset')!,weight=graph.nodes.find(node=>node.type==='weight')!,loss=graph.nodes.find(node=>node.type==='loss')!
    graph.nodes=[source,weight,loss];graph.groups=[]
    graph.edges=[{id:'prediction',source:weight.id,target:loss.id,inputSlot:0},{id:'target',source:source.id,sourceSlot:1,target:loss.id,inputSlot:1}]
    await expectTraceParity(graph,batchSize)
  })
  const python=process.env.PYTORCH_TEST_PYTHON
  it.skipIf(!python)('matches independent PyTorch broadcasting and gradients at batch sizes one, two and three',async()=>{
    const graph=broadcastGraph(),model=new TensorGraph(graph)
    try {
      const weight=graph.nodes.find(node=>node.type==='weight')!,bias=graph.nodes.find(node=>node.type==='bias')!
      const rows=model.examples.slice(0,3)
      const input={x:rows.map(row=>row.features[0].data[0]),y:rows.map(row=>row.target.data[0]),w:toTensor(weight.params.value).data,b:toTensor(bias.params.value).data[0]}
      const program=`import json,sys,torch\nd=json.load(sys.stdin)\nresults=[]\nfor n in [1,2,3]:\n x=torch.tensor(d['x'][:n]).reshape(n,1)\n y=torch.tensor(d['y'][:n]).reshape(n,1)\n w=torch.tensor(d['w'],dtype=torch.float32,requires_grad=True)\n b=torch.tensor(d['b'],dtype=torch.float32,requires_grad=True)\n loss=((x*w+b-y)**2).mean()\n loss.backward()\n results.append({'loss':loss.item(),'w':w.grad.tolist(),'b':b.grad.item()})\nprint(json.dumps(results))\n`
      const run=spawnSync(python!,['-c',program],{input:JSON.stringify(input),encoding:'utf8',timeout:30000})
      expect(run.status,run.stderr).toBe(0)
      const expected=JSON.parse(run.stdout) as {loss:number;w:number[];b:number}[]
      for(const n of [1,2,3]) {
        const actual=model.gradients(rows.slice(0,n))
        try {
          expect((await actual.loss.data())[0]).toBeCloseTo(expected[n-1].loss,5)
          Array.from(await actual.grads[model.variables.get(weight.id)!.name].data()).forEach((value,i)=>expect(value).toBeCloseTo(expected[n-1].w[i],5))
          expect((await actual.grads[model.variables.get(bias.id)!.name].data())[0]).toBeCloseTo(expected[n-1].b,5)
        } finally {tf.dispose([actual.loss,...Object.values(actual.grads)])}
      }
    } finally {model.dispose()}
  },30000)
})

describe('tensor classification metrics',()=>{
  it.each([1,2,3])('uses both logits for binary classification with batch size %i',async batchSize=>{
    const graph=createModelPreset('playground')
    graph.nodes.find(node=>node.id==='layer-2/bias-0')!.params.value=-10
    graph.nodes.find(node=>node.id==='layer-2/bias-1')!.params.value=10
    const source=graph.nodes.find(node=>node.type==='dataset')!
    const expected=evaluateDataset(graph,source.id,'test'),model=new TensorGraph(graph)
    try {
      const rows=model.examples.filter(row=>row.split==='test')
      const metrics=await model.evaluate(rows,batchSize),inference=await model.inference(rows,batchSize)
      expect(metrics.accuracy).toBe(expected.accuracy)
      expect(inference.rows).toEqual(expected.rows)
      expect(inference.accuracy).toBe(expected.accuracy)
      expect(metrics.loss).toBeCloseTo(expected.loss,4)
    } finally {model.dispose()}
  })
  it('trains the built-in Playground with its two-logit output',async()=>{
    const result=await trainTensorGraph(createModelPreset('playground'),{epochs:1,batchSize:3,settings:{...DEFAULT_TRAINING,engine:'tensor',backend:'cpu',patience:0}})
    expect(result.completed).toBe(1)
    expect(result.reports).toHaveLength(2)
    expect(result.reports.every(report=>Number.isFinite(report.train.loss)&&report.validation.accuracy>=0&&report.validation.accuracy<=1)).toBe(true)
  })
  it.each([1,2,3])('counts predicted tokens in fixed-length sequence metrics with batch size %i',async batchSize=>{
    const graph=createModelPreset('decoder'),model=new TensorGraph(graph)
    try {
      const rows=model.examples.slice(0,3)
      const metrics=await model.evaluate(rows,batchSize),inference=await model.inference(rows,batchSize)
      expect(inference.predictions).toBe(9)
      expect(metrics.accuracy).toBe(1)
      expect(metrics.accuracy).toBe(inference.accuracy)
    } finally {model.dispose()}
  })
  it('cancels inference between batches and disposes the active batch',async()=>{
    const model=new TensorGraph(createModelPreset('linear')),controller=new AbortController()
    try {
      const reason=new Error('Cancelled test evaluation')
      const pending=model.inference(model.examples.slice(0,3),1,controller.signal)
      setTimeout(()=>controller.abort(reason),0)
      await expect(pending).rejects.toBe(reason)
      expect(tf.memory().numTensors).toBe(model.variables.size)
    } finally {model.dispose()}
  })
})
