import {regularizedParameters} from './regularization'
import * as tf from '@tensorflow/tfjs'
import {parseArithmetic, type ParsedArithmetic} from './arithmetic'
import {resolveReshape} from './reshape'
import {datasetForNode,datasetTargetSlotForNode,datasetExamplesForNode, type DatasetExample} from './datasets'
import {isLossNode, lossKindForNode, topologicalSort, validateGraph} from './engine'
import {trainingBatches, type DatasetMetrics, type DatasetPrediction} from './datasetTraining'
import {toTensor} from './tensor'
import {DEFAULT_TRAINING, isTrainingSettings, type TrainingSettings} from './trainingSettings'
import type {GraphModel, GraphNode, TensorValue} from './types'

export {tf}
type Axis = 'batch' | 'token' | 'feature'
type Signal = {value:tf.Tensor; axes:Axis[]}
export interface TensorMetrics {loss:number; objective?:number; accuracy:number; examples:number}
export interface TensorReport {epoch:number; train:TensorMetrics; validation:TensorMetrics; improved:boolean}
const SUPPORTED = new Set(['dropout','dataset','weight','bias','input','target','matmul','arithmetic','multiply','add','activation','embedding','one-hot','transpose','tensor-transform','mean','softmax','causal-mask','layer-norm','loss','cross-entropy','concat','reshape','slice'])

/** Probe actual forward AND gradient kernels; merely initializing a backend is insufficient. */
export async function selectTensorBackend(graph:GraphModel, requested:TrainingSettings['backend'], settings:TrainingSettings=DEFAULT_TRAINING) {
  if(!isTrainingSettings(settings)||!Number.isFinite(graph.learningRate)||graph.learningRate<=0) throw Error('Invalid optimizer settings or learning rate.')
  const candidates = requested === 'auto' ? ['webgl','cpu'] : [requested]
  const failures:string[] = []
  for (const backend of candidates) {
    try {
      if (backend === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu')
      if (!await tf.setBackend(backend)) throw Error('Backend unavailable')
      await tf.ready()
      const probe = new TensorGraph(graph)
      const optimizer = new TensorOptimizer(settings,graph.learningRate)
      try {
        const rows=probe.examples.filter(row=>row.split==='train').slice(0,2)
        for(let i=0;i<4;i++) {
          const result=probe.gradients(rows)
          try {
            const loss=(await result.loss.data())[0]
            const gradients=await Promise.all(Object.values(result.grads).map(t=>t.data()))
            if(!Number.isFinite(loss)||gradients.some(g=>g.some(n=>!Number.isFinite(n)))) throw Error('Nonfinite forward/backward probe')
            optimizer.update(probe.variables,result.grads)
          } finally {tf.dispose([result.loss,...Object.values(result.grads)])}
        }
        await probe.snapshot()
        if(!Number.isFinite((await probe.inspect(rows)).loss)) throw Error('Nonfinite optimizer probe')
      } finally {optimizer.dispose();probe.dispose()}
      return {backend, fallback:failures.join('; ')}
    } catch(error) {failures.push(backend+': '+(error instanceof Error?error.message:String(error)))}
  }
  throw Error('No compatible training backend. '+failures.join('; '))
}

function arithmetic(expr:ParsedArithmetic['expression'], inputs:tf.Tensor[]):tf.Tensor {
  if (expr.kind==='number') return tf.scalar(expr.value)
  if (expr.kind==='input') return inputs[expr.index]
  if (expr.kind==='negate') return tf.neg(arithmetic(expr.child,inputs))
  const a=arithmetic(expr.left,inputs), b=arithmetic(expr.right,inputs)
  switch(expr.op) {case '+':return tf.add(a,b);case '-':return tf.sub(a,b);case '*':return tf.mul(a,b);case '/':return tf.div(a,b);case '^':return tf.pow(a,b)}
}
const clipProbability=tf.customGrad((input)=>({value:tf.clipByValue(input as tf.Tensor,1e-7,1-1e-7),gradFunc:(dy:tf.Tensor)=>dy}))

function product(a:tf.Tensor,b:tf.Tensor) {
  // Flatten token rows for a shared parameter matrix. Avoid implicit tiling and its gradient overhead.
  if(a.rank>=3 && b.rank===2) return tf.matMul(a.reshape([-1,a.shape.at(-1)!]),b).reshape([...a.shape.slice(0,-1),b.shape[1]!])
  return tf.matMul(a,b)
}

/** Compiles supported text DAGs, independent of node IDs, labels and visual groups.
 * Batch/token axis provenance controls padding; numerical graph parameters stay unchanged. */
export class TensorGraph {
  readonly examples:DatasetExample[]
  readonly variables = new Map<string,tf.Variable>()
  private order:GraphNode[]
  private inputs = new Map<string,GraphModel['edges']>()
  private dropoutStep = 0
  private expressions = new Map<string,ParsedArithmetic>()
  private source:GraphNode
  private lossNode:GraphNode
  readonly graph:GraphModel
  constructor(graph:GraphModel) {
    this.graph=graph
    const issues=validateGraph(graph).filter(issue=>issue.code!=='disconnected')
    if(issues.length) throw Error(issues[0].message)
    const sources=graph.nodes.filter(node=>node.type==='dataset')
    const losses=graph.nodes.filter(isLossNode)
    if(sources.length!==1 || losses.length!==1) throw Error('Tensor training currently requires one dataset and one Loss block.')
    this.source=sources[0]; this.lossNode=losses[0]
    this.examples=datasetExamplesForNode(this.source)
    this.order=topologicalSort(graph).map(id=>graph.nodes.find(node=>node.id===id)!)
    // Reject unsupported operations before allocating device memory.
    for(const node of this.order) {
      const kind=node.type==='tensor-transform'?node.params.transform:node.type
      if(!SUPPORTED.has(node.type) || node.type==='tensor-transform' && !['transpose','mean','reshape','slice'].includes(kind??'')) throw Error(`Tensor training does not yet support ${node.label} (${kind}). Use the trace engine for this graph.`)
      if(kind==='mean' && node.params.axis!==0) throw Error('Tensor training currently supports Mean over token axis 0 only.')
      this.inputs.set(node.id,graph.edges.filter(edge=>edge.target===node.id).sort((a,b)=>(a.inputSlot??0)-(b.inputSlot??0)))
      if(node.type==='arithmetic') this.expressions.set(node.id,parseArithmetic(node.params.expression??'x1 * x2'))
    }
    const targetInput=this.inputs.get(this.lossNode.id)![1]
    if(targetInput.source!==this.source.id || targetInput.sourceSlot!==datasetTargetSlotForNode(this.source)) {
      const targetNode=graph.nodes.find(node=>node.id===targetInput.source)
      const targetEdge=targetNode?.type==='target'?this.inputs.get(targetNode.id)?.[0]:undefined
      if(targetEdge?.source!==this.source.id || targetEdge.sourceSlot!==datasetTargetSlotForNode(this.source)) throw Error('Tensor training needs the dataset target wired directly, or through Target, to Loss.')
    }
    for(const node of this.order) if(node.type==='weight'||node.type==='bias') {
      const value=toTensor(node.params.value)
      this.variables.set(node.id,tf.tidy(()=>tf.variable(tf.tensor(value.data,value.shape,'float32'))))
    }
  }
  dispose(){for(const variable of this.variables.values()) variable.dispose()}
  private execute(rows:DatasetExample[], training = false): {loss:tf.Scalar; dataLoss:tf.Scalar; prediction:tf.Tensor; target:tf.Tensor; mask:tf.Tensor} {
    const data=this.source.params.textData, count=rows.length
    const generic=!data || data.representation==='facts'
    const lengths=rows.map(row=>generic?1:row.features[1].data.length), length=Math.max(...lengths)
    const mask=tf.tensor2d(lengths.flatMap(n=>Array.from({length},(_,i)=>Number(i<n))),[count,length])
    const pad=(items:number[],n=length)=>[...items,...Array(n-items.length).fill(0)]
    const features:Signal[]=generic ? rows[0].features.map((f,i)=>({value:tf.tensor(rows.flatMap(r=>r.features[i].data),[count,...f.shape]),axes:['batch',...f.shape.map(()=>'feature' as const)]})) : data.representation==='tokens'
      ? [{value:tf.tensor2d(rows.flatMap(row=>pad(row.features[0].data)),[count,length],'int32'),axes:['batch','token']}]
      : [{value:tf.tensor(rows.flatMap(row=>row.features[0].data),[count,1,data.vocabulary.length]),axes:['batch','feature','feature']}]
    if(!generic) features.push({value:tf.tensor2d(lengths.flatMap(n=>pad(Array.from({length:n},(_,i)=>i))),[count,length],'int32'),axes:['batch','token']})
    const sequence=data?.task==='language' && data.targetMode!=='last'
    const target=sequence?tf.tensor2d(rows.flatMap(row=>pad(row.target.data)),[count,length],'int32'):tf.tensor(rows.flatMap(row=>row.target.data),[count,...rows[0].target.shape])
    const targetSignal:Signal={value:target,axes:['batch',...rows[0].target.shape.map(()=>'feature' as const)]}
    if(this.source.params.customCsv) features.splice(datasetTargetSlotForNode(this.source),0,targetSignal)
    else features.push(targetSignal)
    const values=new Map<string,Signal>()
    const read=(edge:GraphModel['edges'][number])=>edge.source===this.source.id?features[edge.sourceSlot??0]:values.get(edge.source)!
    for(const node of this.order) {
      if(node.type==='dataset'||isLossNode(node)) continue
      const args=this.inputs.get(node.id)!.map(read), tensors=args.map(arg=>arg.value), [a,b,c]=tensors
      let axes:Axis[]=args.find(arg=>arg.axes.includes('batch'))?.axes??args[0]?.axes??[], value:tf.Tensor
      const kind=node.type==='tensor-transform'?node.params.transform:node.type
      switch(kind) {
        case 'weight':case 'bias': value=this.variables.get(node.id)!;axes=value.shape.map(()=>'feature');break
        case 'input':case 'target': {const constant=toTensor(node.params.value);value=a??tf.tensor(constant.data,constant.shape);axes=args[0]?.axes??value.shape.map(()=>'feature');break}
        case 'embedding': value=tf.gather(a,b.toInt());axes=[...args[1].axes,'feature'];break
        case 'one-hot':value=tf.oneHot(a.toInt(),node.params.numClasses??2);axes=[...args[0].axes,'feature'];break
        case 'matmul':
          if(args[1].axes[0]==='batch'&&args[0].axes[0]!=='batch') throw Error('Tensor training currently needs batched matrix products to have the batch on the left input.')
          value=product(a,b);axes=[...args[0].axes.slice(0,-1),args[1].axes.at(-1)!];break
        case 'arithmetic':value=arithmetic(this.expressions.get(node.id)!.expression,tensors);break
        case 'add':value=tensors.reduce((x,y)=>tf.add(x,y));break
        case 'multiply':value=tensors.reduce((x,y)=>tf.mul(x,y));break
        case 'dropout':value=training && (node.params.dropoutRate ?? 0.1)>0 ? tf.dropout(a,node.params.dropoutRate ?? 0.1,undefined,137 + this.dropoutStep++) : a;break
        case 'activation':value=node.params.activation==='relu'?tf.relu(a):node.params.activation==='sigmoid'?tf.sigmoid(a):node.params.activation==='tanh'?tf.tanh(a):a;break
        case 'transpose': {
          const batch=args[0].axes[0]==='batch', rank=a.rank-Number(batch)
          const permutation=node.params.axes??Array.from({length:rank},(_,i)=>rank-i-1)
          const actual=batch?[0,...permutation.map(i=>i+1)]:permutation
          value=tf.transpose(a,actual);axes=actual.map(i=>args[0].axes[i]);break
        }
        case 'reshape': {
          const batch=args[0].axes[0]==='batch'
          const shape=resolveReshape(node.params.shape??a.shape.slice(Number(batch)),a.size/(batch?count:1))
          if(!generic && !data?.fixedLength && args[0].axes.includes('token')) throw Error('Reshaping token sequences requires fixed-length inputs.')
          value=a.reshape(batch?[count,...shape]:shape);axes=[...(batch?['batch' as const]:[]),...shape.map(()=>'feature' as const)];break
        }
        case 'slice': {
          const axis=(node.params.axis??0)+Number(args[0].axes[0]==='batch')
          const start=node.params.start??0,end=node.params.end??a.shape[axis]
          const begin=Array(a.rank).fill(0);begin[axis]=start
          const size=[...a.shape];size[axis]=end-start
          value=tf.slice(a,begin,size);break
        }
        case 'mean': {
          const axis=args[0].axes[0]==='batch'?1:0
          if(args[0].axes[axis]==='token') {
            const weights=mask.reshape([...mask.shape,...Array(a.rank-2).fill(1)])
            value=tf.div(tf.sum(tf.mul(a,weights),axis,!!node.params.keepDims),tf.sum(weights,axis,!!node.params.keepDims))
          } else value=tf.mean(a,axis,!!node.params.keepDims)
          axes=[...args[0].axes];if(node.params.keepDims) axes[axis]='feature';else axes.splice(axis,1);break
        }
        case 'causal-mask': {
          const n=a.shape.at(-1)!, triangle=tf.tensor2d(Array.from({length:n*n},(_,i)=>Number(i%n<=Math.floor(i/n))),[n,n])
          value=tf.add(a,tf.mul(tf.sub(1,triangle),-1e9));break
        }
        case 'softmax': {
          const scores=args[0].axes.at(-1)==='token'?tf.add(a,tf.mul(tf.sub(1,mask.expandDims(1)),-1e9)):a
          value=tf.softmax(scores);break
        }
        case 'layer-norm': {if(args[0].axes.at(-1)!=='feature')throw Error('Tensor training normalizes feature axes only.');const {mean,variance}=tf.moments(a,-1,true);value=tf.add(tf.mul(tf.mul(tf.sub(a,mean),tf.rsqrt(tf.add(variance,node.params.epsilon??1e-5))),b),c);break}
        case 'concat': {const axis=(node.params.axis??1)+(axes[0]==='batch'?1:0);if(axes[axis]==='token')throw Error('Tensor training currently concatenates features, not padded token sequences.');value=tf.concat(tensors,axis);break}
        default:throw Error('Unsupported tensor operation '+kind)
      }
      values.set(node.id,{value,axes})
    }
    const prediction=read(this.inputs.get(this.lossNode.id)![0]).value
    const actual=read(this.inputs.get(this.lossNode.id)![1]).value
    let loss:tf.Tensor
    const lossKind=lossKindForNode(this.lossNode,this.graph)
    if(lossKind==='binary-cross-entropy') {
      if(prediction.size!==count) throw Error('A batched sentiment classifier must produce one probability per review. Pool tokens before Loss.')
      const clipped=clipProbability(prediction), labels=actual.reshape(prediction.shape)
      loss=tf.neg(tf.mean(tf.add(tf.mul(labels,tf.log(clipped)),tf.mul(tf.sub(1,labels),tf.log(tf.sub(1,clipped))))))
    } else if(lossKind==='cross-entropy') {
      const width=prediction.shape.at(-1)!
      const labels=actual.reshape(prediction.shape.slice(0,-1)).toInt()
      const losses=tf.neg(tf.sum(tf.mul(tf.oneHot(labels,width),tf.logSoftmax(prediction)),-1))
      loss=sequence?tf.mean(tf.div(tf.sum(tf.mul(losses,mask),1),tf.sum(mask,1))):tf.mean(losses)
    } else if(lossKind==='mse') loss=tf.mean(tf.square(tf.sub(prediction,actual.reshape(prediction.shape))))
    else if(lossKind==='mae') loss=tf.mean(tf.abs(tf.sub(prediction,actual.reshape(prediction.shape))))
    else throw Error('Tensor training supports MSE, MAE, binary cross entropy or cross entropy.')
    const dataLoss=loss as tf.Scalar
    if(this.lossNode.params.regularization && this.lossNode.params.regularization!=='none'){
      const selected=regularizedParameters(this.graph,this.lossNode).map(n=>this.variables.get(n.id)!)
      if(selected.length) loss=tf.add(loss,tf.mul(this.lossNode.params.regularizationStrength??0,tf.addN(selected.map(w=>tf.sum(this.lossNode.params.regularization==='l1'?tf.abs(w):tf.mul(tf.square(w),.5))))))
    }
    return {loss:loss as tf.Scalar,dataLoss,prediction,target,mask}
  }
  gradients(rows:DatasetExample[]) {
    return tf.tidy(()=>{const result=tf.variableGrads(()=>this.execute(rows,true).loss,[...this.variables.values()]);return {loss:result.value,grads:result.grads}})
  }
  async inspect(rows:DatasetExample[]) {
    const result=tf.tidy(()=>{const {loss,prediction}=this.execute(rows);return {loss,prediction:tf.clone(prediction)}})
    try {return {loss:(await result.loss.data())[0],prediction:Array.from(await result.prediction.data()),shape:result.prediction.shape}}
    finally {tf.dispose(result)}
  }
  async evaluate(rows:DatasetExample[],batchSize:number,signal?:AbortSignal):Promise<TensorMetrics> {
    let sum=0,objectives=0,hits=0,total=0
    for(let i=0;i<rows.length;i+=batchSize) {
      if(signal?.aborted) throw Error('Training stopped.')
      const batch=rows.slice(i,i+batchSize), result=tf.tidy(()=>{
        const {loss:objective,dataLoss:loss,prediction,target,mask}=this.execute(batch)
        const binary=datasetForNode(this.source).task==='binary-classification'
        const sequence=this.source.params.textData?.task==='language' && this.source.params.textData.targetMode!=='last'
        const regression=datasetForNode(this.source).task==='regression'
        const correct=regression?tf.zeros([batch.length]):binary?tf.equal(tf.greaterEqual(prediction,.5),tf.cast(target.reshape(prediction.shape),'bool')).cast('float32'):tf.mul(tf.equal(tf.argMax(prediction,-1),target.reshape(prediction.shape.slice(0,-1))).cast('float32'),sequence?mask:tf.scalar(1))
        return {loss,objective,hits:tf.sum(correct),total:sequence?tf.sum(mask):tf.scalar(batch.length)}
      })
      try {const [loss,correct,count,objective]=await Promise.all([result.loss.data(),result.hits.data(),result.total.data(),result.objective.data()]);sum+=loss[0]*batch.length;objectives+=objective[0]*batch.length;hits+=correct[0];total+=count[0]}
      finally {tf.dispose(result)}
      if(i%(batchSize*8)===0) await new Promise(resolve=>setTimeout(resolve,0))
    }
    if(!rows.length) throw Error('Evaluation needs examples.')
    return {loss:sum/rows.length,objective:objectives/rows.length,accuracy:hits/total,examples:rows.length}
  }
  /** Decode predictions in minibatches without blocking the UI on a traced pass per row. */
  async inference(rows: DatasetExample[], batchSize = 64): Promise<DatasetMetrics> {
    if (!rows.length || !Number.isInteger(batchSize) || batchSize < 1) throw Error('Evaluation needs examples and a positive batch size.')
    const dataset = datasetForNode(this.source)
    const categorical = dataset.task.includes('classification') || dataset.task === 'sequence' || lossKindForNode(this.lossNode, this.graph) === 'cross-entropy'
    const binary = dataset.task === 'binary-classification'
    const predictions: DatasetPrediction[] = []
    let sum = 0, hits = 0, scored = 0
    const display = (value: number, isClass: boolean) => isClass
      ? dataset.classLabels?.[value] ?? dataset.vocabulary?.[value] ?? String(value)
      : Number(value.toPrecision(5)).toString()
    for (let offset = 0; offset < rows.length; offset += batchSize) {
      const batch = rows.slice(offset, offset + batchSize)
      const result = tf.tidy(() => {
        const { dataLoss, prediction } = this.execute(batch)
        return { loss: dataLoss, prediction: tf.clone(prediction) }
      })
      try {
        const [loss, values] = await Promise.all([result.loss.data(), result.prediction.data()])
        sum += loss[0] * batch.length
        const width = categorical && !binary ? result.prediction.shape.at(-1)! : 1
        const positions = values.length / (batch.length * width)
        batch.forEach((example, exampleIndex) => example.target.data.forEach((actual, position) => {
          const start = (exampleIndex * positions + position) * width
          const scores = Array.from(values.slice(start, start + width))
          const isClass = categorical && (width > 1 || binary)
          const predicted = width > 1 ? scores.indexOf(Math.max(...scores)) : binary ? Number(scores[0] >= .5) : scores[0]
          if (isClass) { scored++; hits += Number(predicted === actual) }
          predictions.push({
            example: `${example.label ?? `Example ${offset + exampleIndex + 1}`}${example.target.data.length > 1 ? ` · output ${position + 1}` : ''}`,
            actual: display(actual, isClass), predicted: display(predicted, isClass),
            ...(isClass ? { correct: predicted === actual } : {}),
          })
        }))
      } finally { tf.dispose(result) }
      await new Promise(resolve => setTimeout(resolve, 0))
    }
    return { loss: sum / rows.length, examples: rows.length, predictions: scored, rows: predictions, accuracy: scored ? hits / scored : undefined }
  }
  async snapshot():Promise<GraphModel> {
    const params=new Map<string,TensorValue>(await Promise.all([...this.variables].map(async([id,v]):Promise<[string,TensorValue]>=>[id,{shape:Array.from(v.shape),data:Array.from(await v.data())}])))
    if([...params.values()].some(v=>v.data.some(n=>!Number.isFinite(n)))) throw Error('Training diverged; no finite checkpoint available.')
    return {...this.graph,nodes:this.graph.nodes.map(node=>({...node,params:params.has(node.id)?{...node.params,value:params.get(node.id)!}:node.params,value:undefined,grad:undefined,cache:undefined,localDerivative:undefined})),edges:this.graph.edges.map(edge=>({...edge,value:undefined,grad:undefined}))}
  }
}

/** Optimizer slots and parameters remain on the selected device for the entire run. */
export class TensorOptimizer {
  private step=0
  private moments=new Map<string,{m:tf.Tensor;v:tf.Tensor}>()
  readonly settings:TrainingSettings
  readonly rate:number
  constructor(settings:TrainingSettings,rate:number) {
    this.settings=settings;this.rate=rate
    if(!isTrainingSettings(settings)||!Number.isFinite(rate)||rate<=0) throw Error('Invalid optimizer settings or learning rate.')
  }
  update(variables:Map<string,tf.Variable>,grads:tf.NamedTensorMap) {
    this.step++
    tf.tidy(()=>{
      const gradients=[...variables.values()].map(v=>grads[v.name]??tf.zerosLike(v))
      const norm=tf.sqrt(tf.addN(gradients.map(g=>tf.sum(tf.square(g)))))
      const scale=this.settings.clipNorm>0?tf.minimum(1,tf.div(this.settings.clipNorm,tf.add(norm,1e-12))):tf.scalar(1)
      let i=0
      for(const [id,variable] of variables) {
        const g=tf.mul(gradients[i++],scale)
        let update:tf.Tensor=g
        if(this.settings.optimizer!=='sgd') {
          const previous=this.moments.get(id)
          const m=tf.add(tf.mul(previous?.m??tf.zerosLike(variable),.9),tf.mul(g,.1))
          const v=tf.add(tf.mul(previous?.v??tf.zerosLike(variable),.999),tf.mul(tf.square(g),.001))
          update=tf.div(tf.div(m,1-Math.pow(.9,this.step)),tf.add(tf.sqrt(tf.div(v,1-Math.pow(.999,this.step))),1e-8))
          this.moments.set(id,{m:tf.keep(m),v:tf.keep(v)});previous?.m.dispose();previous?.v.dispose()
        }
        const decay=this.settings.optimizer==='adamw'?this.settings.weightDecay:0
        variable.assign(tf.sub(tf.mul(variable,1-this.rate*decay),tf.mul(update,this.rate)))
      }
    })
  }
  dispose(){for(const value of this.moments.values()) tf.dispose([value.m,value.v]);this.moments.clear()}
}

export interface TensorTrainOptions {
  epochs:number; batchSize:number; settings?:TrainingSettings; signal?:AbortSignal; epochOffset?:number; shuffle?:boolean
  onBackend?:(backend:string,fallback:string)=>void
  onProgress?:(done:number,total:number)=>void
  onReport?:(report:TensorReport)=>void
}
export async function trainTensorGraph(graph:GraphModel,options:TensorTrainOptions) {
  const settings=options.settings??{...DEFAULT_TRAINING,engine:'tensor'}
  if(!isTrainingSettings(settings)||!Number.isInteger(options.epochs)||options.epochs<1||!Number.isInteger(options.batchSize)||options.batchSize<1) throw Error('Choose valid training settings, epochs and batch size.')
  const selected=await selectTensorBackend(graph,settings.backend,settings)
  options.onBackend?.(selected.backend,selected.fallback)
  const model=new TensorGraph(graph),optimizer=new TensorOptimizer(settings,graph.learningRate)
  let best=graph,bestEpoch=0,completed=0
  const reports:TensorReport[]=[]
  try {
    const train=model.examples.filter(row=>row.split==='train'),validation=model.examples.filter(row=>row.split==='test')
    if(!train.length||!validation.length||options.batchSize>train.length) throw Error('Use training and validation examples and a batch size no larger than the training set.')
    let bestLoss=Infinity,stale=0
    best=await model.snapshot()
    const report=async(epoch:number)=>{
      const trainMetrics=await model.evaluate(train,options.batchSize,options.signal),val=await model.evaluate(validation,options.batchSize,options.signal)
      if(!Number.isFinite(trainMetrics.loss)||!Number.isFinite(val.loss)) throw Error('Training diverged. Lower the learning rate.')
      const improved=val.loss<bestLoss-settings.minDelta
      if(improved){bestLoss=val.loss;bestEpoch=epoch;best=await model.snapshot();stale=0}else stale++
      const result={epoch:epoch+(options.epochOffset??0),train:trainMetrics,validation:val,improved}
      reports.push(result);options.onReport?.(result)
    }
    await report(0)
    for(let epoch=1;epoch<=options.epochs;epoch++) {
      const batches=trainingBatches(train.map((_,i)=>i),options.batchSize,(options.epochOffset??0)+epoch-1,options.shuffle??true)
      let done=0
      for(const ids of batches) {
        if(options.signal?.aborted) return {graph:best,reports,completed,bestEpoch,backend:selected.backend,stopped:true}
        const result=model.gradients(ids.map(i=>train[i]))
        try {
          // One scalar read per batch synchronizes updates and catches divergence before a long run continues.
          if(!Number.isFinite((await result.loss.data())[0])) throw Error('Training diverged on '+selected.backend+'. Lower the learning rate or choose WebGL / Tensor CPU.')
          optimizer.update(model.variables,result.grads)
        } finally {tf.dispose([result.loss,...Object.values(result.grads)])}
        done+=ids.length;options.onProgress?.((epoch-1)*train.length+done,options.epochs*train.length)
        await new Promise(resolve=>setTimeout(resolve,0))
      }
      completed=epoch
      await report(epoch)
      if(settings.patience>0&&stale>=settings.patience) break
    }
    return {graph:settings.patience>0?best:await model.snapshot(),reports,completed,bestEpoch,backend:selected.backend,stopped:false}
  } catch(error) {
    if(options.signal?.aborted) return {graph:best,reports,completed,bestEpoch,backend:selected.backend,stopped:true}
    throw error
  } finally {optimizer.dispose();model.dispose()}
}
