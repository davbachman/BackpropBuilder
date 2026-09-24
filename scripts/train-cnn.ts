/** Reproduce the tiny CNN checkpoint: node --experimental-strip-types scripts/train-cnn.ts
 * The 400 training and 100 held-out 8×8 UCI digits are checked in; no network or dependencies. */
import { readFileSync, writeFileSync } from 'node:fs'
import { conv2d, conv2dBackward, avgpool2d, avgpool2dBackward } from '../src/learning/cnnMath.ts'
import type { TensorValue } from '../src/domain/types.ts'

type Example = { id: string; label: number; split: string; pixels: number[] }
const examples: Example[] = JSON.parse(readFileSync(new URL('../src/learning/digits.json', import.meta.url), 'utf8'))
const seed = 19, epochs = 180, batchSize = 20, rate = .006
let state = seed
const random = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296 }
const tensor = (shape: number[], scale=0): TensorValue => ({ shape, data: Array.from({length:shape.reduce((a,b)=>a*b,1)},()=>scale ? (random()*2-1)*scale : 0) })
const parameters = { kernels: tensor([4,3,3,1],.5), convBias: tensor([4]), classifier: tensor([36,10],.3), classifierBias: tensor([10]) }
parameters.convBias.data.fill(.05)
const names = Object.keys(parameters) as (keyof typeof parameters)[]
const m=Object.fromEntries(names.map(name=>[name,parameters[name].data.map(()=>0)])) as Record<keyof typeof parameters,number[]>
const v=structuredClone(m)
let step=0
function run(example: Example) {
  const input: TensorValue={shape:[8,8,1],data:example.pixels.map(value=>value/16)}
  const conv=conv2d(input,parameters.kernels,parameters.convBias)
  const relu={shape:conv.shape,data:conv.data.map(value=>Math.max(0,value))}
  const pool=avgpool2d(relu)
  const logits=parameters.classifierBias.data.map((bias,j)=>pool.data.reduce((sum,x,i)=>sum+x*parameters.classifier.data[i*10+j],bias))
  const max=Math.max(...logits), exps=logits.map(x=>Math.exp(x-max)), total=exps.reduce((a,b)=>a+b,0), probabilities=exps.map(x=>x/total)
  return {input,conv,relu,pool,probabilities,loss:-Math.log(Math.max(1e-12,probabilities[example.label]))}
}
const train=examples.filter(e=>e.split==='train'), test=examples.filter(e=>e.split==='test')
function metrics(list: Example[]) {
  let loss=0, correct=0
  for(const example of list){const result=run(example);loss+=result.loss;correct+=Number(result.probabilities.indexOf(Math.max(...result.probabilities))===example.label)}
  return {loss:loss/list.length,accuracy:correct/list.length}
}
const history: {epoch:number;trainingLoss:number;testLoss:number;trainingAccuracy:number;testAccuracy:number}[]=[]
function record(epoch:number){const training=metrics(train),heldOut=metrics(test);const point={epoch,trainingLoss:training.loss,testLoss:heldOut.loss,trainingAccuracy:training.accuracy,testAccuracy:heldOut.accuracy};history.push(point);console.log(JSON.stringify(point))}
record(0)
for(let epoch=1;epoch<=epochs;epoch++){
  const order=[...train]
  for(let i=order.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[order[i],order[j]]=[order[j],order[i]]}
  for(let offset=0;offset<order.length;offset+=batchSize){
    const batch=order.slice(offset,offset+batchSize)
    const gradients=Object.fromEntries(names.map(name=>[name,parameters[name].data.map(()=>0)])) as Record<keyof typeof parameters,number[]>
    for(const example of batch){
      const {input,conv,relu,pool,probabilities}=run(example)
      const dz=[...probabilities];dz[example.label]-=1
      const df=Array(36).fill(0)
      for(let j=0;j<10;j++){
        gradients.classifierBias[j]+=dz[j]
        for(let i=0;i<36;i++){gradients.classifier[i*10+j]+=pool.data[i]*dz[j];df[i]+=parameters.classifier.data[i*10+j]*dz[j]}
      }
      const dRelu=avgpool2dBackward(relu,{shape:[3,3,4],data:df})
      const [,dw,db]=conv2dBackward(input,parameters.kernels,parameters.convBias,{shape:conv.shape,data:dRelu.data.map((value,i)=>conv.data[i]>0?value:0)})
      dw.data.forEach((value,i)=>{gradients.kernels[i]+=value});db.data.forEach((value,i)=>{gradients.convBias[i]+=value})
    }
    step++
    for(const name of names)for(let i=0;i<parameters[name].data.length;i++){
      const gradient=gradients[name][i]/batch.length
      m[name][i]=.9*m[name][i]+.1*gradient;v[name][i]=.999*v[name][i]+.001*gradient*gradient
      parameters[name].data[i]-=rate*(m[name][i]/(1-.9**step))/(Math.sqrt(v[name][i]/(1-.999**step))+1e-8)
    }
  }
  if(epoch%30===0 || epoch===epochs)record(epoch)
}
writeFileSync(new URL('../src/learning/cnn-checkpoint.json',import.meta.url),JSON.stringify({metadata:{seed,epochs,batchSize,learningRate:rate,trainingExamples:train.length,testExamples:test.length,architecture:'8×8×1 → valid3×3 convolution(4) → ReLU → 2×2 average pool → flatten36 → linear10 → softmax',...history.at(-1)},parameters,history},null,2)+'\n')
