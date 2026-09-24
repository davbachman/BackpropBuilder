import checkpoint from './cnn-checkpoint.json'
import digitData from './digits.json'
import { forwardPass } from '../domain/engine'
import type { GraphModel, NodeParams, NodeType, TensorValue } from '../domain/types'

export interface DigitExample { id: string; label: number; split: 'train' | 'test'; pixels: number[] }
export const CNN_DIGITS = digitData as DigitExample[]
export const CNN_CHECKPOINT = checkpoint
export const digitTensor = (sample: DigitExample): TensorValue => ({ shape: [8,8,1], data: sample.pixels.map(pixel => pixel/16) })

export function createCnnPreset(): GraphModel {
  const sample = CNN_DIGITS.find(digit => digit.split === 'test' && digit.label === 3)!
  const graph: GraphModel = { nodes: [], edges: [], groups: [], learningRate: .01, view: { expandedGroupIds: ['features'], semanticZoom: true } }
  const node = (id: string, type: NodeType, label: string, inputs: string[], x: number, y: number, params: NodeParams = {}) => {
    graph.nodes.push({ id,type,label,params,position:{x,y} })
    inputs.forEach((source,inputSlot)=>graph.edges.push({id:`${source}→${id}:${inputSlot}`,source,target:id,inputSlot}))
    return id
  }
  node('digit-data','dataset','Handwritten digits',[],-300,0,{dataset:'digits-8x8',datasetMode:'sample',datasetIndex:CNN_DIGITS.indexOf(sample)})
  node('image-input','input','Handwritten digit · 8 × 8',['digit-data'],0,0,{value:digitTensor(sample)})
  node('conv-weights','weight','4 learned filters · 3 × 3',[],240,-180,{value:structuredClone(checkpoint.parameters.kernels)})
  node('conv-bias','bias','Filter biases',[],240,160,{value:structuredClone(checkpoint.parameters.convBias)})
  node('conv-output','conv2d','Convolve · 4 feature maps',['image-input','conv-weights','conv-bias'],490,0)
  node('conv-relu','activation','Keep positive features',['conv-output'],760,0,{activation:'relu'})
  node('pool-output','avgpool2d','Average each 2 × 2 patch',['conv-relu'],1020,0)
  node('flat-features','reshape','Flatten · 36 features',['pool-output'],1280,0,{shape:[1,36]})
  node('classifier-weights','weight','Digit classifier weights',[],1510,-190,{value:structuredClone(checkpoint.parameters.classifier)})
  node('class-products','matmul','Evidence for each digit',['flat-features','classifier-weights'],1750,0)
  node('classifier-bias','bias','Digit biases',[],1760,210,{value:structuredClone(checkpoint.parameters.classifierBias)})
  node('class-sums','add','Weighted sum + digit bias',['class-products','classifier-bias'],1990,0)
  node('cnn-logits','activation','Class scores · 10 digits',['class-sums'],2200,0,{activation:'identity'})
  node('cnn-probabilities','softmax','Digit probabilities',['cnn-logits'],2250,-90)
  node('digit-target','target','Correct digit',[],1990,300,{value:{shape:[1],data:[sample.label]}})
  node('cnn-loss','cross-entropy','Classification loss',['cnn-logits','digit-target'],2250,220)
  graph.edges.push({id:'digit-data→digit-target',source:'digit-data',sourceSlot:1,target:'digit-target',inputSlot:0})
  graph.groups = [
    { id:'features',kind:'cnn',label:'See strokes → build features',nodeIds:['conv-weights','conv-bias','conv-output','conv-relu','pool-output','flat-features'],position:{x:300,y:0},dimensions:{width:240,height:170},detail:{sampleId:sample.id,inputNodeId:'image-input',outputNodeId:'flat-features'} },
    { id:'convolution',parentId:'features',kind:'convolution',label:'4 learned spatial filters',nodeIds:['conv-weights','conv-bias','conv-output'],position:{x:390,y:0},dimensions:{width:240,height:170} },
    { id:'digit-classifier',kind:'layer',label:'36 features → 10 digit scores',nodeIds:['classifier-weights','class-products','classifier-bias','class-sums','cnn-logits'],position:{x:1600,y:0},dimensions:{width:240,height:170},detail:{inputNodeId:'flat-features',weightNodeId:'classifier-weights',biasNodeId:'classifier-bias',preNodeId:'class-sums',outputNodeId:'cnn-logits',activation:'identity'} },
  ]
  return forwardPass(graph).graph
}

export function selectDigit(graph: GraphModel, sample: DigitExample): GraphModel {
  const next = { ...graph, nodes:graph.nodes.map(node=>node.type==='dataset' && node.params.dataset==='digits-8x8'?{...node,params:{...node.params,datasetIndex:CNN_DIGITS.indexOf(sample),datasetValues:undefined}}:node.id==='image-input'?{...node,params:{...node.params,value:digitTensor(sample)}}:node.id==='digit-target'?{...node,params:{...node.params,value:{shape:[1],data:[sample.label]}}}:node), groups:graph.groups?.map(group=>group.id==='features'?{...group,detail:{...group.detail,sampleId:sample.id}}:group) }
  return forwardPass(next).graph
}

export function restoreCnnWeights(graph: GraphModel): GraphModel {
  const values: Record<string, TensorValue> = { 'conv-weights': checkpoint.parameters.kernels, 'conv-bias': checkpoint.parameters.convBias, 'classifier-weights': checkpoint.parameters.classifier, 'classifier-bias': checkpoint.parameters.classifierBias }
  return forwardPass({ ...graph, nodes:graph.nodes.map(node=>values[node.id]?{...node,params:{...node.params,value:structuredClone(values[node.id])}}:node) }).graph
}
