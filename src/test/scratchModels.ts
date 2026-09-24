import { initializeTensor, type Initializer } from '../domain/authoring'
import { createEmptyGraph, createNode } from '../domain/examples'
import { connectGraphNodes } from '../domain/graphEditing'
import type { GraphModel, NodeParams, NodeType } from '../domain/types'

/** Use the same create/initialize/connect operations as the palette. No preset
 * model or checkpoint, no required node names, and no hidden numerical model. */
export function scratchModel(kind: 'cnn' | 'transformer'): { graph: GraphModel; datasetId: string } {
  let graph = createEmptyGraph(), serial = 0
  graph.learningRate = kind === 'cnn' ? .02 : .005
  const node = (type: NodeType, params: NodeParams = {}, inputs: Array<string | [string,number]> = []) => {
    const created = createNode(type, ++serial)
    created.params = {...created.params,...params}
    graph.nodes.push(created)
    inputs.forEach((source, slot) => {
      const connected = connectGraphNodes(graph,{source:Array.isArray(source)?source[0]:source,sourceHandle:`out-${Array.isArray(source)?source[1]:0}`,target:created.id,targetHandle:`in-${slot}`})
      if (!connected) throw new Error('Palette connection failed.')
      graph = connected
    })
    return created.id
  }
  const weight = (shape: number[], mode: Initializer = 'xavier') => node('weight',{value:initializeTensor(shape,mode,serial+100)})
  const bias = (width: number) => node('bias',{value:initializeTensor([width],'zeros')})
  const dense = (input: string, from: number, to: number) => node('add',{},[node('matmul',{},[input,weight([from,to])]),bias(to)])
  const data = node('dataset',{dataset:kind === 'cnn' ? 'digits-8x8' : 'color-cycle',datasetMode:'sample',datasetIndex:0})
  let logits: string
  if (kind === 'cnn') {
    const conv = node('conv2d',{},[data,weight([2,3,3,1],'he'),bias(2)])
    const relu = node('activation',{activation:'relu'},[conv])
    const pool = node('avgpool2d',{},[relu])
    const flat = node('reshape',{shape:[1,18]},[pool])
    logits = dense(flat,18,10)
  } else {
    const token = node('embedding',{},[weight([5,4]),data])
    const position = node('embedding',{},[weight([12,4]),[data,1]])
    const input = node('add',{},[token,position])
    const norm = (input: string) => node('layer-norm',{},[input,weight([4],'ones'),bias(4)])
    const normalized = norm(input)
    const projections = Array.from({length:3},()=>node('matmul',{},[normalized,weight([4,4])]))
    const heads = [0,1].map(head => {
      const [q,k,v] = projections.map(projection=>node('slice',{axis:1,start:head*2,end:head*2+2},[projection]))
      const kt = node('transpose',{axes:[1,0]},[k])
      const dot = node('matmul',{},[q,kt])
      const scale = node('input',{value:1/Math.sqrt(2)})
      const scaled = node('multiply',{},[dot,scale])
      const mask = node('causal-mask',{},[scaled])
      const probabilities = node('softmax',{},[mask])
      return node('matmul',{},[probabilities,v])
    })
    const attention = node('matmul',{},[node('concat',{axis:1,inputCount:2},heads),weight([4,4])])
    const residual = node('add',{},[input,attention])
    const hidden = node('activation',{activation:'relu'},[dense(norm(residual),4,8)])
    const block = node('add',{},[residual,dense(hidden,8,4)])
    logits = dense(norm(block),4,5)
  }
  node('softmax',{},[logits])
  node('cross-entropy',{},[logits,[data,kind === 'cnn' ? 1 : 2]])
  return {graph,datasetId:data}
}
