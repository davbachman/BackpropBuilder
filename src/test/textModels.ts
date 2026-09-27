/** Instructor pilot recipes. These construct ordinary palette graphs, not new
 * model operators, and are deliberately not part of the student model menu. */
import { createEmptyGraph, createNode } from '../domain/examples'
import { initializeTensor } from '../domain/authoring'
import { mergeNodesIntoVisualGroup } from '../domain/grouping'
import type { GraphModel, NodeParams, NodeType, TextDatasetData } from '../domain/types'

export type TextModelKind = 'counts-linear' | 'counts-mlp' | 'mean' | 'position-mean' | 'attention' | 'position-attention' | 'transformer' | 'alice-baseline' | 'alice-transformer'

export function buildTextModel(data: TextDatasetData, kind: TextModelKind, width = 20, seed = 137): GraphModel {
  const graph = createEmptyGraph()
  graph.learningRate = kind.startsWith('counts') ? .01 : data.task === 'language' ? .03 : .03
  let counter = 0
  const add = (id: string, type: NodeType, params: NodeParams = {}, inputs: Array<string | [string, number]> = []) => {
    const node = {...createNode(type, counter++), id, label: id.replace(/-/g, ' '), params, position: {x: 260 * Math.floor(counter / 5), y: 200 * (counter % 5)}}
    graph.nodes.push(node)
    inputs.forEach((source, inputSlot) => graph.edges.push({id: id + ':' + inputSlot, source: typeof source === 'string' ? source : source[0], sourceSlot: typeof source === 'string' ? 0 : source[1], target: id, inputSlot}))
    return id
  }
  const param = (id: string, shape: number[], mode: 'xavier' | 'zeros' | 'ones' | 'uniform' = 'xavier') => {
    const value = initializeTensor(shape, mode, seed++)
    if (mode === 'uniform') value.data = value.data.map(x => x * 5)
    return add(id, 'weight', {value})
  }
  const linear = (name: string, x: string, from: number, to: number) => {
    const w = param(name + '-weights', [from, to])
    const product = add(name + '-product', 'matmul', {}, [x, w])
    return add(name + '-bias-add', 'arithmetic', {expression: 'x1 + x2'}, [product, param(name + '-bias', [to], 'zeros')])
  }
  const norm = (name: string, x: string) => add(name, 'layer-norm', {}, [x, param(name + '-scale', [width], 'ones'), param(name + '-offset', [width], 'zeros')])
  const representation = kind.startsWith('counts') ? 'counts' : 'tokens'
  add('text-data', 'dataset', {dataset: 'custom-text', textData: {...data, representation}, datasetMode: 'sample', datasetIndex: 0})
  const vocabularySize = data.vocabulary.length
  let x: string
  let features: number
  if (representation === 'counts') {
    x = add('review-counts', 'input', {}, [['text-data', 0]])
    features = vocabularySize
  } else {
    const table = param('word-embeddings', [vocabularySize, width], 'uniform')
    x = add('embedded-tokens', 'embedding', {}, [table, ['text-data', 0]])
    features = width
    if (['position-mean', 'position-attention', 'transformer', 'alice-transformer'].includes(kind)) {
      const positions = add('embedded-positions', 'embedding', {}, [param('position-embeddings', [data.maxLength, width], 'uniform'), ['text-data', 1]])
      x = add('tokens-plus-positions', 'arithmetic', {expression: 'x1 + x2'}, [x, positions])
    }
    const block = kind === 'transformer' || kind === 'alice-transformer'
    if (block || kind === 'attention' || kind === 'position-attention') {
      const residual = x
      if (block) x = norm('attention-norm', x)
      const projections = ['query', 'key', 'value'].map(name => add(name, 'matmul', {}, [x, param(name + '-weights', [width, width])]))
      const keyTranspose = add('key-transpose', 'tensor-transform', {transform: 'transpose', axes: [1, 0]}, [projections[1]])
      let scores = add('attention-scores', 'matmul', {}, [projections[0], keyTranspose])
      scores = add('scaled-scores', 'arithmetic', {expression: 'x1 / ' + Math.sqrt(width)}, [scores])
      if (data.task === 'language') scores = add('causal-scores', 'causal-mask', {}, [scores])
      const weights = add('attention-probabilities', 'softmax', {}, [scores])
      x = add('attended-values', 'matmul', {}, [weights, projections[2]])
      if (block) {
        x = add('attention-output', 'matmul', {}, [x, param('attention-output-weights', [width, width])])
        x = add('attention-residual', 'arithmetic', {expression: 'x1 + x2'}, [residual, x])
        const skip = x
        x = norm('mlp-norm', x)
        x = linear('feed-forward-in', x, width, width * 2)
        x = add('feed-forward-relu', 'activation', {activation: 'relu'}, [x])
        x = linear('feed-forward-out', x, width * 2, width)
        x = add('mlp-residual', 'arithmetic', {expression: 'x1 + x2'}, [skip, x])
        x = norm('output-norm', x)
      }
    }
    if (data.task === 'sentiment') x = add('mean-review', 'tensor-transform', {transform: 'mean', axis: 0, keepDims: true}, [x])
  }
  if (data.task === 'sentiment') {
    if (kind !== 'counts-linear') {
      x = linear('classifier-hidden', x, features, 12)
      x = add('classifier-relu', 'activation', {activation: 'relu'}, [x])
      features = 12
    }
    x = linear('sentiment', x, features, 1)
    x = add('positive-probability', 'activation', {activation: 'sigmoid'}, [x])
    add('loss', 'loss', {loss: 'binary-cross-entropy'}, [x, ['text-data', 2]])
  } else {
    x = linear('vocabulary', x, width, vocabularySize)
    add('loss', 'loss', {loss: 'cross-entropy'}, [x, ['text-data', 2]])
  }
  return organizeTextModel(graph)
}

/** Collapsible teaching modules retain every original operation and weight. */
export function organizeTextModel(input: GraphModel): GraphModel {
  let graph = {...input, groups: [], view: {expandedGroupIds: [], semanticZoom: true}} as GraphModel
  const group = (label: string, predicate: (id: string) => boolean) => {
    const covered = new Set(graph.groups?.flatMap(module => module.nodeIds))
    const ids = graph.nodes.filter(node => !covered.has(node.id) && predicate(node.id)).map(node => node.id)
    const result = mergeNodesIntoVisualGroup(graph, ids)
    if (result.group) result.group.label = label
    graph = result.graph
  }
  group(input.nodes.some(node => node.id === 'position-embeddings') ? 'Token and position embeddings' : 'Token embeddings', id => /^(one-hot|word-embeddings|embedded-tokens|position-embeddings|embedded-positions|tokens-plus-positions)$/.test(id))
  group('Self-attention', id => /^(attention-|query|key|value|scaled-scores|causal-scores|attended-values)/.test(id))
  group('Token-wise MLP + residual', id => /^(mlp-|feed-forward-)/.test(id))
  group(input.nodes[0].params.textData?.task === 'language' ? 'Vocabulary logits' : 'Review classifier', id => id !== 'text-data' && id !== 'loss')
  return graph
}

/** Insert explicit dropout operations without changing any initial parameters. */
export function addTransformerDropout(input: GraphModel, rate: number): GraphModel {
  if (!Number.isFinite(rate) || rate < 0 || rate >= 1) throw Error('Dropout probability must be at least 0 and less than 1.')
  const graph = {...input,nodes:input.nodes.map(n=>({...n,params:{...n.params}})),edges:input.edges.map(e=>({...e}))}
  for (const source of ['attention-output','feed-forward-out-bias-add','classifier-relu']) {
    const parent = graph.nodes.find(n=>n.id===source)
    if(!parent) throw Error('Expected transformer operation '+source)
    const id=source+'-dropout'
    if(graph.nodes.some(n=>n.id===id)) throw Error('Dropout already inserted at '+source)
    graph.edges=graph.edges.map(e=>e.source===source?{...e,source:id}:e)
    graph.nodes.push({id,type:'dropout',label:source+' dropout',params:{dropoutRate:rate},position:{x:parent.position.x+200,y:parent.position.y}})
    graph.edges.push({id:id+':0',source,target:id,inputSlot:0})
  }
  return organizeTextModel(graph)
}

/** Split a single head into two independent 16-wide heads at model width 32.
 * Keep total width, parameter count, and all initial parameter values fixed.
 * Each head owns ordinary projection, score, softmax and weighted-sum blocks. */
export function withTwoAttentionHeads(input: GraphModel): GraphModel {
  const projection = (name: string) => {
    const value = input.nodes.find(n => n.id === name + '-weights')?.params.value
    if (!value || typeof value === 'number' || value.shape.length !== 2 || value.shape[0] !== value.shape[1] || value.shape[1] % 2) throw Error('Two heads require square projections with an even model width.')
    return value
  }
  const width = projection('query').shape[1], headWidth = width / 2
  const source = input.edges.find(e => e.target === 'query' && e.inputSlot === 0)?.source
  if (!source || !input.nodes.some(n => n.id === 'attended-values')) throw Error('Expected single-head attention graph.')
  const causal = input.nodes.some(n => n.id === 'causal-scores')
  const removed = new Set(['query','key','value','query-weights','key-weights','value-weights','key-transpose','attention-scores','scaled-scores','causal-scores','attention-probabilities','attended-values'])
  const graph: GraphModel = {...input,nodes:input.nodes.filter(n=>!removed.has(n.id)).map(n=>({...n,params:{...n.params}})),edges:input.edges.filter(e=>!removed.has(e.target)&&(!removed.has(e.source)||e.source==='attended-values')).map(e=>({...e}))}
  const add = (id: string, type: NodeType, params: NodeParams, sources: string[]) => {
    graph.nodes.push({id,type,label:id.replace(/-/g,' '),params,position:{x:1300,y:graph.nodes.length*40}})
    sources.forEach((source,inputSlot)=>graph.edges.push({id:id+':'+inputSlot,source,target:id,inputSlot}))
    return id
  }
  const outputs = [0,1].map(head=>{
    const prefix = `attention-head-${head+1}-`
    const projected = ['query','key','value'].map(name=>{
      const original = projection(name)
      if (original.shape[1] !== width) throw Error('Projection widths must agree.')
      const data = Array.from({length:width*headWidth},(_,i)=>original.data[Math.floor(i/headWidth)*width+head*headWidth+i%headWidth])
      const weight = add(prefix+name+'-weights','weight',{value:{shape:[width,headWidth],data}},[])
      return add(prefix+name,'matmul',{},[source,weight])
    })
    const transpose = add(prefix+'key-transpose','tensor-transform',{transform:'transpose',axes:[1,0]},[projected[1]])
    let scores = add(prefix+'scores','matmul',{},[projected[0],transpose])
    scores = add(prefix+'scaled-scores','arithmetic',{expression:'x1 / '+Math.sqrt(headWidth)},[scores])
    if(causal) scores=add(prefix+'causal-scores','causal-mask',{},[scores])
    const weights=add(prefix+'probabilities','softmax',{},[scores])
    return add(prefix+'values','matmul',{},[weights,projected[2]])
  })
  add('attended-values','concat',{axis:1},outputs)
  return organizeTextModel(graph)
}
