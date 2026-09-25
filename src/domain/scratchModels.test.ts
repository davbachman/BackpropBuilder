import { describe, expect, it } from 'vitest'
import { scratchModel } from '../test/scratchModels'
import { backwardPass, forwardPass, parameterValues, runTrainingStep, validateGraph } from './engine'
import { datasetExamples, datasetExamplesForNode, datasetForNode, datasetOutputValueForSlot, DATASET_OPTIONS } from './datasets'
import { evaluateDataset, predictionNode, trainDataset, withDatasetExample } from './datasetTraining'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { createEmptyGraph, createNode } from './examples'
import { connectGraphNodes } from './graphEditing'
import { resolveReshape } from './reshape'
import { parseCustomCsv } from './customCsv'
import { tensorValue } from './tensor'

describe('models built from palette primitives', () => {
  it('makes a reproducible, class-balanced train/test split for a custom CSV', () => {
    const data = createNode('dataset', 1)
    const rows = Array.from({ length: 24 }, (_, index) => `${index},${index < 12 ? 'A' : 'B'}`)
    data.params = { dataset: 'custom-csv', customCsv: parseCustomCsv(`feature,label\n${rows.join('\n')}\n`, 'classes.csv'), datasetMode: 'batch', trainPercent: 75 }
    const examples = datasetExamplesForNode(data)
    expect(examples.map(example => example.split)).toEqual(datasetExamplesForNode(data).map(example => example.split))
    for (const label of [0, 1]) {
      const matching = examples.filter(example => example.target.data[0] === label)
      expect(matching.filter(example => example.split === 'train')).toHaveLength(9)
      expect(matching.filter(example => example.split === 'test')).toHaveLength(3)
    }
    expect(datasetOutputValueForSlot({ ...data, params: { ...data.params, datasetSplit: 'train' } }, 0).shape).toEqual([18])
    expect(datasetOutputValueForSlot({ ...data, params: { ...data.params, datasetSplit: 'test' } }, 0).shape).toEqual([6])
  })

  it('scores held-out classes from raw logits without a Softmax or Argmax block', () => {
    const csv = parseCustomCsv('x,label\n-4,A\n-3,A\n-2,A\n-1,A\n1,B\n2,B\n3,B\n4,B\n', 'binary.csv')
    const graph = createEmptyGraph()
    graph.nodes = [
      { id: 'data', type: 'dataset', label: 'Dataset', position: { x: 0, y: 0 }, params: { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', trainPercent: 75 } },
      { id: 'column', type: 'reshape', label: 'Column', position: { x: 100, y: 0 }, params: { shape: [-1, 1] } },
      { id: 'weights', type: 'weight', label: 'Weights', position: { x: 100, y: 150 }, params: { value: tensorValue([1, 2], [-1, 1]) } },
      { id: 'logits', type: 'matmul', label: 'Logits', position: { x: 250, y: 0 }, params: {} },
      { id: 'target', type: 'target', label: 'Target', position: { x: 250, y: 150 }, params: {} },
      { id: 'loss', type: 'loss', label: 'Cross entropy', position: { x: 400, y: 0 }, params: { loss: 'cross-entropy' } },
    ]
    graph.edges = [
      { id: 'feature', source: 'data', sourceSlot: 0, target: 'column', inputSlot: 0 },
      { id: 'column-logits', source: 'column', target: 'logits', inputSlot: 0 },
      { id: 'weights-logits', source: 'weights', target: 'logits', inputSlot: 1 },
      { id: 'label', source: 'data', sourceSlot: 1, target: 'target', inputSlot: 0 },
      { id: 'prediction', source: 'logits', target: 'loss', inputSlot: 0 },
      { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
    ]
    const test = evaluateDataset(graph, 'data', 'test')
    expect(test.accuracy).toBe(1)
    expect(test.predictions).toBe(test.examples)
    expect(test.examples).toBe(2)
    expect(test.rows).toHaveLength(2)
    expect(test.rows.every(row => row.actual === row.predicted && row.correct)).toBe(true)
    expect(new Set(test.rows.map(row => row.predicted))).toEqual(new Set(['A', 'B']))
  })
  it('trains numeric batches with an inferred reshape dimension across train/test splits', async () => {
    let graph = createEmptyGraph()
    graph.learningRate = .03
    const data = createNode('dataset',1), column = createNode('reshape',2), weight = createNode('weight',3), product = createNode('matmul',4), target = createNode('reshape',5), loss = createNode('loss',6)
    column.params.shape = [-1,1]; target.params.shape = [-1,1]
    weight.params.value = {shape:[1,1],data:[0]}; loss.params.loss = 'mse'
    graph.nodes = [data,column,weight,product,target,loss]
    for (const [source,sourceSlot,destination,inputSlot] of [[data.id,0,column.id,0],[column.id,0,product.id,0],[weight.id,0,product.id,1],[data.id,1,target.id,0],[product.id,0,loss.id,0],[target.id,0,loss.id,1]] as const) graph = connectGraphNodes(graph,{source,sourceHandle:`out-${sourceSlot}`,target:destination,targetHandle:`in-${inputSlot}`})!
    const before = evaluateDataset(graph,data.id,'train')
    const trained = await trainDataset(graph,data.id,5)
    expect(evaluateDataset(trained,data.id,'train').loss).toBeLessThan(before.loss)
    expect(evaluateDataset(trained,data.id,'test').examples).toBe(5)
    expect(trained.nodes.find(node=>node.id === column.id)?.value?.shape).toEqual([20,1])
    expect(resolveReshape([-1,2],12)).toEqual([6,2])
    expect(()=>resolveReshape([-1,-1],12)).toThrow()
    expect(()=>resolveReshape([-1,5],12)).toThrow()
  })

  it.each(['cnn','transformer'] as const)('executes, differentiates, trains and saves a %s with arbitrary IDs', async kind => {
    const {graph,datasetId} = scratchModel(kind)
    expect(validateGraph(graph).filter(issue=>issue.code !== 'disconnected')).toEqual([])
    const forward = forwardPass(graph), backward = backwardPass(forward.graph)
    expect(predictionNode(forward.graph)?.value?.shape).toEqual(kind === 'cnn' ? [1,10] : [3,5])
    for (const parameter of backward.graph.nodes.filter(node=>node.type === 'weight' || node.type === 'bias')) {
      expect(parameter.grad?.data.every(Number.isFinite)).toBe(true)
    }
    expect(runTrainingStep(graph).loss).toBeLessThan(forward.loss!)
    const baseline = evaluateDataset(graph,datasetId,'train')
    const trained = await trainDataset(graph,datasetId,1)
    expect(evaluateDataset(trained,datasetId,'train').loss).toBeLessThan(baseline.loss)
    const parameters = parameterValues(trained)
    const test = evaluateDataset(trained,datasetId,'test')
    expect(test.accuracy).toBeGreaterThanOrEqual(0)
    expect(parameterValues(trained)).toEqual(parameters)
    expect(trained.nodes.find(node=>node.id === datasetId)?.params).toEqual(graph.nodes.find(node=>node.id === datasetId)?.params)
    expect(trained.nodes.map(node=>node.position)).toEqual(graph.nodes.map(node=>node.position))
    const file = createProjectStateFile({graph:trained,visualizationGraph:trained,initialParameterValues:parameterValues(graph),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:1,currentLoss:null,display:{showMath:true,showGradient:false,showCode:false,showVisualization:false}})
    const loaded = parseProjectStateFile(JSON.stringify(file))
    expect(loaded.ok).toBe(true)
    if (loaded.ok) expect(forwardPass(loaded.file.state.graph).loss).toBeCloseTo(forwardPass(trained).loss!,12)
  },20000)

  it('never uses held-out labels when training and aborts without mutating the graph', async () => {
    const {graph,datasetId} = scratchModel('transformer'), before = structuredClone(graph)
    const data = datasetForNode(graph.nodes.find(node=>node.id === datasetId)!)
    const original = data.examples!
    const trained = await trainDataset(graph,datasetId,1)
    try {
      data.examples = original.map(example=>example.split === 'test' ? {...example,target:{shape:example.target.shape,data:example.target.data.map(()=>4)}} : example)
      expect(parameterValues(await trainDataset(graph,datasetId,1))).toEqual(parameterValues(trained))
    } finally { data.examples = original }
    const controller = new AbortController(); controller.abort()
    await expect(trainDataset(graph,datasetId,1,{signal:controller.signal})).rejects.toThrow('stopped')
    expect(graph).toEqual(before)
  })

  it.each(LESSONS)('all $id preset data comes through a dataset block', preset => {
    const graph = createModelPreset(preset.id)
    const data = graph.nodes.find(node=>node.type === 'dataset')!
    expect(data).toBeDefined()
    for (const node of graph.nodes.filter(node=>node.type === 'target' || (node.type === 'input' && !node.label.includes('√')))) {
      expect(graph.edges.some(edge=>edge.source === data.id && edge.target === node.id)).toBe(true)
    }
    const before = parameterValues(graph)
    const changed = forwardPass(withDatasetExample(graph,data.id,1)).graph
    expect(parameterValues(changed)).toEqual(before)
    for (const edge of changed.edges.filter(edge=>edge.source === data.id)) expect(edge.value).toEqual(datasetOutputValueForSlot(changed.nodes.find(node=>node.id===data.id)!,edge.sourceSlot ?? 0))
  })

  it('provides aligned finite features/targets and separate splits in every dataset', () => {
    for (const dataset of DATASET_OPTIONS) {
      const examples = datasetExamples(dataset)
      expect(new Set(examples.map(example=>example.split))).toEqual(new Set(['train','test']))
      for (const example of examples) {
        expect(example.features).toHaveLength(dataset.featureLabels.length)
        if (dataset.task.includes('classification') || dataset.task === 'sequence') {
          const classes = dataset.vocabulary?.length ?? (dataset.kind === 'digits-8x8' ? 10 : dataset.kind === 'class-scores' ? 4 : 2)
          expect(example.target.data.every(label=>Number.isInteger(label) && label >= 0 && label < classes)).toBe(true)
        }
        for (const value of [...example.features,example.target]) {
          expect(value.data.length).toBe(value.shape.reduce((a,b)=>a*b,1))
          expect(value.data.every(Number.isFinite)).toBe(true)
        }
      }
    }
  })
})
