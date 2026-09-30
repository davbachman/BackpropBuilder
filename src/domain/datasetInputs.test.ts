import { expect, it } from 'vitest'
import { datasetInputPorts, withDatasetInputAliases } from './datasetInputs'
import { createNode } from './examples'
import { parseCustomCsv } from './customCsv'
import { forwardPass } from './engine'
import type { GraphModel } from './types'

function fixture(): GraphModel {
  const data = createNode('dataset', 0)
  data.params = { dataset: 'custom-csv', customCsv: parseCustomCsv('radius,unused,width,y\n1,9,2,0\n3,8,4,1\n', 'features.csv'), datasetMode: 'batch' }
  const standard = createNode('standardize', 0)
  standard.params.standardization = { mean: [2], scale: [1], count: 2 }
  const sum = createNode('arithmetic', 0)
  sum.params.expression = 'x1 + x2'
  const loss = createNode('loss', 0)
  return { nodes: [data, standard, sum, loss], learningRate: .1, edges: [
    { id: 'radius', source: data.id, target: standard.id, inputSlot: 0 },
    { id: 'standard', source: standard.id, target: sum.id, inputSlot: 0 },
    { id: 'width', source: data.id, sourceSlot: 2, target: sum.id, inputSlot: 1 },
    { id: 'prediction', source: sum.id, target: loss.id, inputSlot: 0 },
    { id: 'target', source: data.id, sourceSlot: 3, target: loss.id, inputSlot: 1 },
  ] }
}

it('infers only dataset ports contributing to predictions through preprocessing', () => {
  const graph = fixture()
  expect(datasetInputPorts(graph)).toEqual([{ nodeId: 'dataset-0', sourceSlot: 0 }, { nodeId: 'dataset-0', sourceSlot: 2 }])
  const snapshot = structuredClone(graph)
  const projected = withDatasetInputAliases(graph)
  expect(forwardPass(projected).loss).toBe(forwardPass(graph).loss)
  expect(graph).toEqual(snapshot)
  graph.nodes.find(node => node.type === 'loss')!.type = 'cross-entropy'
  expect(datasetInputPorts(graph)).toHaveLength(2)
})

it('updates after rewiring and excludes target ancestry and disconnected branches', () => {
  const graph = fixture()
  graph.edges = graph.edges.filter(edge => edge.id !== 'prediction')
  expect(datasetInputPorts(graph)).toEqual([])
  graph.edges.push({ id: 'direct', source: 'dataset-0', sourceSlot: 1, target: 'loss-0', inputSlot: 0 })
  expect(datasetInputPorts(graph)).toEqual([{ nodeId: 'dataset-0', sourceSlot: 1 }])
  // A column used on both sides is a target, not an independent input.
  graph.edges.find(edge => edge.id === 'target')!.sourceSlot = 1
  expect(datasetInputPorts(graph)).toEqual([])
})
