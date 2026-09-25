import { describe, expect, it } from 'vitest'
import { analyzeCustomCsv, parseCustomCsv } from './customCsv'
import { customCsvCardWidth, datasetExamples, datasetForNode, datasetOutputValueForSlot } from './datasets'
import { forwardPass, parameterValues, runTrainingStep } from './engine'
import { createStarterGraph } from './examples'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { layoutSemanticGraph } from './semanticLayout'

describe('custom CSV datasets', () => {
  it('reads quoted headers and numeric columns, keeps train/test rows aligned, and survives project save/import', () => {
    const csv = parseCustomCsv('\uFEFF"length, cm",height,result\r\n1,2,3\r\n2,3,5\r\n3,4,7\r\n4,5,9\r\n', 'measurements.csv')
    expect(csv.hasHeader).toBe(true)
    expect(csv.task).toBe('regression')
    const graph = createStarterGraph(true)
    const source = graph.nodes.find(node => node.type === 'dataset')!
    source.params = { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', datasetSplit: 'train' }
    const dataset = datasetForNode(source)
    expect(dataset.featureLabels).toEqual(['length, cm', 'height'])
    expect(dataset.targetLabel).toBe('result')
    expect(datasetExamples(dataset).map(example => example.split)).toEqual(['test', 'train', 'train', 'train'])
    expect(datasetOutputValueForSlot(source, 0).data).toEqual([2, 3, 4])
    expect(datasetOutputValueForSlot(source, 2).data).toEqual([5, 7, 9])

    const file = createProjectStateFile({ graph, visualizationGraph: graph, initialParameterValues: parameterValues(graph), selectedNodeIds: [], phase: 'edit', traceSteps: [], traceIndex: 0, epoch: 0, currentLoss: null, display: { showMath: true, showGradient: true, showCode: false, showVisualization: false } })
    const imported = parseProjectStateFile(JSON.stringify(file))
    expect(imported.ok).toBe(true)
    if (!imported.ok) return
    const restored = imported.file.state.graph.nodes.find(node => node.type === 'dataset')!
    expect(datasetForNode(restored).targetValue.data).toEqual([3, 5, 7, 9])
  })

  it('encodes string classes, supports a headerless CSV, and rejects invalid feature cells', () => {
    const csv = parseCustomCsv('1,2,no\n2,3,yes\n3,4,no\n4,5,yes\n', 'labels.csv')
    expect(csv.hasHeader).toBe(false)
    expect(csv.task).toBe('binary-classification')
    expect(analyzeCustomCsv(csv).classLabels).toEqual(['no', 'yes'])
    expect(analyzeCustomCsv(csv).targets).toEqual([0, 1, 0, 1])
    expect(() => parseCustomCsv('x,y\n1,2\nnot-a-number,3\n', 'bad.csv')).toThrow(/must be numeric/)
    expect(() => parseCustomCsv('x,y\n1,2\n3\n', 'bad.csv')).toThrow(/same number of columns/)
    const firstColumnLabels = parseCustomCsv('class,length,width\ncat,1,2\ndog,2,3\ncat,3,4\n', 'first.csv')
    expect(firstColumnLabels.targetColumn).toBe(0)
    expect(analyzeCustomCsv(firstColumnLabels).classLabels).toEqual(['cat', 'dog'])
  })

  it('feeds a numeric CSV through an ordinary user-built training graph', () => {
    const csv = parseCustomCsv('x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n', 'line.csv')
    const graph = createStarterGraph(true)
    const source = graph.nodes.find(node => node.type === 'dataset')!
    source.params = { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', datasetSplit: 'train' }
    const initial = forwardPass(graph)
    expect(initial.loss).toBeGreaterThan(0)
    expect(runTrainingStep(initial.graph).loss).toBeLessThan(initial.loss!)
  })

  it('reserves canvas space for long column names', () => {
    const csv = parseCustomCsv('very_long_feature_column_name,second_long_feature_column_name,target_label\n1,2,3\n2,3,5\n3,4,7\n', 'wide.csv')
    const graph = createStarterGraph(true)
    const source = graph.nodes.find(node => node.type === 'dataset')!
    source.params = { dataset: 'custom-csv', customCsv: csv }
    const width = customCsvCardWidth(source)
    expect(width).toBeGreaterThan(350)
    expect(layoutSemanticGraph(graph).nodes.get(source.id)?.width).toBe(width)
  })
})
