import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'
import { createEmptyGraph, createNode } from './examples'
import { parseCustomCsv } from './customCsv'
import { datasetExamplesForNode } from './datasets'
import { forwardPass } from './engine'
import { withDatasetExample } from './datasetTraining'
import { generatePyTorchExport } from './pytorchExport'
import type { GraphModel } from './types'

function customArithmeticGraph(): GraphModel {
  const dataset = createNode('dataset', 1)
  const csv = parseCustomCsv('answer,feature\n2,1\n4,2\n6,3\n8,4\n', 'measurements.csv')
  csv.targetColumn = 0
  dataset.params = { dataset: 'custom-csv', customCsv: csv, datasetMode: 'batch', datasetSplit: 'train' }
  const parameter = createNode('weight', 2)
  parameter.params.value = 0.5
  const arithmetic = createNode('arithmetic', 3)
  arithmetic.params.expression = 'x1 * x2 + 1'
  const target = createNode('target', 4)
  const loss = createNode('loss', 5)
  loss.params.loss = 'mse'
  return { learningRate: 0.01, nodes: [dataset, parameter, arithmetic, target, loss], edges: [
    { id: 'feature', source: dataset.id, sourceSlot: 1, target: arithmetic.id, inputSlot: 0 },
    { id: 'parameter', source: parameter.id, target: arithmetic.id, inputSlot: 1 },
    { id: 'target-column', source: dataset.id, sourceSlot: 0, target: target.id, inputSlot: 0 },
    { id: 'prediction', source: arithmetic.id, target: loss.id, inputSlot: 0 },
    { id: 'loss-target', source: target.id, target: loss.id, inputSlot: 1 },
  ] }
}

describe('PyTorch export', () => {
  it.each(LESSONS.map(lesson => lesson.id))(
    'produces a runnable notebook for the %s preset', (preset) => {
      const exported = generatePyTorchExport(createModelPreset(preset))
      const notebook = JSON.parse(exported.notebook)
      expect(notebook.nbformat).toBe(4)
      expect(notebook.cells.find((cell: { cell_type: string }) => cell.cell_type === 'code').source.join('')).toBe(exported.script)
      expect(exported.script).toContain('class BuilderModel(nn.Module):')
      expect(exported.script).toContain('torch.optim.SGD')
      expect(exported.script).toContain('Test predictions')
    },
  )

  it('preserves the dataset and current parameters in an editable Python file', () => {
    const graph = createModelPreset('linear')
    const parameter = graph.nodes.find(node => node.type === 'weight')!
    parameter.params.value = 3.75
    const script = generatePyTorchExport(graph).script
    expect(script).toContain('torch.tensor([3.75], dtype=torch.float64).reshape([])')
    expect(script).toContain('Line regression (1D)')
    expect(script).toContain('TRAIN_EPOCHS = 10')
  })

  it('exports mini-batch and dual-loss reporting settings', () => {
    const script = generatePyTorchExport(createModelPreset('linear'), { batchSize: 4, shuffleEachEpoch: false, epochs: 3, reportEvery: 2 }).script
    expect(script).toContain('BATCH_MODE = EVALUATE_FULL_BATCH or BATCH_SIZE > 1')
    expect(script).toContain('BATCH_SIZE = 4')
    expect(script).toContain('train_loader = DataLoader(')
    expect(script).toContain('SHUFFLE_EACH_EPOCH = False')
    expect(script).toContain('TRAIN_EPOCHS = 3')
    expect(script).toContain('REPORT_EVERY = 2')
    expect(script).toContain('held_out_loss')
    expect(script).toContain("label='Held-out (validation)'")
    expect(() => generatePyTorchExport(createModelPreset('attention'), { batchSize: 4 })).toThrow('tensor-shaped dataset')
  })

  it('exports a hand-built arithmetic graph with custom CSV columns', () => {
    const exported = generatePyTorchExport(customArithmeticGraph())
    expect(exported.script).toContain('Custom CSV · measurements.csv')
    expect(exported.script).toContain('v_dataset_1_s1 = features[0]')
    expect(exported.script).toContain('v_dataset_1_s0 = target')
    expect(exported.script).toContain(' * ')
  })

  it('refuses incomplete builders with an actionable error', () => {
    expect(() => generatePyTorchExport(createEmptyGraph())).toThrow('Dataset block')
    const graph = createModelPreset('linear')
    graph.edges = graph.edges.filter(edge => edge.target !== 'loss' || edge.inputSlot !== 0)
    expect(() => generatePyTorchExport(graph)).toThrow('Finish the graph')
  })

  const python = process.env.PYTORCH_TEST_PYTHON
  it.skipIf(!python)('runs the exported mini-batch loop and reports both losses', () => {
    const script = generatePyTorchExport(createModelPreset('linear'), { batchSize: 4, epochs: 2, reportEvery: 1 }).script
    const run = spawnSync(python!, ['-c', 'import sys; exec(sys.stdin.read())'], { input: script, encoding: 'utf8', timeout: 120_000, env: { ...process.env, MPLBACKEND: 'Agg' } })
    expect(run.status, run.stderr).toBe(0)
    expect(run.stdout).toContain('Epoch 2: train loss=')
    expect(run.stdout).toContain('held-out loss=')
  }, 120_000)

  it.skipIf(!python)('runs every preset in PyTorch and matches builder losses', () => {
    for (const { id: preset } of LESSONS) {
      const graph = createModelPreset(preset)
      const dataset = graph.nodes.find(node => node.type === 'dataset')!
      const exampleIndex = datasetExamplesForNode(dataset).findIndex(example => example.split === 'train')
      const expected = forwardPass(withDatasetExample(graph, dataset.id, exampleIndex)).loss
      const script = generatePyTorchExport(graph).script
      const harness = `import sys, json\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrow = ns['DATASET']['examples'][${exampleIndex}]\nfeatures = [ns['tensor_value'](value) for value in row['features']]\ntarget = ns['tensor_value'](row['target'])\noutput, loss, _ = model(features, target)\nprint('EXPORT_OUTPUT=' + str(output.numel()))\nprint('EXPORT_LOSS=' + str(float(loss)) if loss is not None else 'EXPORT_LOSS=none')\n`
      const run = spawnSync(python!, ['-c', harness], { input: script, encoding: 'utf8', timeout: 120_000 })
      expect(run.status, `${preset}: ${run.stderr}`).toBe(0)
      expect(Number(run.stdout.match(/EXPORT_OUTPUT=([^\n]+)/)?.[1])).toBeGreaterThan(0)
      if (expected !== undefined) {
        const actual = Number(run.stdout.match(/EXPORT_LOSS=([^\n]+)/)?.[1])
        expect(actual, preset).toBeCloseTo(expected, 7)
      }
    }
  }, 240_000)

  it.skipIf(!python)('runs training and inference exports end to end', () => {
    for (const preset of ['linear', 'attention'] as const) {
      const script = generatePyTorchExport(createModelPreset(preset)).script.replace('TRAIN_EPOCHS = 10', 'TRAIN_EPOCHS = 1')
      const run = spawnSync(python!, ['-c', 'import sys; exec(sys.stdin.read())'], { input: script, encoding: 'utf8', timeout: 120_000 })
      expect(run.status, `${preset}: ${run.stderr}`).toBe(0)
      expect(run.stdout).toContain('Test predictions')
      if (preset === 'linear') expect(run.stdout).toContain('Epoch 1: train loss=')
    }
  }, 180_000)

  it.skipIf(!python)('matches a custom CSV batch and arithmetic expression', () => {
    const graph = customArithmeticGraph()
    const expected = forwardPass(graph).loss!
    const script = generatePyTorchExport(graph).script
    const harness = `import sys\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrows = [row for row in ns['DATASET']['examples'] if row['split'] == 'train']\nfeatures, target = ns['model_inputs'](rows)\n_, loss, _ = model(features, target)\nprint('EXPORT_LOSS=' + str(float(loss)))\n`
    const run = spawnSync(python!, ['-c', harness], { input: script, encoding: 'utf8', timeout: 120_000 })
    expect(run.status, run.stderr).toBe(0)
    expect(Number(run.stdout.match(/EXPORT_LOSS=([^\n]+)/)?.[1])).toBeCloseTo(expected, 7)
  }, 120_000)
})
