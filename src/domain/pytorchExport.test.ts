import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
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

  it('keeps dataset examples outside the Python and notebook files', () => {
    const graph = createModelPreset('linear')
    const parameter = graph.nodes.find(node => node.type === 'weight')!
    parameter.params.value = 3.75
    const exported = generatePyTorchExport(graph)
    const script = exported.script
    expect(script).toContain('torch.tensor([3.75], dtype=torch.float64).reshape([])')
    expect(script).toContain('DATASET = load_dataset()')
    expect(script).toContain('json.load(source)')
    expect(script).not.toContain('-3.62')
    expect(exported.datasetFile?.name).toBe('backprop-builder-dataset.json')
    expect(JSON.parse(exported.datasetFile!.content).examples).toHaveLength(20)
    expect(exported.notebook).not.toContain('-3.62')
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
    expect(exported.script).toContain('DATASET_FILE = "measurements.csv"')
    expect(exported.script).toContain('csv.reader(source)')
    expect(exported.datasetFile).toBeUndefined()
    expect(exported.script).not.toContain('answer,feature')
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
  const syntaxPython = [python, process.env.PYTHON, 'python3'].find(candidate =>
    candidate && spawnSync(candidate, ['-c', 'import ast'], { encoding: 'utf8' }).status === 0)
  it.skipIf(!syntaxPython)('generates syntactically valid Python for built-in and custom datasets', () => {
    for (const graph of [createModelPreset('linear'), createModelPreset('attention'), customArithmeticGraph()]) {
      const { script } = generatePyTorchExport(graph)
      const result = spawnSync(syntaxPython!, ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'], { input: script, encoding: 'utf8' })
      expect(result.status, result.stderr).toBe(0)
    }
  })
  it.skipIf(!syntaxPython)('loads built-in data from a sibling JSON file and custom data from its CSV', () => {
    for (const graph of [createModelPreset('linear'), customArithmeticGraph()]) {
      const exported = generatePyTorchExport(graph)
      const directory = mkdtempSync(join(tmpdir(), 'backprop-data-'))
      try {
        if (exported.datasetFile) writeFileSync(join(directory, exported.datasetFile.name), exported.datasetFile.content)
        else writeFileSync(join(directory, 'measurements.csv'), 'answer,feature\n2,1\n4,2\n6,3\n8,4\n')
        const loader = exported.script.slice(exported.script.indexOf('DATASET_DIR = '), exported.script.indexOf('def binary_cross_entropy'))
        const program = `import csv, json\nfrom pathlib import Path\n${loader}\nprint(len(DATASET['examples']))\nprint(DATASET['examples'][0]['features'][0]['data'][0])\n`
        const result = spawnSync(syntaxPython!, ['-c', program], { cwd: directory, encoding: 'utf8' })
        expect(result.status, result.stderr).toBe(0)
        expect(result.stdout.trim().split('\n')).toEqual(exported.datasetFile ? ['20', '-2.4'] : ['4', '1.0'])
      } finally { rmSync(directory, { recursive: true, force: true }) }
    }
  })
  function runWithDataset(exported: ReturnType<typeof generatePyTorchExport>, script: string, harness?: string) {
    const directory = mkdtempSync(join(tmpdir(), 'backprop-export-'))
    try {
      writeFileSync(join(directory, 'backprop-builder-model.py'), script)
      if (exported.datasetFile) writeFileSync(join(directory, exported.datasetFile.name), exported.datasetFile.content)
      else writeFileSync(join(directory, 'measurements.csv'), 'answer,feature\n2,1\n4,2\n6,3\n8,4\n')
      return spawnSync(python!, harness ? ['-c', harness] : ['backprop-builder-model.py'], {
        cwd: directory, input: harness ? script : undefined, encoding: 'utf8', timeout: 120_000, env: { ...process.env, MPLBACKEND: 'Agg' },
      })
    } finally { rmSync(directory, { recursive: true, force: true }) }
  }
  it.skipIf(!python)('runs the exported mini-batch loop and reports both losses', () => {
    const exported = generatePyTorchExport(createModelPreset('linear'), { batchSize: 4, epochs: 2, reportEvery: 1 })
    const run = runWithDataset(exported, exported.script)
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
      const exported = generatePyTorchExport(graph)
      const script = exported.script
      const harness = `import sys, json\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrow = ns['DATASET']['examples'][${exampleIndex}]\nfeatures = [ns['tensor_value'](value) for value in row['features']]\ntarget = ns['tensor_value'](row['target'])\noutput, loss, _ = model(features, target)\nprint('EXPORT_OUTPUT=' + str(output.numel()))\nprint('EXPORT_LOSS=' + str(float(loss)) if loss is not None else 'EXPORT_LOSS=none')\n`
      const run = runWithDataset(exported, script, harness)
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
      const exported = generatePyTorchExport(createModelPreset(preset))
      const script = exported.script.replace('TRAIN_EPOCHS = 10', 'TRAIN_EPOCHS = 1')
      const run = runWithDataset(exported, script)
      expect(run.status, `${preset}: ${run.stderr}`).toBe(0)
      expect(run.stdout).toContain('Test predictions')
      if (preset === 'linear') expect(run.stdout).toContain('Epoch 1: train loss=')
    }
  }, 180_000)

  it.skipIf(!python)('matches a custom CSV batch and arithmetic expression', () => {
    const graph = customArithmeticGraph()
    const expected = forwardPass(graph).loss!
    const exported = generatePyTorchExport(graph)
    const script = exported.script
    const harness = `import sys\nns = {}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0], ns)\nmodel = ns['BuilderModel']()\nrows = [row for row in ns['DATASET']['examples'] if row['split'] == 'train']\nfeatures, target = ns['model_inputs'](rows)\n_, loss, _ = model(features, target)\nprint('EXPORT_LOSS=' + str(float(loss)))\n`
    const run = runWithDataset(exported, script, harness)
    expect(run.status, run.stderr).toBe(0)
    expect(Number(run.stdout.match(/EXPORT_LOSS=([^\n]+)/)?.[1])).toBeCloseTo(expected, 7)
  }, 120_000)
})
