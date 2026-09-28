import { describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { prepareTextDocuments, textDataset } from './textData'
import { buildTextModel, type TextModelKind } from '../test/textModels'
import { backwardPass, forwardPass } from './engine'
import { generatePyTorchExport } from './pytorchExport'
import type { TextDatasetData } from './types'

const python = process.env.PYTORCH_TEST_PYTHON
const kinds: (TextModelKind | 'one-hot' | 'tied-embeddings')[] = ['counts-linear','counts-mlp','mean','position-mean','attention','position-attention','transformer','alice-baseline','alice-transformer','one-hot','tied-embeddings']
describe('text model PyTorch parity', () => {
  it('exports large count datasets in space proportional to observed tokens', () => {
    const data: TextDatasetData = {
      version: 1, fileName: 'large.csv', task: 'sentiment', tokenizer: 'word', representation: 'counts', lowercase: true,
      vocabulary: ['<unk>', 'good', 'bad', ...Array.from({ length: 3997 }, (_, index) => `w${index}`)],
      maxLength: 3, stride: 3,
      documents: Array.from({ length: 1000 }, (_, index) => ({ text: index % 2 ? 'bad unseen' : 'good good bad ignored', label: index % 2 ? 'negative' : 'positive', split: index < 800 ? 'train' : 'test' })),
    }
    const graph = buildTextModel(data, 'counts-linear', 4)
    const exported = generatePyTorchExport(graph)
    expect(exported.datasetFile!.content.length).toBeLessThan(500_000)
    const saved = JSON.parse(exported.datasetFile!.content)
    expect(saved.examples[0].features[0]).toEqual({ shape: [1, 4000], indices: [1, 2], data: [2, 1] })
    expect(saved.examples[1].features[0]).toEqual({ shape: [1, 4000], indices: [2, 0], data: [1, 1] })
    expect(saved.examples[0].features[1]).toEqual(textDataset(data).examples![0].features[1])
    expect(saved.textData).toEqual(data)
  })

  it.skipIf(!python)('reconstructs sparse fixed-length count inputs with unknown tokens and unchanged padding', () => {
    const data = prepareTextDocuments([
      { text: 'good good film', label: 'positive', split: 'train' },
      { text: 'bad film', label: 'negative', split: 'train' },
      { text: 'good unseen', label: 'positive', split: 'test' },
    ], 'padded.csv', { task: 'sentiment', representation: 'counts', fixedLength: true, maxLength: 4 })
    const graph = buildTextModel(data, 'counts-linear', 4)
    const exported = generatePyTorchExport(graph, { epochs: 1 })
    const directory = mkdtempSync(join(tmpdir(), 'backprop-count-export-'))
    try {
      writeFileSync(join(directory, exported.datasetFile!.name), exported.datasetFile!.content)
      const harness = `import sys,json\nns={}\nexec(sys.stdin.read(),ns)\nprint('FEATURES=' + json.dumps([[ns['tensor_value'](value).tolist() for value in row['features']] for row in ns['DATASET']['examples']]))\n`
      const result = spawnSync(python!, ['-c', harness], { cwd: directory, input: exported.script, encoding: 'utf8', timeout: 30_000 })
      expect(result.status, result.stderr).toBe(0)
      const actual = JSON.parse(result.stdout.match(/FEATURES=([^\n]+)/)![1])
      expect(actual).toEqual(textDataset(data).examples!.map(example => [
        [example.features[0].data], example.features[1].data,
      ]))
      expect(result.stdout).toContain('Epoch 1: train loss=')
    } finally { rmSync(directory, { recursive: true, force: true }) }
  }, 30_000)

  it.skipIf(!python).each(kinds)('matches forward loss and all parameter gradients for %s', kind => {
    const data = prepareTextDocuments([{text:'good film good film',label:'positive',split:'train'}, {text:'bad movie bad movie',label:'negative',split:'train'}, {text:'good new movie',label:'positive',split:'test'}], 'test.csv', {task:kind.startsWith('alice') || kind === 'tied-embeddings' ? 'language' : 'sentiment',maxLength:4})
    const graph = buildTextModel(data,kind === 'one-hot' ? 'mean' : kind === 'tied-embeddings' ? 'alice-transformer' : kind,4)
    if (kind === 'one-hot') {
      graph.nodes = [...graph.nodes.map(node => node.id === 'embedded-tokens' ? {...node,type:'matmul' as const} : node), {id:'one-hot',label:'One-hot',type:'one-hot',params:{numClasses:data.vocabulary.length},position:{x:0,y:0}}]
      graph.edges = [...graph.edges.filter(edge => edge.target !== 'embedded-tokens'), {id:'ids-hot',source:'text-data',sourceSlot:0,target:'one-hot',inputSlot:0}, {id:'hot-matrix',source:'one-hot',target:'embedded-tokens',inputSlot:0}, {id:'table-matrix',source:'word-embeddings',target:'embedded-tokens',inputSlot:1}]
    }
    if (kind === 'tied-embeddings') {
      graph.nodes = graph.nodes.map(node => node.id === 'vocabulary-weights' ? {...node,type:'tensor-transform',params:{transform:'transpose',axes:[1,0]}} : node)
      graph.edges.push({id:'tied-output',source:'word-embeddings',target:'vocabulary-weights',inputSlot:0})
    }
    const forward = forwardPass(graph, false), backward = backwardPass(forward.graph).graph
    const expected = graph.nodes.filter(node => node.type === 'weight' || node.type === 'bias').map(node => backward.nodes.find(result => result.id === node.id)!.grad?.data)
    const exported = generatePyTorchExport(graph, {epochs:1})
    const directory = mkdtempSync(join(tmpdir(),'backprop-text-export-'))
    try {
      writeFileSync(join(directory,exported.datasetFile!.name),exported.datasetFile!.content)
      const harness = `import sys,json\nns={}\nexec(sys.stdin.read().split('\\nmodel = BuilderModel()\\n')[0],ns)\nmodel=ns['BuilderModel']()\nrow=ns['DATASET']['examples'][0]\nfeatures=[ns['tensor_value'](value) for value in row['features']]\ntarget=ns['tensor_value'](row['target'])\noutput,loss,_=model(features,target)\nloss.backward()\nprint(json.dumps({'loss':float(loss.detach()),'gradients':[p.grad.reshape(-1).tolist() if p.grad is not None else None for p in model.parameters()]}))\n`
      const result = spawnSync(python!,['-c',harness], {cwd:directory,input:exported.script,encoding:'utf8',timeout:30000})
      expect(result.status,result.stderr).toBe(0)
      const actual = JSON.parse(result.stdout)
      expect(actual.loss).toBeCloseTo(forward.loss!,9)
      expect(actual.gradients).toHaveLength(expected.length)
      expected.forEach((gradient,i) => gradient?.forEach((value,j) => expect(actual.gradients[i][j]).toBeCloseTo(value,9)))
    } finally {rmSync(directory,{recursive:true,force:true})}
  },30000)
})
