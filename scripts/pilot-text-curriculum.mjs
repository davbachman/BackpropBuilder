import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { createServer } from 'vite'

const output = resolve(process.env.TEXT_PILOT_OUTPUT || 'output/text-curriculum')
const imdbRoot = process.env.IMDB_ROOT || '/tmp/backprop-imdb/aclImdb'
const server = await createServer({ server: {middlewareMode: true, hmr: false}, appType: 'custom', logLevel: 'error' })
try {
  const {prepareTextDocuments} = await server.ssrLoadModule('/src/domain/textData.ts')
  const {buildTextModel} = await server.ssrLoadModule('/src/test/textModels.ts')
  const {evaluateDataset, trainDataset} = await server.ssrLoadModule('/src/domain/datasetTraining.ts')
  const {forwardPass, parameterValues, validateGraph} = await server.ssrLoadModule('/src/domain/engine.ts')
  const {createProjectStateFile, parseProjectStateFile} = await server.ssrLoadModule('/src/domain/session.ts')
  const {generatePyTorchExport} = await server.ssrLoadModule('/src/domain/pytorchExport.ts')
  const {generateText} = await server.ssrLoadModule('/src/domain/textGeneration.ts')
  await mkdir(join(output, 'data'), {recursive: true})
  await mkdir(join(output, 'models'), {recursive: true})
  await mkdir(join(output, 'exports'), {recursive: true})
  const documents = []
  for (const split of ['train', 'test']) {
    for (const label of ['neg', 'pos']) {
      const directory = join(imdbRoot, split, label)
      const names = (await readdir(directory)).filter(name => name.endsWith('.txt')).sort((a,b) => Number(a.split('_')[0])-Number(b.split('_')[0]))
      const count = split === 'train' ? 400 : 100
      // Uniformly spread IDs through the official split; deterministic and balanced.
      for (let i = 0; i < count; i++) documents.push({text: await readFile(join(directory, names[Math.floor(i * names.length/count)]), 'utf8'), label: label === 'pos' ? 'positive' : 'negative', split})
    }
  }
  const reviews = prepareTextDocuments(documents, 'imdb-pilot.csv', {task: 'sentiment', vocabularySize: 1000, maxLength: 64, representation: 'counts'})
  const quote = value => '"' + value.replaceAll('"', '""') + '"'
  await writeFile(join(output, 'data/imdb-pilot.csv'), 'text,label,split\n' + documents.map(doc => [quote(doc.text), doc.label, doc.split].join(',')).join('\n'))
  await writeFile(join(output, 'data/imdb-prepared.json'), JSON.stringify(reviews))
  const raw = await readFile(join(output, 'data/alice-original.txt'), 'utf8')
  const body = raw.split(/\*\*\* START OF[^\n]*\n/)[1]?.split(/\*\*\* END OF/)[0] ?? raw
  const paragraphs = body.split(/\n\s*\n/).map(p => p.replace(/\s+/g, ' ').trim()).filter(p => p.length >= 100)
  const boundary = Math.floor(paragraphs.length * .8)
  const passages = [...paragraphs.slice(0, boundary).filter((_,i) => i % 5 === 0).slice(0, 60).map(text => ({text, split:'train'})), ...paragraphs.slice(boundary).filter((_,i) => i % 3 === 0).slice(0, 20).map(text => ({text, split:'test'}))]
  const alice = prepareTextDocuments(passages, 'alice-pilot.txt', {task:'language', tokenizer:'character', vocabularySize:1000, maxLength:16, stride:16})
  await writeFile(join(output, 'data/alice-prepared.json'), JSON.stringify(alice))
  const configurations = [
    ['counts-linear', reviews, 24, .001], ['counts-mlp', reviews, 24, .001],
    ['mean', reviews, 48, .025], ['position-mean', reviews, 8, .01],
    ['attention', reviews, 24, .01], ['position-attention', reviews, 12, .01], ['transformer', reviews, 20, .003],
    ['alice-baseline', alice, 8, .06], ['alice-transformer', alice, 12, .025],
  ]
  const results = []
  let learnedMean, learnedAttention
  for (const [kind, data, epochs, learningRate] of configurations) {
    if (process.env.TEXT_PILOT_MODELS && !process.env.TEXT_PILOT_MODELS.split(',').includes(kind)) continue
    let graph = buildTextModel(data, kind, data.task === 'language' ? 16 : 20)
    let inherit = kind === 'position-mean' || kind === 'attention' ? learnedMean : kind === 'position-attention' || kind === 'transformer' ? learnedAttention : undefined
    if (!inherit && process.env.TEXT_PILOT_MODELS) {
      const precursor = kind === 'position-mean' || kind === 'attention' ? 'mean' : kind === 'position-attention' || kind === 'transformer' ? 'attention' : undefined
      if (precursor) { try { inherit = JSON.parse(await readFile(join(output,'models',precursor+'.json'),'utf8')).state.graph } catch { throw Error('Run the '+precursor+' pilot before '+kind) } }
    }
    if (inherit) graph.nodes.forEach(node => { const previous = inherit.nodes.find(candidate => candidate.id === node.id); if (previous && ['weight','bias'].includes(node.type) && JSON.stringify(previous.params.value.shape) === JSON.stringify(node.params.value.shape)) node.params.value = structuredClone(previous.params.value) })
    graph.learningRate = learningRate
    const issues = validateGraph(graph).filter(issue => issue.code !== 'disconnected')
    if (issues.length) throw Error(JSON.stringify(issues))
    const initialParameters = parameterValues(graph)
    const record = {kind, vocabulary: data.vocabulary.length, maxLength:data.maxLength, parameters:Object.values(initialParameters).reduce((n,t) => n+t.data.length,0), learningRate, epochs, reports:[]}
    const started = performance.now()
    const report = epoch => {
      const train = evaluateDataset(graph, 'text-data', 'train')
      const test = evaluateDataset(graph, 'text-data', 'test')
      record.reports.push({epoch, trainLoss:train.loss, heldOutLoss:test.loss, trainAccuracy:train.accuracy, heldOutAccuracy:test.accuracy, trainExamples:train.examples, heldOutExamples:test.examples})
      console.log(kind, epoch, 'train', train.loss.toFixed(4), 'held-out', test.loss.toFixed(4), 'accuracy', test.accuracy?.toFixed(3), 'elapsed', ((performance.now()-started)/1000).toFixed(1))
    }
    report(0)
    for (let epoch=1; epoch<=epochs; epoch++) {
      graph = await trainDataset(graph, 'text-data', 1, {epochOffset:epoch-1})
      if (epoch===1 || epoch%4===0 || epoch===epochs) report(epoch)
    }
    if (kind === 'mean') learnedMean = graph
    if (kind === 'attention') learnedAttention = graph
    record.seconds = (performance.now()-started)/1000
    if (data.task === 'language') {
      record.greedy = generateText(graph, 'alice was ', 100)
      let seed = 123
      record.sample = generateText(graph, 'alice was ', 100, {sample:true, temperature:.8, topK:8, random:()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296}})
      console.log(record.sample)
    }
    const evaluated = forwardPass(graph, false)
    // Remove numerical caches from the saved project; the app computes them on import.
    const clean = {...graph, nodes:graph.nodes.map(node => ({...node, value:undefined, grad:undefined, cache:undefined, localDerivative:undefined})), edges:graph.edges.map(edge => ({...edge,value:undefined,grad:undefined}))}
    const file = createProjectStateFile({graph:clean, visualizationGraph:clean, initialParameterValues:initialParameters, selectedNodeIds:[], phase:'edit', traceSteps:[], traceIndex:0, epoch:epochs, currentLoss:evaluated.loss ?? null, display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
    const json = JSON.stringify(file)
    if (!parseProjectStateFile(json).ok) throw Error('Saved model did not validate: '+kind)
    await writeFile(join(output, 'models', kind+'.json'), json)
    const exported = generatePyTorchExport(graph, {epochs:1})
    const exportDir = join(output, 'exports', kind)
    await mkdir(exportDir, {recursive:true})
    await writeFile(join(exportDir,'model.py'), exported.script)
    await writeFile(join(exportDir,'model.ipynb'), exported.notebook)
    if (exported.datasetFile) await writeFile(join(exportDir, exported.datasetFile.name), exported.datasetFile.content)
    results.push(record)
    await writeFile(join(output,kind+'-results.json'),JSON.stringify(record,null,2))
    await writeFile(join(output,'results.json'), JSON.stringify(results,null,2))
  }
} finally { await server.close() }
