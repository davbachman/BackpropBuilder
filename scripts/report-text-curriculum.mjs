import {mkdir, readFile, writeFile} from 'node:fs/promises'
import {join, resolve} from 'node:path'
import {createServer} from 'vite'

const root = resolve(process.env.TEXT_PILOT_OUTPUT || 'output/text-curriculum')
const server = await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
  const {organizeTextModel} = await server.ssrLoadModule('/src/test/textModels.ts')
  const {validateGraph,forwardPass} = await server.ssrLoadModule('/src/domain/engine.ts')
  const {parseProjectStateFile} = await server.ssrLoadModule('/src/domain/session.ts')
  const {generatePyTorchExport} = await server.ssrLoadModule('/src/domain/pytorchExport.ts')
  const {evaluateDataset} = await server.ssrLoadModule('/src/domain/datasetTraining.ts')
  const kinds = ['counts-linear','counts-mlp','mean','position-mean','attention','position-attention','transformer','alice-baseline','alice-transformer']
  const results = [], projects = new Map()
  async function save(kind,file) {
    file.state.graph = organizeTextModel(file.state.graph)
    file.state.visualizationGraph = file.state.graph
    const problems = validateGraph(file.state.graph).filter(issue=>issue.code !== 'disconnected')
    if(problems.length) throw Error(kind+': '+JSON.stringify(problems))
    if(!parseProjectStateFile(JSON.stringify(file)).ok) throw Error('Invalid project '+kind)
    await writeFile(join(root,'models',kind+'.json'),JSON.stringify(file))
    const exported = generatePyTorchExport(file.state.graph,{epochs:1})
    const directory = join(root,'exports',kind)
    await mkdir(directory,{recursive:true})
    await writeFile(join(directory,'model.py'),exported.script)
    await writeFile(join(directory,'model.ipynb'),exported.notebook)
    await writeFile(join(directory,exported.datasetFile.name),exported.datasetFile.content)
    projects.set(kind,file)
  }
  for(const kind of kinds) {
    results.push(JSON.parse(await readFile(join(root,kind+'-results.json'),'utf8')))
    await save(kind,JSON.parse(await readFile(join(root,'models',kind+'.json'),'utf8')))
  }
  const oneHot = structuredClone(projects.get('mean'))
  const graph = oneHot.state.graph, size = graph.nodes[0].params.textData.vocabulary.length
  graph.nodes = [...graph.nodes.map(node=>node.id === 'embedded-tokens' ? {...node,type:'matmul'} : node),{id:'one-hot',label:'One-hot token IDs',type:'one-hot',params:{numClasses:size},position:{x:0,y:0}}]
  graph.edges = [...graph.edges.filter(edge=>edge.target !== 'embedded-tokens'),{id:'ids-onehot',source:'text-data',sourceSlot:0,target:'one-hot',inputSlot:0},{id:'hot-product',source:'one-hot',target:'embedded-tokens',inputSlot:0},{id:'table-product',source:'word-embeddings',target:'embedded-tokens',inputSlot:1}]
  const lookupLoss = forwardPass(projects.get('mean').state.graph,false).loss
  const oneHotLoss = forwardPass(graph,false).loss
  if(Math.abs(lookupLoss-oneHotLoss)>1e-10) throw Error('One-hot and lookup differ')
  await save('mean-onehot',oneHot)

  const causal = structuredClone(projects.get('transformer'))
  const cg = causal.state.graph
  cg.nodes.push({id:'causal-scores',label:'causal scores',type:'causal-mask',params:{},position:{x:0,y:0}})
  cg.edges = cg.edges.map(edge=>edge.target==='attention-probabilities' ? {...edge,source:'causal-scores',sourceSlot:0} : edge)
  cg.edges.push({id:'causal-scores:0',source:'scaled-scores',target:'causal-scores',inputSlot:0})
  causal.state.currentLoss = forwardPass(cg,false).loss
  await save('imdb-causal-mask',causal)
  const causalMetrics = evaluateDataset(cg,'text-data','test')
  await writeFile(join(root,'causal-mask-comparison.json'),JSON.stringify({note:'Same trained bidirectional classifier weights; mask inserted without additional training.',loss:causalMetrics.loss,accuracy:causalMetrics.accuracy},null,2))
  await writeFile(join(root,'results.json'),JSON.stringify(results,null,2))
  const f = n=>Number(n).toFixed(3), pct = n=>(n*100).toFixed(1)+'%'
  const rows = results.map(r=>{const a=r.reports[0],b=r.reports.at(-1);return `| [${r.kind}](models/${r.kind}.json) | ${r.epochs} | ${f(a.trainLoss)} → ${f(b.trainLoss)} | ${f(a.heldOutLoss)} → ${f(b.heldOutLoss)} | ${pct(b.heldOutAccuracy)} | ${Math.round(r.seconds)} s |`}).join('\n')
  const alice = results.find(r=>r.kind === 'alice-transformer')
  const text = `# Completed text-model pilots

These are instructor reference models built from the app's editable blocks. Use **File → Import** to open a project JSON. Groups are visual containers; zoom inside to inspect every operation. No student-facing pretrained model was added.

## Start here

- [Sequence and app instructions](../../docs/TEXT-CURRICULUM.md)
- [IMDb review CSV](data/imdb-pilot.csv) and [prepared dataset](data/imdb-prepared.json)
- [Prepared Alice dataset](data/alice-prepared.json)
- [Word counts](models/counts-linear.json), [learned embeddings](models/mean.json), [one-hot equivalent](models/mean-onehot.json)
- [IMDb transformer](models/transformer.json), [same weights with causal mask](models/imdb-causal-mask.json)
- [Alice transformer / text generator](models/alice-transformer.json)
- [All recorded measurements](results.json)

Python scripts, runnable notebooks and their sibling dataset files are under exports/MODEL-NAME/. Keep each export's three files together. The browser preview at output/text-curriculum/preview.html uses the same App component and trained Alice project; it is only a local development fixture.

## Measured results

IMDb: 800 balanced training reviews and 200 balanced held-out reviews, sampled from the official Stanford splits; a 1,000-entry word vocabulary; first 64 tokens per review; embedding width 20. Alice: 60 training and 20 held-out passages from disjoint portions of the book; character vocabulary 44; context/stride 16; width 16. Accuracy is sentiment accuracy for IMDb and per-character next-token accuracy for Alice. Timings include evaluation on this machine and are not performance guarantees.

| Model | Epochs | Training loss | Held-out loss | Final held-out accuracy | Elapsed |
| --- | ---: | --- | --- | ---: | ---: |
${rows}

These are final checkpoints, not cherry-picked best epochs. Learning rates and full histories are in the JSON reports. The refined IMDb sequence transfers learned embedding/classifier parameters from mean pooling into attention, and from attention into the subsequent models where parameter shapes match. Training budgets differ, so the table is a feasibility record, not a controlled ranking of architectures. This held-out sample was consulted during tuning and should be called validation, not an untouched final test.

The IMDb models demonstrate learning but also substantial overfitting and variation with SGD settings. Attention is not automatically better than counts on this small, truncated sample. Do not turn these final accuracies into student grading thresholds. A later course-design pass should choose a stable baseline, data size, runtime budget and validation protocol.

The causal IMDb checkpoint has the exact bidirectional transformer's trained parameters with only the score mask inserted. Its held-out loss is ${f(causalMetrics.loss)} and accuracy ${pct(causalMetrics.accuracy)}. It is a masking experiment, not a separately trained model.

## Generation observed after Alice training

Prompt: "alice was ".

Greedy:

> ${alice.greedy}

Seeded sample (temperature 0.8, top-k 8):

> ${alice.sample}

This small model learns character patterns and frequent fragments. It does not yet generate coherent prose. Greedy repetition is observable and useful for the decoding comparison. The UI can continue beyond its 16-character context by retaining a sliding window.

## Checks performed

- All nine models validated, differentiated, trained on real datasets, saved, reloaded and exported.
- One-hot matrix multiplication and embedding lookup match in output and repeated-token gradients; the trained reference projects have the same example loss (${f(oneHotLoss)}).
- Mean pooling, positions plus mean, and unpositioned pooled attention are permutation invariant; positioned attention can distinguish order.
- Causal earlier logits do not change when future inputs change.
- Vocabulary fits training text only; windows and train/test passages do not overlap.
- Inference does not update the model or create training targets; target-dependent prediction paths are rejected.
- Native PyTorch checks match forward losses and parameter gradients for every core model variant.
- Browser check: imported 1,000-word model completed an epoch on 800 reviews; trained Alice model generated both greedy and sampled continuations; no browser console errors were observed during those checks.

![Trained Alice generator in the app](alice-generator.png)

## Provenance and reproduction

Reviews: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). Text: [Alice's Adventures in Wonderland, Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11). The original Alice download remains in data/alice-original.txt, including its source notice. The pilot removes the Gutenberg header/footer and selects passages for its prepared JSON. Keep source attribution with redistributed course data.

Run "IMDB_ROOT=/path/to/aclImdb node scripts/pilot-text-curriculum.mjs", then "node scripts/report-text-curriculum.mjs" from the repository root. See the guide for data setup and optional PyTorch tests. Artifacts here are local and Git-ignored; they have not been published.
`
  await writeFile(join(root,'REPORT.md'),text)
  console.log('Wrote report and 11 validated instructor projects to '+root)
} finally {await server.close()}
