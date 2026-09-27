import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {resolve,join} from 'node:path'
import assert from 'node:assert/strict'
import {createServer} from 'vite'
const root=resolve(process.env.DATA_VOCAB_OUTPUT||'output/imdb-data-vocab')
const read=name=>readFile(join(root,name),'utf8').then(JSON.parse)
const protocol=await read('protocol.json'),summary=[]
for(const size of protocol.sizes)for(const vocabulary of protocol.vocabularies){
 const key=`n${size}-v${vocabulary}`,stats=await read(`data/${key}-stats.json`)
 const runs=await Promise.all(protocol.seeds.map(seed=>read(`runs/${key}-seed${seed}-results.json`)))
 for(const r of runs){assert.equal(r.finalTestUsed,false);assert.equal(r.numericSha256,stats.numericSha256);assert.equal(r.trainingSize,size);assert.equal(r.vocabularySize,vocabulary);assert.equal(r.validation.examples,protocol.validation)}
 const mean=fn=>runs.reduce((sum,r)=>sum+fn(r),0)/runs.length
 summary.push({key,trainingSize:size,vocabularySize:vocabulary,validationLoss:mean(r=>r.validation.loss),validationAccuracy:mean(r=>r.validation.accuracy),accuracyRange:[Math.min(...runs.map(r=>r.validation.accuracy)),Math.max(...runs.map(r=>r.validation.accuracy))],trainAccuracy:mean(r=>r.train.accuracy),meanSeconds:mean(r=>r.seconds),parameterCount:runs[0].parameterCount,bestEpochs:runs.map(r=>r.bestEpoch),unknownRate:stats.validationUnknownRate,seeds:runs.map(r=>r.seed)})
}
// Verify the unchanged control reproduces the preceding experiment.
for(const seed of protocol.seeds){
 const now=await read(`runs/n5000-v2000-seed${seed}-results.json`)
 const old=JSON.parse(await readFile(resolve(`output/imdb-dropout/runs/seed${seed}-p0-lr0.001-results.json`),'utf8'))
 assert(Math.abs(now.validation.loss-old.validation.loss)<1e-6,'Baseline loss changed')
 assert.equal(now.validation.accuracy,old.validation.accuracy)
}
const ranked=[...summary].sort((a,b)=>a.validationLoss-b.validationLoss),best=ranked[0]
await writeFile(join(root,'summary.json'),JSON.stringify({protocol,summary,best:best.key,baselineReproduced:true,finalTestUsed:false},null,2))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const {forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 await mkdir(join(root,'models'),{recursive:true})
 for(const key of new Set(['n5000-v2000',best.key])){
  const {graph}=await read(`templates/${key}-seed137.json`),data=await read(`data/${key}-prepared.json`),weights=await read(`runs/${key}-seed137-weights.json`),result=await read(`runs/${key}-seed137-results.json`)
  graph.nodes[0].params.textData=data
  const initialParameterValues=parameterValues(graph)
  for(const node of graph.nodes)if(weights[node.id])node.params.value=weights[node.id]
  const loss=forwardPass(graph,false).loss
  assert(Number.isFinite(loss))
  const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:result.bestEpoch,currentLoss:loss,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
  const serialized=JSON.stringify(file),parsed=parseProjectStateFile(serialized)
  assert(parsed.ok,parsed.error)
  await writeFile(join(root,'models',`transformer-${key}.json`),serialized)
 }
}finally{await server.close()}
const pct=n=>(100*n).toFixed(1)+'%'
const table=summary.map(r=>`| ${r.trainingSize.toLocaleString()} | ${r.vocabularySize.toLocaleString()} | ${r.validationLoss.toFixed(4)} | ${pct(r.validationAccuracy)} | ${r.accuracyRange.map(pct).join('–')} | ${pct(r.unknownRate)} | ${r.bestEpochs.join(', ')} | ${r.meanSeconds.toFixed(1)} |`).join('\n')
const report=`# IMDb training-size and vocabulary experiment

27 runs: nine settings, each at seeds 137, 211 and 307. Balanced nested subsets of 5,000 / 10,000 / 20,000 training reviews, vocabulary sizes 2,000 / 4,000 / 8,000 including the unknown token, and exactly the same 1,000 validation reviews as the earlier experiments. Vocabularies are fitted on training reviews only. Split manifests and tokenized-data hashes are retained. The 5,000 / 2,000 control reproduces the previous no-dropout, learning-rate 0.001 results for all three seeds.

The editable transformer is unchanged: width 32, one attention head, context 128, no dropout; AdamW learning rate 0.001, batch 32, weight decay 0.01, global gradient clipping 1. Up to 30 epochs, patience 3, minimum validation-loss improvement 0.001; restore each run's best validation-loss checkpoint. Native PyTorch CPU training uses the app's graph and primitive operations. Times below are native CPU measurements, not browser estimates.

| Training reviews | Vocabulary | Mean validation loss | Mean accuracy | Accuracy range | Unknown validation tokens | Best epochs (three seeds) | Mean seconds |
|---:|---:|---:|---:|---:|---:|---|---:|
${table}

Selected by lowest mean validation loss: **${best.key}**, accuracy **${pct(best.validationAccuracy)}**, loss **${best.validationLoss.toFixed(4)}**. These are exploratory validation results; the final-test split was never evaluated. Repeated tuning against the same validation reviews makes a fresh held-out evaluation necessary before claiming generalization performance.

This compares practical training recipes, not equal compute budgets: larger datasets receive more updates per epoch; larger vocabularies create more parameters and change token identity/initialization. Unknown-token percentages describe the retained first 128 tokens. Longer contexts and larger hidden layers were not varied. Every reported range is across initialization seeds, not a confidence interval.

Import models/transformer-${best.key}.json for the selected setting or models/transformer-n5000-v2000.json for the original control. Both use fixed seed 137, rather than selecting the luckiest seed, and include training/validation reviews and initial parameters for Reset. Choose Tensor engine, batch 32 and up to 30 epochs; optimizer settings are stored. Checkpoints do not store optimizer moments, and a new training run resets those moments. All 27 checkpoints and epoch histories remain in runs/.

Reproduce with node scripts/prepare-data-vocab-study.mjs, then a PyTorch-enabled Python running scripts/train-data-vocab-study.py, then node scripts/report-data-vocab-study.mjs. Preparation requires the extracted official IMDb dataset at /tmp/backprop-imdb/aclImdb (or IMDB_ROOT), plus the preceding accelerated-imdb split artifacts. Reporting checks against the preceding dropout study. Generated datasets, models and reports remain under git-ignored output/.
`
await writeFile(join(root,'RESULTS.md'),report)
console.log(JSON.stringify({best,summary},null,2))
