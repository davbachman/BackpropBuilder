import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {join,resolve} from 'node:path'
import {createHash} from 'node:crypto'
import assert from 'node:assert/strict'
import {createServer} from 'vite'
const root=resolve(process.env.FINAL_IMDB_OUTPUT||'output/imdb-final'),read=path=>readFile(join(root,path),'utf8').then(JSON.parse)
const frozenRaw=await readFile(join(root,'frozen.json')),frozen=JSON.parse(frozenRaw),test=await read('test-results.json'),selection=await read('context-selection.json')
assert.equal(test.frozenSha256,createHash('sha256').update(frozenRaw).digest('hex'))
const summary=frozen.plan.models.map(kind=>{
 const runs=test.results.filter(r=>r.kind===kind);assert.equal(runs.length,3)
 const mean=fn=>runs.reduce((s,r)=>s+fn(r),0)/runs.length
 return {kind,validationLoss:mean(r=>r.validation.loss),validationAccuracy:mean(r=>r.validation.accuracy),testLoss:mean(r=>r.test.loss),testAccuracy:mean(r=>r.test.accuracy),testAccuracyRange:[Math.min(...runs.map(r=>r.test.accuracy)),Math.max(...runs.map(r=>r.test.accuracy))],bestEpochs:runs.map(r=>r.bestEpoch)}
})
await writeFile(join(root,'summary.json'),JSON.stringify({contextComparison:selection,models:summary,protocol:frozen.protocol,testExamples:5000,selectionUsedTest:false},null,2))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const {forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 const data=await read('data/n20000-v4000-prepared.json');await mkdir(join(root,'models'),{recursive:true})
 for(const kind of frozen.plan.models){
  const name=kind+'-seed137',{graph}=await read('templates/'+name+'.json'),weights=await read('runs/'+name+'-weights.json'),result=frozen.jobs.find(j=>j.name===name)
  const node=graph.nodes.find(n=>n.type==='dataset');node.params.textData={...data,representation:node.params.textData.representation}
  const initialParameterValues=parameterValues(graph)
  for(const n of graph.nodes)if(weights[n.id])n.params.value=weights[n.id]
  const loss=forwardPass(graph,false).loss;assert(Number.isFinite(loss))
  const serialized=JSON.stringify(createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:result.bestEpoch,currentLoss:loss,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}}))
  const parsed=parseProjectStateFile(serialized);assert(parsed.ok,parsed.error)
  await writeFile(join(root,'models',kind+'.json'),serialized)
 }
}finally{await server.close()}
const pct=n=>(100*n).toFixed(1)+'%',names={'counts-linear':'Linear word counts','counts-mlp':'MLP on word counts',mean:'Pooled learned embeddings','transformer-two-head':'Two-head transformer'}
const table=summary.map(r=>`| ${names[r.kind]} | ${pct(r.validationAccuracy)} | ${pct(r.testAccuracy)} | ${r.testLoss.toFixed(4)} | ${r.testAccuracyRange.map(pct).join('–')} | ${r.bestEpochs.join(', ')} |`).join('\n')
const contexts=selection.comparisons.map(r=>`| ${r.context} | ${pct(r.meanValidationAccuracy)} | ${r.meanValidationLoss.toFixed(4)} |`).join('\n')
const report=`# Final matched IMDb comparison

## Context selection (validation only)

| Context tokens | Mean validation accuracy | Mean validation loss |
|---:|---:|---:|
${contexts}

Selected context: ${selection.chosen.context}, by mean validation loss across seeds 137, 211 and 307. The same 20,000 training / 1,000 validation reviews and training-fitted vocabulary of 4,000 were used. The 256-token run preserves the 128-token prefixes and shared initial weights, adding 128 position-embedding rows. Transformer width 32, two heads of width 16; no dropout.

## Frozen comparison on fresh test reviews

| Model | Mean validation accuracy | Mean test accuracy | Mean test loss | Test accuracy range | Selected epochs |
|---|---:|---:|---:|---|---|
${table}

Every figure averages the same three initializations; ranges are across seeds, not confidence intervals. Test accuracy uses 5,000 fresh, balanced official-test reviews, excluding the 1,000 reviews evaluated in the earlier accelerated study and the 200 pilot reviews. All twelve checkpoints and their hashes were frozen before preparing or evaluating this sample. Test results did not select architecture, context, epochs or a preferred seed. See frozen.json, holdout-manifest.json and test-results.json for the audit trail and per-review probabilities.

All four models use identical review IDs, vocabulary, and the same first ${selection.chosen.context} tokens. Word-count features are raw counts of that prefix, including unknown tokens. The MLP on counts has 12 hidden ReLU units. Pooled embeddings have width 32 and the same 12-unit classifier as the transformer, but no positions or attention. These controls separate nonlinear classification and learned embeddings from contextual processing, although architecture parameter counts differ.

Training: native PyTorch CPU using the app's graph primitives; AdamW learning rate 0.001, batch 32, decay 0.01, global norm clipping 1, no dropout, maximum 30 epochs, patience 3 and minimum validation-loss improvement 0.001. Validation loss selects each checkpoint. The optimizer and stopping budget are matched, not individually tuned to each architecture. This comparison does not establish a universal model ranking or isolate word order as the cause of a performance difference. Repeated prior validation tuning remains a limitation; the new test sample provides a separate assessment of the frozen recipes. Do not continue tuning against these test results.

## Inspect or reproduce

Import models/counts-linear.json, models/counts-mlp.json, models/mean.json or models/transformer-two-head.json. All projects use seed 137 chosen in advance and include only training/validation reviews, saved optimizer settings and initial weights for Reset. The seed-137 checkpoint accuracy can differ from the table mean. Set batch 32 and up to 30 epochs manually. New runs reset optimizer moments. Generated datasets, models and reports remain git-ignored under output/imdb-final/.

After the preceding data/vocabulary and two-head studies:

\`\`\`sh
node scripts/prepare-final-imdb-study.mjs
python3 scripts/train-data-vocab-study.py --root output/imdb-final/context256
node scripts/prepare-final-imdb-baselines.mjs
python3 scripts/train-final-imdb-baselines.py
node scripts/freeze-final-imdb-study.mjs
python3 scripts/evaluate-final-imdb-study.py
node scripts/report-final-imdb-study.mjs
\`\`\`

Python requires PyTorch. Use a new output directory to reproduce; freezing refuses to overwrite an existing frozen.json, and test evaluation refuses to replace test-results.json. Preparation/reporting accept FINAL_IMDB_OUTPUT; Python scripts accept --root (use its context256 subdirectory for the context run). Raw IMDb defaults to /tmp/backprop-imdb/aclImdb; freezing accepts IMDB_ROOT. Prior-study directories are fixed as recorded in the preparation scripts.

The app now materializes word-count vectors on demand, removing the former dense-corpus allocation limit while retaining per-vector and text import limits. This allows the full 20,000-review / 4,000-word comparison to run in the app without caching 84 million dense count cells.
`
await writeFile(join(root,'RESULTS.md'),report)
console.log(JSON.stringify({selection,summary},null,2))
