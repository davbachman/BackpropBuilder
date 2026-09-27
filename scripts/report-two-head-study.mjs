import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {resolve,join} from 'node:path'
import assert from 'node:assert/strict'
import {createServer} from 'vite'
const root=resolve(process.env.TWO_HEAD_OUTPUT||'output/imdb-two-head')
const read=path=>readFile(path,'utf8').then(JSON.parse)
const protocol=await read(join(root,'protocol.json')),key='n20000-v4000',summary=[]
for(const heads of [1,2]){
 const folder=heads===1?protocol.baselineRoot:root
 const runs=await Promise.all(protocol.seeds.map(seed=>read(join(folder,'runs',`${key}-seed${seed}-results.json`))))
 const stats=await read(join(root,'data',key+'-stats.json'))
 for(const run of runs){assert.equal(run.finalTestUsed,false);assert.equal(run.numericSha256,stats.numericSha256);assert.equal(run.parameterCount,140985);assert.equal(run.validation.examples,1000)}
 const mean=fn=>runs.reduce((sum,r)=>sum+fn(r),0)/runs.length
 summary.push({heads,parameters:runs[0].parameterCount,validationLoss:mean(r=>r.validation.loss),validationAccuracy:mean(r=>r.validation.accuracy),accuracyRange:[Math.min(...runs.map(r=>r.validation.accuracy)),Math.max(...runs.map(r=>r.validation.accuracy))],trainAccuracy:mean(r=>r.train.accuracy),meanSeconds:mean(r=>r.seconds),runs:runs.map(({seed,bestEpoch,completed,validation,train})=>({seed,bestEpoch,completed,validation,train}))})
}
await writeFile(join(root,'summary.json'),JSON.stringify({protocol,summary,finalTestUsed:false},null,2))
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const {forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 const {graph}=await read(join(root,'templates',key+'-seed137.json'))
 graph.nodes.find(n=>n.type==='dataset').params.textData=await read(join(root,'data',key+'-prepared.json'))
 const initialParameterValues=parameterValues(graph),weights=await read(join(root,'runs',key+'-seed137-weights.json'))
 for(const node of graph.nodes)if(weights[node.id])node.params.value=weights[node.id]
 const loss=forwardPass(graph,false).loss;assert(Number.isFinite(loss))
 const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:summary[1].runs[0].bestEpoch,currentLoss:loss,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
 const serialized=JSON.stringify(file),parsed=parseProjectStateFile(serialized);assert(parsed.ok,parsed.error)
 await mkdir(join(root,'models'),{recursive:true})
 await writeFile(join(root,'models','transformer-two-heads.json'),serialized)
}finally{await server.close()}
const pct=n=>(n*100).toFixed(1)+'%'
const table=summary.map(r=>`| ${r.heads} | ${r.validationLoss.toFixed(4)} | ${pct(r.validationAccuracy)} | ${r.accuracyRange.map(pct).join('–')} | ${r.runs.map(t=>t.bestEpoch).join(', ')} | ${r.meanSeconds.toFixed(1)} |`).join('\n')
const report=`# Two-head IMDb transformer comparison

Three new runs (seeds 137, 211, 307) compared with the existing three one-head controls. Exactly the same 20,000 training reviews, 1,000 validation reviews, training-fitted 4,000-entry vocabulary, context 128, total model width 32 and 140,985 parameters. No final-test evaluation.

Two heads each have width 16 and independent query/key/value projections, scaled scores (divide by sqrt(16)), and softmax. Their outputs concatenate to width 32 before the unchanged output projection. Initial projection matrices are column partitions of the one-head matrices. Every other initial weight is unchanged. This avoids increasing model capacity by doubling its width. All components remain editable elementary blocks; no prebuilt attention operator is introduced.

AdamW learning rate 0.001, no dropout, batch 32, weight decay 0.01, global gradient clipping 1, up to 30 epochs, patience 3, minimum validation-loss improvement 0.001. Each run restores its selected validation-loss checkpoint. Training uses the native PyTorch graph interpreter. Reported times are native CPU training measurements, not browser timings, and controls were run earlier.

| Heads | Mean validation loss | Mean accuracy | Accuracy range | Selected epochs | Mean seconds |
|---:|---:|---:|---:|---|---:|
${table}

Accuracy change: ${((summary[1].validationAccuracy-summary[0].validationAccuracy)*100).toFixed(2)} percentage points. Validation-loss change: ${(summary[1].validationLoss-summary[0].validationLoss).toFixed(4)} (negative is better). Ranges describe seed variation, not confidence intervals. These are exploratory results on repeatedly used validation reviews.

Import models/transformer-two-heads.json to inspect the two-head checkpoint (fixed seed 137, accuracy ${pct(summary[1].runs[0].validation.accuracy)}). Initial parameters are included for Reset. Set batch 32 and up to 30 epochs explicitly; stored optimizer settings do not include optimizer moments. All generated datasets and models stay git-ignored. The one-head control is ../imdb-data-vocab/models/transformer-n20000-v4000.json.

Reproduce after the data/vocabulary study: node scripts/prepare-two-head-study.mjs; python3 scripts/train-data-vocab-study.py --root output/imdb-two-head; node scripts/report-two-head-study.mjs. Python requires PyTorch. Preparation accepts DATA_VOCAB_OUTPUT and TWO_HEAD_OUTPUT; reporting accepts TWO_HEAD_OUTPUT and reads the saved baseline path from protocol.json.
`
await writeFile(join(root,'RESULTS.md'),report)
console.log(JSON.stringify(summary,null,2))
