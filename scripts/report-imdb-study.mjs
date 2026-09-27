import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises'
import {resolve,join} from 'node:path'
import {createServer} from 'vite'
const root=resolve(process.env.IMDB_STUDY_OUTPUT||'output/accelerated-imdb')
const read=name=>readFile(join(root,name),'utf8').then(JSON.parse)
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const {forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 const prepared=await read('prepared.json')
 const runs=await Promise.all((await readdir(join(root,'runs'))).filter(n=>n.endsWith('-results.json')).map(n=>read('runs/'+n)))
 await mkdir(join(root,'models'),{recursive:true})
 const summary=[]
 for(const kind of ['counts-linear','mean','attention','position-attention','transformer']) {
  const trials=runs.filter(r=>r.kind===kind),mean=trials.reduce((sum,r)=>sum+r.test.accuracy,0)/trials.length
  summary.push({kind,seeds:trials.map(r=>r.seed),testMean:mean,testMin:Math.min(...trials.map(r=>r.test.accuracy)),testMax:Math.max(...trials.map(r=>r.test.accuracy)),testSampleSD:Math.sqrt(trials.reduce((sum,r)=>sum+(r.test.accuracy-mean)**2,0)/(trials.length-1))})
  // Fixed seed, chosen before inspecting results, rather than each model's best test seed.
  const {graph}=await read('templates/'+kind+'-137.json'),weights=await read('runs/'+kind+'-137-weights.json'),result=trials.find(r=>r.seed===137)
  graph.nodes[0].params.textData={...prepared,representation:kind.startsWith('counts')?'counts':'tokens'}
  const initialParameterValues=parameterValues(graph)
  for(const node of graph.nodes) if(weights[node.id])node.params.value=weights[node.id]
  const loss=forwardPass(graph,false).loss
  if(!Number.isFinite(loss))throw Error('Invalid restored checkpoint '+kind)
  const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:result.bestEpoch,currentLoss:loss,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
  const serialized=JSON.stringify(file),parsed=parseProjectStateFile(serialized)
  if(!parsed.ok)throw Error(parsed.error)
  await writeFile(join(root,'models',kind+'.json'),serialized)
 }
 await writeFile(join(root,'summary.json'),JSON.stringify(summary,null,2))
 const table=summary.map(r=>`| ${r.kind} | ${(100*r.testMean).toFixed(1)}% | ${(100*r.testMin).toFixed(1)}–${(100*r.testMax).toFixed(1)}% |`).join('\n')
 await writeFile(join(root,'RESULTS.md'),`# Controlled IMDb study\n\n5,000 training, 1,000 validation, 1,000 fresh final-test reviews. Vocabulary 2,000, first 128 tokens, embedding width 32. Three initialization seeds: 137, 211, 307. AdamW, learning rate 0.0003, batch 32, decay 0.01, global gradient norm limit 1; at most 12 epochs, validation-loss patience 3, minimum improvement 0.001. Final-test labels were used only after checkpoint selection.\n\n| Model | Mean test accuracy | Range across seeds |\n|---|---:|---:|\n${table}\n\nThese are stronger baselines, not optimized settings or evidence that each architectural addition improves sentiment accuracy. Counts and mean models reached the epoch cap. Position attention and the transformer did not reliably outperform counts. The runs used an independently checked native PyTorch CPU interpreter of the same editable graphs. Browser performance is measured separately in browser-benchmark.json. The final test is now observed and must not become a tuning set.\n\nImport models/*.json through Open project. These are fixed-seed 137 checkpoints, with the 6,000 training/validation reviews and the tokenizer included; final-test reviews are deliberately omitted from the projects. In the app, the split called test serves as validation for early stopping. Choose Train → Tensor engine, batch 32, and the settings above to run. Each new run starts fresh optimizer moments. Full manifests, per-epoch metrics, and all 15 checkpoints are retained locally.\n`)
 console.log(JSON.stringify(summary,null,2))
}finally{await server.close()}
