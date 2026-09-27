import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises'
import {resolve,join} from 'node:path'
import {createServer} from 'vite'
const source=resolve(process.env.IMDB_STUDY_OUTPUT||'output/accelerated-imdb')
const root=resolve(process.env.DROPOUT_STUDY_OUTPUT||'output/imdb-dropout')
const read=name=>readFile(join(root,name),'utf8').then(JSON.parse)
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
 const {createProjectStateFile,parseProjectStateFile}=await server.ssrLoadModule('/src/domain/session.ts')
 const {forwardPass,parameterValues}=await server.ssrLoadModule('/src/domain/engine.ts')
 const prepared=JSON.parse(await readFile(join(source,'prepared.json'),'utf8')),protocol=await read('protocol.json'),selection=await read('selection.json')
 const runs=await Promise.all((await readdir(join(root,'runs'))).filter(n=>n.endsWith('-results.json')).map(n=>read('runs/'+n)))
 if(runs.some(r=>r.finalTestUsed!==false||r.test))throw Error('This report must use validation only.')
 const summary=[]
 await mkdir(join(root,'models'),{recursive:true})
 for(const [rate,lr] of selection.confirmationIncludingBaseline){
  const trials=runs.filter(r=>r.dropoutRate===rate&&r.learningRate===lr)
  if(trials.length!==3)throw Error('Missing confirmation seed')
  const average=field=>trials.reduce((sum,r)=>sum+r.validation[field],0)/trials.length
  summary.push({dropoutRate:rate,learningRate:lr,validationLoss:average('loss'),validationAccuracy:average('accuracy'),validationRange:[Math.min(...trials.map(r=>r.validation.accuracy)),Math.max(...trials.map(r=>r.validation.accuracy))],seeds:trials.map(r=>r.seed),bestEpochs:trials.map(r=>r.bestEpoch)})
  const result=trials.find(r=>r.seed===137),{graph}=await read(`templates/transformer-137-p${rate}.json`),weights=await read('runs/'+result.name+'-weights.json')
  graph.learningRate=lr;graph.nodes[0].params.textData={...prepared,representation:'tokens'}
  const initialParameterValues=parameterValues(graph)
  for(const node of graph.nodes)if(weights[node.id])node.params.value=weights[node.id]
  const loss=forwardPass(graph,false).loss
  if(!Number.isFinite(loss))throw Error('Invalid restored checkpoint')
  const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues,selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:result.bestEpoch,currentLoss:loss,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
  const serialized=JSON.stringify(file),parsed=parseProjectStateFile(serialized)
  if(!parsed.ok)throw Error(parsed.error)
  await writeFile(join(root,'models',`transformer-p${rate}-lr${lr}.json`),serialized)
 }
 summary.sort((a,b)=>a.validationLoss-b.validationLoss)
 const screening=runs.filter(r=>r.seed===137).sort((a,b)=>a.dropoutRate-b.dropoutRate||a.learningRate-b.learningRate)
 await writeFile(join(root,'summary.json'),JSON.stringify({protocol,screening:screening.map(({name,dropoutRate,learningRate,bestEpoch,validation})=>({name,dropoutRate,learningRate,bestEpoch,validation})),confirmation:summary,totalRuns:runs.length,finalTestUsed:false},null,2))
 const pct=n=>(n*100).toFixed(1)+'%'
 const grid=screening.map(r=>`| ${r.dropoutRate} | ${r.learningRate} | ${r.bestEpoch} | ${r.validation.loss.toFixed(4)} | ${pct(r.validation.accuracy)} |`).join('\n')
 const table=summary.map(r=>`| ${r.dropoutRate} | ${r.learningRate} | ${r.validationLoss.toFixed(4)} | ${pct(r.validationAccuracy)} | ${r.validationRange.map(pct).join('–')} |`).join('\n')
 const report=`# IMDb dropout / learning-rate experiment\n\n${runs.length} native PyTorch CPU runs of the app's editable transformer: nine settings at seed 137, then two additional seeds (211, 307) for the two lowest-validation-loss settings, the original baseline, and a no-dropout control at the selected learning rate. The matched-rate control was added after screening to separate dropout from the learning-rate effect. No final-test evaluation was performed.\n\nThe same 5,000 training / 1,000 validation reviews, 2,000-token vocabulary, 128-token context, width 32, one head and classifier were used. Dropout sites: attention output before residual addition, feed-forward output before residual addition, and classifier ReLU before its final linear layer. AdamW: batch 32, decay 0.01, clipping 1, up to 30 epochs, patience 3, minimum validation-loss improvement 0.001. Every setting starts from the same parameter initialization for its seed. Each run has a fixed PyTorch dropout seed.\n\n## Nine-setting screen (seed 137)\n\n| Dropout | Learning rate | Selected epoch | Validation loss | Validation accuracy |\n|---|---|---:|---:|---:|\n${grid}\n\n## Three-seed comparisons\n\n| Dropout | Learning rate | Mean validation loss | Mean validation accuracy | Accuracy range |\n|---|---|---:|---:|---:|\n${table}\n\nThe best three-seed mean validation loss and accuracy came from learning rate 0.001 with dropout disabled. Increasing the learning rate helped this model; dropout at these sites and rates did not provide an additional consistent benefit. These are exploratory validation results, not new test accuracies or a statistically established improvement. They must not be compared directly to the previous final-test table.\n\n## Files and use\n\nImport models/transformer-p0-lr0.001.json for the strongest mean setting, or models/transformer-p0.2-lr0.001.json to inspect the dropout variant. All importable projects use fixed seed 137 and contain only training/validation reviews. Initial parameters are preserved for reset. Each Dropout block has an editable probability in Details. Choose Tensor engine, batch 32, up to 30 epochs, and the stored optimizer settings to train; Run forward and Test disable dropout. New runs reset optimizer state, so these files are checkpoints, not exact resumable training sessions.\n\nPer-epoch reports and all trained weights live in runs/. protocol.json and selection.json record the comparison and selection rules. All files in this directory remain git-ignored. Browser and PyTorch random generators differ, so the same model/configuration has the same dropout semantics but does not guarantee identical training trajectories across engines.\n`
 await writeFile(join(root,'RESULTS.md'),report)
 console.log(JSON.stringify(summary,null,2))
}finally{await server.close()}
