import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {resolve,join} from 'node:path'
import {createServer} from 'vite'
const source=resolve(process.env.IMDB_STUDY_OUTPUT||'output/accelerated-imdb')
const root=resolve(process.env.DROPOUT_STUDY_OUTPUT||'output/imdb-dropout')
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try {
 const {addTransformerDropout}=await server.ssrLoadModule('/src/test/textModels.ts')
 const {forwardPass}=await server.ssrLoadModule('/src/domain/engine.ts')
 const prepared=JSON.parse(await readFile(join(source,'prepared.json'),'utf8'))
 await mkdir(join(root,'templates'),{recursive:true})
 const protocol={training:5000,validation:1000,vocabulary:2000,context:128,width:32,rates:[0,.1,.2],learningRates:[.0001,.0003,.001],screeningSeed:137,confirmationSeeds:[211,307],selection:'Two lowest validation-loss settings from seed 137, then compare means over three seeds; include p=0/lr=0.0003 baseline and p=0 controls at each selected learning rate.',optimizer:'AdamW',batchSize:32,weightDecay:.01,clipNorm:1,maxEpochs:30,patience:3,minDelta:.001,dropoutSites:['attention output before residual','feed-forward output before residual','classifier ReLU before final linear layer'],finalTestUsed:false}
 await writeFile(join(root,'protocol.json'),JSON.stringify(protocol,null,2))
 for(const seed of [137,211,307])for(const rate of protocol.rates){
  const template=JSON.parse(await readFile(join(source,'templates',`transformer-${seed}.json`),'utf8'))
  template.graph=addTransformerDropout(template.graph,rate)
  template.graph.nodes[0].params.textData={...prepared,representation:'tokens'}
  const loss=forwardPass(template.graph,false).loss
  if(Math.abs(loss-template.expectedLoss)>1e-10)throw Error('Dropout changed evaluation at initialization')
  template.graph.nodes[0].params.textData={...template.graph.nodes[0].params.textData,documents:[]}
  await writeFile(join(root,'templates',`transformer-${seed}-p${rate}.json`),JSON.stringify({...template,dropoutRate:rate}))
 }
 console.log('Prepared frozen nine-setting validation-only study with unchanged initial weights.')
}finally{await server.close()}
