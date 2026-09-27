import {readFile,writeFile,mkdir} from 'node:fs/promises'
import {createServer} from 'vite'
const root='output/semester-pilot',server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',logLevel:'error'})
try{
 const frozen=JSON.parse(await readFile(root+'/frozen.json','utf8')),{textDataset}=await server.ssrLoadModule('/src/domain/textData.ts')
 await mkdir(root+'/final',{recursive:true})
 let facts=[],qa1=[]
 for(const line of (await readFile('output/babi-pilot/raw/qa1_single-supporting-fact_test.txt','utf8')).trim().split('\n')){
  const first=line.indexOf(' '),number=line.slice(0,first),text=line.slice(first+1);if(number==='1')facts=[]
  if(!text.includes('\t'))facts.push(text);else{const [question,answer]=text.split('\t');qa1.push({facts:[...facts],question,answer})}
 }
 const qa0=JSON.parse(await readFile('output/babi-pilot/data/qa0-test.json','utf8')),alice=JSON.parse(await readFile(root+'/data/alice-final-passages.json','utf8'))
 for(const job of frozen.jobs){
  const template=JSON.parse(await readFile(root+'/templates/'+job.name+'.json','utf8')),source=template.graph.nodes.find(n=>n.type==='dataset'),data=source.params.textData
  let rows
  if(data){
   let documents
   if(job.name.startsWith('qa'))documents=(job.name.startsWith('qa0')?qa0:qa1).map(r=>({text:r.facts.join(' ')+' '+r.question,facts:r.facts,question:r.question,label:r.answer,split:'test'}))
   else documents=alice.map(text=>({text,split:'test'}))
   const finalData={...data,documents:[data.documents.find(d=>d.split==='train'),...documents]}
   rows=textDataset(finalData).examples.filter(r=>r.split==='test').map(r=>({features:r.features,target:r.target}))
  }else{
   const family=job.name.split('-')[0],raw=JSON.parse(await readFile(root+'/data/'+family+'.json','utf8'))
   let scaling;if(job.name.startsWith('mpg-small'))scaling=JSON.parse(await readFile(root+'/data/mpg-small-scaling.json','utf8'))
   rows=raw.rows.filter(r=>r.split==='final-test').map(r=>({features:r.x.map((x,i)=>({shape:[],data:[scaling?(x-scaling.mean[i])/scaling.scale[i]:x]})),target:{shape:raw.task==='regression'?[]:[1],data:[scaling?(r.y-scaling.targetMean)/scaling.targetScale:r.y]}}))
  }
  await writeFile(root+'/final/'+job.name+'.json',JSON.stringify(rows))
 }
}finally{await server.close()}
