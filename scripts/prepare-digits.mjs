/** Refresh the checked-in educational subset from scikit-learn's public UCI mirror.
 * Run only when updating data; ordinary training and the app work offline. */
import { gunzipSync } from 'node:zlib'
import { writeFileSync } from 'node:fs'

const source='https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv.gz'
const response=await fetch(source)
if(!response.ok)throw new Error(`Dataset download failed: ${response.status}`)
const rows=gunzipSync(Buffer.from(await response.arrayBuffer())).toString().trim().split('\n').map(row=>row.split(',').map(Number))
let seed=37
const random=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296}
const examples=[]
for(let label=0;label<10;label++){
  const pool=rows.map((row,index)=>({row,index})).filter(item=>item.row[64]===label)
  for(let i=pool.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[pool[i],pool[j]]=[pool[j],pool[i]]}
  for(let i=0;i<50;i++)examples.push({id:`digit-${pool[i].index}`,label,split:i<40?'train':'test',pixels:pool[i].row.slice(0,64)})
}
writeFileSync(new URL('../src/learning/digits.json',import.meta.url),JSON.stringify(examples)+'\n')
console.log(`Saved ${examples.length} digit images: 400 training, 100 held out.`)
