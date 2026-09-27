import {expect,it} from 'vitest'
import {textDataset} from './textData'
import type {TextDatasetData} from './types'
it('materializes large-corpus count vectors on demand without retaining dense rows',()=>{
 const data:TextDatasetData={version:1,fileName:'large.csv',task:'sentiment',tokenizer:'word',representation:'counts',lowercase:true,vocabulary:['<unk>','good','bad',...Array.from({length:3997},(_,i)=>'w'+i)],maxLength:3,stride:3,documents:Array.from({length:21000},(_,i)=>({text:i%2?'bad unseen':'good good bad ignored',label:i%2?'negative':'positive',split:i<20000?'train':'test'}))}
 const dataset=textDataset(data)
 expect(dataset.examples).toHaveLength(21000)
 const features=dataset.examples![0].features
 expect(Object.getOwnPropertyDescriptor(features,0)?.get).toBeTypeOf('function')
 expect(features[0].shape).toEqual([1,4000]);expect(features[0].data.slice(0,4)).toEqual([0,2,1,0])
 expect(features[0].data.reduce((a,b)=>a+b,0)).toBe(3)
 expect(dataset.examples![1].features[0].data.slice(0,3)).toEqual([1,0,1])
 const first=features[0];first.data[1]=999
 expect(features[0].data[1]).toBe(2)
 expect(textDataset(data)).toBe(dataset)
})
