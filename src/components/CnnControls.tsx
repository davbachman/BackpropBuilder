import { useState } from 'react'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
import { datasetExampleIndex, datasetOutputValueForSlot } from '../domain/datasets'
import { forwardPass } from '../domain/engine'
import { toTensor } from '../domain/tensor'
import type { GraphModel } from '../domain/types'
import { CNN_CHECKPOINT, CNN_DIGITS, restoreCnnWeights, selectDigit } from '../learning/cnn'
import './cnn.css'

interface Props { graph: GraphModel; onGraphChange: (graph: GraphModel) => void }
const heat = (value: number, maximum: number) => {
  const intensity = Math.min(1, Math.abs(value) / Math.max(.01,maximum))
  return value < 0 ? `rgba(98,129,192,${.08+intensity*.92})` : `rgba(51,151,134,${.06+intensity*.94})`
}

export function CnnControls({ graph, onGraphChange }: Props) {
  const [filter,setFilter]=useState(0), [weightCell,setWeightCell]=useState(4)
  const [patch,setPatch]=useState({y:2,x:2}), [mode,setMode]=useState<'conv'|'relu'>('relu')
  const sampleId=graph.groups?.find(group=>group.id==='features')?.detail?.sampleId
  const datasetNode=graph.nodes.find(node=>node.type==='dataset' && node.params.dataset==='digits-8x8')
  const sample=datasetNode ? CNN_DIGITS[datasetExampleIndex(datasetNode)] : CNN_DIGITS.find(example=>example.id===sampleId)
  const split=sample?.split??'test'
  const target=datasetNode ? datasetOutputValueForSlot(datasetNode,1).data[0] : toTensor(graph.nodes.find(node=>node.id==='digit-target')?.params.value).data[0]??0
  const examples=CNN_DIGITS.filter(example=>example.split===split && example.label===target)
  const image=datasetNode ? datasetOutputValueForSlot(datasetNode,0) : toTensor(graph.nodes.find(node=>node.id==='image-input')?.params.value)
  const kernels=toTensor(graph.nodes.find(node=>node.id==='conv-weights')?.params.value)
  const biases=toTensor(graph.nodes.find(node=>node.id==='conv-bias')?.params.value)
  const raw=graph.nodes.find(node=>node.id==='conv-output')?.value?.data??[]
  const relu=graph.nodes.find(node=>node.id==='conv-relu')?.value?.data??[]
  const pooled=graph.nodes.find(node=>node.id==='pool-output')?.value?.data??[]
  const probabilities=graph.nodes.find(node=>node.id==='cnn-probabilities')?.value?.data??[]
  const maximum=Math.max(...probabilities), prediction=probabilities.indexOf(maximum)
  const maps=mode==='relu'?relu:raw, mapMaximum=Math.max(.001,...maps.map(Math.abs)), weightMaximum=Math.max(.001,...kernels.data.map(Math.abs))
  const selectedOffset=(patch.y*6+patch.x)*4+filter
  const choose=(id:string)=>{const next=CNN_DIGITS.find(example=>example.id===id);if(next)onGraphChange(selectDigit(graph,next))}
  const changeWeight=(value:number)=>{
    if(!Number.isFinite(value))return
    const values={...kernels,data:kernels.data.map((item,index)=>index===filter*9+weightCell?value:item)}
    onGraphChange(forwardPass({...graph,nodes:graph.nodes.map(node=>node.id==='conv-weights'?{...node,params:{...node.params,value:values}}:node)}).graph)
  }
  return <section className="cnn-controls">
    <p className="eyebrow">Pixels become evidence</p><h2>A tiny digit recognizer</h2>
    <p className="cnn-intro">Watch four learned filters find strokes in an 8 × 8 handwritten image.</p>
    <div className="cnn-sample-controls"><label>Examples<select aria-label="Digit dataset split" value={split} onChange={event=>{const next=event.target.value as 'train'|'test';choose(CNN_DIGITS.find(example=>example.split===next && example.label===target)!.id)}}><option value="test">Held out · 100 images</option><option value="train">Training · 400 images</option></select></label><div className="cnn-digit-picker">{Array.from({length:10},(_,digit)=><button key={digit} type="button" className={digit===target?'is-active':''} aria-label={`Show handwritten digit ${digit}`} onClick={()=>choose(CNN_DIGITS.find(example=>example.split===split && example.label===digit)!.id)}>{digit}</button>)}</div></div>
    <div className="cnn-input-row"><div className="cnn-image-grid" aria-label={`8 by 8 handwritten digit ${target}`}>{image.data.map((pixel,index)=>{const y=Math.floor(index/8),x=index%8;const inside=y>=patch.y && y<patch.y+3 && x>=patch.x && x<patch.x+3;return <span key={index} title={`Pixel (${y+1}, ${x+1}) = ${pixel.toFixed(4)}`} className={inside?'in-patch':''} style={{background:`rgb(${Math.round(245-pixel*208)},${Math.round(248-pixel*203)},${Math.round(252-pixel*181)})`}}/>})}</div><div className="cnn-prediction"><span>Model sees</span><strong>{prediction<0?'?':prediction}</strong><span>{prediction<0?'Run forward':`${(maximum*100).toFixed(1)}% confidence`}</span><small className={prediction===target?'is-correct':'is-wrong'}>Label {target} · {prediction===target?'correct':'try another example'}</small></div></div>
    <div className="cnn-example-nav"><button type="button" aria-label="Previous digit image" onClick={()=>choose(examples[(Math.max(0,examples.findIndex(example=>example.id===sample?.id))-1+examples.length)%examples.length].id)}><ChevronLeft size={14}/></button><span>{sample?.split==='test'?'Held-out':'Training'} image {Math.max(0,examples.findIndex(example=>example.id===sample?.id))+1} / {examples.length}</span><button type="button" aria-label="Next digit image" onClick={()=>choose(examples[(Math.max(0,examples.findIndex(example=>example.id===sample?.id))+1)%examples.length].id)}><ChevronRight size={14}/></button></div>
    <div className="cnn-probability-bars" aria-label="Predicted digit probabilities">{probabilities.map((probability,digit)=><div key={digit} title={`${digit}: ${(probability*100).toFixed(3)}%`}><span style={{height:`${Math.max(2,probability*48)}px`}} className={digit===prediction?'is-largest':''}/><b>{digit}</b></div>)}</div>
    <div className="cnn-section-heading"><h3>Shared filters</h3><span>3 × 3 · editable</span></div>
    <div className="cnn-filter-picker">{Array.from({length:4},(_,index)=><button type="button" key={index} className={filter===index?'is-selected':''} aria-label={`Inspect convolution filter ${index+1}`} onClick={()=>setFilter(index)}><span>Filter {index+1}</span><span className="cnn-kernel-mini">{kernels.data.slice(index*9,index*9+9).map((weight,i)=><i key={i} style={{background:heat(weight,weightMaximum)}}/>)}</span></button>)}</div>
    <div className="cnn-filter-editor"><div className="cnn-kernel-editor">{kernels.data.slice(filter*9,filter*9+9).map((weight,index)=><button type="button" key={index} aria-label={`Select filter weight row ${Math.floor(index/3)+1} column ${index%3+1}`} className={weightCell===index?'is-selected':''} style={{background:heat(weight,weightMaximum)}} onClick={()=>setWeightCell(index)} title={weight.toFixed(5)}>{weight.toFixed(1)}</button>)}</div><label>Selected weight<input aria-label="Convolution filter weight" type="number" step="0.05" value={Number((kernels.data[filter*9+weightCell]??0).toFixed(4))} onChange={event=>changeWeight(Number(event.target.value))}/><small>One weight, reused at every location.</small></label></div>
    <div className="cnn-section-heading"><h3>Feature maps</h3><select aria-label="Feature map activation" value={mode} onChange={event=>setMode(event.target.value as 'conv'|'relu')}><option value="relu">After ReLU</option><option value="conv">Before ReLU</option></select></div>
    <p className="cnn-map-hint">Choose a cell to trace its 3 × 3 input patch.</p>
    <div className="cnn-feature-maps">{Array.from({length:4},(_,feature)=><div key={feature} className={filter===feature?'is-selected':''}><span>Filter {feature+1} <small>6 × 6</small></span><div className="cnn-feature-grid">{Array.from({length:36},(_,cell)=>{const y=Math.floor(cell/6),x=cell%6,value=maps[cell*4+feature]??0;return <button key={cell} type="button" aria-label={`Feature ${feature+1} row ${y+1} column ${x+1}`} title={value.toFixed(5)} className={filter===feature && y===patch.y && x===patch.x?'is-selected':''} style={{background:heat(value,mapMaximum)}} onClick={()=>{setFilter(feature);setPatch({y,x})}}/>})}</div><div className="cnn-pool-line"><small>pool →</small><div>{Array.from({length:9},(_,cell)=><i key={cell} title={(pooled[cell*4+feature]??0).toFixed(5)} style={{background:heat(pooled[cell*4+feature]??0,mapMaximum)}}/>)}</div></div></div>)}</div>
    <div className="cnn-patch-readout"><strong>Filter {filter+1} · patch ({patch.y+1}, {patch.x+1})</strong><span>Σ(pixel × weight) + {biases.data[filter]?.toFixed(3)} = {raw[selectedOffset]?.toFixed(4)??'—'}</span><span>ReLU → {relu[selectedOffset]?.toFixed(4)??'—'}</span></div>
    <p className="cnn-checkpoint-note">Included checkpoint: {Math.round(CNN_CHECKPOINT.metadata.testAccuracy*100)} / 100 held-out images correct. These are UCI 8 × 8 digits; the network has 410 trainable parameters.</p>
    <button className="cnn-restore" type="button" onClick={()=>onGraphChange(restoreCnnWeights(graph))}><RotateCcw size={13}/> Restore trained weights</button>
    <a className="cnn-source" href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits" target="_blank" rel="noreferrer">Alpaydin &amp; Kaynak · UCI Handwritten Digits · CC BY 4.0</a>
  </section>
}
