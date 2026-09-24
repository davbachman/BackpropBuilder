import { useMemo, useState } from 'react'
import { forwardPass } from '../domain/engine'
import { datasetOutputValueForSlot } from '../domain/datasets'
import { toTensor } from '../domain/tensor'
import type { GraphModel, GraphNode, TensorValue } from '../domain/types'
import { TensorHeatmap } from './TensorHeatmap'

/** Every convolution, including copied and hand-built ones, is inspectable. */
export function ConvolutionInspector({graph, node, onValue}: {graph:GraphModel; node:GraphNode; onValue:(id:string,value:TensorValue)=>void}) {
  const [chosenFilter,setFilter] = useState(0), [chosenChannel,setChannel] = useState(0)
  const evaluated = useMemo(() => { try { return forwardPass(graph).graph } catch { return graph } }, [graph])
  const input = (slot: number) => {
    const edge = graph.edges.find(edge => edge.target === node.id && (edge.inputSlot ?? 0) === slot)
    return {edge, source:evaluated.nodes.find(node => node.id === edge?.source)}
  }
  const imageSource = input(0), kernelSource = input(1)
  const image = imageSource.source?.type === 'dataset' ? datasetOutputValueForSlot(imageSource.source,imageSource.edge?.sourceSlot ?? 0) : imageSource.source?.value
  const kernel = kernelSource.source && toTensor(kernelSource.source.params.value ?? kernelSource.source.value)
  if (!kernel || kernel.shape.length !== 4) return <p className="coordinate-note">Connect a filter tensor [count, height, width, channels] to see its filters here.</p>
  const [filters, kh, kw, channels] = kernel.shape
  const filter = Math.min(chosenFilter,filters-1), channel = Math.min(chosenChannel,channels-1)
  const indices = Array.from({length:kh*kw},(_,cell)=>(filter*kh*kw+cell)*channels+channel)
  const output = evaluated.nodes.find(candidate => candidate.id === node.id)?.value
  return <section aria-label="Convolution filter visualizer">
    <div className="operation-fields"><label className="inspector-field">Filter<select aria-label="Filter to inspect" value={filter} onChange={event=>setFilter(Number(event.target.value))}>{Array.from({length:filters},(_,i)=><option key={i} value={i}>{i+1}</option>)}</select></label><label className="inspector-field">Input channel<select aria-label="Filter input channel" value={channel} onChange={event=>setChannel(Number(event.target.value))}>{Array.from({length:channels},(_,i)=><option key={i} value={i}>{i+1}</option>)}</select></label></div>
    <p className="eyebrow">Shared spatial filter</p>
    <TensorHeatmap key={`${node.id}:${filter}:${channel}`} value={{shape:[kh,kw],data:indices.map(index=>kernel.data[index])}} onEdit={kernelSource.source?.type === 'weight' ? (index,value) => onValue(kernelSource.source!.id,{shape:kernel.shape,data:kernel.data.map((old,i)=>i===indices[index]?value:old)}) : undefined}/>
    {image?.shape.length === 3 && <><p className="eyebrow">Input channel {channel+1}</p><TensorHeatmap value={{shape:image.shape.slice(0,2),data:image.data.filter((_,i)=>i%channels===channel)}}/></>}
    {output?.shape.length === 3 && <><p className="eyebrow">Feature map · before activation</p><TensorHeatmap value={{shape:output.shape.slice(0,2),data:output.data.filter((_,i)=>i%filters===filter)}}/></>}
    <p className="coordinate-note">Each filter weight is reused at every spatial location. Click a cell to inspect its value or edit the shared weight.</p>
  </section>
}
