import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { scratchModel } from '../test/scratchModels'
import { createNode } from '../domain/examples'
import { blockPalette } from '../domain/blockPalette'
import { forwardPass } from '../domain/engine'
import { datasetOutputValueForSlot } from '../domain/datasets'
import { tensorValue } from '../domain/tensor'
import { predictionNode } from '../domain/datasetTraining'
import type { GraphModel, GraphNode } from '../domain/types'
import { ModelInspector } from './ModelInspector'
import { DecoderControls } from './DecoderControls'
import { ConvolutionInspector } from './ConvolutionInspector'
import { DatasetWorkbench } from './DatasetWorkbench'

const callbacks = {onParams:vi.fn(),onValue:vi.fn(),onDataset:vi.fn(),onOpen:vi.fn(),onInspectNeuron:vi.fn(),onGroup:vi.fn(),selectionCount:1}
function InspectorHarness({initial}: {initial:GraphNode}) {
  const [node,setNode] = useState(initial)
  return <ModelInspector {...callbacks} graph={{nodes:[node],edges:[],learningRate:.01}} node={node} onValue={(_,value)=>setNode({...node,params:{...node.params,value}})}/>
}

describe('scratch model authoring controls',()=>{
  it('builds both scratch architectures using only blocks in the current palette',()=>{
    const available = new Set(blockPalette.map(item => item.type))
    for (const kind of ['cnn', 'transformer'] as const) {
      const { graph } = scratchModel(kind)
      expect(graph.nodes.every(node => available.has(node.type))).toBe(true)
      expect(forwardPass(graph).loss).toBeGreaterThan(0)
    }
  })
  it('creates an entire filter tensor and retains the selected initializer',()=>{
    render(<InspectorHarness initial={createNode('weight',1)}/>)
    fireEvent.change(screen.getByLabelText('Tensor shape'),{target:{value:'4,3,3,1'}})
    fireEvent.change(screen.getByLabelText('Tensor initializer'),{target:{value:'he'}})
    fireEvent.click(screen.getByRole('button',{name:'Initialize tensor'}))
    expect(screen.getByLabelText('Tensor shape')).toHaveValue('4, 3, 3, 1')
    expect(screen.getByLabelText('Tensor initializer')).toHaveValue('he')
    const values = (screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',').map(Number)
    expect(values).toHaveLength(36)
    expect(values.every(Number.isFinite)).toBe(true)
    fireEvent.change(screen.getByLabelText('Tensor shape'),{target:{value:'-3,0'}})
    fireEvent.click(screen.getByRole('button',{name:'Initialize tensor'}))
    expect(screen.getByRole('alert')).toHaveTextContent('positive dimensions')
  })

  it('accepts parenthesized parameter shapes and initializes every entry on Enter',()=>{
    render(<InspectorHarness initial={createNode('weight',1)}/>)
    const shape = screen.getByLabelText('Tensor shape')
    fireEvent.change(shape,{target:{value:'(3,1)'}})
    fireEvent.keyDown(shape,{key:'Enter'})
    expect(shape).toHaveValue('3, 1')
    expect((screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',')).toHaveLength(3)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('supports default transpose axes and a whole-tensor mean with keepDims',()=>{
    const node = createNode('transpose',1), onParams=vi.fn()
    const props={...callbacks,onParams,graph:{nodes:[node],edges:[],learningRate:.01}}
    const view=render(<ModelInspector {...props} node={node}/>)
    expect(screen.getByLabelText('Operation axes')).toHaveValue('')
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith(node.id,{axes:undefined})
    view.rerender(<ModelInspector {...props} node={createNode('mean',1)}/>)
    fireEvent.click(screen.getByLabelText('Keep dimensions'))
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith('mean-1',{axis:undefined,keepDims:true})
  })

  it('edits tensor transform parameters from Details at any zoom level',()=>{
    const node=createNode('tensor-transform',1), onParams=vi.fn()
    const props={...callbacks,onParams,graph:{nodes:[node],edges:[],learningRate:.01}}
    const view=render(<ModelInspector {...props} node={node}/>)
    expect(screen.getByLabelText('Tensor transform operation')).toHaveValue('reshape')
    fireEvent.change(screen.getByLabelText('Tensor transform operation'),{target:{value:'mean'}})
    expect(onParams).toHaveBeenCalledWith(node.id,{transform:'mean'})
    const changed={...node,params:{...node.params,transform:'mean' as const}}
    view.rerender(<ModelInspector {...props} graph={{...props.graph,nodes:[changed]}} node={changed}/>)
    expect(screen.getByLabelText('Operation axis')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Keep dimensions'))
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith(node.id,{axis:undefined,keepDims:true})
  })

  it('keeps loss editing visible and exact numbers available on demand',()=>{
    const prediction={...createNode('input',1),params:{value:tensorValue([2],[.2,.8])}}
    const target={...createNode('target',1),params:{value:tensorValue([2],[0,1])}}
    const loss=createNode('loss',1)
    const graph:GraphModel={nodes:[prediction,target,loss],edges:[
      {id:'prediction-loss',source:prediction.id,target:loss.id,inputSlot:0},
      {id:'target-loss',source:target.id,target:loss.id,inputSlot:1},
    ],learningRate:.01}
    const evaluated=forwardPass(graph).graph
    render(<ModelInspector {...callbacks} graph={evaluated} node={evaluated.nodes.find(node=>node.id===loss.id)}/>)
    expect(screen.getByLabelText('Loss function')).toHaveValue('mse')
    const disclosure=screen.getByText('Exact values and gradients').closest('details')!
    expect(disclosure).not.toHaveAttribute('open')
    fireEvent.click(screen.getByText('Exact values and gradients'))
    expect(disclosure).toHaveAttribute('open')
    expect(disclosure).toHaveTextContent('Value')
  })

  it('renames a selected block from its title, without the old selection heading',()=>{
    const node=createNode('weight',1), onRename=vi.fn()
    render(<ModelInspector {...callbacks} graph={{nodes:[node],edges:[],learningRate:.01}} node={node} onRename={onRename}/>)
    expect(screen.queryByText('Selected block')).not.toBeInTheDocument()
    expect(screen.queryByText('Rename block')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button',{name:'Rename Param 1'}))
    const name=screen.getByRole('textbox',{name:'Block name'})
    fireEvent.change(name,{target:{value:'Kernel'}})
    fireEvent.keyDown(name,{key:'Enter'})
    expect(onRename).toHaveBeenCalledWith(node.id,'Kernel')
  })

  it('renames a group from its title while keeping its kind selector',()=>{
    const group={id:'group-1',label:'Group 1',kind:'module',nodeIds:[],position:{x:0,y:0},dimensions:{width:200,height:100}}
    const onGroupChange=vi.fn()
    render(<ModelInspector {...callbacks} graph={{nodes:[],edges:[],groups:[group],learningRate:.01}} group={group} onGroupChange={onGroupChange}/>)
    fireEvent.click(screen.getByRole('button',{name:'Rename Group 1'}))
    const name=screen.getByRole('textbox',{name:'Block name'})
    fireEvent.change(name,{target:{value:'Attention head'}})
    fireEvent.blur(name)
    expect(onGroupChange).toHaveBeenCalledWith(group.id,{label:'Attention head'})
    fireEvent.click(screen.getByText('Group settings'))
    fireEvent.change(screen.getByRole('combobox',{name:'Block kind'}),{target:{value:'head'}})
    expect(onGroupChange).toHaveBeenCalledWith(group.id,{kind:'head'})
  })

  it('generates on a transformer whose nodes have only palette IDs',()=>{
    const {graph,datasetId}=scratchModel('transformer'), changed=vi.fn()
    render(<DecoderControls graph={graph} onGraphChange={changed}/>)
    fireEvent.change(screen.getByLabelText('Prompt'),{target:{value:'<bos> red green blue'}})
    fireEvent.click(screen.getByRole('button',{name:'Apply prompt'}))
    const next:GraphModel=changed.mock.calls[0][0]
    expect(datasetOutputValueForSlot(next.nodes.find(node=>node.id === datasetId)!,0).data).toEqual([0,1,2,3])
    expect(predictionNode(next)?.value?.shape).toEqual([4,5])
    fireEvent.click(screen.getByRole('button',{name:'Generate next token'}))
    expect(changed).toHaveBeenCalledTimes(2)
  })

  it('inspects arbitrary CNN filter counts and edits the actual shared kernel',()=>{
    const {graph}=scratchModel('cnn'), evaluated=forwardPass(graph).graph
    const node=evaluated.nodes.find(node=>node.type === 'conv2d')!, onValue=vi.fn()
    render(<ConvolutionInspector graph={evaluated} node={node} onValue={onValue}/>)
    expect((screen.getByLabelText('Filter to inspect') as HTMLSelectElement).options).toHaveLength(2)
    fireEvent.change(screen.getByLabelText('Filter to inspect'),{target:{value:'1'}})
    fireEvent.change(screen.getByLabelText('Selected matrix coordinate'),{target:{value:'0.75'}})
    expect(onValue.mock.calls[0][1].data[9]).toBe(.75)
    expect(screen.getByText('Feature map · before activation')).toBeVisible()
  })

  it('keeps dataset details for configuration, including a selectable train/test split',()=>{
    const {graph,datasetId}=scratchModel('transformer'), onParams=vi.fn()
    const node=graph.nodes.find(candidate=>candidate.id === datasetId)!
    render(<DatasetWorkbench graph={graph} node={node} onParams={onParams} onDataset={vi.fn()}/>)
    fireEvent.change(screen.getByLabelText('Train/test split'),{target:{value:'80'}})
    expect(onParams).toHaveBeenCalledWith(datasetId,expect.objectContaining({trainPercent:80,datasetSplit:'train'}))
    expect(screen.queryByRole('button',{name:/Train 1 epoch|Run inference|Evaluate training/i})).not.toBeInTheDocument()
  })
})
