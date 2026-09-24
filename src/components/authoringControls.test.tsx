import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { scratchModel } from '../test/scratchModels'
import { createNode } from '../domain/examples'
import { forwardPass } from '../domain/engine'
import { datasetOutputValueForSlot } from '../domain/datasets'
import { predictionNode } from '../domain/datasetTraining'
import type { GraphModel, GraphNode } from '../domain/types'
import { ModelInspector } from './ModelInspector'
import { DecoderControls } from './DecoderControls'
import { ConvolutionInspector } from './ConvolutionInspector'
import { DatasetWorkbench } from './DatasetWorkbench'

const callbacks = {onParams:vi.fn(),onValue:vi.fn(),onDataset:vi.fn(),onOpen:vi.fn(),onInspectNeuron:vi.fn(),onCopy:vi.fn(),onDuplicate:vi.fn(),onGroup:vi.fn(),selectionCount:1}
function InspectorHarness({initial}: {initial:GraphNode}) {
  const [node,setNode] = useState(initial)
  return <ModelInspector {...callbacks} graph={{nodes:[node],edges:[],learningRate:.01}} node={node} onValue={(_,value)=>setNode({...node,params:{...node.params,value}})}/>
}

describe('scratch model authoring controls',()=>{
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

  it('trains a user-created graph through dataset controls and evaluates held-out examples',async()=>{
    const {graph,datasetId}=scratchModel('transformer'), changed=vi.fn()
    render(<DatasetWorkbench graph={graph} node={graph.nodes.find(node=>node.id === datasetId)!} onParams={vi.fn()} onGraphChange={changed}/>)
    fireEvent.click(screen.getByRole('button',{name:'Train 1 epoch'}))
    await waitFor(()=>expect(changed).toHaveBeenCalledTimes(1))
    expect(changed.mock.calls[0][1]).toBe(1)
    fireEvent.click(screen.getByRole('button',{name:'Evaluate training & held-out data'}))
    expect(screen.getByText('Held out · 8 examples')).toBeVisible()
    expect(screen.getByRole('status')).toHaveTextContent('without updating parameters')
  })
})
