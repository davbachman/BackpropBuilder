import { describe, expect, it } from 'vitest'
import { backwardPass, forwardPass, runTrainingStep } from '../domain/engine'
import { CNN_CHECKPOINT, CNN_DIGITS, createCnnPreset, selectDigit } from './cnn'
import { conv2d } from './cnnMath'
import { toTensor } from '../domain/tensor'

describe('the connected handwritten-digit CNN', () => {
  it('uses disjoint, balanced training and held-out data', () => {
    expect(new Set(CNN_DIGITS.map(sample=>sample.id)).size).toBe(500)
    for(let label=0;label<10;label++) {
      expect(CNN_DIGITS.filter(sample=>sample.label===label && sample.split==='train')).toHaveLength(40)
      expect(CNN_DIGITS.filter(sample=>sample.label===label && sample.split==='test')).toHaveLength(10)
    }
    expect(CNN_DIGITS.every(sample=>sample.pixels.length===64 && sample.pixels.every(value=>value>=0 && value<=16))).toBe(true)
  })

  it('runs the included checkpoint through the same editable computation graph', () => {
    const graph=createCnnPreset()
    let correct=0
    for(const sample of CNN_DIGITS.filter(example=>example.split==='test')) {
      const result=selectDigit(graph,sample)
      const probabilities=result.nodes.find(node=>node.id==='cnn-probabilities')!.value!.data
      correct+=Number(probabilities.indexOf(Math.max(...probabilities))===sample.label)
    }
    expect(correct/100).toBeCloseTo(CNN_CHECKPOINT.metadata.testAccuracy,10)
    expect(correct).toBeGreaterThanOrEqual(90)
    const image=toTensor(graph.nodes.find(node=>node.id==='image-input')!.params.value)
    const kernels=toTensor(graph.nodes.find(node=>node.id==='conv-weights')!.params.value)
    const bias=toTensor(graph.nodes.find(node=>node.id==='conv-bias')!.params.value)
    expect(graph.nodes.find(node=>node.id==='conv-output')!.value).toEqual(conv2d(image,kernels,bias))
  })

  it('backpropagates to the shared filters and updates those exact parameters', () => {
    const graph=createCnnPreset(), backward=backwardPass(graph).graph
    const weight=backward.nodes.find(node=>node.id==='conv-weights')!
    expect(weight.grad?.data.some(value=>Math.abs(value)>1e-10)).toBe(true)
    const step=runTrainingStep(graph)
    expect(step.graph.nodes.find(node=>node.id==='conv-weights')!.params.value).not.toEqual(graph.nodes.find(node=>node.id==='conv-weights')!.params.value)
    expect(forwardPass(step.graph).loss).toBeLessThan(forwardPass(graph).loss!)
  })

  it('changes the image and target together while preserving learned filters', () => {
    const graph=createCnnPreset(),sample=CNN_DIGITS.find(example=>example.label===8 && example.split==='test')!
    const changed=selectDigit(graph,sample)
    expect(toTensor(changed.nodes.find(node=>node.id==='digit-target')!.params.value).data).toEqual([8])
    expect(changed.nodes.find(node=>node.id==='conv-weights')!.params).toEqual(graph.nodes.find(node=>node.id==='conv-weights')!.params)
    expect(changed.nodes.find(node=>node.id==='conv-output')!.value).not.toEqual(graph.nodes.find(node=>node.id==='conv-output')!.value)
  })
})
