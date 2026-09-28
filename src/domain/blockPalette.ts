import type { NodeType } from './types'

export const blockPalette: Array<{ type: NodeType; label: string }> = [
  { type: 'dataset', label: 'Dataset' },
  { type: 'input', label: 'Input' },
  { type: 'weight', label: 'Param' },
  { type: 'arithmetic', label: 'Arithmetic' },
  { type: 'matmul', label: 'Matrix product' },
  { type: 'activation', label: 'Activation' },
  { type: 'standardize', label: 'Standardize features' },
  { type: 'dropout', label: 'Dropout' },
  { type: 'target', label: 'Target' },
  { type: 'loss', label: 'Loss' },
  { type: 'embedding', label: 'Embedding lookup' },
  { type: 'one-hot', label: 'One-hot' },
  { type: 'tensor-transform', label: 'Tensor transform' },
  { type: 'concat', label: 'Concatenate' },
  { type: 'softmax', label: 'Softmax' },
  { type: 'causal-mask', label: 'Causal mask' },
  { type: 'layer-norm', label: 'Layer norm' },
  { type: 'conv2d', label: 'Convolution' },
  { type: 'avgpool2d', label: 'Average pooling' },
]

export const blockCategories: Array<{ id: string; label: string; types: NodeType[] }> = [
  { id: 'core', label: 'Core model', types: ['dataset', 'input', 'weight', 'arithmetic', 'matmul', 'target', 'loss'] },
  { id: 'features', label: 'Features and tensors', types: ['standardize', 'tensor-transform', 'concat', 'one-hot'] },
  { id: 'neural', label: 'Neural networks', types: ['activation', 'softmax', 'dropout', 'layer-norm'] },
  { id: 'sequences', label: 'Sequences and attention', types: ['embedding', 'causal-mask'] },
  { id: 'images', label: 'Images', types: ['conv2d', 'avgpool2d'] },
]
