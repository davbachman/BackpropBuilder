import type { NodeType } from './types'

export const blockPalette: Array<{ type: NodeType; label: string }> = [
  { type: 'dataset', label: 'Dataset' },
  { type: 'input', label: 'Input' },
  { type: 'weight', label: 'Param' },
  { type: 'arithmetic', label: 'Arithmetic' },
  { type: 'matmul', label: 'Matrix product' },
  { type: 'activation', label: 'Activation' },
  { type: 'target', label: 'Target' },
  { type: 'loss', label: 'Loss' },
  { type: 'embedding', label: 'Embedding lookup' },
  { type: 'transpose', label: 'Transpose' },
  { type: 'slice', label: 'Slice tensor' },
  { type: 'concat', label: 'Concatenate' },
  { type: 'softmax', label: 'Softmax' },
  { type: 'causal-mask', label: 'Causal mask' },
  { type: 'layer-norm', label: 'Layer norm' },
  { type: 'reshape', label: 'Reshape' },
  { type: 'mean', label: 'Mean' },
  { type: 'cross-entropy', label: 'Cross-entropy' },
  { type: 'conv2d', label: 'Convolution' },
  { type: 'avgpool2d', label: 'Average pooling' },
]
