import { customCsvCardHeight, customCsvCardWidth, customCsvOutputTop } from './datasets'
import { FLEX_INPUT_HEIGHT_STEP, NODE_WIDTH, heightForInputCount, inputArityForNode, isFlexibleInputNodeType, outputArityForNode } from './engine'
import type { GraphNode } from './types'

/** Dimensions and port locations shared by BuilderNode, the zoom scene, and
 * its wire router. A card must keep the same connection geometry at every scale. */
export function builderCardWidth(node: GraphNode): number {
  return node.type === 'dataset' && node.params.dataset === 'custom-csv' ? customCsvCardWidth(node) : NODE_WIDTH
}

export function builderCardHeight(node: GraphNode): number {
  if (node.type === 'dataset' && node.params.dataset === 'custom-csv') return customCsvCardHeight(node, true)
  const inputCount = inputArityForNode(node)
  if (node.type === 'arithmetic' || isFlexibleInputNodeType(node.type)) return Math.max(190, heightForInputCount(inputCount) + 30)
  if (node.type === 'dataset') return Math.max(205, 65 + outputArityForNode(node) * 30)
  if (['weight', 'bias', 'activation', 'loss', 'tensor-transform'].includes(node.type)) return 205
  return 190
}

export function builderInputPortY(node: GraphNode, index: number): number {
  const count = inputArityForNode(node)
  return node.type === 'loss' ? builderCardHeight(node) * (index + 1) / (count + 1) : FLEX_INPUT_HEIGHT_STEP * (index + 1)
}

export function builderOutputPortY(node: GraphNode, index: number): number {
  const count = outputArityForNode(node)
  if (node.type === 'dataset' && node.params.dataset === 'custom-csv') return customCsvOutputTop(index, true)
  return count > 1 ? FLEX_INPUT_HEIGHT_STEP + index * 30 : builderCardHeight(node) / 2
}
