import {toTensor} from './tensor'
import type {GraphModel,GraphNode} from './types'
/** The objective is mean data loss + lambda*sum(abs(w)) or lambda/2*sum(w*w).
 * A shared parameter is counted once. Biases are excluded by default. */
export function regularizedParameters(graph:GraphModel,loss:GraphNode) {
 const ids=loss.params.regularizationParameterIds
 return graph.nodes.filter(n=>(n.type==='weight'||n.type==='bias')&&(ids?ids.includes(n.id):n.type==='weight'))
}
export function parameterPenalty(graph:GraphModel,loss=graph.nodes.find(n=>n.type==='loss'||n.type==='cross-entropy')) {
 if(!loss || !loss.params.regularization || loss.params.regularization==='none') return 0
 const strength=loss.params.regularizationStrength??0
 return strength*regularizedParameters(graph,loss).reduce((sum,n)=>sum+toTensor(n.params.value).data.reduce((s,w)=>s+(loss.params.regularization==='l1'?Math.abs(w):w*w/2),0),0)
}
export function penaltyGradients(graph:GraphModel,loss:GraphNode) {
 const kind=loss.params.regularization,strength=loss.params.regularizationStrength??0
 return !kind||kind==='none'?[]:regularizedParameters(graph,loss).map(node=>({node,value:{shape:toTensor(node.params.value).shape,data:toTensor(node.params.value).data.map(w=>strength*(kind==='l1'?Math.sign(w):w))}}))
}
