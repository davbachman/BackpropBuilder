/** A single inferred dimension keeps user-built layers usable across sample
 * and batch sizes without rewriting their graph or stored layout. */
export function resolveReshape(shape: number[], size: number): number[] {
  if (shape.filter(d => d === -1).length > 1 || shape.some(d => !Number.isInteger(d) || (d < 1 && d !== -1))) throw new Error('Reshape needs positive dimensions and at most one -1 to infer a dimension.')
  const known = shape.reduce((a,b)=>b === -1 ? a : a*b,1)
  const result = shape.map(d=>d === -1 ? size/known : d)
  if (result.some(d=>!Number.isInteger(d) || d<1) || result.reduce((a,b)=>a*b,1) !== size) throw new Error('Reshape must preserve element count.')
  return result
}
