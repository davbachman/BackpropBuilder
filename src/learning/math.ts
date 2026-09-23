/** A deliberately bounded, row-major tensor engine. `mul` is elementwise;
 * `matmul` is the separately named matrix product. Backward uses a DAG, so
 * shared parameter contributions add before the parameter is visited. */
import type { TensorValue } from '../domain/types'

const MAX_ELEMENTS = 200_000
const size = (shape: number[]) => shape.reduce((a, b) => a * b, 1)
function checkedSize(shape: number[]) {
  const count = size(shape)
  if (shape.some(d => !Number.isInteger(d) || d < 1) || !Number.isSafeInteger(count) || count > MAX_ELEMENTS) throw new Error(`Invalid or oversized tensor shape [${shape}].`)
  return count
}
const strides = (shape: number[]) => shape.map((_, i) => size(shape.slice(i + 1)))
const coords = (index: number, shape: number[]) => strides(shape).map((stride, i) => Math.floor(index / stride) % shape[i])
const offset = (indices: number[], shape: number[]) => indices.reduce((n, c, i) => n + c * strides(shape)[i], 0)

export class Tensor implements TensorValue {
  shape: number[]
  data: number[]
  grad: number[]
  requiresGrad: boolean
  /** Exact exclusions used by softmax; the finite display sentinel is never
   * relied on as an approximation of negative infinity. */
  excluded?: boolean[]
  parents: Tensor[] = []
  backwardRule: () => void = () => {}
  constructor(shape: number[], data: number[], requiresGrad = false) {
    if (checkedSize(shape) !== data.length) throw new Error(`Invalid tensor shape [${shape}]: element count differs.`)
    if (data.some(v => !Number.isFinite(v))) throw new Error('Tensor values must be finite.')
    this.shape = [...shape]
    this.data = [...data]
    this.grad = data.map(() => 0)
    this.requiresGrad = requiresGrad
  }
  toValue(): TensorValue { return { shape: [...this.shape], data: [...this.data] } }
  zeroGrad() { this.grad.fill(0) }
  backward(seed?: number[]) {
    if (!seed && this.data.length !== 1) throw new Error('A non-scalar backward call requires an explicit seed.')
    if (seed && (seed.length !== this.data.length || seed.some(v => !Number.isFinite(v)))) throw new Error('Backward seed shape mismatch or non-finite value.')
    const seen = new Set<Tensor>(), order: Tensor[] = []
    const visit = (t: Tensor) => { if (!seen.has(t)) { seen.add(t); t.parents.forEach(visit); order.push(t) } }
    visit(this)
    // Intermediate adjoints belong to this pass. Leaf gradients accumulate until zeroGrad.
    for (const t of order) if (t.parents.length) t.zeroGrad()
    const incoming = seed ?? [1]
    for (let i = 0; i < incoming.length; i++) this.grad[i] += incoming[i]
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardRule()
  }
}
export const tensor = (shape: number[], data: number[], requiresGrad = false) => new Tensor(shape, data, requiresGrad)
export const scalar = (value: number, requiresGrad = false) => tensor([], [value], requiresGrad)
const result = (shape: number[], data: number[], parents: Tensor[], backward: (out: Tensor) => void) => {
  const out = tensor(shape, data, parents.some(p => p.requiresGrad))
  if (out.requiresGrad) { out.parents = parents; out.backwardRule = () => backward(out) }
  return out
}
function broadcast(a: number[], b: number[]) {
  const n = Math.max(a.length, b.length), out = Array(n).fill(1)
  for (let i = 0; i < n; i++) {
    const x = a[a.length - n + i] ?? 1, y = b[b.length - n + i] ?? 1
    if (x !== y && x !== 1 && y !== 1) throw new Error(`Cannot broadcast [${a}] and [${b}].`)
    out[i] = Math.max(x, y)
  }
  return out
}
function broadcastIndex(index: number, outShape: number[], inputShape: number[]) {
  const c = coords(index, outShape).slice(outShape.length - inputShape.length)
  return offset(c.map((v, i) => inputShape[i] === 1 ? 0 : v), inputShape)
}
function binary(a: Tensor, b: Tensor, op: 'add' | 'mul') {
  const shape = broadcast(a.shape, b.shape), ai: number[] = [], bi: number[] = []
  const data = Array.from({ length: checkedSize(shape) }, (_, i) => {
    ai[i] = broadcastIndex(i, shape, a.shape); bi[i] = broadcastIndex(i, shape, b.shape)
    return op === 'add' ? a.data[ai[i]] + b.data[bi[i]] : a.data[ai[i]] * b.data[bi[i]]
  })
  return result(shape, data, [a, b], out => {
    for (let i = 0; i < data.length; i++) {
      if (a.requiresGrad) a.grad[ai[i]] += out.grad[i] * (op === 'add' ? 1 : b.data[bi[i]])
      if (b.requiresGrad) b.grad[bi[i]] += out.grad[i] * (op === 'add' ? 1 : a.data[ai[i]])
    }
  })
}
export const add = (a: Tensor, b: Tensor) => binary(a, b, 'add')
export const mul = (a: Tensor, b: Tensor) => binary(a, b, 'mul')
export const scale = (a: Tensor, factor: number) => mul(a, scalar(factor))
export const sub = (a: Tensor, b: Tensor) => add(a, scale(b, -1))
export function matmul(a: Tensor, b: Tensor) {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) throw new Error(`Matmul needs [m,k] @ [k,n]; received [${a.shape}] @ [${b.shape}].`)
  const [m, k] = a.shape, n = b.shape[1], data = Array(checkedSize([m, n])).fill(0)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let p = 0; p < k; p++) data[i * n + j] += a.data[i * k + p] * b.data[p * n + j]
  return result([m, n], data, [a, b], out => {
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let p = 0; p < k; p++) {
      if (a.requiresGrad) a.grad[i * k + p] += out.grad[i * n + j] * b.data[p * n + j]
      if (b.requiresGrad) b.grad[p * n + j] += out.grad[i * n + j] * a.data[i * k + p]
    }
  })
}
export function reshape(a: Tensor, shape: number[]) {
  if (size(shape) !== a.data.length) throw new Error('Reshape must preserve element count.')
  return result(shape, a.data, [a], out => { if (a.requiresGrad) out.grad.forEach((g, i) => { a.grad[i] += g }) })
}
export function transpose(a: Tensor, axes = a.shape.map((_, i) => a.shape.length - i - 1)) {
  if (axes.length !== a.shape.length || new Set(axes).size !== axes.length || axes.some(i => i < 0 || i >= axes.length || !Number.isInteger(i))) throw new Error('Invalid transpose permutation.')
  const shape = axes.map(i => a.shape[i]), map = a.data.map((_, i) => {
    const c = coords(i, shape), original = Array(c.length).fill(0)
    axes.forEach((axis, j) => { original[axis] = c[j] })
    return offset(original, a.shape)
  })
  return result(shape, map.map(i => a.data[i]), [a], out => { if (a.requiresGrad) map.forEach((index, i) => { a.grad[index] += out.grad[i] }) })
}
export function slice(a: Tensor, axis: number, start: number, end: number) {
  if (![axis, start, end].every(Number.isInteger) || axis < 0 || axis >= a.shape.length || start < 0 || end > a.shape[axis] || end <= start) throw new Error('Invalid slice bounds.')
  const shape = [...a.shape]; shape[axis] = end - start
  const map = Array.from({ length: size(shape) }, (_, i) => { const c = coords(i, shape); c[axis] += start; return offset(c, a.shape) })
  return result(shape, map.map(i => a.data[i]), [a], out => { if (a.requiresGrad) map.forEach((index, i) => { a.grad[index] += out.grad[i] }) })
}
export function concat(values: Tensor[], axis: number) {
  if (!values.length || axis < 0 || axis >= values[0].shape.length) throw new Error('Invalid concatenation.')
  const shape = [...values[0].shape]; shape[axis] = 0
  for (const t of values) {
    if (t.shape.length !== shape.length || t.shape.some((d, i) => i !== axis && d !== shape[i])) throw new Error('Concatenation shapes do not match.')
    shape[axis] += t.shape[axis]
  }
  const map = Array.from({ length: size(shape) }, (_, i) => {
    const c = coords(i, shape); let p = 0
    while (c[axis] >= values[p].shape[axis]) { c[axis] -= values[p].shape[axis]; p++ }
    return [p, offset(c, values[p].shape)]
  })
  return result(shape, map.map(([p, i]) => values[p].data[i]), values, out => map.forEach(([p, index], i) => { if (values[p].requiresGrad) values[p].grad[index] += out.grad[i] }))
}
export function sum(a: Tensor, axis?: number, keepDims = false) {
  if (axis !== undefined && (!Number.isInteger(axis) || axis < 0 || axis >= a.shape.length)) throw new Error('Invalid reduction axis.')
  const shape = axis === undefined ? (keepDims ? a.shape.map(() => 1) : []) : a.shape.flatMap((d, i) => i === axis ? (keepDims ? [1] : []) : [d])
  const data = Array(checkedSize(shape)).fill(0)
  const map = a.data.map((v, i) => {
    const c = coords(i, a.shape), target = axis === undefined ? [] : c.flatMap((v, j) => j === axis ? (keepDims ? [0] : []) : [v])
    const index = axis === undefined ? 0 : offset(target, shape); data[index] += v; return index
  })
  return result(shape, data, [a], out => { if (a.requiresGrad) map.forEach((index, i) => { a.grad[i] += out.grad[index] }) })
}
export const mean = (a: Tensor, axis?: number, keepDims = false) => scale(sum(a, axis, keepDims), 1 / (axis === undefined ? a.data.length : a.shape[axis]))
export function relu(a: Tensor) {
  return result(a.shape, a.data.map(v => Math.max(0, v)), [a], out => { if (a.requiresGrad) a.data.forEach((v, i) => { a.grad[i] += v > 0 ? out.grad[i] : 0 }) })
}
export function embedding(table: Tensor, ids: number[]) {
  if (table.shape.length !== 2 || !ids.length || ids.some(id => !Number.isInteger(id) || id < 0 || id >= table.shape[0])) throw new Error('Embedding token ID is out of range.')
  const d = table.shape[1]
  checkedSize([ids.length, d])
  const data = ids.flatMap(id => table.data.slice(id * d, (id + 1) * d))
  return result([ids.length, d], data, [table], out => { if (table.requiresGrad) ids.forEach((id, row) => { for (let j = 0; j < d; j++) table.grad[id * d + j] += out.grad[row * d + j] }) })
}
export function softmax(a: Tensor) {
  if (!a.shape.length) throw new Error('Softmax needs a final class axis.')
  const d = a.shape.at(-1)!, data = Array(a.data.length).fill(0)
  for (let row = 0; row < a.data.length / d; row++) {
    const start = row * d, entries = a.data.slice(start, start + d)
    const max = entries.reduce((largest, value, j) => a.excluded?.[start + j] ? largest : Math.max(largest, value), -Infinity)
    if (max === -Infinity) throw new Error('Softmax needs at least one allowed entry per row.')
    const e = entries.map((v, j) => a.excluded?.[start + j] ? 0 : Math.exp(v - max)), denom = e.reduce((n, v) => n + v, 0)
    for (let j = 0; j < d; j++) data[start + j] = e[j] / denom
  }
  return result(a.shape, data, [a], out => {
    if (!a.requiresGrad) return
    for (let row = 0; row < a.data.length / d; row++) {
      const start = row * d; let dot = 0
      for (let j = 0; j < d; j++) dot += data[start + j] * out.grad[start + j]
      for (let j = 0; j < d; j++) a.grad[start + j] += data[start + j] * (out.grad[start + j] - dot)
    }
  })
}
export function causalMask(a: Tensor) {
  if (a.shape.length !== 2 || a.shape[0] !== a.shape[1]) throw new Error('Causal mask expects square scores.')
  const n = a.shape[0], allowed = (i: number) => i % n <= Math.floor(i / n)
  const out = result(a.shape, a.data.map((v, i) => allowed(i) ? v : -1e9), [a], out => { if (a.requiresGrad) a.data.forEach((_, i) => { if (allowed(i)) a.grad[i] += out.grad[i] }) })
  out.excluded = a.data.map((_, i) => !allowed(i))
  return out
}
export function layerNorm(a: Tensor, gamma?: Tensor, beta?: Tensor, epsilon = 1e-5) {
  const d = a.shape.at(-1)
  if (!d || !Number.isFinite(epsilon) || epsilon <= 0 || (gamma && (gamma.shape.length !== 1 || gamma.shape[0] !== d)) || (beta && (beta.shape.length !== 1 || beta.shape[0] !== d))) throw new Error('Layer norm needs matching final-axis scale and bias.')
  const inv: number[] = [], normalized: number[] = [], data: number[] = []
  for (let row = 0; row < a.data.length / d; row++) {
    const entries = a.data.slice(row * d, (row + 1) * d), avg = entries.reduce((n, v) => n + v, 0) / d
    inv[row] = 1 / Math.sqrt(entries.reduce((n, v) => n + (v - avg) ** 2, 0) / d + epsilon)
    entries.forEach((v, j) => { const i = row * d + j; normalized[i] = (v - avg) * inv[row]; data[i] = normalized[i] * (gamma?.data[j] ?? 1) + (beta?.data[j] ?? 0) })
  }
  return result(a.shape, data, [a, ...(gamma ? [gamma] : []), ...(beta ? [beta] : [])], out => {
    for (let row = 0; row < a.data.length / d; row++) {
      const start = row * d; let total = 0, centered = 0
      for (let j = 0; j < d; j++) { const g = out.grad[start + j] * (gamma?.data[j] ?? 1); total += g; centered += g * normalized[start + j] }
      for (let j = 0; j < d; j++) {
        const i = start + j
        if (a.requiresGrad) a.grad[i] += inv[row] / d * (d * out.grad[i] * (gamma?.data[j] ?? 1) - total - normalized[i] * centered)
        if (gamma?.requiresGrad) gamma.grad[j] += out.grad[i] * normalized[i]
        if (beta?.requiresGrad) beta.grad[j] += out.grad[i]
      }
    }
  })
}
/** Mean next-token negative log likelihood, computed with log-sum-exp. */
export function crossEntropy(logits: Tensor, targets: number[]) {
  if (logits.shape.length !== 2 || targets.length !== logits.shape[0] || targets.some(t => !Number.isInteger(t) || t < 0 || t >= logits.shape[1])) throw new Error('Cross entropy target shape or token ID mismatch.')
  const [n, d] = logits.shape, probabilities = softmax(tensor(logits.shape, logits.data)).data
  let loss = 0
  for (let row = 0; row < n; row++) {
    const entries = logits.data.slice(row * d, (row + 1) * d), max = entries.reduce((max, value) => Math.max(max, value), -Infinity)
    loss += max + Math.log(entries.reduce((sum, v) => sum + Math.exp(v - max), 0)) - entries[targets[row]]
  }
  return result([], [loss / n], [logits], out => { if (logits.requiresGrad) probabilities.forEach((p, i) => { logits.grad[i] += out.grad[0] * (p - (targets[Math.floor(i / d)] === i % d ? 1 : 0)) / n }) })
}
