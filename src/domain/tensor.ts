import type { TensorValue } from './types'

export function tensorValue(shape: number[], data: number[]): TensorValue {
  const normalizedShape = shape.map((dimension) => {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(`Invalid tensor shape dimension ${dimension}.`)
    }
    return dimension
  })
  const expectedSize = tensorSize(normalizedShape)
  if (data.length !== expectedSize) {
    throw new Error(`Tensor shape ${formatShape(normalizedShape)} expects ${expectedSize} value${expectedSize === 1 ? '' : 's'} but received ${data.length}.`)
  }
  if (data.some((value) => !Number.isFinite(value))) {
    throw new Error('Tensor data must contain only finite numbers.')
  }
  return { shape: normalizedShape, data: [...data] }
}

export function scalarValue(value: number): TensorValue {
  return tensorValue([], [value])
}

export function isTensorValue(value: unknown): value is TensorValue {
  if (!value || typeof value !== 'object') return false
  const candidate = value as TensorValue
  return (
    Array.isArray(candidate.shape) &&
    Array.isArray(candidate.data) &&
    candidate.shape.every((dimension) => Number.isInteger(dimension) && dimension >= 0) &&
    candidate.data.every((entry) => typeof entry === 'number' && Number.isFinite(entry)) &&
    tensorSize(candidate.shape) === candidate.data.length
  )
}

export function toTensor(value: TensorValue | number | undefined, fallback = 0): TensorValue {
  if (isTensorValue(value)) return cloneTensor(value)
  if (typeof value === 'number' && Number.isFinite(value)) return scalarValue(value)
  return scalarValue(fallback)
}

export function cloneTensor(value: TensorValue): TensorValue {
  return { shape: [...value.shape], data: [...value.data] }
}

export function tensorSize(shape: number[]): number {
  return shape.reduce((size, dimension) => size * dimension, 1)
}

export function isScalarTensor(value: TensorValue): boolean {
  return value.shape.length === 0
}

export function scalarFromTensor(value: TensorValue): number {
  if (!isScalarTensor(value)) {
    throw new Error(`Expected a scalar tensor but received shape ${formatShape(value.shape)}.`)
  }
  return value.data[0] ?? 0
}

export function zeroLike(value: TensorValue): TensorValue {
  return fillLike(value, 0)
}

export function oneLike(value: TensorValue): TensorValue {
  return fillLike(value, 1)
}

export function fillLike(value: TensorValue, fill: number): TensorValue {
  return tensorValue(value.shape, Array.from({ length: value.data.length }, () => fill))
}

export function tensorShapesEqual(first: number[], second: number[]): boolean {
  return first.length === second.length && first.every((dimension, index) => dimension === second[index])
}

export function broadcastShapeForShapes(shapes: number[][]): number[] | undefined {
  const nonScalarShapes = shapes.filter((shape) => shape.length > 0)
  if (nonScalarShapes.length === 0) return []
  const [firstShape] = nonScalarShapes
  if (nonScalarShapes.every((shape) => tensorShapesEqual(shape, firstShape))) return [...firstShape]
  return undefined
}

export function broadcastShapeForTensors(values: TensorValue[]): number[] | undefined {
  return broadcastShapeForShapes(values.map((value) => value.shape))
}

export function elementwiseTensors(
  values: TensorValue[],
  operation: (entries: number[]) => number,
): TensorValue {
  const shape = broadcastShapeForTensors(values)
  if (!shape) {
    throw new Error(`Incompatible tensor shapes: ${values.map((value) => formatShape(value.shape)).join(', ')}.`)
  }
  const size = tensorSize(shape)
  return tensorValue(
    shape,
    Array.from({ length: size }, (_, index) => operation(values.map((value) => valueAtBroadcastIndex(value, index)))),
  )
}

export function addTensorsExact(first: TensorValue, second: TensorValue): TensorValue {
  if (!tensorShapesEqual(first.shape, second.shape)) {
    throw new Error(`Cannot add exact tensor shapes ${formatShape(first.shape)} and ${formatShape(second.shape)}.`)
  }
  return tensorValue(first.shape, first.data.map((value, index) => value + second.data[index]))
}

export function subtractTensors(first: TensorValue, second: TensorValue): TensorValue {
  return elementwiseTensors([first, second], ([left, right]) => left - right)
}

export function multiplyTensors(first: TensorValue, second: TensorValue): TensorValue {
  return elementwiseTensors([first, second], ([left, right]) => left * right)
}

export function scaleTensor(value: TensorValue, scale: number): TensorValue {
  return tensorValue(value.shape, value.data.map((entry) => entry * scale))
}

export function reduceToShape(value: TensorValue, targetShape: number[]): TensorValue {
  if (tensorShapesEqual(value.shape, targetShape)) return cloneTensor(value)
  if (targetShape.length === 0) {
    return scalarValue(value.data.reduce((sum, entry) => sum + entry, 0))
  }
  if (isScalarTensor(value)) {
    return tensorValue(targetShape, Array.from({ length: tensorSize(targetShape) }, () => scalarFromTensor(value)))
  }
  throw new Error(`Cannot reduce tensor shape ${formatShape(value.shape)} to ${formatShape(targetShape)}.`)
}

export function sumTensor(value: TensorValue): number {
  return value.data.reduce((sum, entry) => sum + entry, 0)
}

export function formatTensor(value: TensorValue | number | undefined, digits = 3): string {
  return formatFullTensor(value, digits)
}

export function formatCompactTensor(value: TensorValue | number | undefined, digits = 3): string {
  if (value === undefined) return '--'
  const tensor = toTensor(value)
  if (isScalarTensor(tensor)) return formatScalarNumber(tensor.data[0], digits)
  if (tensor.data.length === 0) return '[]'
  return `[${formatScalarNumber(tensor.data[0], digits)},...]`
}

export function formatFullTensor(value: TensorValue | number | undefined, digits = 3): string {
  if (value === undefined) return '--'
  const tensor = toTensor(value)
  if (isScalarTensor(tensor)) return formatScalarNumber(tensor.data[0], digits)
  return `${formatShape(tensor.shape)} [${tensor.data.map((entry) => formatScalarNumber(entry, digits)).join(', ')}]`
}

export function formatTensorInput(value: TensorValue | number | undefined): string {
  const tensor = toTensor(value)
  if (isScalarTensor(tensor)) return String(tensor.data[0] ?? 0)
  return JSON.stringify(tensorToNested(tensor))
}

export function parseTensorInput(text: string): TensorValue | undefined {
  const trimmed = text.trim()
  if (trimmed.length === 0) return undefined
  const numeric = Number(trimmed)
  if (Number.isFinite(numeric)) return scalarValue(numeric)

  try {
    return tensorFromNested(JSON.parse(trimmed))
  } catch {
    return undefined
  }
}

export function formatShape(shape: number[]): string {
  return `[${shape.join(', ')}]`
}

function formatScalarNumber(value: number | undefined, digits: number): string {
  if (value === undefined || Number.isNaN(value)) return '--'
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2)
  }
  return value.toFixed(digits)
}

function valueAtBroadcastIndex(value: TensorValue, index: number): number {
  return isScalarTensor(value) ? value.data[0] ?? 0 : value.data[index] ?? 0
}

function tensorFromNested(value: unknown): TensorValue | undefined {
  const collected = collectNestedTensor(value)
  if (!collected) return undefined
  return tensorValue(collected.shape, collected.data)
}

function collectNestedTensor(value: unknown): { shape: number[]; data: number[] } | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return { shape: [], data: [value] }
  if (!Array.isArray(value)) return undefined

  if (value.length === 0) return { shape: [0], data: [] }

  const children = value.map((entry) => collectNestedTensor(entry))
  if (children.some((child) => !child)) return undefined
  const childValues = children as Array<{ shape: number[]; data: number[] }>
  const [firstChild] = childValues
  if (!childValues.every((child) => tensorShapesEqual(child.shape, firstChild.shape))) return undefined

  return {
    shape: [value.length, ...firstChild.shape],
    data: childValues.flatMap((child) => child.data),
  }
}

function tensorToNested(value: TensorValue): number | unknown[] {
  if (isScalarTensor(value)) return value.data[0] ?? 0
  return buildNested(value.shape, value.data, 0).value
}

function buildNested(shape: number[], data: number[], offset: number): { value: unknown[]; offset: number } {
  const [length, ...restShape] = shape
  if (restShape.length === 0) {
    return { value: data.slice(offset, offset + length), offset: offset + length }
  }

  const value: unknown[] = []
  let nextOffset = offset
  for (let index = 0; index < length; index += 1) {
    const child = buildNested(restShape, data, nextOffset)
    value.push(child.value)
    nextOffset = child.offset
  }
  return { value, offset: nextOffset }
}
