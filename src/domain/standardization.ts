import type { TensorValue } from './types'

export interface StandardizationStats {
  mean: number[]
  scale: number[]
  count: number
}

export function isStandardizationStats(value: unknown): value is StandardizationStats {
  if (!value || typeof value !== 'object') return false
  const stats = value as StandardizationStats
  return Array.isArray(stats.mean) && stats.mean.length > 0 && stats.mean.length <= 129 &&
    stats.mean.every(Number.isFinite) && Array.isArray(stats.scale) &&
    stats.scale.length === stats.mean.length && stats.scale.every(x => Number.isFinite(x) && x > 0) &&
    Number.isInteger(stats.count) && stats.count > 0
}

export function standardize(value: TensorValue, stats?: StandardizationStats) {
  if (!isStandardizationStats(stats)) throw Error('Fit Standardize features on training rows first.')
  const width = stats.mean.length
  if (width !== 1 && value.shape.at(-1) !== width) throw Error('Standardization feature count changed. Refit on the training data.')
  return {
    value: { shape: [...value.shape], data: value.data.map((x, i) => (x - stats.mean[i % width]) / stats.scale[i % width]) },
    derivative: { shape: [...value.shape], data: value.data.map((_, i) => 1 / stats.scale[i % width]) },
  }
}

