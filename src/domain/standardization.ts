import type { TensorValue } from './types'

export interface StandardizationStats {
  mean: number[]
  scale: number[]
  count: number
}

export function isStandardizationStats(value: unknown): value is StandardizationStats {
  if (!value || typeof value !== 'object') return false
  const stats = value as StandardizationStats
  return Array.isArray(stats.mean) && stats.mean.length > 0 &&
    stats.mean.every(Number.isFinite) && Array.isArray(stats.scale) &&
    stats.scale.length === stats.mean.length && stats.scale.every(x => Number.isFinite(x) && x > 0) &&
    Number.isInteger(stats.count) && stats.count > 0
}

/** A rank-one scalar batch may share one statistic; matrices require one per column. */
export function standardizationShapeMatches(shape: number[], width: number): boolean {
  return shape.length < 2 && width === 1 || shape.at(-1) === width
}

/** Streaming population statistics, independently along the final feature axis. */
export function standardizationAccumulator() {
  let count = 0, width = 0
  let mean: number[] = [], m2: number[] = []
  return {
    add(value: TensorValue) {
      const columns = value.shape.at(-1) ?? 1
      if (!count) { width = columns; mean = Array(width).fill(0); m2 = Array(width).fill(0) }
      if (!width || columns !== width || !value.data.length || value.data.length % width || value.data.some(x => !Number.isFinite(x))) {
        throw Error('Features must have a fixed width and finite values.')
      }
      for (let offset = 0; offset < value.data.length; offset += width) {
        count++
        for (let column = 0; column < width; column++) {
          const delta = value.data[offset + column] - mean[column]
          mean[column] += delta / count
          m2[column] += delta * (value.data[offset + column] - mean[column])
        }
      }
    },
    finish(): StandardizationStats {
      const stats = { mean, scale: m2.map(x => Math.sqrt(Math.max(0, x / count)) || 1), count }
      if (!isStandardizationStats(stats)) throw Error('Standardization needs finite numeric features and at least one observation.')
      return stats
    },
  }
}

export function standardize(value: TensorValue, stats?: StandardizationStats) {
  if (!isStandardizationStats(stats)) throw Error('Fit Standardize features on training rows first.')
  const width = stats.mean.length
  if (!standardizationShapeMatches(value.shape, width)) throw Error('Standardization feature count changed. Refit on the training data.')
  return {
    value: { shape: [...value.shape], data: value.data.map((x, i) => (x - stats.mean[i % width]) / stats.scale[i % width]) },
    derivative: { shape: [...value.shape], data: value.data.map((_, i) => 1 / stats.scale[i % width]) },
  }
}

