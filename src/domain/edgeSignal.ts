import type { TensorValue } from './types'

/** Fixed logarithmic scale makes intensity comparable across wires and runs.
 * Causal exclusions are display sentinels, not numerical signal. */
export function edgeSignalIntensity(value: TensorValue | undefined): number {
  if (!value) return 0.2
  const entries = value.data.filter((_, index) => !value.excluded?.[index])
  const magnitude = entries.reduce((mean, entry) => mean + Math.abs(entry) / Math.max(1, entries.length), 0)
  return 0.24 + 0.76 * Math.min(1, Math.log1p(magnitude) / Math.log(11))
}
