export function numberLabel(value: number): string {
  if (Math.abs(value) < 0.00005) return '0'
  return Number(value.toFixed(4)).toString()
}
