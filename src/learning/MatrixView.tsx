import type { TensorValue } from '../domain/types'
import { numberLabel } from './format'

export function MatrixView({
  label,
  value,
  row,
  selected,
  onCell,
  parameter = false,
  columnLabels,
  causal = false,
}: {
  label: string
  value: TensorValue
  row?: number
  selected?: [number, number]
  onCell?: (row: number, col: number) => void
  parameter?: boolean
  columnLabels?: string[]
  causal?: boolean
}) {
  const width = value.shape.at(-1) ?? value.data.length
  const height = value.data.length / width
  return (
    <div className={`matrix-view ${parameter ? 'matrix-parameter' : ''}`}>
      <div className="matrix-caption">
        <strong>{label}</strong>
        <span>
          {parameter ? 'learned parameter' : 'computed value'} ·{' '}
          {value.shape.join(' × ') || 'scalar'}
        </span>
      </div>
      <div className="matrix-scroll">
        <table aria-label={label}>
          <thead>
            <tr>
              <th scope="col">{height > 1 ? 'row' : ''}</th>
              {Array.from({ length: width }, (_, j) => (
                <th scope="col" key={j}>
                  {columnLabels?.[j] ?? j}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: height }, (_, i) => (
              <tr key={i} className={row === i ? 'matrix-selected-row' : ''}>
                <th scope="row">{i}</th>
                {Array.from({ length: width }, (_, j) => {
                  const n = value.data[i * width + j]
                  const masked = causal && j > i
                  const labelValue = masked ? 'masked' : numberLabel(n)
                  return (
                    <td key={j}>
                      <button
                        type="button"
                        aria-label={`${label} row ${i} column ${j}: ${labelValue}`}
                        aria-pressed={
                          selected?.[0] === i && selected?.[1] === j
                        }
                        className={masked ? 'masked-cell' : ''}
                        onClick={() => onCell?.(i, j)}
                        disabled={!onCell}
                        style={{
                          background: masked
                            ? undefined
                            : n < 0
                              ? `rgba(209, 134, 42, ${Math.min(0.55, 0.07 + Math.abs(n) * 0.18)})`
                              : `rgba(122, 83, 184, ${Math.min(0.55, 0.07 + Math.abs(n) * 0.18)})`,
                        }}
                      >
                        {labelValue}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
