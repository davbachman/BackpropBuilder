import { useState } from 'react'
import { formatNumber } from '../domain/engine'
import type { TensorValue } from '../domain/types'

/** Readable tensor coordinates, shared by attention and learned matrices. */
export function TensorHeatmap({
  value,
  onEdit,
}: {
  value: TensorValue
  onEdit?: (index: number, next: number) => void
}) {
  const [cell, setCell] = useState(0)
  const columns = value.shape.at(-1) ?? 1
  const rows = Math.ceil(value.data.length / columns)
  const pageRows = Math.min(24, rows)
  const [page, setPage] = useState(0)
  const start =
    Math.min(page * pageRows, Math.max(0, rows - pageRows)) * columns
  const visible = value.data.slice(start, start + pageRows * columns)
  const finite = value.data.filter(
    (number, index) => Number.isFinite(number) && !value.excluded?.[index],
  )
  const scale = Math.max(0.00001, ...finite.map(Math.abs))
  const selected = Math.min(cell, value.data.length - 1)
  return (
    <div className="tensor-heatmap">
      <div className="heatmap-caption">
        <span>{value.shape.join(' × ')} coordinates</span>
        <span>
          − <i className="negative-key" />
          <i className="positive-key" /> +
        </span>
      </div>
      <div className="tensor-grid-scroll">
        <div
          className="tensor-grid"
          style={{
            gridTemplateColumns: `repeat(${columns}, minmax(12px,1fr))`,
            minWidth: columns * 12,
          }}
        >
          {visible.map((number, offset) => {
            const index = start + offset,
              masked = value.excluded?.[index],
              intensity = Math.min(1, Math.abs(number) / scale)
            return (
              <button
                key={index}
                className={selected === index ? 'selected' : ''}
                style={{
                  background: masked
                    ? '#f0eef2'
                    : number < 0
                      ? `rgba(215,139,87,${0.08 + 0.84 * intensity})`
                      : `rgba(94,127,181,${0.08 + 0.84 * intensity})`,
                }}
                aria-label={`Row ${Math.floor(index / columns) + 1}, column ${(index % columns) + 1}: ${masked ? 'masked' : formatNumber(number)}`}
                title={masked ? 'Masked future position' : String(number)}
                onClick={() => setCell(index)}
              />
            )
          })}
        </div>
      </div>
      {rows > pageRows && (
        <label className="heatmap-pagination">
          Rows
          <select
            aria-label="Tensor row page"
            value={page}
            onChange={(event) => setPage(Number(event.target.value))}
          >
            {Array.from({ length: Math.ceil(rows / pageRows) }, (_, i) => (
              <option key={i} value={i}>
                {i * pageRows + 1}–{Math.min(rows, (i + 1) * pageRows)}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="heatmap-coordinate">
        <span>
          [{Math.floor(selected / columns)}, {selected % columns}]
        </span>
        {onEdit ? (
          <input
            aria-label="Selected matrix coordinate"
            type="number"
            step="0.05"
            value={value.data[selected]}
            onChange={(event) => onEdit(selected, Number(event.target.value))}
          />
        ) : (
          <strong>
            {value.excluded?.[selected]
              ? 'masked'
              : formatNumber(value.data[selected])}
          </strong>
        )}
      </div>
    </div>
  )
}
