import { useState } from 'react'
import { datasetExampleIndex, datasetExamplesForNode, datasetMode, datasetOutputCountForNode, datasetOutputLabelForSlot, datasetOutputValueForSlot, datasetTargetSlotForNode } from '../domain/datasets'
import { toTensor } from '../domain/tensor'
import type { GraphEdge, GraphGroup, GraphModel, GraphNode, TensorValue } from '../domain/types'
import './DataInspector.css'

interface Signal {
  key: string
  label: string
  value?: TensorValue | number
  gradient?: TensorValue
}

interface Props {
  graph: GraphModel
  node?: GraphNode
  edge?: GraphEdge
  group?: GraphGroup
}

const PAGE_ROWS = 12
const PAGE_COLUMNS = 8

function shortNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  return magnitude >= 100000 || magnitude < 0.0001
    ? value.toExponential(3)
    : Number(value.toPrecision(5)).toString()
}

function shapeLabel(value: TensorValue): string {
  return value.shape.length ? `[${value.shape.join(' × ')}]` : 'scalar'
}

function signalList(graph: GraphModel, node?: GraphNode, edge?: GraphEdge, group?: GraphGroup): Signal[] {
  if (edge) return [{ key: edge.id, label: 'Connection', value: edge.value, gradient: edge.grad }]
  if (node?.type === 'dataset') {
    return Array.from({ length: datasetOutputCountForNode(node) }, (_, slot) => ({
      key: `${node.id}:${slot}`,
      label: datasetOutputLabelForSlot(node, slot),
      value: datasetOutputValueForSlot(node, slot),
    }))
  }
  if (node) {
    const inputs = graph.edges.filter(candidate => candidate.target === node.id).sort((a, b) => (a.inputSlot ?? 0) - (b.inputSlot ?? 0))
    return [
      { key: node.id, label: 'Output', value: node.value ?? node.params.value, gradient: node.grad },
      ...inputs.map(candidate => ({
        key: candidate.id,
        label: `Input ${(candidate.inputSlot ?? 0) + 1} · ${graph.nodes.find(source => source.id === candidate.source)?.label ?? candidate.source}`,
        value: candidate.value,
        gradient: candidate.grad,
      })),
    ]
  }
  if (!group) return []

  const members = new Set(group.nodeIds)
  const boundary = graph.edges.filter(candidate => members.has(candidate.source) && !members.has(candidate.target))
  if (boundary.length) return boundary.map(candidate => ({
    key: candidate.id,
    label: `${graph.nodes.find(item => item.id === candidate.source)?.label ?? candidate.source} → ${graph.nodes.find(item => item.id === candidate.target)?.label ?? candidate.target}`,
    value: candidate.value,
    gradient: candidate.grad,
  }))
  const terminal = graph.nodes.filter(candidate => members.has(candidate.id) && !graph.edges.some(link => link.source === candidate.id && members.has(link.target)))
  return terminal.map(candidate => ({ key: candidate.id, label: candidate.label, value: candidate.value, gradient: candidate.grad }))
}

function DatasetPreview({ node, signals }: { node: GraphNode; signals: Signal[] }) {
  const [page, setPage] = useState(0)
  const examples = datasetExamplesForNode(node)
  const rows = examples.length
  if (!rows) return null
  const pages = Math.ceil(rows / PAGE_ROWS)
  const currentPage = Math.min(page, pages - 1)
  const first = currentPage * PAGE_ROWS
  const targetSlot = datasetTargetSlotForNode(node)
  const activeExample = datasetMode(node) === 'sample' ? datasetExampleIndex(node) : -1
  return <section className="data-dataset-preview" aria-label="Dataset rows">
    <div className="data-section-heading"><h3>Dataset rows</h3><span>{rows} {rows === 1 ? 'example' : 'examples'}</span></div>
    <div className="data-table-scroll"><table className="data-table">
      <thead><tr><th scope="col">#</th><th scope="col">Split</th>{signals.map(signal => <th scope="col" key={signal.key} title={signal.label}>{signal.label}</th>)}</tr></thead>
      <tbody>{Array.from({ length: Math.min(PAGE_ROWS, rows - first) }, (_, offset) => {
        const row = first + offset
        const example = examples[row]
        return <tr key={row} className={row === activeExample ? 'is-current' : undefined}><th scope="row" title={example.label}>{row}</th><td>{example.split}</td>{signals.map((signal, slot) => {
          const featureSlot = slot < targetSlot ? slot : slot - 1
          const value = slot === targetSlot ? example.target : example.features[featureSlot]
          const content = !value ? '—' : value.data.length === 1 ? shortNumber(value.data[0]) : `${shapeLabel(value)} tensor`
          return <td key={signal.key} title={content}>{content}</td>
        })}</tr>
      })}</tbody>
    </table></div>
    {pages > 1 && <PageControls label="Rows" page={currentPage} pages={pages} onPage={setPage} range={`${first}–${Math.min(rows - 1, first + PAGE_ROWS - 1)}`} />}
    {activeExample >= 0 && <p className="data-current-note">Highlighted row: the example currently sent through the graph.</p>}
  </section>
}

function PageControls({ label, page, pages, onPage, range }: { label: string; page: number; pages: number; onPage: (page: number) => void; range: string }) {
  return <div className="data-page-controls">
    <span>{label} {range}</span>
    <div><button type="button" aria-label={`Previous ${label.toLowerCase()}`} disabled={page === 0} onClick={() => onPage(page - 1)}>←</button><button type="button" aria-label={`Next ${label.toLowerCase()}`} disabled={page >= pages - 1} onClick={() => onPage(page + 1)}>→</button></div>
  </div>
}

function flatIndex(coordinates: number[], shape: number[]): number {
  return shape.reduce((index, width, axis) => index * width + coordinates[axis], 0)
}

function TensorViewer({ value }: { value: TensorValue }) {
  const [rowPage, setRowPage] = useState(0)
  const [columnPage, setColumnPage] = useState(0)
  const [selectedCoordinates, setSelectedCoordinates] = useState<number[]>([])
  const [slice, setSlice] = useState<number[]>([])
  if (value.data.length === 0) return <p className="data-empty">This tensor is empty.</p>

  let min: number | undefined
  let max: number | undefined
  value.data.forEach((entry, index) => {
    if (!Number.isFinite(entry) || value.excluded?.[index]) return
    min = min === undefined ? entry : Math.min(min, entry)
    max = max === undefined ? entry : Math.max(max, entry)
  })
  const scale = Math.max(Math.abs(min ?? 0), Math.abs(max ?? 0), 0.000001)
  const rank = value.shape.length
  const rows = rank ? value.shape[0] : 1
  const columns = rank > 1 ? value.shape[1] : 1
  const rowPages = Math.max(1, Math.ceil(rows / PAGE_ROWS))
  const columnPages = Math.max(1, Math.ceil(columns / PAGE_COLUMNS))
  const visibleRowPage = Math.min(rowPage, rowPages - 1)
  const visibleColumnPage = Math.min(columnPage, columnPages - 1)
  const firstRow = visibleRowPage * PAGE_ROWS
  const firstColumn = visibleColumnPage * PAGE_COLUMNS
  const visibleRows = Math.min(PAGE_ROWS, rows - firstRow)
  const visibleColumns = Math.min(PAGE_COLUMNS, columns - firstColumn)
  const tail = value.shape.slice(2).map((size, index) => Math.min(Math.max(0, slice[index] ?? 0), size - 1))
  const coordinates = value.shape.map((size, index) => Math.min(Math.max(0, selectedCoordinates[index] ?? (index > 1 ? tail[index - 2] : 0)), size - 1))
  const selectedIndex = rank ? flatIndex(coordinates, value.shape) : 0
  const selectedValue = value.data[selectedIndex]
  const selectedMasked = value.excluded?.[selectedIndex]
  const color = (entry: number, masked: boolean | undefined) => masked ? undefined : {
    backgroundColor: entry < 0
      ? `rgba(220, 112, 91, ${0.08 + 0.48 * Math.abs(entry) / scale})`
      : `rgba(62, 166, 151, ${0.08 + 0.48 * Math.abs(entry) / scale})`,
  }

  return <>
    <div className="data-stats">
      <div><span>Shape</span><strong>{shapeLabel(value)}</strong></div>
      <div><span>Elements</span><strong>{value.data.length}</strong></div>
      <div><span>Range</span><strong>{min === undefined ? '—' : `${shortNumber(min)} to ${shortNumber(max!)}`}</strong></div>
    </div>
    {rank === 0 ? <div className="data-scalar" aria-label="Scalar value">{selectedMasked ? 'masked' : shortNumber(selectedValue)}</div> : <>
      {value.shape.slice(2).map((size, axis) => <label className="data-slice-control" key={axis}>Axis {axis + 2} index
        <select aria-label={`Axis ${axis + 2} index`} value={tail[axis]} onChange={event => { const next = [...tail]; next[axis] = Number(event.target.value); setSlice(next); setSelectedCoordinates([0, 0, ...next]) }}>
          {Array.from({ length: size }, (_, index) => <option key={index} value={index}>{index}</option>)}
        </select>
      </label>)}
      <div className="data-section-heading"><h3>{rank === 1 ? 'Values' : 'Tensor cells'}</h3><span>Click a value for its exact number</span></div>
      <div className="data-table-scroll"><table className="data-table data-value-table">
        <thead><tr><th scope="col">{rank === 1 ? 'Index' : 'Row'}</th>{rank === 1 ? <th scope="col">Value</th> : Array.from({ length: visibleColumns }, (_, offset) => <th scope="col" key={offset}>{firstColumn + offset}</th>)}</tr></thead>
        <tbody>{Array.from({ length: visibleRows }, (_, rowOffset) => {
          const row = firstRow + rowOffset
          return <tr key={row}><th scope="row">{row}</th>{Array.from({ length: visibleColumns }, (_,columnOffset) => {
            const column = firstColumn + columnOffset
            const cellCoordinates = rank === 1 ? [row] : [row, column, ...tail]
            const index = flatIndex(cellCoordinates, value.shape)
            const entry = value.data[index]
            const masked = value.excluded?.[index]
            return <td key={column}><button type="button" className={`data-cell ${selectedIndex === index ? 'is-selected' : ''}`} style={color(entry, masked)} aria-pressed={selectedIndex === index} aria-label={`${rank === 1 ? `Index ${row}` : `Row ${row}, column ${column}`}: ${masked ? 'masked' : String(entry)}`} onClick={() => setSelectedCoordinates(cellCoordinates)}>{masked ? 'masked' : shortNumber(entry)}</button></td>
          })}</tr>
        })}</tbody>
      </table></div>
      {rowPages > 1 && <PageControls label="Rows" page={visibleRowPage} pages={rowPages} onPage={setRowPage} range={`${firstRow}–${firstRow + visibleRows - 1}`} />}
      {columnPages > 1 && <PageControls label="Columns" page={visibleColumnPage} pages={columnPages} onPage={setColumnPage} range={`${firstColumn}–${firstColumn + visibleColumns - 1}`} />}
      <div className="data-exact"><span>{`[${coordinates.join(', ')}]`}</span><strong>{selectedMasked ? 'masked' : String(selectedValue)}</strong></div>
    </>}
  </>
}

export function DataInspector({ graph, node, edge, group }: Props) {
  const [signalIndex, setSignalIndex] = useState(0)
  const [mode, setMode] = useState<'value' | 'gradient'>('value')
  const signals = signalList(graph, node, edge, group)
  if (!node && !edge && !group) return <section className="data-inspector data-no-selection"><p className="eyebrow">Data viewer</p><h2>Select a block or wire</h2><p>Inspect the numbers moving forward and the gradients moving backward. Select a Dataset block to preview its rows and columns.</p></section>

  const selected = signals[Math.min(signalIndex, signals.length - 1)]
  const source = graph.nodes.find(candidate => candidate.id === edge?.source)
  const target = graph.nodes.find(candidate => candidate.id === edge?.target)
  const title = edge ? `${source?.label ?? edge.source} → ${target?.label ?? edge.target}` : group?.label ?? node?.label ?? ''
  const hasValue = selected?.value !== undefined
  const hasGradient = selected?.gradient !== undefined
  const shownMode = mode === 'gradient' && hasGradient || !hasValue && hasGradient ? 'gradient' : 'value'
  const tensor = shownMode === 'gradient' ? selected?.gradient : hasValue ? toTensor(selected.value) : undefined

  return <section className="data-inspector">
    <header className="data-header"><p className="eyebrow">{edge ? 'Connection data' : group ? 'Group data' : node?.type === 'dataset' ? 'Dataset data' : 'Block data'}</p><h2>{title}</h2><p>{edge ? 'Forward values travel along this wire; gradient contributions return through it during backpropagation.' : group ? 'Inspect signals leaving this group.' : node?.type === 'dataset' ? 'Browse all rows, then inspect the columns currently supplied by this Dataset block.' : 'Inspect this block’s output, inputs, and gradients.'}</p></header>
    {node?.type === 'dataset' && <DatasetPreview node={node} signals={signals} />}
    {signals.length > 1 && <label className="data-signal-select">{node?.type === 'dataset' ? 'Inspect column' : 'Inspect signal'}<select aria-label={node?.type === 'dataset' ? 'Inspect column' : 'Inspect signal'} value={Math.min(signalIndex, signals.length - 1)} onChange={event => setSignalIndex(Number(event.target.value))}>{signals.map((signal, index) => <option key={signal.key} value={index}>{signal.label}</option>)}</select></label>}
    {selected && <div className="data-view-tabs" role="tablist" aria-label="Data direction"><button type="button" role="tab" aria-selected={shownMode === 'value'} disabled={!hasValue} onClick={() => setMode('value')}>Forward value</button><button type="button" role="tab" aria-selected={shownMode === 'gradient'} disabled={!hasGradient} onClick={() => setMode('gradient')}>← Gradient</button></div>}
    {tensor ? <TensorViewer key={`${selected.key}:${shownMode}`} value={tensor} /> : <p className="data-empty">{signals.length === 0 ? 'This selection has no output signal to inspect.' : 'No value has reached this signal yet. Run forward or take a Step to calculate it.'}</p>}
  </section>
}
