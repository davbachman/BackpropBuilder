import type { CustomCsvData } from './types'

const MAX_ROWS = 10_000
const MAX_COLUMNS = 32
const MAX_FILE_CHARACTERS = 10_000_000

export interface CsvColumns {
  headers: string[]
  features: number[][]
  targets: number[]
  classLabels?: string[]
}

/** Read a local, numeric-feature CSV. Quoted commas, quotes and newlines are supported. */
export function parseCustomCsv(text: string, fileName: string): CustomCsvData {
  if (text.length > MAX_FILE_CHARACTERS) throw new Error('CSV is too large for the browser (10 MB maximum).')
  const rows = parseRows(text.replace(/^\uFEFF/, '')).filter(row => row.some(cell => cell.trim()))
  if (rows.length < 2) throw new Error('CSV needs at least two data rows.')
  const width = rows[0].length
  if (width < 2 || width > MAX_COLUMNS) throw new Error(`CSV needs 2–${MAX_COLUMNS} columns, including a target.`)
  if (rows.length > MAX_ROWS + 1) throw new Error(`CSV supports at most ${MAX_ROWS} data rows.`)
  if (rows.some(row => row.length !== width)) throw new Error('Every CSV row must have the same number of columns.')
  if (rows.some(row => row.some(cell => cell.length > 256))) throw new Error('A CSV cell is too long (256 characters maximum).')
  const hasHeader = rows[0].some((cell, index) => !isNumeric(cell) && isNumeric(rows[1][index]))
  const dataRows = rows.slice(hasHeader ? 1 : 0)
  const textColumns = Array.from({ length: width }, (_, index) => index)
    .filter(index => dataRows.every(row => row[index].trim() && !isNumeric(row[index])))
  if (textColumns.length > 1) throw new Error('Only one CSV column can contain text labels; model features must be numeric.')
  const targetColumn = textColumns[0] ?? width - 1
  const targets = dataRows.map(row => row[targetColumn].trim())
  const distinct = new Set(targets)
  const task = targets.every(isNumeric)
    ? distinct.size === 2 ? 'binary-classification' : 'regression'
    : distinct.size === 2 ? 'binary-classification' : 'classification'
  const csv: CustomCsvData = { fileName: (fileName || 'data.csv').slice(0, 256), rows, hasHeader, targetColumn, task }
  analyzeCustomCsv(csv)
  return csv
}

export function analyzeCustomCsv(csv: CustomCsvData): CsvColumns {
  const width = csv.rows[0]?.length ?? 0
  if (width < 2 || width > MAX_COLUMNS || csv.rows.length < 2 || csv.rows.length > MAX_ROWS + 1 ||
      csv.rows.some(row => row.length !== width || row.some(cell => typeof cell !== 'string' || cell.length > 256)) ||
      !Number.isInteger(csv.targetColumn) || csv.targetColumn < 0 || csv.targetColumn >= width ||
      !['regression', 'binary-classification', 'classification'].includes(csv.task)) {
    throw new Error('The CSV table or target column is invalid.')
  }
  const dataRows = csv.rows.slice(csv.hasHeader ? 1 : 0)
  if (dataRows.length < 2 || dataRows.length > MAX_ROWS) throw new Error('CSV needs 2–10,000 data rows.')
  const rawHeaders = csv.hasHeader ? csv.rows[0] : Array.from({ length: width }, (_, index) => index === csv.targetColumn ? 'y' : `x${index + 1}`)
  const headers = rawHeaders.map((header, index) => header.trim() || `column ${index + 1}`)
  const featureColumns = Array.from({ length: width }, (_, index) => index).filter(index => index !== csv.targetColumn)
  const features = featureColumns.map(index => dataRows.map((row, rowIndex) => {
    const raw = row[index].trim()
    if (!isNumeric(raw)) throw new Error(`Row ${rowIndex + 1}: ${headers[index]} must be numeric and nonempty.`)
    return Number(raw)
  }))
  const rawTargets = dataRows.map((row, rowIndex) => {
    const raw = row[csv.targetColumn].trim()
    if (!raw) throw new Error(`Row ${rowIndex + 1}: target ${headers[csv.targetColumn]} is empty.`)
    return raw
  })
  if (csv.task === 'regression') {
    if (rawTargets.some(value => !isNumeric(value))) throw new Error('Regression targets must be numeric. Choose a classification task for text labels.')
    return { headers, features, targets: rawTargets.map(Number) }
  }
  const classLabels = [...new Set(rawTargets)].sort((a, b) => isNumeric(a) && isNumeric(b) ? Number(a) - Number(b) : a.localeCompare(b))
  if (classLabels.length < 2) throw new Error('Classification needs at least two target classes.')
  if (csv.task === 'binary-classification' && classLabels.length !== 2) throw new Error('Binary classification needs exactly two target classes.')
  return { headers, features, targets: rawTargets.map(value => classLabels.indexOf(value)), classLabels }
}

export function isCustomCsvData(value: unknown): value is CustomCsvData {
  if (!value || typeof value !== 'object') return false
  const csv = value as Partial<CustomCsvData>
  if (typeof csv.fileName !== 'string' || csv.fileName.length > 256 || typeof csv.hasHeader !== 'boolean' ||
      !Array.isArray(csv.rows) || !Number.isInteger(csv.targetColumn)) return false
  try { analyzeCustomCsv(csv as CustomCsvData); return true } catch { return false }
}

function isNumeric(value: string): boolean {
  return value.trim() !== '' && Number.isFinite(Number(value))
}

function parseRows(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = [], cell = '', quoted = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { cell += '"'; index++ }
      else if (char === '"') quoted = false
      else cell += char
    } else if (char === '"') {
      if (cell.trim()) throw new Error('CSV has an unexpected quote.')
      quoted = true
    } else if (char === ',') {
      row.push(cell.trim()); cell = ''
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') index++
      row.push(cell.trim()); rows.push(row); row = []; cell = ''
    } else cell += char
  }
  if (quoted) throw new Error('CSV has an unclosed quoted value.')
  if (row.length || cell.trim()) { row.push(cell.trim()); rows.push(row) }
  return rows
}
