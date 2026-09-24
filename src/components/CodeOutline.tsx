import { ChevronDown, ChevronRight, Code2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { buildCodeOutline, type CodeOutlineLine } from '../domain/codeOutline'
import type { GraphModel } from '../domain/types'
import './codeOutline.css'

export interface CodeTarget { kind: 'group' | 'node'; id: string }

function syntax(code: string): ReactElement {
  const assignment = code.indexOf(' = ')
  const left = assignment < 0 ? '' : code.slice(0, assignment)
  const right = assignment < 0 ? code : code.slice(assignment + 3)
  const parts = right.split(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_]\w*(?=\()|[=+*/@()[\],])/g)
  return <code className="code-outline-statement">
    {left ? <><span className="code-token-variable">{left}</span><span className="code-token-operator"> = </span></> : null}
    {parts.map((part, index) => {
      const kind = /^['"]/.test(part) ? 'string' : /^\d/.test(part) ? 'number' : /^[A-Za-z_]\w*$/.test(part) && parts[index + 1] === '(' ? 'function' : /^[=+*/@()[\],]$/.test(part) ? 'operator' : ''
      return kind ? <span className={`code-token-${kind}`} key={index}>{part}</span> : part
    })}
  </code>
}

interface CodeOutlineProps {
  graph: GraphModel
  selected?: CodeTarget
  active: boolean
  onNavigate: (target: CodeTarget) => void
}

export function CodeOutline({ graph, selected, active, onNavigate }: CodeOutlineProps): ReactElement {
  const lines = useMemo(() => buildCodeOutline(graph), [graph])
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const rows = useRef(new Map<string, HTMLButtonElement>())
  const parents = useMemo(() => {
    const result = new Map<string, string | undefined>()
    const walk = (items: CodeOutlineLine[]) => items.forEach(line => { result.set(line.id, line.parentId); walk(line.children) })
    walk(lines)
    return result
  }, [lines])

  useEffect(() => {
    if (!selected || !parents.has(selected.id)) return
    const ancestors: string[] = []
    let parent = parents.get(selected.id)
    while (parent) { ancestors.push(parent); parent = parents.get(parent) }
    const frame = window.requestAnimationFrame(() => setExpanded(current => ancestors.every(id => current.has(id)) ? current : new Set([...current, ...ancestors])))
    return () => window.cancelAnimationFrame(frame)
  }, [parents, selected])

  useEffect(() => {
    if (!active || !selected) return
    const frame = window.requestAnimationFrame(() => rows.current.get(selected.id)?.scrollIntoView?.({ block: 'center', behavior: 'smooth' }))
    return () => window.cancelAnimationFrame(frame)
  }, [active, selected, expanded, lines])

  const toggle = (id: string) => setExpanded(current => {
    const next = new Set(current)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })

  let lineNumber = 0
  const renderLines = (items: CodeOutlineLine[], depth = 0): ReactElement[] => items.flatMap(line => {
    const number = ++lineNumber
    const group = line.kind === 'group'
    const opened = expanded.has(line.id)
    const selectedLine = selected?.id === line.id && selected.kind === line.kind
    const comment = !group && !line.code.startsWith(`${line.label} = `) ? line.label : undefined
    const row = <div className={`code-outline-row ${selectedLine ? 'is-selected' : ''}`} key={line.id}>
      <span className="code-outline-number" aria-hidden="true">{number}</span>
      <span className="code-outline-indent" aria-hidden="true" style={{ width: `${depth * 15}px` }} />
      {group ? <button type="button" className="code-outline-disclosure" aria-label={`${opened ? 'Collapse' : 'Expand'} ${line.label}`} aria-expanded={opened} onClick={() => toggle(line.id)}>{opened ? <ChevronDown size={13} /> : <ChevronRight size={13} />}</button>
        : <span className="code-outline-spacer" aria-hidden="true" />}
      <button type="button" className="code-outline-line" aria-current={selectedLine ? 'true' : undefined} aria-label={`${line.label}: ${line.code}`} title={line.label}
        ref={element => { if (element) rows.current.set(line.id, element); else rows.current.delete(line.id) }}
        onClick={() => onNavigate({ kind: line.kind, id: line.id })}>
        {syntax(line.code)}{comment ? <span className="code-outline-comment">  # {comment}</span> : null}
      </button>
    </div>
    return [row, ...(group && opened ? renderLines(line.children, depth + 1) : [])]
  })

  return <section className="code-outline" aria-label="Model code">
    <div className="code-outline-intro"><Code2 size={15} /><div><strong>model.pseudo</strong><p>Read-only pseudocode · select a line to find its block</p></div></div>
    {lines.length ? <div className="code-outline-list">{renderLines(lines)}</div> : <p className="code-outline-empty">Add a block to start the model.</p>}
  </section>
}
