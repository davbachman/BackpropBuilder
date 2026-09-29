import { useState } from 'react'
import { Calculator, ChevronDown, ChevronRight } from 'lucide-react'
import { blockPalette, blockCategories } from '../domain/blockPalette'
import type { NodeType } from '../domain/types'
import './BlockPalette.css'

const storageKey = 'neural-canvas-block-categories'
function initialOpen(): string[] {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(storageKey) ?? localStorage.getItem('backprop-builder-block-categories') ?? 'null')
    if (Array.isArray(saved) && saved.every(value => typeof value === 'string')) {
      return saved.filter(value => blockCategories.some(category => category.id === value))
    }
  } catch { /* Storage is optional. */ }
  return ['core']
}

export function BlockPalette({ selected, onSelect }: { selected?: NodeType; onSelect: (type: NodeType) => void }) {
  const [open, setOpen] = useState(initialOpen)
  const [query, setQuery] = useState('')
  const search = query.trim().toLowerCase()
  const toggle = (id: string) => {
    const next = open.includes(id) ? open.filter(value => value !== id) : [...open, id]
    setOpen(next)
    try { localStorage.setItem(storageKey, JSON.stringify(next)) } catch { /* Storage is optional. */ }
  }
  const groups = blockCategories.map(category => ({
    ...category,
    items: category.types.map(type => blockPalette.find(item => item.type === type)!).filter(item =>
      !search || `${item.label} ${item.type} ${category.label}`.toLowerCase().includes(search)),
  })).filter(category => category.items.length)
  return <div className="block-categories">
    <div className="block-search">
      <input type="search" aria-label="Search blocks" placeholder="Search blocks…" value={query} onChange={event => setQuery(event.target.value)} />
      {query && <button type="button" aria-label="Clear block search" onClick={() => setQuery('')}>Clear</button>}
    </div>
    {groups.map(category => {
      const expanded = !!search || open.includes(category.id)
      return <section className="block-category" key={category.id} aria-label={category.label}>
        {search ? <h3 className="block-category-heading">{category.label}</h3> :
          <h3><button type="button" className="block-category-toggle" aria-expanded={expanded} aria-controls={`block-category-${category.id}`} onClick={() => toggle(category.id)}>
            {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
            <span>{category.label}</span><span className="block-category-count" aria-hidden="true">{category.items.length}</span>
          </button></h3>}
        <div id={`block-category-${category.id}`} className="palette-grid" hidden={!expanded}>
          {category.items.map(item => <button key={item.type} type="button" className={`palette-button ${selected === item.type ? 'is-selected' : ''}`} aria-pressed={selected === item.type} onClick={() => onSelect(item.type)}>
            <Calculator size={15} />{item.label}
          </button>)}
        </div>
      </section>
    })}
    {!groups.length && <p role="status" className="palette-intro">No blocks match “{query.trim()}”.</p>}
  </div>
}
