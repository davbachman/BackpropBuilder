import { useRef, useState } from 'react'
import './editableBlockTitle.css'

interface Props {
  label: string
  onRename?: (label: string) => void
}

/** The visible title is also the rename affordance for every block detail. */
export function EditableBlockTitle({ label, onRename }: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(label)
  const canceled = useRef(false)
  const finish = () => {
    setEditing(false)
    if (canceled.current) { canceled.current = false; return }
    const next = draft.trim()
    if (next && next !== label) onRename?.(next)
  }

  return <h2 className="block-detail-title">
    {editing ? <input
      autoFocus
      aria-label="Block name"
      value={draft}
      onChange={event => setDraft(event.target.value)}
      onBlur={finish}
      onKeyDown={event => {
        if (event.key === 'Enter') event.currentTarget.blur()
        if (event.key === 'Escape') { canceled.current = true; event.currentTarget.blur() }
      }}
    /> : onRename ? <button
      type="button"
      className="editable-block-title"
      aria-label={`Rename ${label}`}
      title="Click to rename"
      onClick={() => { setDraft(label); setEditing(true) }}
    >{label}</button> : label}
  </h2>
}
