import { useState, type KeyboardEvent } from 'react'
import { validGoogleClientId } from '../domain/googleColab'

interface Props {
  clientId: string
  ready: boolean
  loadError?: string
  connected: boolean
  connecting: boolean
  status?: string
  onSave: (clientId: string) => void
  onConnect: (clientId: string) => void
  onDisconnect: () => void
  onOpenNotebook: () => void
  onClose: () => void
}

export function ColabConnectionDialog({ clientId, ready, loadError, connected, connecting, status, onSave, onConnect, onDisconnect, onOpenNotebook, onClose }: Props) {
  const [draft, setDraft] = useState(clientId)
  const cleanId = draft.trim()
  const valid = validGoogleClientId(cleanId)
  const matchesConnection = connected && cleanId === clientId
  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); onClose(); return }
    if (event.key !== 'Tab') return
    const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('a[href], input, button:not([disabled])')]
    const first = controls[0], last = controls[controls.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
  }

  return <div className="colab-connection-backdrop" onPointerDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="colab-connection-title" className="colab-connection-dialog" onKeyDown={handleKeyDown}>
      <p className="eyebrow">File export</p>
      <h2 id="colab-connection-title">Colab connection</h2>
      <p>Connect Google Drive to upload a generated notebook and open it directly in Colab. The notebook can contain your model parameters and dataset. Nothing uploads until you choose <strong>Upload &amp; open</strong>.</p>
      <label htmlFor="colab-client-id">Google OAuth web client ID</label>
      <input id="colab-client-id" type="text" autoFocus autoComplete="off" spellCheck={false} value={draft} onChange={event => setDraft(event.target.value)} placeholder="123…apps.googleusercontent.com" />
      {cleanId && !valid ? <p role="alert" className="colab-connection-error">Enter a web client ID ending in .apps.googleusercontent.com.</p> : null}
      <p className="colab-connection-help">Create a Google Cloud OAuth web client, enable the Drive API, and authorize this site's origin. Enter the public client ID, never a client secret. See the <a href="https://github.com/davbachman/BackpropBuilder/blob/main/docs/FILES-AND-EXPORT.md#direct-colab-connection" target="_blank" rel="noopener noreferrer">setup guide</a>.</p>
      <div className="colab-connection-actions">
        <button type="button" onClick={() => onSave(cleanId)} disabled={!valid || cleanId === clientId}>Save client ID</button>
        <button type="button" className="colab-connect-button" onClick={() => onConnect(cleanId)} disabled={!valid || !ready || connecting}>{connecting ? 'Connecting…' : 'Connect Google Drive'}</button>
      </div>
      <p className="colab-connection-status" role="status">{loadError || (cleanId !== clientId ? 'Save or connect this client ID to use it.' : status || (matchesConnection ? 'Connected for this browser session.' : ready ? 'Ready to connect. Google will ask for permission to create files made by this app.' : 'Loading Google sign-in…'))}</p>
      {matchesConnection ? <div className="colab-connection-actions">
        <button type="button" className="colab-connect-button" onClick={onOpenNotebook}>Upload &amp; open current notebook</button>
        <button type="button" onClick={onDisconnect}>Disconnect</button>
      </div> : null}
      <button type="button" className="colab-connection-close" onClick={onClose}>Close</button>
    </section>
  </div>
}
