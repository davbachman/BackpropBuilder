import { useState } from 'react'
import { fitStandardizer } from '../domain/standardizationFit'
import type { GraphModel, GraphNode, NodeParams } from '../domain/types'

export function StandardizationInspector({ graph, node, onParams }: { graph: GraphModel; node: GraphNode; onParams: (id: string, params: NodeParams) => void }) {
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const stats = node.params.standardization
  const fit = async () => {
    setBusy(true); setError('')
    try { onParams(node.id, { standardization: await fitStandardizer(graph, node.id) }) }
    catch (error) { setError(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }
  return <fieldset><legend>Feature standardization</legend>
    <p>z = (x − training mean) / training standard deviation</p>
    <button type="button" disabled={busy} onClick={() => void fit()}>{busy ? 'Fitting…' : stats ? 'Refit on training rows' : 'Fit on training rows'}</button>
    <p>Fit after connecting your features. Statistics stay fixed during training and prediction. Refit when changing the training split or feature construction. Do not refit on new test data.</p>
    {stats && <><p>Fitted on {stats.count} training observations. Constant columns use scale 1.</p><table><thead><tr><th>Feature</th><th>Mean</th><th>Scale</th></tr></thead><tbody>{stats.mean.map((mean, i) => <tr key={i}><td>{i + 1}</td><td>{mean.toPrecision(6)}</td><td>{stats.scale[i].toPrecision(6)}</td></tr>)}</tbody></table></>}
    {error && <p role="alert">{error}</p>}
  </fieldset>
}
