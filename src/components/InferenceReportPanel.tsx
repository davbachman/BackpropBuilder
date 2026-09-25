import { useState } from 'react'
import type { DatasetMetrics } from '../domain/datasetTraining'
import { formatNumber } from '../domain/engine'
import './inferenceReportPanel.css'

export function InferenceReportPanel({ metrics, split, task }: { metrics?: DatasetMetrics; split?: 'train' | 'test'; task?: string }) {
  const [predictionPage, setPredictionPage] = useState(0)
  const rows = metrics?.rows ?? []
  const pageCount = Math.max(1, Math.ceil(rows.length / 25))
  const page = Math.min(predictionPage, pageCount - 1)
  return <section className="inference-report-panel" aria-label="Inference report">
    <p className="eyebrow">Model evaluation</p>
    <h3>{split === 'train' ? 'Training-set predictions' : 'Test-set predictions'}</h3>
    {metrics ? <>
      <div className="inference-report-summary">
        <span>{metrics.examples} {split === 'train' ? 'training' : 'held-out'} examples</span>
        <strong>{formatNumber(metrics.loss)} <small>loss</small></strong>
        {metrics.accuracy !== undefined && <strong aria-label={`${split === 'train' ? 'Training' : 'Test'} accuracy`}>{(metrics.accuracy * 100).toFixed(1)}% <small>{task === 'sequence' ? 'token' : 'class'} accuracy</small></strong>}
      </div>
      {rows.length > 0 && <div className="inference-predictions" aria-label={`${split === 'train' ? 'Training' : 'Test'} predictions`}>
        <div className="inference-predictions-heading"><h4>Predictions</h4><span>{rows.length} outputs</span></div>
        <div className="inference-predictions-scroll"><table><thead><tr><th>Example</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>{rows.slice(page * 25, (page + 1) * 25).map((row, index) => <tr key={`${page}-${index}`} className={row.correct === undefined ? '' : row.correct ? 'is-correct' : 'is-incorrect'}><td>{row.example}</td><td>{row.actual}</td><td>{row.predicted}{row.correct !== undefined && <span className="inference-prediction-result" aria-label={row.correct ? 'Correct' : 'Incorrect'}>{row.correct ? '✓' : '×'}</span>}</td></tr>)}</tbody></table></div>
        {pageCount > 1 && <div className="inference-predictions-pages"><button disabled={page === 0} onClick={() => setPredictionPage(page - 1)}>Previous</button><span>Page {page + 1} of {pageCount}</span><button disabled={page >= pageCount - 1} onClick={() => setPredictionPage(page + 1)}>Next</button></div>}
      </div>}
    </> : <p className="coordinate-note">Use Test in the left sidebar to run inference and inspect predictions.</p>}
  </section>
}
