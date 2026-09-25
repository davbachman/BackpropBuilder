import { formatNumber } from '../domain/engine'
import './lossReportPanel.css'

export interface LossReport { epoch: number; loss: number; heldOutLoss?: number }

export function LossReportPanel({ reports, warning }: { reports: LossReport[]; warning?: string }) {
  const download = () => {
    const data = ['epoch,train_loss,held_out_loss', ...reports.map(report => `${report.epoch},${report.loss},${report.heldOutLoss ?? ''}`)].join('\n')
    const url = URL.createObjectURL(new Blob([data], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'backpropbuilder-loss.csv'
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }
  const width = 300, height = 126, padding = 18
  const minimum = reports.reduce((value, report) => Math.min(value, report.loss, report.heldOutLoss ?? Infinity), Infinity)
  const maximum = reports.reduce((value, report) => Math.max(value, report.loss, report.heldOutLoss ?? -Infinity), -Infinity)
  const span = maximum - minimum || Math.max(Math.abs(maximum), 1) * .1
  const firstEpoch = reports[0]?.epoch ?? 0, lastEpoch = reports.at(-1)?.epoch ?? 0
  const epochSpan = Math.max(1, lastEpoch - firstEpoch)
  const chartReports = reports.length <= 600 ? reports : reports.filter((_, index) => index === 0 || index === reports.length - 1 || index % Math.ceil(reports.length / 600) === 0)
  const shownReports = reports.slice(-250)
  const hasHeldOut = reports.some(report => report.heldOutLoss !== undefined)
  const point = (report: LossReport, value: number, continuation: boolean) => {
    const x = padding + (report.epoch - firstEpoch) / epochSpan * (width - padding * 2)
    const y = height - padding - (value - minimum) / span * (height - padding * 2)
    return `${continuation ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`
  }
  const trainPath = chartReports.map((report, index) => point(report, report.loss, index > 0)).join(' ')
  const heldOutPath = chartReports.map((report, index) => report.heldOutLoss === undefined ? ''
    : point(report, report.heldOutLoss, index > 0 && chartReports[index - 1].heldOutLoss !== undefined)).join(' ')

  return <section className="loss-report-panel" aria-label="Training loss history">
    <div className="loss-report-heading"><div><p className="eyebrow">Training record</p><h3>Loss by epoch</h3></div><span>{reports.length} reports</span></div>
    {reports.length ? <>
      <p className="loss-report-note">Loss over the training set{hasHeldOut ? ' and held-out examples' : ''}; the graph above may show one example. Held-out evaluation never updates parameters.</p>
      {warning && <p className="loss-report-warning" role="status">{warning}</p>}
      <button type="button" className="loss-report-download" onClick={download}>Download all losses · CSV</button>
      <div className="loss-report-legend"><span className="is-train">Training</span>{hasHeldOut && <span className="is-held-out">Held-out (validation)</span>}</div>
      <svg role="img" aria-label={hasHeldOut ? 'Training and held-out loss chart' : 'Training loss chart'} viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <path className="is-train" d={trainPath} />
        {hasHeldOut && <path className="is-held-out" d={heldOutPath} />}
      </svg>
      <div className="loss-report-range"><span>Epoch {firstEpoch}</span><span>Epoch {lastEpoch}</span></div>
      {reports.length > shownReports.length && <p className="loss-report-note">Showing the latest {shownReports.length} values below; the chart spans the full run.</p>}
      <div className="loss-report-list" role="list" aria-label="Reported losses">
        {shownReports.map(report => <div role="listitem" key={report.epoch}><span>Epoch {report.epoch}</span><strong>Train {formatNumber(report.loss)}{report.heldOutLoss !== undefined ? <> · Held-out {formatNumber(report.heldOutLoss)}</> : null}</strong></div>)}
      </div>
    </> : <p className="coordinate-note">Run training from the Train tab to record loss here.</p>}
  </section>
}
