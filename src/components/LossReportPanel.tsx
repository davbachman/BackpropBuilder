import { formatNumber } from '../domain/engine'
import './lossReportPanel.css'

export interface LossReport { epoch: number; loss: number }

export function LossReportPanel({ reports }: { reports: LossReport[] }) {
  const download = () => {
    const data = ['epoch,loss', ...reports.map(report => `${report.epoch},${report.loss}`)].join('\n')
    const url = URL.createObjectURL(new Blob([data], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'backpropbuilder-loss.csv'
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }
  const width = 300, height = 126, padding = 18
  const minimum = reports.reduce((value, report) => Math.min(value, report.loss), Infinity)
  const maximum = reports.reduce((value, report) => Math.max(value, report.loss), -Infinity)
  const span = maximum - minimum || Math.max(Math.abs(maximum), 1) * .1
  const firstEpoch = reports[0]?.epoch ?? 0, lastEpoch = reports.at(-1)?.epoch ?? 0
  const epochSpan = Math.max(1, lastEpoch - firstEpoch)
  const chartReports = reports.length <= 600 ? reports : reports.filter((_, index) => index === 0 || index === reports.length - 1 || index % Math.ceil(reports.length / 600) === 0)
  const shownReports = reports.slice(-250)
  const path = chartReports.map((report, index) => {
    const x = padding + (report.epoch - firstEpoch) / epochSpan * (width - padding * 2)
    const y = height - padding - (report.loss - minimum) / span * (height - padding * 2)
    return `${index ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`
  }).join(' ')

  return <section className="loss-report-panel" aria-label="Training loss history">
    <div className="loss-report-heading"><div><p className="eyebrow">Training record</p><h3>Loss by epoch</h3></div><span>{reports.length} reports</span></div>
    {reports.length ? <>
      <p className="loss-report-note">Loss over the training set; the graph above may show one example.</p>
      <button type="button" className="loss-report-download" onClick={download}>Download all losses · CSV</button>
      <svg role="img" aria-label="Training loss chart" viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <path d={path} />
      </svg>
      <div className="loss-report-range"><span>Epoch {firstEpoch}</span><span>Epoch {lastEpoch}</span></div>
      {reports.length > shownReports.length && <p className="loss-report-note">Showing the latest {shownReports.length} values below; the chart spans the full run.</p>}
      <div className="loss-report-list" role="list" aria-label="Reported losses">
        {shownReports.map(report => <div role="listitem" key={report.epoch}><span>Epoch {report.epoch}</span><strong>{formatNumber(report.loss)}</strong></div>)}
      </div>
    </> : <p className="coordinate-note">Run training from the Train tab to record loss here.</p>}
  </section>
}
