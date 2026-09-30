import { useEffect, useMemo, useState } from 'react'
import { Brush, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { IconArrowRight, IconPlayerSkipForward } from '@tabler/icons-react'
import { api, fmt, readNumber, type Item, type Metric } from './api'
import MatrixRain from './MatrixRain'
import type { WorkstationCopy } from './workstation-copy'

export const number = (value: unknown, digits = 0, fallback = '—') =>
  typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : fallback

export function clock(seconds: number | null | undefined, fallback = '—') {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return fallback
  const n = Math.floor(seconds)
  return `${String(Math.floor(n / 3600)).padStart(2, '0')}:${String(Math.floor(n % 3600 / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}`
}

export function bytes(value: unknown, fallback = '—') {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value / 1024 ** 3).toFixed(1)} GB` : fallback
}

export function StatusDot({ state = 'idle' }: { state?: 'live' | 'ready' | 'idle' | 'error' }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />
}

export function Boot({ onDone, s, gpuName, online }: { onDone: () => void; s: WorkstationCopy; gpuName?: unknown; online: boolean }) {
  const [step, setStep] = useState(0)
  const lines = [s.bootInit, s.bootCuda, s.bootTorch, s.bootManifest, s.bootCheckpoints,
    s.bootRender, gpuName ? s.bootGpu : s.gpuUnavailable, s.bootReady, online ? s.bootOnline : s.offline]
  useEffect(() => {
    const timer = window.setInterval(() => setStep(value => Math.min(value + 1, lines.length)), 330)
    const finish = window.setTimeout(onDone, 4050)
    return () => { clearInterval(timer); clearTimeout(finish) }
  }, [onDone, lines.length])
  return <div className="boot-screen" role="status" aria-live="off">
    <MatrixRain boot tier="balanced" />
    <div className="boot-vignette" />
    <div className="boot-body">
      <div className="boot-cursor" aria-hidden="true" />
      <div className="boot-log">{lines.slice(0, step).map((line, index) => <div key={index} className="boot-line"><span>{String(index + 1).padStart(2, '0')}</span><strong>{line}</strong><i>{index === step - 1 ? '▌' : 'OK'}</i></div>)}</div>
      <div className={`boot-wordmark ${step >= 5 ? 'visible' : ''}`}><span>NERF</span><strong>RESEARCH CONSOLE</strong><small>NEURAL RENDERING EXPERIMENTAL WORKSTATION</small></div>
      <div className="boot-meter"><span style={{ width: `${step / lines.length * 100}%` }} /></div>
      <button className="boot-skip" onClick={onDone}>{s.skip}<IconPlayerSkipForward size={16} /></button>
    </div>
  </div>
}

export function PreviewFrame({ image, title, step, label, empty, className = '', s }: {
  image?: Item; title: string; step?: number | null; label?: string; empty: string; className?: string; s: WorkstationCopy
}) {
  const url = api.asset(image?.url)
  return <div className={`preview-frame ${className}`}>
    <div className="preview-frame-head"><span>{title}</span><span>{typeof step === 'number' ? `${s.iteration} ${number(step)}` : label || '—'}</span></div>
    <div className="preview-frame-media">{url ? <img key={url} src={url} alt={`${title} · iteration ${number(step)}`} /> : <div className="preview-empty"><div className="preview-empty-ring" /><span>{empty}</span></div>}</div>
    <div className="preview-frame-foot"><span>{s.view} {String(image?.view_index ?? 0).padStart(3, '0')}</span><span>{image?.split?.toUpperCase() || 'VAL'} / {image?.diagnostic_preview ? s.previewKind : s.savedFrame}</span></div>
    <span className="corner top-left" /><span className="corner top-right" /><span className="corner bottom-left" /><span className="corner bottom-right" />
  </div>
}

export function Timeline({ steps, checkpoints, selected, onSelect, s, maxIteration }: {
  steps: number[]; checkpoints: Item[]; selected: number | null; onSelect: (step: number) => void;
  s: WorkstationCopy; maxIteration: number
}) {
  const current = selected ?? steps[steps.length - 1] ?? 0
  const index = Math.max(0, steps.indexOf(current))
  const max = Math.max(maxIteration, steps[steps.length - 1] ?? 1, 1)
  const checkpointSteps = [...new Set(checkpoints.map(item => item.iteration).filter((v): v is number => typeof v === 'number'))]
  return <div className="timeline">
    <div className="timeline-heading"><div><span className="eyebrow">{s.timeMachine}</span><h3>{s.snapshot} <strong>{steps.length ? number(current) : '—'}</strong></h3></div><p>{s.timeMachineHint}</p></div>
    <div className="timeline-rail" aria-hidden="true"><div className="timeline-energy" style={{ width: `${current / max * 100}%` }} />
      {checkpointSteps.map(step => <i key={`checkpoint-${step}`} className="timeline-node checkpoint" style={{ left: `${step / max * 100}%` }} title={`${s.checkpoints} ${step} · ${steps.includes(step) ? s.savedFrame : s.imageUnavailable}`} />)}
      {steps.map(step => <button type="button" key={`frame-${step}`} className={`timeline-node frame${current === step ? ' current' : ''}`} style={{ left: `${step / max * 100}%` }} title={`${s.snapshot} ${step}`} aria-label={`${s.snapshot} ${number(step)}`} onClick={() => onSelect(step)} />)}
    </div>
    <input className="timeline-input" aria-label={s.timeMachine} type="range" min="0" max={Math.max(steps.length - 1, 0)} step="1" value={index} disabled={steps.length < 2} onChange={event => onSelect(steps[Number(event.target.value)])} />
    <div className="timeline-labels"><span>0</span>{[5000, 10000, 15000, 30000, 50000].filter(step => step < max).map(step => <span key={step} style={{ left: `${step / max * 100}%` }}>{step / 1000}K</span>)}<span>{max >= 1000 ? `${max / 1000}K` : max}</span></div>
    {!steps.length && <div className="timeline-empty">{s.noTimeline}</div>}
  </div>
}

type PlotSeries = { key: string; name: string; color: string }
export function MetricPlot({ rows, series, height = 170, expanded = false, s }: {
  rows: Metric[]; series: PlotSeries[]; height?: number; expanded?: boolean; s: WorkstationCopy
}) {
  const [resetKey, setResetKey] = useState(0)
  const points = useMemo(() => {
    if (!rows.length) return []
    const stride = Math.max(1, Math.floor(rows.length / 800))
    return rows.filter((row, index) => index % stride === 0 || index === rows.length - 1 || readNumber(row, 'validation_psnr', 'ssim', 'lpips') !== null)
      .map(row => ({ ...row, iteration: readNumber(row, 'iteration') ?? 0 }))
  }, [rows])
  if (!points.length) return <div className="plot-empty">{s.unavailable}</div>
  const axisFor = (key: string) => key === 'total_loss' ? 'loss' : key === 'ssim' || key === 'lpips' ? 'ratio' : 'psnr'
  const groups = [...new Set(series.map(item => axisFor(item.key)))]
  return <div className="metric-plot" style={{ height }}>
    {expanded && <button className="plot-reset" onClick={() => setResetKey(value => value + 1)}>{s.reset}<IconArrowRight size={13} /></button>}
    <ResponsiveContainer width="100%" height="100%"><LineChart data={points} margin={{ top: 9, right: groups.length > 1 ? 25 : 12, left: -22, bottom: expanded ? 4 : 0 }}>
      <CartesianGrid stroke="#1f392c" strokeDasharray="2 7" vertical={false} />
      <XAxis dataKey="iteration" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: '#749383', fontSize: 10 }} axisLine={{ stroke: '#264434' }} tickLine={false} tickFormatter={value => value >= 1000 ? `${Math.round(value / 1000)}K` : String(value)} />
      {groups.map((group, index) => <YAxis key={group} yAxisId={group} orientation={index === 0 ? 'left' : 'right'} hide={index > 1} domain={['auto', 'auto']} tick={{ fill: '#749383', fontSize: 10 }} axisLine={false} tickLine={false} width={64} tickFormatter={value => Number(value).toFixed(Math.abs(value) < .1 ? 3 : Math.abs(value) < 1 ? 2 : 1)} />)}
      <Tooltip contentStyle={{ background: '#101c17', border: '1px solid #42614d', color: '#e5f2e8', fontSize: 11, fontFamily: 'var(--mono)' }} labelFormatter={value => `${s.iteration} ${fmt(value, 0)}`} />
      {series.map(item => <Line key={item.key} yAxisId={axisFor(item.key)} type="linear" dataKey={item.key} name={item.name} stroke={item.color} strokeWidth={2} dot={false} activeDot={{ r: 3 }} connectNulls isAnimationActive={false} />)}
      {expanded && points.length > 8 && <Brush key={resetKey} dataKey="iteration" height={19} stroke="#a4f66b" fill="#132119" travellerWidth={7} tickFormatter={value => String(value)} />}
    </LineChart></ResponsiveContainer>
  </div>
}
