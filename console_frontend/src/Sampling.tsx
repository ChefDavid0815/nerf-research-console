import { useEffect, useRef, useState } from 'react'
import { IconArrowRight, IconCircleX, IconRadar } from '@tabler/icons-react'
import { api, fmt, type Item, type RayDiagnostic, type RayStage } from './api'
import { useI18n } from './i18n'
import './Sampling.css'

type SamplingData = { available: boolean; reason?: string; items?: Item[]; static_figures?: Item[] } | null
type TraceMetric = 'density' | 'alpha' | 'weights' | 'transmittance'
const metrics: TraceMetric[] = ['density', 'alpha', 'weights', 'transmittance']
const tones: Record<TraceMetric, string> = { density: '#b7ffcb', alpha: '#a8d9e1', weights: '#f0dd8f', transmittance: '#80c7a3' }

function Trace({ stage, metric, title }: { stage: RayStage; metric: TraceMetric; title: string }) {
  const { t } = useI18n()
  const values = stage[metric]
  const depth = stage.depths
  const loX = depth[0] ?? 0
  const hiX = depth[depth.length - 1] ?? 1
  const loY = metric === 'density' ? Math.min(0, ...values) : 0
  const hiY = Math.max(metric === 'transmittance' || metric === 'alpha' ? 1 : 0, ...values, 0.000001)
  const xx = (value: number) => 23 + (value - loX) / Math.max(hiX - loX, 0.000001) * 534
  const yy = (value: number) => 120 - (value - loY) / Math.max(hiY - loY, 0.000001) * 102
  const points = values.map((value, index) => `${xx(depth[index])},${yy(value)}`).join(' ')
  return <div className="ray-trace">
    <div className="ray-trace-head"><strong>{title}</strong><span>{t('{count} 个样本', { count: depth.length })} · {t(metric)}</span></div>
    <svg viewBox="0 0 580 150" role="img" aria-label={t('{stage} 的 {metric} 曲线', { stage: title, metric: t(metric) })}>
      <line x1="23" y1="120" x2="557" y2="120" stroke="#426a4e" />
      <line x1="23" y1="69" x2="557" y2="69" stroke="#284a37" strokeDasharray="3 6" />
      <line x1="23" y1="18" x2="557" y2="18" stroke="#284a37" strokeDasharray="3 6" />
      <polyline points={points} fill="none" stroke={tones[metric]} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <text x="23" y="142">t {loX.toFixed(2)}</text><text x="490" y="142">t {hiX.toFixed(2)}</text>
      <text x="512" y="16">{fmt(hiY, 3)}</text>
    </svg>
  </div>
}

function DepthRail({ data }: { data: RayDiagnostic }) {
  const { t } = useI18n()
  const all = data.combined_depths
  const lo = all[0] ?? 0
  const hi = all[all.length - 1] ?? 1
  const left = (value: number) => `${((value - lo) / Math.max(hi - lo, 0.000001) * 100).toFixed(2)}%`
  return <div className="ray-depth-rail">
    <div><span>{t('粗采样深度')}</span><div className="ray-depth-track">{data.coarse_depths.map((value, index) => <i key={index} className="coarse" style={{ left: left(value) }} />)}</div><b>{data.coarse_depths.length}</b></div>
    <div><span>{t('新增细采样深度')}</span><div className="ray-depth-track">{data.fine_depths.map((value, index) => <i key={index} className="fine" style={{ left: left(value) }} />)}</div><b>{data.fine_depths.length}</b></div>
    <small>{t('深度 t 为射线参数，不是物理世界距离。细网络在合并并排序后的深度上计算。')}</small>
  </div>
}

function RayPanel({ data, metric }: { data: RayDiagnostic; metric: TraceMetric }) {
  const { t } = useI18n()
  return <div className="ray-result">
    <div className="ray-result-top"><div><span>{t('检查点')}</span><strong>{fmt(data.checkpoint, 0)}</strong></div><div><span>{t('预测 RGB')}</span><strong>{data.fine.rgb.map(value => value.toFixed(3)).join(' / ')}</strong></div><div><span>{t('累积不透明度')}</span><strong>{fmt(data.fine.accumulated_opacity, 4)}</strong></div></div>
    <DepthRail data={data} />
    <Trace stage={data.coarse} metric={metric} title="COARSE" />
    <Trace stage={data.fine} metric={metric} title="FINE" />
    <p className="ray-source">SHA256 {String(data.source.checkpoint_sha256 || '').slice(0, 16)}… · {t('CPU 只读重算')}</p>
  </div>
}

export default function Sampling({ runId, data, artifacts, checks }: { runId: string; data: SamplingData; artifacts: Item[]; checks: Item[] }) {
  const { t } = useI18n()
  const [first, setFirst] = useState<number | null>(null)
  const [second, setSecond] = useState<number | null>(null)
  const [split, setSplit] = useState('val')
  const [view, setView] = useState(0)
  const [x, setX] = useState(400)
  const [y, setY] = useState(400)
  const [metric, setMetric] = useState<TraceMetric>('weights')
  const [left, setLeft] = useState<RayDiagnostic | null>(null)
  const [right, setRight] = useState<RayDiagnostic | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [showAllArtifacts, setShowAllArtifacts] = useState(false)
  const requestVersion = useRef(0)
  const options = [...new Set(checks.map(item => item.iteration).filter((value): value is number => typeof value === 'number'))].sort((a, b) => a - b)
  const checkpointA = first !== null && options.includes(first) ? first : options.includes(500) ? 500 : options[0]
  const checkpointB = second !== null && options.includes(second) ? second : options[options.length - 1]
  useEffect(() => { requestVersion.current++; setLeft(null); setRight(null); setError(''); setFirst(null); setSecond(null); setShowAllArtifacts(false) }, [runId])
  useEffect(() => () => { requestVersion.current++ }, [])

  const inspect = async () => {
    if (!runId || checkpointA === undefined || checkpointB === undefined) return
    const version = ++requestVersion.current
    setBusy(true)
    setError('')
    setLeft(null)
    setRight(null)
    const params = { split, view_index: view, x, y }
    try {
      const a = await api.rayDiagnostic(runId, { checkpoint: checkpointA, ...params })
      if (version !== requestVersion.current) return
      setLeft(a)
      const b = checkpointB === checkpointA ? a : await api.rayDiagnostic(runId, { checkpoint: checkpointB, ...params })
      if (version !== requestVersion.current) return
      setRight(b)
    } catch (reason) {
      if (version === requestVersion.current) setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (version === requestVersion.current) setBusy(false)
    }
  }

  return <div className="sampling-page">
    <div className="section-head"><div><h2>{t('分层采样诊断')}</h2><p>{t('从真实检查点按需重算单条射线；不会修改科研产物。')}</p></div></div>
    <div className="panel ray-controls">
      <div className="panel-heading"><h3><IconRadar size={19} /> {t('单射线检查台')}</h3><span>{t('按需确定性推理 · 非训练期记录')}</span></div>
      <div className="ray-control-grid">
        <label>{t('检查点 A')}<select value={checkpointA ?? ''} onChange={event => setFirst(Number(event.target.value))}>{options.map(value => <option key={value} value={value}>{t('Iteration')} {fmt(value, 0)}</option>)}</select></label>
        <label>{t('检查点 B')}<select value={checkpointB ?? ''} onChange={event => setSecond(Number(event.target.value))}>{options.map(value => <option key={value} value={value}>{t('Iteration')} {fmt(value, 0)}</option>)}</select></label>
        <label>{t('划分')}<select value={split} onChange={event => setSplit(event.target.value)}><option value="val">VAL</option><option value="test">TEST</option><option value="train">TRAIN</option></select></label>
        <label>{t('视角索引')}<input type="number" min="0" value={view} onChange={event => setView(Number(event.target.value))} /></label>
        <label>{t('Pixel X')}<input type="number" min="0" max="799" value={x} onChange={event => setX(Number(event.target.value))} /></label>
        <label>{t('Pixel Y')}<input type="number" min="0" max="799" value={y} onChange={event => setY(Number(event.target.value))} /></label>
      </div>
      <div className="ray-control-foot"><p>{t('两个检查点按顺序在 CPU 上读取同一像素。每次读取真实模型权重，可能需要几秒。')}</p><button className="action action-primary" type="button" onClick={inspect} disabled={busy || !runId || options.length === 0 || !Number.isInteger(view) || !Number.isInteger(x) || !Number.isInteger(y) || view < 0 || x < 0 || x > 799 || y < 0 || y > 799}>{busy ? t('正在重算真实射线…') : t('读取并对比射线')}</button></div>
      {error && <div className="inline-error" role="alert">{t('单射线读取失败')}: {error}</div>}
    </div>
    {(left || right) && <div className="panel ray-analysis"><div className="panel-heading"><h3>{t('检查点射线对照')}</h3><span>{split.toUpperCase()} / VIEW {String(view).padStart(3, '0')} / ({x}, {y})</span></div><div className="ray-metric-switch" role="group" aria-label={t('显示量')}>{metrics.map(value => <button key={value} className={metric === value ? 'active' : ''} onClick={() => setMetric(value)} style={{ '--trace-tone': tones[value] } as React.CSSProperties}>{t(value)}</button>)}</div><div className="ray-comparison">{left && <RayPanel data={left} metric={metric} />}{right ? <RayPanel data={right} metric={metric} /> : busy && <div className="ray-waiting">{t('正在读取第二个检查点…')}</div>}</div><p className="ray-integrity-note">{t('这些数值从冻结配置和对应 checkpoint 重新推理，未在原训练中保存。密度与细网络合并采样深度一一对应。')}</p></div>}
    <div className="panel sampling-figure-panel"><div className="panel-heading"><h3>{t('已有采样验证图')}</h3><span>{t('{count} 张 · 非逐射线数值', { count: data?.static_figures?.length || 0 })}</span></div>{data?.static_figures?.length ? <div className="sampling-figures">{data.static_figures.map((figure, index) => <figure key={index}>{figure.url && <a href={api.asset(figure.url)} target="_blank" rel="noreferrer"><img src={api.asset(figure.url)} alt={t('项目原有分层采样验证图 {index}', { index: index + 1 })} /></a>}<figcaption>{figure.path || t('验证图 {index}', { index: index + 1 })} · {t('项目既有验证产物')}</figcaption></figure>)}</div> : <div className="empty-state">{t('暂无可用验证图')}</div>}</div>
    <div className="panel"><div className="panel-heading"><h3>{t('其他已登记产物')}</h3><span>{t('{artifacts} 件 · {checkpoints} 个检查点', { artifacts: artifacts.length, checkpoints: checks.length })}</span></div>{artifacts.length ? <><div className="artifact-list">{(showAllArtifacts ? artifacts : artifacts.slice(0, 24)).map((item, index) => <a key={index} href={api.asset(item.url)} target="_blank" rel="noreferrer">{item.name || item.path || item.url || t('产物 {index}', { index: index + 1 })} <IconArrowRight size={15} /></a>)}</div>{artifacts.length > 24 && <button className="ray-artifact-more" type="button" onClick={() => setShowAllArtifacts(value => !value)}>{showAllArtifacts ? t('收起产物列表') : t('显示全部 {count} 件产物', { count: artifacts.length })}</button>}</> : <div className="empty-state"><IconCircleX size={20} />{t('暂无可用产物')}</div>}</div>
  </div>
}
