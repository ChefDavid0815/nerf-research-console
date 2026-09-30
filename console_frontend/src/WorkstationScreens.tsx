import { useEffect, useMemo, useState } from 'react'
import { IconAdjustments, IconArrowRight, IconCheck, IconChevronDown, IconCpu, IconPlayerPause, IconPlayerPlay, IconRefresh, IconSearch, IconX } from '@tabler/icons-react'
import { api, readNumber, type Dict, type Item, type LiveEvent, type Metric, type Run, type System } from './api'
import type { Language } from './i18n'
import type { AnalysisTab, RunData, SystemTab } from './Workstation'
import type { WorkstationCopy } from './workstation-copy'
import { availableFrames, buildQuickConfig, encodedDimension, frameAt, mergePlotRows, modelParameterCount } from './workstation-model'
import { bytes, clock, MetricPlot, number, PreviewFrame, StatusDot, Timeline } from './WorkstationWidgets'

const liveStatus = (status?: string) => status === 'running' || status === 'stopping' || status === 'training'
const formatMetric = (source: Dict | Metric | null | undefined, keys: string[], digits = 3, empty = '—') => number(readNumber(source, ...keys), digits, empty)
const statusLabel = (status: string | undefined, s: WorkstationCopy) => status === 'completed' ? s.completedStatus : liveStatus(status) ? s.runningStatus : status === 'stopped' ? s.stoppedStatus : status === 'failed' ? s.failedStatus : s.createdStatus
const statusTone = (status?: string): 'live' | 'ready' | 'error' | 'idle' => liveStatus(status) ? 'live' : status === 'completed' ? 'ready' : status === 'failed' ? 'error' : 'idle'
const evaluationAt = (data: RunData, step: number): Item | undefined => data.evaluations.find(item => item.iteration === step && item.split === 'val' && item.view_index === 0)

function SectionIntro({ index, eyebrow, title, subtitle, action }: { index: string; eyebrow: string; title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="screen-intro"><div className="intro-index">{index}<span /></div><div className="intro-text"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{subtitle}</p></div>{action && <div className="intro-action">{action}</div>}</div>
}

function Stat({ label, value, unit, source, large = false }: { label: string; value: string | number | null | undefined; unit?: string; source?: string; large?: boolean }) {
  return <div className={`instrument-stat${large ? ' large' : ''}`}><span>{label}</span><strong>{value ?? '—'}{unit && <small>{unit}</small>}</strong>{source && <em>{source}</em>}</div>
}

type TrainProps = {
  s: WorkstationCopy; language: Language; defaults: Dict | null; scenes: Dict[]; system: System; connected: boolean;
  reference?: Item; data: RunData; selectedRun: Run | null; showLive: boolean;
  onNew: () => void; onLaunch: (config: Dict) => Promise<void>; onAction: (action: 'stop' | 'resume' | 'evaluate') => Promise<void>;
  onOpenAnalyze: () => void; acting: boolean; launchPhase: 'idle' | 'lock' | 'request' | 'wait' | 'done';
  selectedSnapshot: number | null; onSnapshot: (step: number | null) => void; lastEvent: LiveEvent | null; loading: boolean
}

const advancedFields = [
  ['batch_size', 'advancedBatch', 1, 65536], ['learning_rate', 'advancedLearning', 0.0000001, 1],
  ['checkpoint_interval', 'advancedCheckpoints', 1, 1000000], ['preview_interval', 'advancedPreview', 1, 1000000],
  ['network_depth', 'advancedDepth', 2, 32], ['network_width', 'advancedWidth', 16, 1024],
  ['direction_encoding_L', 'advancedDirection', 0, 15], ['render_chunk_size', 'advancedRender', 1, 65536],
] as const

export function TrainScreen(props: TrainProps) {
  const { s, defaults, scenes, system, connected, reference, showLive, onNew, onLaunch, launchPhase } = props
  const [L, setL] = useState(10)
  const [iterations, setIterations] = useState(30000)
  const [seed, setSeed] = useState(42)
  const [preset, setPreset] = useState<'demo' | 'standard' | 'high' | 'custom'>('standard')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [advanced, setAdvanced] = useState<Dict>({})
  const [draftError, setDraftError] = useState('')
  const sceneReady = scenes.some(scene => scene.id === 'lego' && scene.available === true)
  const gpuName = typeof system.gpu?.name === 'string' ? system.gpu.name : null
  const gpuLoad = readNumber(system.gpu, 'utilization_percent')
  const prepared = useMemo(() => {
    if (!defaults) return null
    try { return { ...buildQuickConfig(defaults, { L, iterations, seed }), ...advanced } }
    catch { return null }
  }, [defaults, L, iterations, seed, advanced])
  const choosePreset = (next: typeof preset) => {
    setPreset(next)
    if (next === 'demo') setIterations(5000)
    if (next === 'standard') setIterations(30000)
    if (next === 'high') setIterations(50000)
  }
  const start = () => {
    if (!defaults) return
    try {
      const config = { ...buildQuickConfig(defaults, { L, iterations, seed }), ...advanced }
      setDraftError('')
      onLaunch(config)
    } catch (cause) { setDraftError(cause instanceof Error ? cause.message : String(cause)) }
  }
  if (showLive) return <LiveTraining {...props} onNew={onNew} />
  const canStart = connected && sceneReady && !!prepared && launchPhase === 'idle'
  const phase = launchPhase === 'lock' ? s.launchLock : launchPhase === 'request' ? s.launchRequest : launchPhase === 'wait' ? s.launchWait : s.launchDone
  return <div className="train-screen setup-screen">
    <SectionIntro index="01" eyebrow={s.launcher} title={s.setupTitle} subtitle={s.setupSubtitle} action={<div className="intro-ready"><StatusDot state={canStart ? 'ready' : 'error'} /><span>{canStart ? s.systemReady : s.systemOffline}</span></div>} />
    <div className="setup-grid">
      <section className="setup-controls instrument-panel"><div className="panel-kicker"><span>{s.quickSetup}</span><span>01 / 04</span></div>
        <div className="scene-selector"><label>{s.scene}</label><button type="button" className="scene-option selected" disabled={!sceneReady} title={s.sceneNote}><span className="scene-glyph">◈</span><span><strong>LEGO</strong><small>NERF SYNTHETIC</small></span><IconCheck size={18} /></button><p>{s.sceneNote}</p></div>
        <div className="control-block"><div className="control-heading"><label htmlFor="bandwidth">{s.positionBandwidth}</label><span>L = {L}</span></div><div className="bandwidth-control"><button onClick={() => setL(value => Math.max(0, value - 1))} aria-label={`${s.positionBandwidth} −`} disabled={L === 0}>−</button><div className="bandwidth-value"><strong>{L}</strong><div className="frequency-bars" aria-hidden="true">{Array.from({ length: 15 }, (_, index) => <i key={index} className={index < L ? 'on' : ''} style={{ height: `${17 + index * 3}%` }} />)}</div></div><button onClick={() => setL(value => Math.min(15, value + 1))} aria-label={`${s.positionBandwidth} +`} disabled={L === 15}>+</button></div></div>
        <div className="control-block"><div className="control-heading"><label>{s.iterations}</label><span>{number(iterations)}</span></div><div className="segmented iterations">{[5000, 15000, 30000, 50000].map(value => <button key={value} className={iterations === value ? 'active' : ''} onClick={() => { setIterations(value); setPreset(value === 5000 ? 'demo' : value === 30000 ? 'standard' : value === 50000 ? 'high' : 'custom') }}>{value / 1000}K</button>)}</div>{preset === 'custom' && <input className="number-field custom-iterations" type="number" min="1" max="1000000" step="1" aria-label={s.iterations} value={iterations} onChange={event => setIterations(Number(event.target.value))} />}</div>
        <div className="control-block seed-block"><label htmlFor="seed">{s.seed}</label><input id="seed" className="number-field" type="number" min="0" max="2147483647" step="1" value={seed} onChange={event => setSeed(Number(event.target.value))} /></div>
        <div className="preset-block"><span className="eyebrow">{s.presets}</span><div className="preset-list">{([['demo', s.demo, s.demoDesc], ['standard', s.standard, s.standardDesc], ['high', s.highQuality, s.highDesc], ['custom', s.custom, ''] ] as const).map(([id, label, description]) => <button key={id} className={preset === id ? 'active' : ''} onClick={() => choosePreset(id)}><StatusDot state={preset === id ? 'ready' : 'idle'} /><strong>{label}</strong><small>{description}</small></button>)}</div></div>
        <div className="advanced-wrap"><button className="advanced-trigger" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen(value => !value)}><IconAdjustments size={16} />{s.advanced}<IconChevronDown className={advancedOpen ? 'up' : ''} size={15} /></button>{advancedOpen && <div className="advanced-fields"><p>{s.baseline}</p>{advancedFields.map(([key, label, min, max]) => <label key={key}><span>{s[label]}</span><input type="number" min={min} max={max} step={key === 'learning_rate' ? 'any' : 1} value={typeof advanced[key] === 'number' ? advanced[key] : typeof defaults?.[key] === 'number' ? defaults[key] as number : ''} onChange={event => setAdvanced(current => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}</div>}</div>
      </section>
      <section className="setup-visual instrument-panel"><div className="panel-kicker"><span>{s.sceneInput}</span><span>RGB / 256²</span></div><div className="scene-visual"><div className="scene-orbit orbit-one" /><div className="scene-orbit orbit-two" /><div className="scene-coordinate axis-x">X</div><div className="scene-coordinate axis-y">Y</div><div className="scene-coordinate axis-z">Z</div>{reference?.url ? <img src={api.asset(reference.url)} alt={s.groundTruth} /> : <div className="scene-visual-empty">{s.noSceneImage}</div>}<div className="scene-scan" /></div><div className="scene-caption"><span>01 / INPUT SCENE</span><strong>LEGO</strong><small>{s.reference}</small></div><div className="encoding-flow"><div><span>XYZ</span><small>3D INPUT</small></div><i /><div><span>γ<sub>{L}</sub>(x)</span><small>POSITIONAL ENCODING</small></div><i /><div><span>MLP</span><small>NEURAL FIELD</small></div><i /><div><span>RGB + σ</span><small>RENDER</small></div></div></section>
      <aside className="setup-readiness instrument-panel"><div className="panel-kicker"><span>{s.systemReady}</span><IconCpu size={17} /></div><div className="readiness-status"><StatusDot state={connected && sceneReady ? 'ready' : 'error'} /><strong>{connected && sceneReady ? s.systemReady : s.systemOffline}</strong><small>{connected ? s.online : s.offline}</small></div><div className="machine-name"><span>{s.gpuName}</span><strong>{gpuName || s.gpuUnavailable}</strong><small>{gpuName ? system.backend?.cuda_runtime ? `CUDA ${system.backend.cuda_runtime}` : s.cpuMode : s.cpuMode}</small></div><div className="readiness-list"><div><span>{s.backend}</span><b className={connected ? 'good' : 'bad'}>{connected ? s.online : s.offline}</b></div><div><span>{s.scene}</span><b className={sceneReady ? 'good' : 'bad'}>{sceneReady ? s.datasetReady : s.datasetMissing}</b></div><div><span>{s.gpuLoad}</span><b>{gpuLoad === null ? s.unavailable : `${number(gpuLoad)}%`}</b></div></div><div className="encoding-readout"><span>{s.positionBandwidth}</span><strong>L = {L}</strong><div><Stat label={s.encoding} value={`${encodedDimension(L)}D`} /><Stat label={s.maxBand} value={L ? `2^${L - 1}` : '—'} /><Stat label={s.parameters} value={prepared ? number(modelParameterCount(prepared)) : '—'} source={s.estimate} /></div></div></aside>
    </div>
    <div className="launch-deck"><div className="launch-readiness"><StatusDot state={canStart ? 'ready' : 'error'} /><span>{canStart ? s.systemReady : s.systemOffline}</span><small>{gpuName || s.cpuMode} / {sceneReady ? s.datasetReady : s.datasetMissing}</small></div><button className="launch-trigger" onClick={start} disabled={!canStart}><span className="launch-glyph"><IconPlayerPlay size={24} fill="currentColor" /></span><strong>{connected ? s.start : s.startUnavailable}</strong><span className="launch-arrow">↗</span></button><p>{s.startNote}</p>{draftError && <div className="field-error" role="alert">{draftError}</div>}</div>
    {launchPhase !== 'idle' && <div className="launch-transition" role="status"><div className="launch-transition-ring" /><span>{phase}</span><strong>{launchPhase === 'done' ? 'READY' : 'PROCESSING'}</strong></div>}
  </div>
}

function LiveTraining({ s, system, data, selectedRun, onNew, onAction, onOpenAnalyze, acting, selectedSnapshot, onSnapshot, lastEvent, loading }: TrainProps) {
  const frames = availableFrames(data.reconstructions, 'val', 0)
  const selectedStep = selectedSnapshot !== null && frames.includes(selectedSnapshot) ? selectedSnapshot : frames[frames.length - 1]
  const frame = typeof selectedStep === 'number' ? frameAt(data.reconstructions, selectedStep, 'prediction', 'val', 0) : undefined
  const metric = data.metrics[data.metrics.length - 1] || selectedRun?.latest_metrics
  const iteration = selectedRun?.iteration ?? readNumber(metric, 'iteration') ?? 0
  const budget = selectedRun?.effective_target_iteration ?? selectedRun?.training_iterations ?? 0
  const progress = budget > 0 ? Math.min(100, iteration / budget * 100) : 0
  const currentEval = typeof selectedStep === 'number' ? evaluationAt(data, selectedStep) : undefined
  const elapsed = readNumber(metric, 'elapsed_time') ?? selectedRun?.duration_seconds ?? null
  const eta = liveStatus(selectedRun?.status) && typeof elapsed === 'number' && iteration > 0 && budget > iteration ? elapsed / iteration * (budget - iteration) : null
  const gpuLoad = readNumber(system.gpu, 'utilization_percent')
  const complete = selectedRun?.status === 'completed'
  const failed = selectedRun?.status === 'failed'
  const noRun = !selectedRun && !loading
  const eventNames: Record<string, string> = { checkpoint_saved: 'CHECKPOINT SAVED', preview_ready: 'RECONSTRUCTION UPDATED', evaluation_started: 'VALIDATING', training_completed: 'TRAINING COMPLETE', training_failed: 'TRAINING INTERRUPTED' }
  return <div className="train-screen live-screen"><SectionIntro index="02" eyebrow={liveStatus(selectedRun?.status) ? s.live : complete ? s.completed : failed ? s.interrupted : s.noRun} title={selectedRun ? `${String(selectedRun.scene || 'LEGO').toUpperCase()} / ${s.reconstruction}` : s.noRun} subtitle={selectedRun ? `RUN ${selectedRun.id}` : s.previewPendingBody} action={<div className="live-actions">{liveStatus(selectedRun?.status) && <button className="quiet-button danger" disabled={acting} onClick={() => onAction('stop')}><IconPlayerPause size={16} />{s.stop}</button>}{selectedRun?.status === 'stopped' && <button className="quiet-button" disabled={acting} onClick={() => onAction('resume')}><IconPlayerPlay size={16} />{s.resume}</button>}{selectedRun && !liveStatus(selectedRun.status) && data.checkpoints.length > 0 && <button className="quiet-button" disabled={acting} onClick={() => onAction('evaluate')}><IconRefresh size={16} />{s.evaluate}</button>}<button className="quiet-button" onClick={onNew}>{s.newExperiment}<IconArrowRight size={16} /></button></div>} />
    {(complete || failed) && <div className={`completion-strip ${failed ? 'failed' : ''}`}><div><StatusDot state={failed ? 'error' : 'ready'} /><strong>{failed ? s.interrupted : s.completed}</strong><span>{failed ? selectedRun?.error || selectedRun?.termination_reason || '' : selectedRun?.latest_evaluation ? `PSNR ${formatMetric(selectedRun.latest_evaluation, ['psnr'])} dB · SSIM ${formatMetric(selectedRun.latest_evaluation, ['ssim'], 4)} · LPIPS ${formatMetric(selectedRun.latest_evaluation, ['lpips'], 4)}` : s.evaluationUnavailable}</span></div><button onClick={failed ? () => document.getElementById('run-log-anchor')?.scrollIntoView() : onOpenAnalyze}>{failed ? s.showLogs : s.viewResults}<IconArrowRight size={16} /></button></div>}
    {noRun ? <div className="no-run-state"><h2>{s.noRun}</h2><p>{s.noRunsBody}</p><button className="primary-small" onClick={onNew}>{s.newExperiment}</button></div> : <>
      <div className="live-grid"><div className="live-main"><div className="viewport-panel"><div className="viewport-top"><span><StatusDot state={liveStatus(selectedRun?.status) ? 'live' : 'ready'} />{selectedSnapshot !== null ? s.savedFrame : s.reconstruction}</span><span>{s.iteration} {number(selectedStep ?? iteration)}</span></div><PreviewFrame s={s} image={frame} title={frame?.diagnostic_preview ? s.previewKind : s.evaluatedKind} step={selectedStep ?? iteration} empty={s.previewPending} className="live-viewport" /><div className="viewport-caption"><span>{frame?.diagnostic_preview ? s.previewKind : frame ? s.evaluatedKind : s.previewPendingBody}</span>{selectedSnapshot !== null && <button onClick={() => onSnapshot(null)}>{s.returnLive}<IconArrowRight size={14} /></button>}</div></div><Timeline steps={frames} checkpoints={data.checkpoints} selected={selectedSnapshot} onSelect={onSnapshot} s={s} maxIteration={budget || frames[frames.length - 1] || 1} /></div>
      <aside className="telemetry-stack"><div className="telemetry-head"><span className="eyebrow">{s.telemetry}</span><StatusDot state={liveStatus(selectedRun?.status) ? 'live' : 'ready'} /></div><div className="hero-iteration"><span>{s.iteration}</span><strong>{number(iteration)}</strong><small>/ {number(budget)}</small><div className="progress-signal"><span style={{ width: `${progress}%` }} /></div><em>{progress.toFixed(1)}%</em></div><div className="telemetry-grid"><Stat label={s.loss} value={formatMetric(metric, ['total_loss', 'loss'], 6)} source={s.trainBatch} /><Stat label={s.psnr} value={formatMetric(metric, ['psnr', 'fine_psnr'], 2)} unit="dB" source={s.trainBatch} /><Stat label={s.ssim} value={currentEval ? formatMetric(currentEval, ['ssim'], 3) : s.notEvaluated} source={s.fullView} /><Stat label={s.lpips} value={currentEval ? formatMetric(currentEval, ['lpips'], 3) : s.notEvaluated} source={s.fullView} /><Stat label={s.gpuLoad} value={gpuLoad === null ? s.unavailable : `${number(gpuLoad)}%`} /><Stat label={s.vram} value={bytes(system.gpu?.memory_used_bytes, s.unavailable)} /><Stat label={s.rays} value={formatMetric(metric, ['rays_per_second'], 0)} /><Stat label={s.elapsed} value={clock(elapsed, s.unavailable)} /><Stat label={s.eta} value={clock(eta, s.unavailable)} /></div><div className="telemetry-source">{s.metricSource}</div>{lastEvent && <div className="event-ticker"><span>EVENT</span><strong>{eventNames[lastEvent.type] || lastEvent.type.toUpperCase()}</strong></div>}</aside></div>
      <div className="training-plots"><div className="plot-panel"><div className="plot-heading"><span>{s.lossChart}</span><strong>{formatMetric(metric, ['total_loss', 'loss'], 6)}</strong></div><MetricPlot rows={data.metrics} series={[{ key: 'total_loss', name: s.loss, color: '#a9ff77' }]} s={s} /></div><div className="plot-panel"><div className="plot-heading"><span>{s.psnrChart}</span><strong>{formatMetric(metric, ['psnr', 'fine_psnr'], 2)} dB</strong></div><MetricPlot rows={data.metrics} series={[{ key: 'psnr', name: s.psnr, color: '#78d7b1' }]} s={s} /></div></div>
      {failed && <div className="run-traceback" id="run-log-anchor"><h3>{s.showLogs}</h3><pre>{selectedRun?.traceback || selectedRun?.error || s.unavailable}</pre><div><button className="quiet-button" onClick={() => onAction('resume')} disabled={acting || !data.checkpoints.length}>{s.resume}</button><button className="quiet-button" onClick={onNew}>{s.return}</button></div></div>}
    </>}</div>
}

export function RunsScreen({ s, language, runs, selected, onChoose, loading }: { s: WorkstationCopy; language: Language; runs: Run[]; selected: string; onChoose: (id: string) => void; loading: boolean }) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'running' | 'completed' | 'stopped' | 'failed'>('all')
  const shown = [...runs].sort((a, b) => String(b.timestamp_utc || '').localeCompare(String(a.timestamp_utc || '')))
    .filter(run => (filter === 'all' || (filter === 'running' ? liveStatus(run.status) : run.status === filter)) && `${run.id} ${run.scene}`.toLowerCase().includes(query.toLowerCase()))
  return <div className="runs-screen"><SectionIntro index="03" eyebrow={s.runs} title={s.runArchive} subtitle={s.runArchiveHint} action={<div className="archive-count"><strong>{number(runs.length)}</strong><span>{s.run}</span></div>} />
    <div className="runs-toolbar"><div className="search-control"><IconSearch size={17} /><input aria-label={s.selectRun} placeholder={`${s.run} / ID`} value={query} onChange={event => setQuery(event.target.value)} /></div><div className="filter-control">{(['all', 'running', 'completed', 'stopped', 'failed'] as const).map(value => <button key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{value === 'all' ? language === 'zh' ? '全部' : 'ALL' : statusLabel(value, s)}</button>)}</div></div>
    <div className="run-table-wrap">{loading && !runs.length ? <div className="data-loading">{s.loading}</div> : !shown.length ? <div className="no-run-state"><h2>{query || filter !== 'all' ? s.unavailable : s.noRuns}</h2><p>{s.noRunsBody}</p></div> : <table className="run-table"><thead><tr><th>{s.run}</th><th>{s.scene}</th><th>L</th><th>{s.iterations}</th><th>{s.psnr} · VAL</th><th>{s.status}</th><th>{s.date}</th><th /></tr></thead><tbody>{shown.map(run => <tr key={run.id} className={selected === run.id ? 'selected' : ''} onClick={() => onChoose(run.id)} tabIndex={0} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onChoose(run.id) } }}><td><div className="run-cell-name"><strong>{run.id}</strong>{run.featured_baseline && <small>FEATURED / OBSERVED</small>}</div></td><td>{String(run.scene || '—').toUpperCase()}</td><td>{number(run.position_encoding_L)}</td><td>{number(run.iteration)} / {number(run.effective_target_iteration ?? run.training_iterations)}</td><td>{formatMetric(run.latest_evaluation, ['psnr'], 2, s.notEvaluated)}</td><td><span className="table-status"><StatusDot state={statusTone(run.status)} />{statusLabel(run.status, s)}</span></td><td>{run.timestamp_utc ? new Date(run.timestamp_utc).toLocaleDateString(language === 'zh' ? 'zh-CN' : 'en-US') : '—'}</td><td><IconArrowRight size={17} /></td></tr>)}</tbody></table>}</div>
  </div>
}

export function AnalyzeScreen({ s, data, runs, selected, onChoose, tab, onTab, loading }: {
  s: WorkstationCopy; language: Language; data: RunData; runs: Run[]; selected: string;
  onChoose: (id: string) => void; tab: AnalysisTab; onTab: (tab: AnalysisTab) => void; loading: boolean
}) {
  const [requestedStep, setRequestedStep] = useState<number | null>(null)
  const [series, setSeries] = useState<string[]>(['total_loss', 'psnr'])
  const [encodingL, setEncodingL] = useState(10)
  const [encoding, setEncoding] = useState<Awaited<ReturnType<typeof api.encoding>> | null>(null)
  const [encodingError, setEncodingError] = useState('')
  useEffect(() => { setRequestedStep(null); setEncodingL(data.detail?.position_encoding_L ?? 10) }, [selected, data.detail?.position_encoding_L])
  useEffect(() => {
    if (tab !== 'encoding') return
    let active = true
    setEncoding(null); setEncodingError('')
    api.encoding(encodingL).then(value => { if (active) setEncoding(value) }).catch(cause => { if (active) setEncodingError(String(cause)) })
    return () => { active = false }
  }, [tab, encodingL])
  const frames = availableFrames(data.reconstructions, 'val', 0)
  const step = requestedStep !== null && frames.includes(requestedStep) ? requestedStep : frames[frames.length - 1]
  const prediction = typeof step === 'number' ? frameAt(data.reconstructions, step, 'prediction', 'val', 0) : undefined
  const truth = typeof step === 'number' ? frameAt(data.reconstructions, step, 'ground_truth', 'val', 0) : undefined
  const difference = typeof step === 'number' ? frameAt(data.reconstructions, step, 'absolute_difference', 'val', 0) : undefined
  const evalRow = typeof step === 'number' ? evaluationAt(data, step) : undefined
  const run = data.detail || runs.find(item => item.id === selected)
  const rows = useMemo(() => mergePlotRows(data.metrics, data.evaluationSummaries), [data.metrics, data.evaluationSummaries])
  const definitions = [
    { key: 'total_loss', name: s.loss, color: '#a9ff77' },
    { key: 'psnr', name: `${s.psnr} · ${s.trainBatch}`, color: '#7bd9b2' },
    { key: 'validation_psnr', name: `${s.psnr} · ${s.fullView}`, color: '#b8c9a4' },
    { key: 'ssim', name: s.ssim, color: '#8cb6a6' },
    { key: 'lpips', name: s.lpips, color: '#c0a686' },
  ]
  const availableSeries = definitions.filter(def => rows.some(row => typeof row[def.key] === 'number'))
  const tabs: [AnalysisTab, string][] = [['reconstruction', s.reconstructionTab], ['compare', s.compareTab], ['metrics', s.metricsTab], ['encoding', s.encodingTab], ['sampling', s.samplingTab]]
  return <div className="analyze-screen"><SectionIntro index="04" eyebrow={s.analyze} title={s.analyzeTitle} subtitle={s.analyzeSubtitle} action={<div className="analysis-run"><span>{s.selectRun}</span><div className="analysis-run-options">{runs.map(item => <button key={item.id} className={selected === item.id ? 'active' : ''} title={item.id} onClick={() => onChoose(item.id)}>{item.featured_baseline ? 'LEGO / 50K' : item.id.length > 20 ? `${item.id.slice(0, 16)}…` : item.id}</button>)}</div></div>} />
    <div className="view-tabs" role="tablist" aria-label={s.analyze}>{tabs.map(([id, label]) => <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'active' : ''} onClick={() => onTab(id)}>{label}</button>)}</div>
    {!selected ? <div className="no-run-state"><h2>{s.selectRun}</h2><p>{s.noRunsBody}</p></div> : loading && !data.detail ? <div className="data-loading">{s.loading}</div> : <>
      {(tab === 'reconstruction' || tab === 'compare') && <>
        <div className="analysis-frame-nav"><div><span>{s.timeMachine}</span><strong>{typeof step === 'number' ? `${s.iteration} ${number(step)}` : s.noTimeline}</strong></div><div className="frame-quick">{[500, 2000, 5000, 10000, 15000, 30000, 50000].filter(value => frames.includes(value)).map(value => <button key={value} className={step === value ? 'active' : ''} onClick={() => setRequestedStep(value)}>{value >= 1000 ? `${value / 1000}K` : value}</button>)}</div></div>
        {tab === 'reconstruction' ? <div className="analysis-reconstruction"><PreviewFrame s={s} image={prediction} title={s.prediction} step={step} empty={s.imageUnavailable} className="analysis-primary-frame" /><aside className="analysis-inspector"><div className="inspector-title"><span>{s.run}</span><strong title={selected}>{selected}</strong></div><PreviewFrame s={s} image={truth} title={s.groundTruth} step={step} empty={s.imageUnavailable} className="inspector-preview" /><div className="inspector-stats"><Stat label={`${s.psnr} · VAL`} value={evalRow ? formatMetric(evalRow, ['psnr'], 2) : s.notEvaluated} unit={evalRow ? 'dB' : undefined} /><Stat label={s.ssim} value={evalRow ? formatMetric(evalRow, ['ssim'], 3) : s.notEvaluated} /><Stat label={s.lpips} value={evalRow ? formatMetric(evalRow, ['lpips'], 3) : s.notEvaluated} /></div><p>{evalRow ? s.fullView : s.evaluationUnavailable}</p></aside></div> : <><p className="comparison-note">{s.compareNote}</p><div className="comparison-grid"><PreviewFrame s={s} image={truth} title={s.groundTruth} step={step} empty={s.imageUnavailable} /><PreviewFrame s={s} image={prediction} title={s.prediction} step={step} empty={s.imageUnavailable} /><PreviewFrame s={s} image={difference} title={s.difference} step={step} empty={s.imageUnavailable} /></div><div className="comparison-metrics"><Stat label="PSNR · VAL" value={evalRow ? formatMetric(evalRow, ['psnr'], 2) : s.notEvaluated} unit={evalRow ? 'dB' : undefined} /><Stat label={s.ssim} value={evalRow ? formatMetric(evalRow, ['ssim'], 3) : s.notEvaluated} /><Stat label={s.lpips} value={evalRow ? formatMetric(evalRow, ['lpips'], 3) : s.notEvaluated} /></div></>}
        <Timeline steps={frames} checkpoints={data.checkpoints} selected={requestedStep} onSelect={setRequestedStep} s={s} maxIteration={run?.effective_target_iteration ?? run?.training_iterations ?? frames[frames.length - 1] ?? 1} />
      </>}
      {tab === 'metrics' && <div className="metrics-lab"><div className="metrics-lab-head"><div><span className="eyebrow">{s.metricsView}</span><h2>{s.metrics}</h2><p>{s.metricSource}</p></div><div className="metric-series">{availableSeries.map(def => <button key={def.key} className={series.includes(def.key) ? 'active' : ''} onClick={() => setSeries(current => current.includes(def.key) ? current.length > 1 ? current.filter(key => key !== def.key) : current : [...current, def.key])}><i style={{ background: def.color }} />{def.name}</button>)}</div></div><MetricPlot rows={rows} series={definitions.filter(def => series.includes(def.key))} height={440} expanded s={s} /><div className="metrics-summary"><Stat label={s.loss} value={formatMetric(data.metrics[data.metrics.length - 1], ['total_loss'], 6)} source={s.trainBatch} /><Stat label={s.psnr} value={formatMetric(data.metrics[data.metrics.length - 1], ['psnr'], 2)} unit="dB" source={s.trainBatch} /><Stat label={`${s.psnr} · VAL`} value={formatMetric(run?.latest_evaluation, ['psnr'], 2, s.notEvaluated)} source={s.fullView} /></div></div>}
      {tab === 'encoding' && <div className="encoding-lab"><div className="encoding-left"><div className="encoding-symbol">γ<sub>L</sub>(x)</div><span className="eyebrow">{s.encodingTitle}</span><h2>{s.encodingNote}</h2><div className="encoding-stepper"><button onClick={() => setEncodingL(value => Math.max(0, value - 1))} disabled={encodingL === 0}>−</button><strong>L = {encodingL}</strong><button onClick={() => setEncodingL(value => Math.min(15, value + 1))} disabled={encodingL === 15}>+</button></div><div className="encoding-facts"><Stat label={s.encoding} value={encoding ? `${encoding.encoded_dim}D` : `${encodedDimension(encodingL)}D`} /><Stat label={s.maxBand} value={encodingL ? `2^${encodingL - 1}` : '—'} /><Stat label={s.parameters} value={number(modelParameterCount({ ...(run?.config || {}), position_encoding_L: encodingL }))} source={s.estimate} /></div></div><div className="encoding-right"><div className="panel-kicker"><span>{s.spectralPreview}</span><span>sin / cos</span></div>{encodingError && <p className="field-error">{encodingError}</p>}{encoding ? <div className="spectrum-list">{encoding.curves.map(curve => <div className="spectrum-row" key={curve.k}><span>2<sup>{curve.k}</sup></span><svg viewBox="0 0 300 54" preserveAspectRatio="none" aria-label={`Frequency band ${curve.k}`}><line x1="0" x2="300" y1="27" y2="27" stroke="#244a33" /><polyline points={curve.sin.map((value, index) => `${index / (curve.sin.length - 1) * 300},${27 - value * 20}`).join(' ')} fill="none" stroke="#a9ff77" strokeWidth="1.5" /><polyline points={curve.cos.map((value, index) => `${index / (curve.cos.length - 1) * 300},${27 - value * 20}`).join(' ')} fill="none" stroke="#5bbda9" strokeWidth="1" /></svg><small>k {String(curve.k).padStart(2, '0')}</small></div>)}</div> : <div className="data-loading">{s.loading}</div>}</div></div>}
      {tab === 'sampling' && <div className="sampling-lab"><div className="panel-kicker"><span>{s.samplingTitle}</span><span>{s.samplingNote}</span></div>{data.sampling?.available ? <div className="sampling-figures">{(data.sampling.static_figures || data.sampling.items || []).filter(item => item.url).map((item, index) => <figure key={`${item.url}-${index}`}><img src={api.asset(item.url)} alt={String(item.kind || item.name || s.samplingTitle)} /><figcaption>{String(item.name || item.kind || s.samplingTitle)}</figcaption></figure>)}</div> : <div className="no-run-state"><h2>{s.unavailable}</h2><p>{data.sampling?.reason || s.samplingNote}</p></div>}</div>}
    </>}
  </div>
}

function logLines(raw: unknown): string[] {
  if (Array.isArray(raw)) return raw.map(item => typeof item === 'string' ? item : JSON.stringify(item))
  if (raw && typeof raw === 'object') {
    const value = raw as { items?: unknown[]; lines?: unknown[] }
    return logLines(value.items || value.lines || [])
  }
  return typeof raw === 'string' ? raw.split('\n') : []
}

export function SystemScreen({ s, language, data, system, connected, tab, onTab, playBoot, onPlayBoot, onRefresh }: {
  s: WorkstationCopy; language: Language; data: RunData; system: System; connected: 'checking' | 'online' | 'offline';
  tab: SystemTab; onTab: (tab: SystemTab) => void; playBoot: boolean; onPlayBoot: (value: boolean) => void; onRefresh: () => void
}) {
  const [samples, setSamples] = useState<number[]>([])
  const load = readNumber(system.gpu, 'utilization_percent')
  useEffect(() => { if (load !== null) setSamples(current => [...current.slice(-29), load]) }, [load])
  const used = readNumber(system.gpu, 'memory_used_bytes')
  const total = readNumber(system.gpu, 'memory_total_bytes')
  const lines = logLines(data.logs)
  const pointString = samples.map((value, index) => `${index / Math.max(samples.length - 1, 1) * 100},${100 - value}`).join(' ')
  return <div className="system-screen"><SectionIntro index="05" eyebrow={s.system} title={s.systemTitle} subtitle={s.systemSubtitle} action={<button className="quiet-button" onClick={onRefresh}><IconRefresh size={16} />{s.reset}</button>} /><div className="view-tabs" role="tablist" aria-label={s.system}>{([['machine', s.machine], ['logs', s.logs], ['settings', s.settings]] as [SystemTab, string][]).map(([id, label]) => <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'active' : ''} onClick={() => onTab(id)}>{label}</button>)}</div>
    {tab === 'machine' && <div className="machine-grid"><section className="machine-hero instrument-panel"><div className="panel-kicker"><span>{s.gpuName}</span><StatusDot state={system.gpu?.name ? 'ready' : 'error'} /></div><div className="machine-gpu-symbol"><IconCpu size={74} stroke={.8} /><div className="chip-lines" /></div><h2>{typeof system.gpu?.name === 'string' ? system.gpu.name : s.gpuUnavailable}</h2><p>{connected === 'online' ? s.online : s.offline} / {system.backend?.cuda_runtime ? `CUDA ${system.backend.cuda_runtime}` : s.cpuMode}</p><div className="machine-big-numbers"><Stat label={s.utilization} value={load === null ? s.unavailable : number(load)} unit={load === null ? undefined : '%'} large /><Stat label={s.vram} value={used !== null ? bytes(used) : s.unavailable} source={total !== null ? `/ ${bytes(total)}` : undefined} large /></div></section><section className="machine-activity instrument-panel"><div className="panel-kicker"><span>{s.activity}</span><span>GPU / LIVE</span></div><div className="sparkline"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label={s.activity}><line x1="0" y1="100" x2="100" y2="100" stroke="#44644e" /><line x1="0" y1="50" x2="100" y2="50" stroke="#284334" strokeDasharray="1 5" />{samples.length > 1 && <polyline points={pointString} fill="none" stroke="#a9ff77" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}</svg>{!samples.length && <span>{s.unavailable}</span>}</div><div className="machine-sensors"><Stat label={s.temperature} value={number(readNumber(system.gpu, 'temperature_c'), 0, s.unavailable)} unit={readNumber(system.gpu, 'temperature_c') !== null ? '°C' : undefined} /><Stat label={s.power} value={number(readNumber(system.gpu, 'power_w'), 0, s.unavailable)} unit={readNumber(system.gpu, 'power_w') !== null ? 'W' : undefined} /><Stat label={s.cpu} value={number(readNumber(system.cpu, 'utilization_percent'), 0, s.unavailable)} unit={readNumber(system.cpu, 'utilization_percent') !== null ? '%' : undefined} /><Stat label={s.ram} value={bytes(readNumber(system.ram, 'used_bytes'), s.unavailable)} /></div></section><section className="machine-runtime instrument-panel"><div className="panel-kicker"><span>{s.runtime}</span><span>LOCAL / PYTHON</span></div><div className="runtime-rows"><div><span>{s.backend}</span><strong>{connected === 'online' ? s.online : s.offline}</strong></div><div><span>PYTHON</span><strong>{String(system.backend?.python || s.unavailable)}</strong></div><div><span>{s.pytorch}</span><strong>{String(system.backend?.pytorch || s.unavailable)}</strong></div><div><span>{s.cuda}</span><strong>{String(system.backend?.cuda_runtime || s.unavailable)}</strong></div><div><span>{s.cpu}</span><strong>{String(system.cpu?.name || s.unavailable)}</strong></div></div></section></div>}
    {tab === 'logs' && <div className="logs-panel instrument-panel"><div className="panel-kicker"><span>{s.runLogs}</span><span>{number(lines.length)} LINES</span></div>{lines.length ? <div className="log-lines" role="log">{lines.map((line, index) => <div key={index}><span>{String(index + 1).padStart(4, '0')}</span><code>{line}</code></div>)}</div> : <div className="no-run-state"><h2>{s.noLogs}</h2></div>}{data.detail?.traceback && <pre className="traceback">{data.detail.traceback}</pre>}</div>}
    {tab === 'settings' && <div className="settings-grid"><section className="instrument-panel setting-card"><div className="panel-kicker"><span>{s.playBoot}</span><span>STARTUP / MOTION</span></div><h2>{s.playBoot}</h2><p>{s.bootPreference}</p><div className="setting-choice"><button className={playBoot ? 'active' : ''} onClick={() => onPlayBoot(true)}>{s.playBoot}</button><button className={!playBoot ? 'active' : ''} onClick={() => onPlayBoot(false)}>{s.skipBoot}</button></div></section><section className="instrument-panel setting-card"><div className="panel-kicker"><span>{s.language}</span><span>DISPLAY / LOCAL</span></div><h2>{s.language}</h2><p>{language === 'zh' ? '可在窗口顶栏随时切换中文与英文。' : 'Switch between Chinese and English in the title bar at any time.'}</p><div className="setting-readout">{language === 'zh' ? '简体中文' : 'ENGLISH'}</div></section></div>}
  </div>
}

export function PresentationScreen({ s, data, selectedRun, reference, system, selectedSnapshot, onSnapshot, onExit }: {
  s: WorkstationCopy; data: RunData; selectedRun: Run | null; reference?: Item; system: System;
  selectedSnapshot: number | null; onSnapshot: (step: number | null) => void; onExit: () => void
}) {
  const frames = availableFrames(data.reconstructions, 'val', 0)
  const step = selectedSnapshot !== null && frames.includes(selectedSnapshot) ? selectedSnapshot : frames[frames.length - 1]
  const prediction = typeof step === 'number' ? frameAt(data.reconstructions, step, 'prediction', 'val', 0) : undefined
  const truth = typeof step === 'number' ? frameAt(data.reconstructions, step, 'ground_truth', 'val', 0) || reference : reference
  const evalRow = typeof step === 'number' ? evaluationAt(data, step) : undefined
  const metric = data.metrics[data.metrics.length - 1]
  const gpuLoad = readNumber(system.gpu, 'utilization_percent')
  return <div className="presentation-screen"><header><div><span className="presentation-mark">N∴R</span><div><small>{s.presentation}</small><strong>NeRF RESEARCH CONSOLE</strong></div></div><div className="presentation-state"><StatusDot state={liveStatus(selectedRun?.status) ? 'live' : 'ready'} />{liveStatus(selectedRun?.status) ? s.live : statusLabel(selectedRun?.status, s)}</div><button onClick={onExit} title="Esc / F11">{s.exitPresentation}<IconX size={18} /></button></header><div className="presentation-body"><div className="presentation-frames"><PreviewFrame s={s} image={prediction} title={s.prediction} step={step} empty={s.imageUnavailable} className="presentation-main-frame" /><PreviewFrame s={s} image={truth} title={s.groundTruth} step={step} empty={s.imageUnavailable} className="presentation-truth-frame" /></div><aside><div className="presentation-sequence">{s.presentationHint}</div><div className="presentation-big"><span>{s.scene} / LEGO</span><strong>L = {number(selectedRun?.position_encoding_L)}</strong><small>{s.encoding} {typeof selectedRun?.position_encoding_L === 'number' ? encodedDimension(selectedRun.position_encoding_L) : '—'}D</small></div><div className="presentation-big"><span>{s.iteration}</span><strong>{number(step ?? selectedRun?.iteration)}</strong><small>/ {number(selectedRun?.effective_target_iteration ?? selectedRun?.training_iterations)}</small></div><div className="presentation-metrics"><Stat label={`${s.psnr} · ${s.fullView}`} value={evalRow ? formatMetric(evalRow, ['psnr'], 2) : s.notEvaluated} unit={evalRow ? 'dB' : undefined} /><Stat label={`${s.psnr} · ${s.trainBatch}`} value={formatMetric(metric, ['psnr'], 2, s.unavailable)} unit="dB" /><Stat label={s.gpuLoad} value={gpuLoad === null ? s.unavailable : `${number(gpuLoad)}%`} /></div></aside></div><Timeline steps={frames} checkpoints={data.checkpoints} selected={selectedSnapshot} onSelect={onSnapshot} s={s} maxIteration={selectedRun?.effective_target_iteration ?? selectedRun?.training_iterations ?? frames[frames.length - 1] ?? 1} /><footer><span>SCENE → ENCODE → TRAIN → RENDER</span><span>{s.metricSource}</span></footer></div>
}
