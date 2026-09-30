import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { IconActivity, IconArrowsMaximize, IconChartDots3, IconCpu, IconFlask2, IconMinus, IconRectangle, IconX } from '@tabler/icons-react'
import { api, readNumber, type Dict, type Item, type LiveEvent, type Metric, type Run, type System } from './api'
import { useI18n } from './i18n'
import MatrixRain from './MatrixRain'
import { ambientTier, chooseRunSelection, createCommandGate } from './workstation-model'
import { copy } from './workstation-copy'
import { Boot, StatusDot, number } from './WorkstationWidgets'
import { AnalyzeScreen, PresentationScreen, RunsScreen, SystemScreen, TrainScreen } from './WorkstationScreens'

declare global {
  interface Window {
    nerfDesktop?: {
      minimize: () => void; maximize: () => void; close: () => void;
      setFullscreen: (value: boolean) => Promise<void>; isMaximized: () => Promise<boolean>;
      onMaximized: (callback: (value: boolean) => void) => () => void;
    }
  }
}

export type Screen = 'train' | 'runs' | 'analyze' | 'system'
export type AnalysisTab = 'reconstruction' | 'compare' | 'metrics' | 'encoding' | 'sampling'
export type SystemTab = 'machine' | 'logs' | 'settings'
export type RunData = {
  detail: Run | null; metrics: Metric[]; checkpoints: Item[]; reconstructions: Item[];
  evaluations: Item[]; evaluationSummaries: Metric[];
  sampling: { available: boolean; reason?: string; items?: Item[]; static_figures?: Item[] } | null;
  logs: unknown
}
const emptyRun: RunData = { detail: null, metrics: [], checkpoints: [], reconstructions: [], evaluations: [], evaluationSummaries: [], sampling: null, logs: null }
const isLive = (status?: string) => status === 'running' || status === 'stopping' || status === 'training'

export default function Workstation() {
  const { language, setLanguage } = useI18n()
  const s = copy[language]
  const [screen, setScreen] = useState<Screen>('train')
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>('reconstruction')
  const [systemTab, setSystemTab] = useState<SystemTab>('machine')
  const [trainView, setTrainView] = useState<'setup' | 'live'>('setup')
  const [connected, setConnected] = useState<'checking' | 'online' | 'offline'>('checking')
  const [system, setSystem] = useState<System>({})
  const [runs, setRuns] = useState<Run[]>([])
  const [scenes, setScenes] = useState<Dict[]>([])
  const [defaults, setDefaults] = useState<Dict | null>(null)
  const [selected, setSelected] = useState('')
  const [data, setData] = useState<RunData>(emptyRun)
  const [reference, setReference] = useState<Item | undefined>()
  const [selectedSnapshot, setSelectedSnapshot] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [runLoading, setRunLoading] = useState(false)
  const [error, setError] = useState('')
  const [acting, setActing] = useState(false)
  const [launchPhase, setLaunchPhase] = useState<'idle' | 'lock' | 'request' | 'wait' | 'done'>('idle')
  const [lastEvent, setLastEvent] = useState<LiveEvent | null>(null)
  const [presentation, setPresentation] = useState(false)
  const [maximized, setMaximized] = useState(false)
  const [playBoot, setPlayBoot] = useState(() => localStorage.getItem('nerf-play-boot') === 'true')
  const [boot, setBoot] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches &&
    (localStorage.getItem('nerf-play-boot') === 'true' || localStorage.getItem('nerf-boot-seen') !== 'true'))
  const doneBoot = useCallback(() => { localStorage.setItem('nerf-boot-seen', 'true'); setBoot(false) }, [])
  const selectedRef = useRef(selected)
  const loadRef = useRef(0)
  const connectionErrorRef = useRef('')
  const launchGate = useRef(createCommandGate())
  const actionGate = useRef(createCommandGate())
  const launchResetTimer = useRef(0)
  selectedRef.current = selected

  const refreshGlobal = useCallback(async () => {
    try {
      const health = await api.health()
      if (health.status !== 'ok') throw new Error('Backend not ready')
      const [nextRuns, nextSystem, nextDefaults, nextScenes] = await Promise.all([api.runs(), api.system(), api.defaults(), api.scenes()])
      setConnected('online')
      setRuns(nextRuns)
      setSystem(nextSystem)
      setDefaults(nextDefaults.config)
      setScenes(nextScenes)
      const choice = chooseRunSelection(nextRuns, selectedRef.current)
      if (choice.id !== selectedRef.current) setSelected(choice.id)
      if (choice.openLive) setTrainView('live')
      if (connectionErrorRef.current) {
        const previous = connectionErrorRef.current
        setError(current => current === previous ? '' : current)
        connectionErrorRef.current = ''
      }
    } catch (cause) {
      setConnected('offline')
      connectionErrorRef.current = cause instanceof Error ? cause.message : String(cause)
      setError(connectionErrorRef.current)
    } finally { setLoading(false) }
  }, [])

  const refreshRun = useCallback(async (id: string) => {
    if (!id) return
    const seq = ++loadRef.current
    setRunLoading(true)
    const results = await Promise.allSettled([
      api.run(id), api.metrics(id), api.checkpoints(id), api.reconstructions(id),
      api.evaluations(id), api.sampling(id), api.logs(id),
    ])
    if (seq !== loadRef.current || id !== selectedRef.current) return
    const value = <T,>(index: number, fallback: T): T => results[index].status === 'fulfilled'
      ? (results[index] as PromiseFulfilledResult<unknown>).value as T : fallback
    const evaluation = value<{ per_view?: Item[]; summary?: Metric[] }>(4, {})
    setData(current => ({ detail: value(0, current.detail), metrics: value(1, current.metrics),
      checkpoints: value(2, current.checkpoints), reconstructions: value(3, current.reconstructions),
      evaluations: evaluation.per_view || current.evaluations, evaluationSummaries: evaluation.summary || current.evaluationSummaries,
      sampling: value(5, current.sampling), logs: value(6, current.logs) }))
    setRunLoading(false)
    const failure = results.find(result => result.status === 'rejected')
    if (failure && failure.status === 'rejected') setError(String(failure.reason))
  }, [])

  useEffect(() => { refreshGlobal(); const timer = window.setInterval(refreshGlobal, 8000); return () => clearInterval(timer) }, [refreshGlobal])
  useEffect(() => { loadRef.current++; setData(emptyRun); setSelectedSnapshot(null); if (selected) refreshRun(selected) }, [selected, refreshRun])
  useEffect(() => () => clearTimeout(launchResetTimer.current), [])
  const live = isLive(data.detail?.status)
  useEffect(() => {
    if (!selected || !live) return
    const timer = window.setInterval(() => refreshRun(selected), 12000)
    let refreshTimer = 0
    const ws = api.ws(selected)
    ws.onmessage = event => {
      try {
        const message = JSON.parse(event.data) as LiveEvent
        setLastEvent(message)
        if (['iteration_update', 'metric_update', 'checkpoint_saved', 'preview_ready', 'training_completed', 'training_failed', 'evaluation_completed'].includes(message.type)) {
          clearTimeout(refreshTimer)
          refreshTimer = window.setTimeout(() => { refreshRun(selected); refreshGlobal() }, 500)
        }
      } catch { /* A malformed event cannot alter scientific values. */ }
    }
    return () => { clearInterval(timer); clearTimeout(refreshTimer); ws.close() }
  }, [selected, live, refreshRun, refreshGlobal])
  useEffect(() => {
    const featured = runs.find(run => run.featured_baseline)?.id
    if (!featured) return
    let current = true
    api.reconstructions(featured).then(items => {
      if (!current) return
      const frames = items.filter(item => item.kind === 'ground_truth' && item.split === 'val' && item.view_index === 0 && item.url)
      setReference(frames.sort((a, b) => (b.iteration || 0) - (a.iteration || 0))[0])
    }).catch(() => {})
    return () => { current = false }
  }, [runs.find(run => run.featured_baseline)?.id])
  useEffect(() => {
    let active = true
    window.nerfDesktop?.isMaximized().then(value => { if (active) setMaximized(value) })
    const dispose = window.nerfDesktop?.onMaximized(setMaximized)
    return () => { active = false; dispose?.() }
  }, [])

  const chooseRun = (id: string) => { setSelected(id); setScreen('analyze'); setAnalysisTab('reconstruction') }
  const launch = async (config: Dict) => {
    if (launchPhase !== 'idle' || !launchGate.current.enter()) return
    setError(''); setLaunchPhase('lock')
    const phase1 = window.setTimeout(() => setLaunchPhase('request'), 260)
    const phase2 = window.setTimeout(() => setLaunchPhase('wait'), 640)
    try {
      const run = await api.create(config)
      setLaunchPhase('done')
      setSelected(run.id); setTrainView('live'); setScreen('train')
      await refreshGlobal()
      launchResetTimer.current = window.setTimeout(() => setLaunchPhase('idle'), 550)
    } catch (cause) {
      setLaunchPhase('idle'); setError(cause instanceof Error ? cause.message : String(cause))
    } finally { clearTimeout(phase1); clearTimeout(phase2); launchGate.current.leave() }
  }
  const runAction = async (name: 'stop' | 'resume' | 'evaluate') => {
    if (!selected || acting || !actionGate.current.enter()) return
    setActing(true); setError('')
    try { await api.action(selected, name); await Promise.all([refreshGlobal(), refreshRun(selected)]) }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
    finally { setActing(false); actionGate.current.leave() }
  }
  const togglePresentation = useCallback(async () => {
    const next = !presentation
    try {
      if (window.nerfDesktop) await window.nerfDesktop.setFullscreen(next)
      else if (next) await document.documentElement.requestFullscreen()
      else if (document.fullscreenElement) await document.exitFullscreen()
      setPresentation(next)
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
  }, [presentation])
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'F11') { event.preventDefault(); togglePresentation(); return }
      if (event.key === 'Escape' && presentation) { event.preventDefault(); togglePresentation(); return }
      if (event.ctrlKey && /^[1-4]$/.test(event.key) && !(event.target instanceof HTMLInputElement)) {
        event.preventDefault(); setScreen((['train', 'runs', 'analyze', 'system'] as Screen[])[Number(event.key) - 1])
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [presentation, togglePresentation])

  const selectedRun = data.detail || runs.find(run => run.id === selected) || null
  const chromeRun = runs.find(run => isLive(run.status)) || (screen === 'train' && trainView === 'live' ? selectedRun : null)
  const currentMetric = data.metrics[data.metrics.length - 1]
  const iteration = selectedRun?.iteration ?? readNumber(currentMetric, 'iteration') ?? 0
  const gpuLoad = readNumber(system.gpu, 'utilization_percent')
  const tier = ambientTier(gpuLoad, live || presentation, window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const nav = useMemo(() => [
    { id: 'train' as Screen, label: s.train, icon: IconFlask2 },
    { id: 'runs' as Screen, label: s.runs, icon: IconActivity },
    { id: 'analyze' as Screen, label: s.analyze, icon: IconChartDots3 },
    { id: 'system' as Screen, label: s.system, icon: IconCpu },
  ], [s])

  return <><MatrixRain active={live} tier={tier} />
    <div className={`workstation${presentation ? ' presentation-active' : ''}`}>
      {!presentation && <>
        <header className="window-chrome">
          <div className="chrome-drag"><div className="chrome-mark"><span>N</span><i /></div><div className="chrome-title"><strong>NERF<span> / RESEARCH CONSOLE</span></strong><small>{s.workstation}</small></div><span className="chrome-divider" /><span className={`chrome-status ${isLive(chromeRun?.status) ? 'live' : ''}`}><StatusDot state={isLive(chromeRun?.status) ? 'live' : chromeRun?.status === 'completed' ? 'ready' : 'idle'} />{isLive(chromeRun?.status) ? s.live : chromeRun?.status === 'completed' ? s.completed : s.noRun}</span></div>
          <div className="chrome-telemetry"><span>{s.gpu} <b>{gpuLoad === null ? '—' : `${number(gpuLoad)}%`}</b></span><span>{s.backend} <b className={connected === 'online' ? 'online' : 'offline'}>{connected === 'online' ? s.online : connected === 'checking' ? s.connecting : s.offline}</b></span></div>
          <button className="chrome-presentation" onClick={togglePresentation} title="F11"><IconArrowsMaximize size={15} />{s.presentation}</button>
          <div className="language-control" role="group" aria-label={s.language}><button onClick={() => setLanguage('zh')} className={language === 'zh' ? 'active' : ''} aria-pressed={language === 'zh'}>中</button><button onClick={() => setLanguage('en')} className={language === 'en' ? 'active' : ''} aria-pressed={language === 'en'}>EN</button></div>
          {window.nerfDesktop && <div className="window-controls"><button title={s.minimize} aria-label={s.minimize} onClick={() => window.nerfDesktop?.minimize()}><IconMinus size={16} /></button><button title={maximized ? s.restore : s.maximize} aria-label={maximized ? s.restore : s.maximize} onClick={() => window.nerfDesktop?.maximize()}><IconRectangle size={15} /></button><button className="close" title={s.close} aria-label={s.close} onClick={() => window.nerfDesktop?.close()}><IconX size={16} /></button></div>}
        </header>
        <div className="workstation-core"><aside className="nav-rail"><div className="nav-rail-top"><span>NR / 01</span></div><nav aria-label="Workstation">{nav.map(({ id, label, icon: Icon }, index) => <button key={id} className={screen === id ? 'active' : ''} onClick={() => setScreen(id)} title={`${label} · Ctrl+${index + 1}`}><Icon size={21} stroke={1.5} /><span>{label}</span></button>)}</nav><div className="nav-rail-bottom"><span className="rail-line" /><small>NERF<br />LAB</small></div></aside>
          <main className="screen-surface">
            {error && <div className="global-error" role="alert"><StatusDot state="error" /><span>{s.actionFailed}: {error}</span><button onClick={() => { setError(''); refreshGlobal(); if (selected) refreshRun(selected) }}>{s.retry}</button><button aria-label={s.close} onClick={() => setError('')}><IconX size={16} /></button></div>}
            {screen === 'train' && <TrainScreen s={s} language={language} defaults={defaults} scenes={scenes} system={system} connected={connected === 'online'} reference={reference} data={data} selectedRun={selectedRun} showLive={trainView === 'live'} onNew={() => setTrainView('setup')} onLaunch={launch} onAction={runAction} onOpenAnalyze={() => { setScreen('analyze'); setAnalysisTab('reconstruction') }} acting={acting} launchPhase={launchPhase} selectedSnapshot={selectedSnapshot} onSnapshot={setSelectedSnapshot} lastEvent={lastEvent} loading={loading || runLoading} />}
            {screen === 'runs' && <RunsScreen s={s} language={language} runs={runs} selected={selected} onChoose={chooseRun} loading={loading} />}
            {screen === 'analyze' && <AnalyzeScreen s={s} language={language} data={data} runs={runs} selected={selected} onChoose={setSelected} tab={analysisTab} onTab={setAnalysisTab} loading={runLoading} />}
            {screen === 'system' && <SystemScreen s={s} language={language} data={data} system={system} connected={connected} tab={systemTab} onTab={setSystemTab} playBoot={playBoot} onPlayBoot={value => { setPlayBoot(value); localStorage.setItem('nerf-play-boot', String(value)) }} onRefresh={refreshGlobal} />}
          </main>
        </div>
        <footer className="workstation-footer"><span><StatusDot state={connected === 'online' ? 'ready' : 'error'} />{connected === 'online' ? s.systemReady : s.systemOffline}</span><span>{selected ? `RUN / ${selected}` : s.noRun}</span><span>{s.iteration} / {selected ? number(iteration) : '—'}</span><span>LOCAL · VANILLA NeRF</span></footer>
      </>}
      {presentation && <PresentationScreen s={s} data={data} selectedRun={selectedRun} reference={reference} system={system} selectedSnapshot={selectedSnapshot} onSnapshot={setSelectedSnapshot} onExit={togglePresentation} />}
    </div>
    {boot && <Boot onDone={doneBoot} s={s} gpuName={system.gpu?.name} online={connected === 'online'} />}
  </>
}
