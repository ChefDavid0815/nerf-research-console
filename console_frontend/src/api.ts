import { getLanguage, t } from './i18n'

export type Dict = Record<string, unknown>
export type Metric = Record<string, number | string | null>
export type Run = {
  id: string; scene?: string; status?: string; iteration?: number; training_iterations?: number;
  effective_target_iteration?: number; parameter_count?: number; duration_seconds?: number | null;
  position_encoding_L?: number; direction_encoding_L?: number; seed?: number;
  latest_metrics?: Metric; latest_evaluation?: Dict; timestamp_utc?: string; imported?: boolean; featured_baseline?: boolean;
  config?: Dict; manifest?: Dict; environment?: Dict; availability?: Dict;
  termination_reason?: string; error?: string; traceback?: string
}
export type Item = { iteration?: number; kind?: string; url?: string; split?: string; view_index?: number; metrics?: Dict; size_bytes?: number; name?: string; path?: string; [key: string]: unknown }
export type System = { backend?: Dict; gpu?: Dict; cpu?: Dict; ram?: Dict }
export type LiveEvent = { type: string; run_id?: string; data?: Dict; timestamp?: string; [key: string]: unknown }
export type RayStage = { depths: number[]; density: number[]; alpha: number[]; weights: number[]; transmittance: number[]; rgb: number[]; depth_map: number; accumulated_opacity: number; sampled_depths?: number[] }
export type RayDiagnostic = {
  available: true; run_id: string; checkpoint: number; split: string; view_index: number;
  pixel: { x: number; y: number }; image_size: { width: number; height: number };
  ray: { origin: number[]; direction: number[]; direction_normalized: boolean };
  coarse_depths: number[]; fine_depths: number[]; combined_depths: number[];
  coarse: RayStage; fine: RayStage; source: Dict;
}

const BASE = import.meta.env.VITE_API_BASE || ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}/api${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  let payload: unknown
  try { payload = await response.json() } catch { payload = null }
  if (!response.ok) {
    const detail = (payload && typeof payload === 'object' && ('detail' in payload)) ? (payload as {detail: unknown}).detail : response.statusText
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return payload as T
}

const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`
const items = <T>(value: T[] | { items?: T[] } | null | undefined): T[] => Array.isArray(value) ? value : value?.items || []

export const api = {
  health: () => request<Dict>('/health'),
  system: () => request<System>('/system'),
  scenes: () => request<Dict[] | { items?: Dict[] }>('/scenes').then(items),
  cameras: (scene='lego',pixel?:{x:number;y:number}) => request<Dict>(`/scenes/${encodeURIComponent(scene)}/cameras?limit=500${pixel?`&pixel_x=${pixel.x}&pixel_y=${pixel.y}`:''}`),
  encoding: (L:number) => request<{L:number;input_dim:number;encoded_dim:number;include_input:boolean;highest_frequency_band:number|null;x:number[];curves:{k:number;frequency_multiplier:number;sin:number[];cos:number[]}[]}>(`/positional-encoding/${L}`),
  defaults: () => request<{config: Dict; validated_baseline?: boolean; position_encoding?: Dict}>('/config/defaults'),
  runs: () => request<Run[] | {items?: Run[]}>('/runs').then(items),
  run: (id: string) => request<Run>(runPath(id)),
  metrics: (id: string) => request<{rows?: Metric[]; total?: number; latest?: Metric} | Metric[]>(`${runPath(id)}/metrics`).then(v => Array.isArray(v) ? v : v.rows || []),
  checkpoints: (id: string) => request<Item[] | {items?: Item[]}>(`${runPath(id)}/checkpoints`).then(items),
  reconstructions: (id: string) => request<Item[] | {items?: Item[]}>(`${runPath(id)}/reconstructions`).then(items),
  evaluations: (id: string) => request<{per_view?: Item[]; summary?: Metric[]}>(`${runPath(id)}/evaluations`),
  sampling: (id: string) => request<{available: boolean; reason?: string; items?: Item[]}>(`${runPath(id)}/sampling`),
  rayDiagnostic: (id: string, params: {checkpoint: number; split: string; view_index: number; x: number; y: number}) =>
    request<RayDiagnostic>(`${runPath(id)}/sampling/ray?${new URLSearchParams(Object.entries(params).map(([key,value])=>[key,String(value)]))}`),
  logs: (id: string) => request<Dict | string[] | {items?: Dict[]}>(`${runPath(id)}/logs`),
  artifacts: (id: string) => request<Item[] | {items?: Item[]}>(`${runPath(id)}/artifacts`).then(items),
  create: (config: Dict) => request<Run>('/runs', {method:'POST', body:JSON.stringify({config})}),
  action: (id: string, action: 'start'|'stop'|'resume'|'evaluate'|'replay') => request<Dict>(`${runPath(id)}/${action}`, {method:'POST'}),
  ws: (id: string) => {
    const url = new URL(`${BASE}/ws/runs/${encodeURIComponent(id)}`, window.location.href)
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    return new WebSocket(url)
  },
  asset: (path?: string) => path ? new URL(path.startsWith('/api') ? path : `${BASE}${path}`, window.location.href).toString() : undefined,
}

export function readNumber(source: Dict | undefined | null, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = source?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string' && value.trim() && Number.isFinite(Number(value))) return Number(value)
  }
  return null
}
export function readText(source: Dict | undefined | null, ...keys: string[]): string | null {
  for (const key of keys) { const value = source?.[key]; if (typeof value === 'string' && value.trim()) return value }
  return null
}
export function fmt(value: unknown, digits=2): string {
  if (typeof value === 'number' && Number.isFinite(value)) return value.toLocaleString(getLanguage()==='zh'?'zh-CN':'en-US', {maximumFractionDigits:digits})
  if (typeof value === 'string' && value !== '') return value
  return t('暂无数据')
}
