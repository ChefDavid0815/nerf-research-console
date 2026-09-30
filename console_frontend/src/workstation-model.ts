import type { Dict, Item, Metric } from './api'

export type QuickSettings = { L: number; iterations: number; seed: number }
export type AmbientTier = 'full' | 'balanced' | 'reduced' | 'off'

export function buildQuickConfig(defaults: Dict, settings: QuickSettings): Dict {
  if (!Number.isInteger(settings.L) || settings.L < 0 || settings.L > 15) throw new Error('L must be an integer from 0 to 15')
  if (!Number.isInteger(settings.iterations) || settings.iterations < 1 || settings.iterations > 1_000_000) throw new Error('iterations must be from 1 to 1,000,000')
  if (!Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 2_147_483_647) throw new Error('seed must be a nonnegative integer')
  return { ...defaults, position_encoding_L: settings.L, training_iterations: settings.iterations, random_seed: settings.seed }
}

export function encodedDimension(L: number): number { return 3 + 6 * L }
export function maxFrequencyBand(L: number): number | null { return L > 0 ? 2 ** (L - 1) : null }

/** Mirrors the validated VanillaNeRF Linear layers; display estimate, backend remains authoritative. */
export function modelParameterCount(config: Dict): number | null {
  const values = ['position_encoding_L', 'direction_encoding_L', 'network_depth', 'network_width', 'skip_connection_layer', 'view_width']
    .map(key => config[key])
  if (values.some(value => typeof value !== 'number' || !Number.isInteger(value))) return null
  const [L, directionL, depth, width, skip, viewWidth] = values as number[]
  if (L < 0 || directionL < 0 || depth < 2 || width < 1 || skip < 0 || skip >= depth - 1 || viewWidth < 1) return null
  const positionDim = encodedDimension(L)
  const directionDim = encodedDimension(directionL)
  let count = 0
  for (let layer = 0; layer < depth; layer++) {
    const input = layer === 0 ? positionDim : layer === skip + 1 ? width + positionDim : width
    count += input * width + width
  }
  count += width + 1 // density
  count += width * width + width // feature
  count += (width + directionDim) * viewWidth + viewWidth // view branch
  count += viewWidth * 3 + 3 // RGB
  return count
}

export function availableFrames(items: Item[], split: string, view: number): number[] {
  return [...new Set(items.filter(item => item.kind === 'prediction' && item.split === split && item.view_index === view && item.url && typeof item.iteration === 'number')
    .map(item => item.iteration as number))].sort((a, b) => a - b)
}

export function frameAt(items: Item[], step: number, kind: string, split: string, view: number): Item | undefined {
  return items.find(item => item.iteration === step && item.kind === kind && item.split === split && item.view_index === view && !!item.url)
}

export function ambientTier(load: number | null, active: boolean, reducedMotion = false): AmbientTier {
  if (reducedMotion) return 'off'
  if (load === null) return active ? 'reduced' : 'balanced'
  if (load >= 90) return 'reduced'
  if (load >= 60) return 'balanced'
  return 'full'
}

export function mergePlotRows(training: Metric[], evaluation: Metric[]): Metric[] {
  const rows = new Map<number, Metric>()
  for (const row of training) {
    const step = Number(row.iteration)
    if (Number.isFinite(step)) rows.set(step, { ...row })
  }
  for (const row of evaluation) {
    if (row.split && row.split !== 'val') continue
    const step = Number(row.iteration)
    if (!Number.isFinite(step)) continue
    const result: Metric = { ...(rows.get(step) || { iteration: step }) }
    for (const [target, keys] of Object.entries({ validation_psnr: ['psnr', 'psnr_mean'], ssim: ['ssim', 'ssim_mean'], lpips: ['lpips', 'lpips_mean'] })) {
      const value = keys.map(key => row[key]).find(value => typeof value === 'number' && Number.isFinite(value))
      if (typeof value === 'number') result[target] = value
    }
    rows.set(step, result)
  }
  return [...rows.values()].sort((a, b) => Number(a.iteration) - Number(b.iteration))
}

export function chooseRunSelection(runs: { id: string; status?: string; featured_baseline?: boolean }[], current: string): { id: string; openLive: boolean } {
  if (current && runs.some(run => run.id === current)) return { id: current, openLive: false }
  const active = runs.find(run => run.status === 'running' || run.status === 'stopping' || run.status === 'training')
  return { id: active?.id || runs.find(run => run.featured_baseline)?.id || runs[0]?.id || '', openLive: !current && !!active }
}

export function createCommandGate(): { enter: () => boolean; leave: () => void } {
  let pending = false
  return {
    enter: () => { if (pending) return false; pending = true; return true },
    leave: () => { pending = false },
  }
}
