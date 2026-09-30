import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { build } from 'esbuild'

let i18nSource
let apiSource
const originalWindow = globalThis.window
const originalFetch = globalThis.fetch

before(async () => {
  const i18n = await build({ entryPoints: ['src/i18n.ts'], bundle: true, format: 'esm', platform: 'node', write: false })
  const api = await build({ entryPoints: ['src/api.ts'], bundle: true, format: 'esm', platform: 'node', write: false,
    define: { 'import.meta.env.VITE_API_BASE': '""' } })
  i18nSource = Buffer.from(i18n.outputFiles[0].contents).toString('base64')
  apiSource = Buffer.from(api.outputFiles[0].contents).toString('base64')
})

after(() => {
  globalThis.window = originalWindow
  globalThis.fetch = originalFetch
})

const load = (source, salt) => import(`data:text/javascript;base64,${source}#${salt}`)
const response = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})

test('system language is used once, then the manual language switch persists', async () => {
  const saved = new Map()
  globalThis.window = {
    navigator: { language: 'zh-CN' },
    localStorage: { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) },
    document: { documentElement: { lang: '' } },
    addEventListener: () => {},
  }
  const first = await load(i18nSource, 'first')
  assert.equal(first.getLanguage(), 'zh')
  assert.equal(first.t('暂无数据'), '暂无数据')
  first.setLanguage('en')
  assert.equal(first.t('暂无数据'), 'Not available')
  assert.equal(first.t('{count} 个检查点', { count: 15 }), '15 checkpoints')
  assert.equal(saved.get('nerf-console-language'), 'en')
  assert.equal(globalThis.window.document.documentElement.lang, 'en')
  const reopened = await load(i18nSource, 'reopened')
  assert.equal(reopened.getLanguage(), 'en')
})

test('saved history and reconstruction responses retain measured records and status', async () => {
  const seen = []
  globalThis.fetch = async url => {
    seen.push(String(url))
    if (url === '/api/runs') return response({ items: [
      { id: 'baseline', status: 'completed', iteration: 50000, latest_evaluation: { psnr: 27.3 } },
      { id: 'active', status: 'running', iteration: 12 },
      { id: 'failed', status: 'failed', traceback: 'CUDA out of memory' },
    ] })
    if (url === '/api/runs/baseline/metrics') return response({ rows: [
      { iteration: 1, total_loss: 0.4 }, { iteration: 50000, total_loss: 0.004 },
    ] })
    if (url === '/api/runs/baseline/reconstructions') return response({ items: [
      { iteration: 50000, split: 'val', view_index: 0, kind: 'prediction', url: '/api/artifacts/real.png', metrics: { psnr: 27.3 } },
    ] })
    if (url === '/api/runs/baseline/evaluations') return response({ per_view: [
      { iteration: 50000, split: 'val', view_index: 0, psnr: 27.3 },
    ], summary: [] })
    throw new Error(`unexpected API call: ${url}`)
  }
  const { api } = await load(apiSource, 'history')
  const runs = await api.runs()
  assert.deepEqual(runs.map(run => run.status), ['completed', 'running', 'failed'])
  assert.equal(runs[2].traceback, 'CUDA out of memory')
  assert.deepEqual((await api.metrics('baseline')).map(row => row.iteration), [1, 50000])
  const renders = await api.reconstructions('baseline')
  assert.equal(renders[0].metrics.psnr, 27.3)
  assert.equal(renders[0].url, '/api/artifacts/real.png')
  assert.equal((await api.evaluations('baseline')).per_view[0].psnr, 27.3)
  assert.equal(seen.length, 4)
})

test('experiment creation forwards the exact draft and surfaces backend validation failure', async () => {
  const draft = { scene: 'lego', position_encoding_L: 10, direction_encoding_L: 4,
    training_iterations: 500, batch_size: 256, random_seed: 0 }
  const requests = []
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init })
    return response({ detail: 'learning_rate must be positive' }, 422)
  }
  const { api } = await load(apiSource, 'builder')
  await assert.rejects(api.create(draft), /learning_rate must be positive/)
  assert.equal(requests[0].url, '/api/runs')
  assert.equal(requests[0].init.method, 'POST')
  assert.deepEqual(JSON.parse(requests[0].init.body), { config: draft })
  assert.equal(requests.length, 1)
  assert.deepEqual(draft, { scene: 'lego', position_encoding_L: 10, direction_encoding_L: 4,
    training_iterations: 500, batch_size: 256, random_seed: 0 })
})

test('stop and resume remain explicit backend actions with recoverable errors', async () => {
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, method: init?.method })
    if (url === '/api/runs/active/stop') return response({ status: 'stopped', termination_reason: 'user_requested' })
    if (url === '/api/runs/active/resume') return response({ detail: 'checkpoint configuration mismatch' }, 409)
    throw new Error(`unexpected API call: ${url}`)
  }
  const { api } = await load(apiSource, 'actions')
  assert.deepEqual(await api.action('active', 'stop'), { status: 'stopped', termination_reason: 'user_requested' })
  await assert.rejects(api.action('active', 'resume'), /checkpoint configuration mismatch/)
  assert.deepEqual(calls, [
    { url: '/api/runs/active/stop', method: 'POST' },
    { url: '/api/runs/active/resume', method: 'POST' },
  ])
})
