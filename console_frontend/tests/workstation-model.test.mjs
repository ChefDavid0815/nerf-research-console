import assert from 'node:assert/strict'
import { test } from 'node:test'
import { build } from 'esbuild'

const bundled = await build({ entryPoints: ['src/workstation-model.ts'], bundle: true, format: 'esm', platform: 'node', write: false })
const source = Buffer.from(bundled.outputFiles[0].contents).toString('base64')
const model = await import(`data:text/javascript;base64,${source}`)

const baseline = {
  scene: 'lego', position_encoding_L: 10, direction_encoding_L: 4,
  network_depth: 8, network_width: 256, skip_connection_layer: 4, view_width: 128,
  training_iterations: 500000, random_seed: 0, batch_size: 4096,
  checkpoint_interval: 10000, preview_interval: 10000,
}

test('quick launch changes only the requested experimental controls', () => {
  const draft = model.buildQuickConfig(baseline, { L: 10, iterations: 30000, seed: 42 })
  assert.deepEqual(draft, { ...baseline, training_iterations: 30000, random_seed: 42 })
  assert.equal(baseline.training_iterations, 500000)
  assert.equal(baseline.random_seed, 0)
  assert.throws(() => model.buildQuickConfig(baseline, { L: 16, iterations: 30000, seed: 42 }), /L/)
  assert.throws(() => model.buildQuickConfig(baseline, { L: 10, iterations: 0, seed: 42 }), /iterations/)
})

test('model size reflects the validated NeRF architecture and L bandwidth', () => {
  assert.equal(model.modelParameterCount(baseline), 595844)
  assert.equal(model.modelParameterCount({ ...baseline, position_encoding_L: 6 }), 583556)
  assert.equal(model.encodedDimension(10), 63)
  assert.equal(model.maxFrequencyBand(10), 512)
})

test('time machine exposes only saved prediction iterations and exact frames', () => {
  const images = [
    { iteration: 500, kind: 'prediction', split: 'val', view_index: 0, url: '/500.png' },
    { iteration: 500, kind: 'ground_truth', split: 'val', view_index: 0, url: '/gt.png' },
    { iteration: 2000, kind: 'prediction', split: 'val', view_index: 0, url: '/2000.png' },
    { iteration: 5000, kind: 'prediction', split: 'test', view_index: 1, url: '/other.png' },
  ]
  assert.deepEqual(model.availableFrames(images, 'val', 0), [500, 2000])
  assert.equal(model.frameAt(images, 2000, 'prediction', 'val', 0)?.url, '/2000.png')
  assert.equal(model.frameAt(images, 1000, 'prediction', 'val', 0), undefined)
  assert.equal(model.frameAt(images, 2000, 'absolute_difference', 'val', 0), undefined)
})

test('ambient load reduces effects under measured GPU pressure', () => {
  assert.equal(model.ambientTier(null, false), 'balanced')
  assert.equal(model.ambientTier(42, false), 'full')
  assert.equal(model.ambientTier(77, false), 'balanced')
  assert.equal(model.ambientTier(94, true), 'reduced')
  assert.equal(model.ambientTier(10, false, true), 'off')
})

test('evaluation metrics remain distinct from training-batch PSNR in plots', () => {
  const training = [{ iteration: 500, total_loss: 0.04, psnr: 18.2 }]
  const evaluation = [{ iteration: 500, psnr: 16.7, ssim: 0.71, lpips: 0.29 }]
  const rows = model.mergePlotRows(training, evaluation)
  assert.equal(rows[0].psnr, 18.2)
  assert.equal(rows[0].validation_psnr, 16.7)
  assert.equal(rows[0].ssim, 0.71)
  assert.equal(rows[0].lpips, 0.29)
})

test('run selection preserves the user choice and auto-opens only an initial live run', () => {
  const runs = [{ id: 'history', status: 'completed' }, { id: 'active', status: 'running' }]
  assert.deepEqual(model.chooseRunSelection(runs, ''), { id: 'active', openLive: true })
  assert.deepEqual(model.chooseRunSelection(runs, 'history'), { id: 'history', openLive: false })
  assert.deepEqual(model.chooseRunSelection(runs, 'missing'), { id: 'active', openLive: false })
})

test('command gate prevents duplicate writes while a command is pending', () => {
  const gate = model.createCommandGate()
  assert.equal(gate.enter(), true)
  assert.equal(gate.enter(), false)
  gate.leave()
  assert.equal(gate.enter(), true)
})
