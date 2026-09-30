import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

const require = createRequire(import.meta.url)
const { findProjectRoot, safeWindowBounds } = require('../desktop/runtime.cjs')

test('desktop executable discovers the local research workspace from a nested release folder', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'nerf-desktop-'))
  try {
    for (const folder of ['configs', 'src/nerf_console', '.venv/Scripts', 'console_frontend/dist', 'console_frontend/release/win-unpacked']) mkdirSync(path.join(root, folder), { recursive: true })
    for (const file of ['configs/baseline.yaml', '.venv/Scripts/python.exe', 'console_frontend/dist/index.html']) writeFileSync(path.join(root, file), '')
    assert.equal(findProjectRoot(path.join(root, 'console_frontend/release/win-unpacked')), root)
  } finally { rmSync(root, { recursive: true, force: true }) }
})

test('window bounds are restored only when visible on a current display', () => {
  const fallback = { width: 1600, height: 920 }
  const displays = [{ x: 0, y: 0, width: 1920, height: 1040 }]
  assert.deepEqual(safeWindowBounds({ x: 100, y: 100, width: 1400, height: 850 }, displays, fallback), { x: 100, y: 100, width: 1400, height: 850 })
  assert.deepEqual(safeWindowBounds({ x: 6000, y: 6000, width: 1400, height: 850 }, displays, fallback), fallback)
  assert.deepEqual(safeWindowBounds({ x: 100, y: 100, width: 200, height: 100 }, displays, fallback), fallback)
})
