const fs = require('node:fs')
const path = require('node:path')

function findProjectRoot(start) {
  let current = path.resolve(start)
  for (let depth = 0; depth < 9; depth++) {
    if (['configs/baseline.yaml', '.venv/Scripts/python.exe', 'console_frontend/dist/index.html']
      .every(file => fs.existsSync(path.join(current, file)))) return current
    const parent = path.dirname(current)
    if (parent === current) break
    current = parent
  }
  return null
}

function safeWindowBounds(saved, workAreas, fallback) {
  if (!saved || !['x', 'y', 'width', 'height'].every(key => Number.isFinite(saved[key]))) return fallback
  if (saved.width < 980 || saved.height < 650) return fallback
  const display = workAreas.find(area => {
    const overlapWidth = Math.max(0, Math.min(saved.x + saved.width, area.x + area.width) - Math.max(saved.x, area.x))
    const overlapHeight = Math.max(0, Math.min(saved.y + saved.height, area.y + area.height) - Math.max(saved.y, area.y))
    return overlapWidth >= 100 && overlapHeight >= 100
  })
  if (!display) return fallback
  const width = Math.min(saved.width, display.width)
  const height = Math.min(saved.height, display.height)
  return {
    x: Math.max(display.x, Math.min(saved.x, display.x + display.width - width)),
    y: Math.max(display.y, Math.min(saved.y, display.y + display.height - height)),
    width, height,
  }
}
module.exports = { findProjectRoot, safeWindowBounds }
