const { app, BrowserWindow, dialog, ipcMain, Menu, screen } = require('electron')
const fs = require('node:fs')
const path = require('node:path')
const net = require('node:net')
const { spawn } = require('node:child_process')
const { findProjectRoot, safeWindowBounds } = require('./runtime.cjs')

let window = null
let backend = null
let backendLog = null
let backendOrigin = ''
let windowStateTimer = null

const instanceLock = app.requestSingleInstanceLock()
if (!instanceLock) app.quit()
else app.on('second-instance', () => {
  if (!window) return
  if (window.isMinimized()) window.restore()
  window.focus()
})

function findWorkspace() {
  const starts = [process.env.PORTABLE_EXECUTABLE_DIR, path.dirname(process.execPath), process.cwd(), __dirname].filter(Boolean)
  for (const start of starts) {
    const root = findProjectRoot(start)
    if (root) return root
  }
  return null
}

function probePort(preferred) {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(preferred, '127.0.0.1', () => {
      const address = server.address()
      const port = address && typeof address === 'object' ? address.port : null
      server.close(() => port ? resolve(port) : reject(new Error('No local port available')))
    })
  })
}

async function availablePort() {
  try { return await probePort(8765) }
  catch (error) {
    if (error.code !== 'EADDRINUSE' && error.code !== 'EACCES') throw error
    return probePort(0)
  }
}

async function waitForBackend(origin, processHandle) {
  const started = Date.now()
  while (Date.now() - started < 30000) {
    if (processHandle.exitCode !== null) throw new Error(`Research backend exited with code ${processHandle.exitCode}`)
    try {
      const response = await fetch(`${origin}/api/health`, { signal: AbortSignal.timeout(1500) })
      if (response.ok && (await response.json()).status === 'ok') return
    } catch { /* Python and PyTorch are still starting. */ }
    await new Promise(resolve => setTimeout(resolve, 230))
  }
  throw new Error('Research backend did not become ready within 30 seconds')
}

function loadSavedWindowState() {
  try { return JSON.parse(fs.readFileSync(path.join(app.getPath('userData'), 'window-state.json'), 'utf8')) }
  catch { return null }
}

function saveWindowState() {
  if (!window || window.isDestroyed()) return
  const saved = loadSavedWindowState() || {}
  const next = { ...saved, maximized: window.isMaximized() }
  if (!window.isMaximized() && !window.isFullScreen()) next.bounds = window.getBounds()
  const target = path.join(app.getPath('userData'), 'window-state.json')
  try {
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(`${target}.tmp`, JSON.stringify(next))
    fs.renameSync(`${target}.tmp`, target)
  } catch { /* Window remains usable if persistence is unavailable. */ }
}

function buildWindow() {
  const display = screen.getPrimaryDisplay().workArea
  const fallback = { width: Math.min(1600, display.width), height: Math.min(960, display.height) }
  const saved = loadSavedWindowState()
  const bounds = safeWindowBounds(saved?.bounds, screen.getAllDisplays().map(item => item.workArea), fallback)
  window = new BrowserWindow({
    ...bounds,
    minWidth: 980, minHeight: 650,
    frame: false, thickFrame: true, show: false,
    backgroundColor: '#050907',
    title: 'NeRF Research Console',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      backgroundThrottling: true,
    },
  })
  Menu.setApplicationMenu(null)
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  window.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(`${backendOrigin}/`)) event.preventDefault()
  })
  window.on('maximize', () => window?.webContents.send('desktop:maximized', true))
  window.on('unmaximize', () => window?.webContents.send('desktop:maximized', false))
  const queueSave = () => {
    clearTimeout(windowStateTimer)
    windowStateTimer = setTimeout(saveWindowState, 280)
  }
  window.on('resize', queueSave)
  window.on('move', queueSave)
  window.on('close', saveWindowState)
  if (saved?.maximized || !saved) window.maximize()
  window.once('ready-to-show', () => window?.show())
  window.show()
  return window
}

function validateSender(event) {
  return !!window && !window.isDestroyed() && event.sender === window.webContents && event.senderFrame?.url.startsWith(`${backendOrigin}/`)
}

ipcMain.on('desktop:minimize', event => { if (validateSender(event)) window.minimize() })
ipcMain.on('desktop:maximize', event => { if (validateSender(event)) window.isMaximized() ? window.unmaximize() : window.maximize() })
ipcMain.on('desktop:close', event => { if (validateSender(event)) window.close() })
ipcMain.handle('desktop:isMaximized', event => validateSender(event) ? window.isMaximized() : false)
ipcMain.handle('desktop:setFullscreen', (event, value) => {
  if (!validateSender(event) || typeof value !== 'boolean') return
  window.setFullScreen(value)
})

async function start() {
  app.setAppUserModelId('org.nerfresearch.console')
  const projectRoot = findWorkspace()
  if (!projectRoot) {
    dialog.showErrorBox('NeRF Research Console', 'The local research workspace could not be found. Keep this desktop app inside the NeRF-Extended Essay workspace, with its .venv and dataset.')
    app.quit()
    return
  }
  const port = await availablePort()
  backendOrigin = `http://127.0.0.1:${port}`
  const target = buildWindow()
  const python = path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
  backendLog = fs.createWriteStream(path.join(app.getPath('userData'), 'backend.log'), { flags: 'a' })
  backend = spawn(python, ['-m', 'uvicorn', 'nerf_console.api:app', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: projectRoot, windowsHide: true,
    env: { ...process.env, PYTHONPATH: path.join(projectRoot, 'src'), PYTHONUNBUFFERED: '1',
      NERF_FRONTEND_DIST: path.join(path.dirname(process.execPath), 'frontend') },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  backend.stdout.pipe(backendLog, { end: false })
  backend.stderr.pipe(backendLog, { end: false })
  try {
    await waitForBackend(backendOrigin, backend)
    if (!target.isDestroyed()) await target.loadURL(`${backendOrigin}/`)
  } catch (cause) {
    const response = await dialog.showMessageBox(target, {
      type: 'error', title: 'Research backend unavailable',
      message: 'The local research runtime could not start.',
      detail: `${cause instanceof Error ? cause.message : String(cause)}\n\nBackend log: ${path.join(app.getPath('userData'), 'backend.log')}`,
      buttons: ['Close'],
    })
    if (response.response === 0) app.quit()
  }
}

app.whenReady().then(start).catch(cause => {
  dialog.showErrorBox('NeRF Research Console', cause instanceof Error ? cause.message : String(cause))
  app.quit()
})
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => {
  clearTimeout(windowStateTimer)
  if (backend && backend.exitCode === null) backend.kill()
  backendLog?.end()
})
