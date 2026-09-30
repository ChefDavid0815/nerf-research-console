"use strict"

const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const asar = require('@electron/asar')

const frontend = path.resolve(__dirname, '..')
const workspace = path.resolve(frontend, '..')
const output = path.join(workspace, 'artifacts', 'NeRF Research Console Windows')
const electronBinary = require('electron')
const electronFiles = path.dirname(electronBinary)
const stage = fs.mkdtempSync(path.join(os.tmpdir(), 'nerf-console-package-'))

async function main() {
  if (!fs.existsSync(path.join(frontend, 'dist', 'index.html'))) throw new Error('Build the frontend before packaging')
  const updating = fs.existsSync(output)
  if (updating && !fs.existsSync(path.join(output, 'NeRF Research Console.exe'))) throw new Error(`Incomplete delivery folder: ${output}`)
  fs.mkdirSync(path.dirname(output), { recursive: true })
  if (!updating) fs.cpSync(electronFiles, output, { recursive: true })
  fs.cpSync(path.join(frontend, 'dist'), path.join(output, 'frontend'), { recursive: true, force: true })
  fs.cpSync(path.join(frontend, 'dist'), path.join(stage, 'dist'), { recursive: true })
  fs.cpSync(path.join(frontend, 'desktop'), path.join(stage, 'desktop'), { recursive: true })
  fs.copyFileSync(path.join(frontend, 'package.json'), path.join(stage, 'package.json'))
  const appArchive = path.join(output, 'resources', 'app.asar')
  if (updating) {
    const nextArchive = path.join(output, 'resources', 'app.next.asar')
    await asar.createPackage(stage, nextArchive)
    fs.copyFileSync(appArchive, `${appArchive}.previous`)
    fs.copyFileSync(nextArchive, appArchive)
    fs.unlinkSync(nextArchive)
  } else {
    await asar.createPackage(stage, appArchive)
  }

  const executable = path.join(output, 'NeRF Research Console.exe')
  if (!updating) {
    try { fs.renameSync(path.join(output, 'electron.exe'), executable) }
    catch { fs.copyFileSync(path.join(output, 'electron.exe'), executable) }
  }

  const rcedit = path.join(frontend, 'node_modules', 'electron-winstaller', 'vendor', 'rcedit.exe')
  if (fs.existsSync(rcedit)) {
    const result = spawnSync(rcedit, [executable,
      '--set-icon', path.join(__dirname, 'icon.ico'),
      '--set-version-string', 'ProductName', 'NeRF Research Console',
      '--set-version-string', 'FileDescription', 'Neural Rendering Research Workstation',
      '--set-version-string', 'CompanyName', 'NeRF Research Console',
      '--set-version-string', 'InternalName', 'NeRF Research Console',
      '--set-version-string', 'OriginalFilename', 'NeRF Research Console.exe',
      '--set-file-version', '1.0.0', '--set-product-version', '1.0.0',
    ], { encoding: 'utf8' })
    if (result.status !== 0) throw new Error(`Windows icon update failed: ${result.stderr || result.stdout || result.error}`)
  }
  const readme = [
    'NeRF Research Console / Windows',
    '',
    '双击 NeRF Research Console.exe。应用会自动启动本地研究后端，无需终端。',
    'Double-click NeRF Research Console.exe. The local research backend starts automatically.',
    '',
    '请将本目录保留在本项目工作区的 artifacts 文件夹中。',
    'Keep this folder inside the project workspace artifacts directory.',
    '前端文件随桌面包提供；训练仍使用工作区现有的 .venv、data、configs 和 runs。',
    'Frontend assets are included. Training uses the workspace .venv, data, configs and runs.',
    '此文件夹不是独立安装包。 / This folder is not a standalone installer.',
    '',
    'F11: presentation mode / 演示模式',
    'Ctrl+1…4: Train, Runs, Analyze, System / 训练、运行档案、分析、系统',
  ].join('\r\n')
  fs.writeFileSync(path.join(output, 'START HERE.txt'), readme, 'utf8')
  process.stdout.write(`${output}\n`)
}

main().catch(error => { process.stderr.write(`${error.stack || error}\n`); process.exitCode = 1 }).finally(() => {
  const tempRoot = path.resolve(os.tmpdir()) + path.sep
  const resolvedStage = path.resolve(stage)
  if (resolvedStage.startsWith(tempRoot) && path.basename(resolvedStage).startsWith('nerf-console-package-')) {
    fs.rmSync(resolvedStage, { recursive: true, force: true })
  }
})
