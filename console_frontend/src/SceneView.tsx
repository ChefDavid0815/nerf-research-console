import { Component, useEffect, useMemo, useState } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import { Grid, Line, OrbitControls } from '@react-three/drei'
import { BufferGeometry, Float32BufferAttribute } from 'three'
import type { Dict } from './api'
import { t, useI18n } from './i18n'
import './SceneView.css'

type Vec3 = [number, number, number]
type Ray = { pixel_xy: [number, number]; direction: Vec3 }
type Camera = { key: string; split: string; index: number; position: Vec3; forward: Vec3; centerRay: Vec3; cornerRays: Vec3[]; selectedRay: Ray | null }
type Geometry = { cameras: Camera[]; total: number; width: number; height: number; convention: string }

function vec(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null
  const result = value.map(Number)
  return result.every(Number.isFinite) ? result as Vec3 : null
}
function parse(data: Dict | null): Geometry {
  const cameras: Camera[] = []
  for (const entry of Array.isArray(data?.cameras) ? data.cameras : []) {
    if (!entry || typeof entry !== 'object') continue
    const c = entry as Dict
    const position = vec(c.position), forward = vec(c.forward), centerRay = vec(c.center_ray)
    const corners = Array.isArray(c.corner_rays) ? c.corner_rays.map(vec) : []
    const index = Number(c.index), split = String(c.split || '')
    if (!position || !forward || !centerRay || corners.length !== 4 || corners.some(v => !v) || !Number.isInteger(index) || !['train', 'val', 'test'].includes(split)) continue
    const rawRay = c.selected_ray && typeof c.selected_ray === 'object' ? c.selected_ray as Dict : null
    const rayDirection = rawRay && vec(rawRay.direction)
    const pixel = rawRay && Array.isArray(rawRay.pixel_xy) ? rawRay.pixel_xy.map(Number) : []
    const selectedRay = rayDirection && pixel.length === 2 && pixel.every(Number.isInteger) ? { direction: rayDirection, pixel_xy: pixel as [number, number] } : null
    cameras.push({ key: `${split}:${index}`, split, index, position, forward, centerRay, cornerRays: corners as Vec3[], selectedRay })
  }
  const intrinsics = data?.intrinsics && typeof data.intrinsics === 'object' ? data.intrinsics as Dict : {}
  return { cameras, total: Number(data?.total) || cameras.length, width: Number(intrinsics.width) || 800, height: Number(intrinsics.height) || 800, convention: String(data?.coordinate_system || 'OpenGL camera-to-world') }
}
const add = (a: Vec3, b: Vec3, scale: number): Vec3 => [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale]

function linesFor(cameras: Camera[], allFrustums: boolean) {
  const values: number[] = []
  const line = (a: Vec3, b: Vec3) => values.push(...a, ...b)
  for (const c of cameras) {
    line(c.position, add(c.position, c.forward, 0.24))
    if (allFrustums) {
      const corners = c.cornerRays.map(ray => add(c.position, ray, 0.48))
      for (let i = 0; i < 4; i++) { line(c.position, corners[i]); line(corners[i], corners[(i + 1) % 4]) }
    }
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(values, 3))
  return geometry
}

function CameraScene({ cameras, selected, allFrustums, select }: { cameras: Camera[]; selected: Camera; allFrustums: boolean; select: (key: string) => void }) {
  const markers = useMemo(() => {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new Float32BufferAttribute(cameras.flatMap(c => c.position), 3))
    return geometry
  }, [cameras])
  const backgroundLines = useMemo(() => linesFor(cameras, allFrustums), [cameras, allFrustums])
  useEffect(() => () => { markers.dispose(); backgroundLines.dispose() }, [markers, backgroundLines])
  const corners = selected.cornerRays.map(ray => add(selected.position, ray, 0.76))
  const activeRay = selected.selectedRay?.direction || selected.centerRay
  return <>
    <color attach="background" args={['#07110d']} />
    <fog attach="fog" args={['#07110d', 10, 24]} />
    <Grid position={[0, -0.02, 0]} infiniteGrid fadeDistance={18} fadeStrength={1.7} sectionSize={2} cellSize={0.5} sectionColor="#315c45" cellColor="#1a3729" sectionThickness={0.9} cellThickness={0.42} />
    <axesHelper args={[0.7]} />
    <mesh position={[0, 0, 0]}><octahedronGeometry args={[0.075]} /><meshBasicMaterial color="#d5fce6" /></mesh>
    <lineSegments geometry={backgroundLines}><lineBasicMaterial color="#3a7454" transparent opacity={allFrustums ? 0.31 : 0.56} /></lineSegments>
    <points geometry={markers} onClick={event => { event.stopPropagation(); if (event.index != null && cameras[event.index]) select(cameras[event.index].key) }}><pointsMaterial color="#69c493" size={0.105} sizeAttenuation transparent opacity={0.9} depthWrite={false} /></points>
    <mesh position={selected.position}><sphereGeometry args={[0.085, 14, 14]} /><meshBasicMaterial color="#b4ffd0" /></mesh>
    {corners.map((corner, i) => <Line key={`ray-${i}`} points={[selected.position, corner]} color="#91ebae" lineWidth={1.5} transparent opacity={0.78} />)}
    {corners.map((corner, i) => <Line key={`edge-${i}`} points={[corner, corners[(i + 1) % 4]]} color="#a5ffc5" lineWidth={1.7} transparent opacity={0.9} />)}
    <Line points={[selected.position, add(selected.position, activeRay, 3.7)]} color="#d8fca0" lineWidth={2.4} transparent opacity={0.94} />
    <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={1.2} maxDistance={26} target={[0, 0, 0]} />
  </>
}

class SceneErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('3D camera view failed', error, info) }
  render() { return this.state.failed ? <div className="scene-webgl-error">{t('3D 渲染不可用。右侧真实相机列表和数值仍可使用。')}</div> : this.props.children }
}

export default function SceneView({ data, onPixelChange }: { data: Dict | null; onPixelChange?: (x: number, y: number) => void }) {
  const { t } = useI18n()
  const geometry = useMemo(() => parse(data), [data])
  const [selectedKey, setSelectedKey] = useState('')
  const [allFrustums, setAllFrustums] = useState(false)
  const [split, setSplit] = useState('all')
  const [search, setSearch] = useState('')
  const [pixelX, setPixelX] = useState(400)
  const [pixelY, setPixelY] = useState(400)
  useEffect(() => { setPixelX(Math.floor(geometry.width / 2)); setPixelY(Math.floor(geometry.height / 2)) }, [geometry.width, geometry.height])
  const selected = geometry.cameras.find(c => c.key === selectedKey) || geometry.cameras[0]
  const visible = geometry.cameras.filter(c => (split === 'all' || c.split === split) && `${c.split} ${c.index}`.includes(search.trim().toLowerCase()))
  const listed = visible.slice(0, 60)
  const chooseSplit = (next: string) => { setSplit(next); const first = geometry.cameras.find(c => next === 'all' || c.split === next); if (first) setSelectedKey(first.key) }
  if (!selected) return <div className="empty-state">{t('经验证的相机数据尚不可用。3D 视图不会在浏览器端推测相机坐标。')}</div>
  return <div className="scene-view">
    <div className="scene-toolbar"><div><strong>LEGO / CAMERA SPACE</strong><span>{t('{shown} / {total} 台相机 · {convention}', { shown: geometry.cameras.length, total: geometry.total, convention: geometry.convention })}</span></div><label className="scene-toggle"><input type="checkbox" checked={allFrustums} onChange={event => setAllFrustums(event.target.checked)} />{t('全部视锥')}</label></div>
    <div className="scene-workspace">
      <div className="scene-stage" role="img" aria-label={t('Lego 相机位置、方向、视锥与选中射线的三维视图')}>
        <SceneErrorBoundary><Canvas camera={{ position: [6.3, 4.9, 7.2], fov: 44 }} dpr={[1, 1.6]} gl={{ antialias: true, powerPreference: 'high-performance' }}><CameraScene cameras={geometry.cameras} selected={selected} allFrustums={allFrustums} select={setSelectedKey} /></Canvas></SceneErrorBoundary>
        <div className="scene-stage-hint">{t('左键旋转 / 滚轮缩放 / 右键平移')}</div><div className="scene-legend"><span>{t('● 相机')}</span><span>{t('◇ 原点')}</span><span>{t('━ 选中射线')}</span></div>
      </div>
      <aside className="scene-inspector">
        <div className="scene-inspector-head"><span>{t('相机检视')}</span><strong>{selected.split.toUpperCase()} / {String(selected.index).padStart(3, '0')}</strong></div>
        <dl className="scene-vector-list"><div><dt>{t('世界位置')}</dt><dd>{selected.position.map(v => v.toFixed(3)).join(' / ')}</dd></div><div><dt>{t('前向方向')}</dt><dd>{selected.forward.map(v => v.toFixed(3)).join(' / ')}</dd></div><div><dt>{t('射线来源')}</dt><dd>{selected.selectedRay ? t('像素 {pixel}', { pixel: selected.selectedRay.pixel_xy.join(', ') }) : t('中心像素')}</dd></div></dl>
        <p className="scene-source-note">{t('位置、方向及四角射线由 Python 端按 Step 1 约定计算。图中射线长度只用于空间显示。')}</p>
        <div className="scene-pixel-controls"><span>{t('像素射线')}</span><div><label>X <input type="number" min={0} max={geometry.width - 1} value={pixelX} onChange={e => setPixelX(Number(e.target.value))} /></label><label>Y <input type="number" min={0} max={geometry.height - 1} value={pixelY} onChange={e => setPixelY(Number(e.target.value))} /></label><button type="button" onClick={() => onPixelChange?.(pixelX, pixelY)} disabled={!onPixelChange || !Number.isInteger(pixelX) || !Number.isInteger(pixelY) || pixelX < 0 || pixelY < 0 || pixelX >= geometry.width || pixelY >= geometry.height}>{t('读取')}</button></div></div>
        <div className="scene-list-head"><strong>{t('相机索引')}</strong><small>{visible.length > listed.length ? t('显示前 {shown} / {total} 项，搜索编号可定位其余视角', { shown: listed.length, total: visible.length }) : t('{count} 项', { count: visible.length })}</small></div>
        <div className="scene-filters" role="group" aria-label={t('数据集划分')}>{['all', 'train', 'val', 'test'].map(value => <button key={value} type="button" className={split === value ? 'active' : ''} onClick={() => chooseSplit(value)}>{value === 'all' ? t('全部') : value.toUpperCase()}</button>)}</div>
        <label className="scene-search"><span className="sr-only">{t('搜索相机')}</span><input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder={t('按划分或编号查找')} /></label>
        <div className="scene-camera-list" role="listbox" aria-label={t('选择相机')}>{visible.length ? listed.map(c => <button key={c.key} role="option" aria-selected={c.key === selected.key} type="button" onClick={() => setSelectedKey(c.key)}><span>{c.split.toUpperCase()}</span><strong>{String(c.index).padStart(3, '0')}</strong><small>{c.position.map(v => v.toFixed(1)).join(' / ')}</small></button>) : <div className="scene-no-results">{t('无匹配相机，调整筛选或搜索。')}</div>}</div>
      </aside>
    </div>
  </div>
}
