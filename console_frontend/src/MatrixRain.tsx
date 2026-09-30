import { useEffect, useRef } from 'react'
import type { AmbientTier } from './workstation-model'
import './MatrixRain.css'

type Stream = { x: number; head: number; speed: number; length: number; glyphs: string[]; tint: boolean }
type Bit = { x: number; y: number; size: number; depth: number; speed: number; value: string }
const alphabet = ['0', '1', 'x', 'y', 'z', 'σ', 'L', 'sin', 'cos', 'RGB', 'ray', 'CUDA', '256', '64', '128', 'PSNR', '0.0048']
const glyph = () => alphabet[Math.floor(Math.random() * alphabet.length)]

/** Decorative instrumentation layer. It contains no measured or simulated training values. */
export default function MatrixRain({ active = false, boot = false, tier = 'balanced' }: {
  active?: boolean; boot?: boolean; tier?: AmbientTier
}) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d', { alpha: true })
    if (!canvas || !ctx) return
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    let width = 1, height = 1, frame = 0, last = 0, offset = 0
    let streams: Stream[] = [], bits: Bit[] = []
    let quality: AmbientTier = motion.matches ? 'off' : tier
    const spacing = () => boot ? 48 : quality === 'full' ? 66 : quality === 'balanced' ? 92 : 142
    const interval = () => boot ? 35 : quality === 'full' ? 48 : quality === 'balanced' ? 70 : 120

    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.25)
      width = Math.max(1, window.innerWidth)
      height = Math.max(1, window.innerHeight)
      canvas.width = Math.round(width * ratio)
      canvas.height = Math.round(height * ratio)
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
      const cell = spacing()
      streams = Array.from({ length: Math.ceil(width / cell) }, (_, i) => ({
        x: i * cell + Math.random() * 13,
        head: Math.random() * height,
        speed: (boot ? 22 : active ? 16 : 9) + Math.random() * 21,
        length: 6 + Math.floor(Math.random() * 10),
        glyphs: Array.from({ length: 18 }, glyph),
        tint: i % 13 === 0,
      }))
      const count = boot ? 34 : quality === 'full' ? 28 : quality === 'balanced' ? 15 : 7
      bits = Array.from({ length: count }, () => ({
        x: Math.random() * width, y: Math.random() * height,
        size: 9 + Math.random() * 7, depth: Math.random(),
        speed: 1 + Math.random() * 7, value: Math.random() > .5 ? '1' : '0',
      }))
      draw(0)
    }

    const draw = (seconds: number) => {
      ctx.clearRect(0, 0, width, height)
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'center'
      for (const stream of streams) {
        if (seconds) stream.head += stream.speed * seconds
        if (stream.head - stream.length * 22 > height) stream.head = -Math.random() * height * .25
        for (let n = 0; n < stream.length; n++) {
          const y = stream.head - n * 22
          if (y < -20 || y > height + 20) continue
          const alpha = (boot ? .27 : .10) * (1 - n / stream.length) ** 1.8
          ctx.fillStyle = stream.tint ? `rgba(92,214,194,${alpha})` : `rgba(121,250,151,${alpha})`
          ctx.font = `${n === 0 ? 12 : 10}px "Cascadia Code", Consolas, monospace`
          if (seconds && n === 0 && Math.random() < .09) {
            const slot = ((Math.floor(stream.head / 22) % stream.glyphs.length) + stream.glyphs.length) % stream.glyphs.length
            stream.glyphs[slot] = glyph()
          }
          const index = Math.abs(Math.floor(y / 22)) % stream.glyphs.length
          ctx.fillText(stream.glyphs[index], stream.x + offset * .2, y)
        }
      }
      for (const bit of bits) {
        if (seconds) bit.y -= bit.speed * seconds
        if (bit.y < -20) bit.y = height + 20
        const alpha = (boot ? .15 : .05) * (.3 + bit.depth)
        ctx.fillStyle = `rgba(164,255,182,${alpha})`
        ctx.font = `${bit.size}px "Cascadia Code", Consolas, monospace`
        ctx.fillText(bit.value, bit.x + offset * bit.depth, bit.y)
      }
    }

    const tick = (now: number) => {
      if (now - last > interval()) {
        draw(last ? Math.min((now - last) / 1000, .1) : 0)
        last = now
      }
      frame = requestAnimationFrame(tick)
    }
    const start = () => {
      cancelAnimationFrame(frame)
      last = 0
      if (!document.hidden && quality !== 'off') frame = requestAnimationFrame(tick)
      else draw(0)
    }
    const pointer = (event: PointerEvent) => { offset = (event.clientX / Math.max(width, 1) - .5) * 6 }
    const onMotion = () => { quality = motion.matches ? 'off' : tier; resize(); start() }
    resize(); start()
    window.addEventListener('resize', resize)
    window.addEventListener('pointermove', pointer, { passive: true })
    document.addEventListener('visibilitychange', start)
    motion.addEventListener('change', onMotion)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', pointer)
      document.removeEventListener('visibilitychange', start)
      motion.removeEventListener('change', onMotion)
    }
  }, [active, boot, tier])

  return <canvas ref={ref} className={`matrix-rain${boot ? ' matrix-rain-boot' : ''}`} aria-hidden="true" />
}
