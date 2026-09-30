import { useMemo, useState } from 'react'
import { api, fmt, type Item } from './api'
import { t } from './i18n'

type Sort='high_psnr'|'low_psnr'|'high_ssim'|'low_lpips'
const sortLabels:Record<Sort,string>={high_psnr:'最高 PSNR',low_psnr:'最低 PSNR',high_ssim:'最高 SSIM',low_lpips:'最低 LPIPS'}
function image(items:Item[],step:number|undefined,split:string|undefined,index:number|undefined,kind:string){return items.find(item=>item.iteration===step&&item.split===split&&item.view_index===index&&item.kind===kind)}
export default function Reconstruction({items,checkpoints}:{items:Item[];checkpoints:Item[]}){
  const [step,setStep]=useState<number|null>(null),[split,setSplit]=useState('val'),[index,setIndex]=useState(0),[zoom,setZoom]=useState(1),[sort,setSort]=useState<Sort>('high_psnr')
  const steps=useMemo(()=>[...new Set(items.map(item=>item.iteration).filter((v):v is number=>typeof v==='number'))].sort((a,b)=>a-b),[items])
  const current=step!==null&&steps.includes(step)?step:steps[steps.length-1]
  const splits=[...new Set(items.filter(item=>item.iteration===current).map(item=>item.split).filter((v):v is string=>typeof v==='string'))]
  const currentSplit=splits.includes(split)?split:splits[0]
  const views=[...new Set(items.filter(item=>item.iteration===current&&item.split===currentSplit).map(item=>item.view_index).filter((v):v is number=>typeof v==='number'))]
  const metric=sort.includes('psnr')?'psnr':sort.includes('ssim')?'ssim':'lpips'
  const descending=sort==='high_psnr'||sort==='high_ssim'
  const sortedViews=[...views].sort((a,b)=>{
    const av=image(items,current,currentSplit,a,'prediction')?.metrics?.[metric],bv=image(items,current,currentSplit,b,'prediction')?.metrics?.[metric]
    const an=typeof av==='number'?av:null,bn=typeof bv==='number'?bv:null
    if(an===null&&bn===null)return a-b
    if(an===null)return 1
    if(bn===null)return -1
    return descending?bn-an:an-bn
  })
  const currentIndex=views.includes(index)?index:views[0]
  return <><div className="section-head"><div><h2>{t('重建实验室')}</h2><p>{t('Ground Truth、Prediction 与绝对差异来自保存的真实渲染文件。')}</p></div></div>{!items.length?<div className="empty-state"><strong>{t('尚无重建图像')}</strong><span>{t('运行完成预览或评估后，后端提供的图像会在这里出现。')}</span></div>:<><div className="viewer-controls"><label>{t('检查点')}<select value={current??''} onChange={e=>setStep(Number(e.target.value))}>{steps.map(n=><option key={n} value={n}>Iteration {n.toLocaleString()}</option>)}</select></label><label>{t('数据分割')}<select value={currentSplit||''} onChange={e=>{setSplit(e.target.value);setIndex(0)}}>{splits.map(s=><option key={s}>{s}</option>)}</select></label><label>{t('视角索引')}<select value={currentIndex??''} onChange={e=>setIndex(Number(e.target.value))}>{views.map(n=><option key={n} value={n}>{n}</option>)}</select></label><label>{t('缩放')}<input type="range" min="1" max="3" step="0.1" value={zoom} onChange={e=>setZoom(Number(e.target.value))}/></label></div><div className="image-triptych">{[['ground_truth','真实图像'],['prediction','模型预测'],['absolute_difference','绝对差异']].map(([kind,label])=>{const frame=image(items,current,currentSplit,currentIndex,kind);return <div className="panel" key={kind}><div className="image-label"><span>{t(label)}</span><span>{frame?t('真实产物'):t('暂无数据')}</span></div>{frame?.url?<div className="image-window"><img src={api.asset(frame.url)} alt={t('{label}，{split} 视角 {index}',{label:t(label),split:currentSplit,index:currentIndex})} style={{transform:`scale(${zoom})`}}/></div>:<div className="empty-state">{t('暂无数据')}</div>}{frame?.metrics&&<div className="image-metrics">PSNR {fmt(frame.metrics.psnr)} · SSIM {fmt(frame.metrics.ssim)} · LPIPS {fmt(frame.metrics.lpips)}</div>}</div>})}</div><div className="panel timeline"><div className="panel-heading"><h3>{t('训练时间线')}</h3><span>{t('{count} 个检查点',{count:checkpoints.length})}</span></div><div className="timeline-rail">{steps.map(n=><button key={n} className={n===current?'selected':''} onClick={()=>setStep(n)}><i/><span>{n.toLocaleString()}</span></button>)}</div></div><div className="panel"><div className="panel-heading"><h3>{t('验证视角图库')}</h3><label className="gallery-sort">{t('排序')}<select value={sort} onChange={e=>setSort(e.target.value as Sort)}>{(Object.keys(sortLabels) as Sort[]).map(key=><option key={key} value={key}>{t(sortLabels[key])}</option>)}</select></label></div><div className="gallery-grid">{sortedViews.map(n=>{const frame=image(items,current,currentSplit,n,'prediction');return frame?.url?<button key={n} className={`gallery-card ${n===currentIndex?'selected':''}`} onClick={()=>setIndex(n)}><img src={api.asset(frame.url)} alt={t('视角 {index} 预测',{index:n})}/><span>VIEW {String(n).padStart(3,'0')}<br/>PSNR {fmt(frame.metrics?.psnr)} · SSIM {fmt(frame.metrics?.ssim)}<br/>LPIPS {fmt(frame.metrics?.lpips)}</span></button>:null})}</div></div></>}</>
}
