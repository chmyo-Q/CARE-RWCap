"""Frozen deployment metric definitions; ported from the validated evaluator.

Only file locations and CLI are adapted. No training, labels or thresholds change.
GPU imports are kept out of the public CLI's --help and preflight paths.
"""
from pathlib import Path
import ctypes,csv,gzip,hashlib,json
import numpy as np
import torch
import torch_tensorrt
import tensorrt
from common import ROOT
N=23
DL=N**3
GL=6*N*N
SL=DL+112+GL
EPS=1e-12
DATA=None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
 return h.hexdigest()


def dump(path, obj):
 Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows):
 opener=gzip.open if str(path).endswith('.gz') else open
 with opener(path,'wt',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(v):
 a=np.asarray(v,dtype=np.float64)
 if not a.size:return {'n':0}
 assert np.isfinite(a).all()
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)),max=float(a.max()))


def change(raw, new):return 100*(new/raw-1) if raw else None


class Dataset:
 def __init__(self,path):
  self.path=path
  m=np.memmap(path,dtype='<f8',mode='r')
  assert m[:2].tolist()==[23.,1.] and m.size==2+100000*SL
  self.rows=m[2:].reshape(100000,SL)
 def read(self,ids):
  r=self.rows[ids]
  x=np.asarray(r[:,:DL],np.float32).reshape(-1,N,N,N).copy()
  maximum=x.max((1,2,3),keepdims=True)
  x/=np.where(maximum==0,1,maximum)
  y=np.asarray(r[:,-GL:],np.float64).reshape(-1,6,N,N).copy()
  assert np.isfinite(x).all() and np.isfinite(y).all()
  return x,y


def masks(x,side):
 a=np.all(x==x[:,:,:,::-1],axis=(1,2,3))
 b=np.all(x==(x[:,::-1,:,:] if side else x[:,:,::-1,:]),axis=(1,2,3))
 return a.astype(np.uint8)+2*b.astype(np.uint8)


def project(q,mask,side):
 out=q.copy();sign=-1 if side else 1
 for k in (1,2,3):
  ii=mask==k;a=q[ii]
  if k==1:out[ii]=(a+a[:,:,::-1])*.5
  elif k==2:out[ii]=(a+sign*a[:,::-1,:])*.5
  else:out[ii]=((a+a[:,:,::-1])+sign*(a[:,::-1,:]+a[:,::-1,::-1]))*.25
 return out


def transform(x,face):
 if face==0:return x[:,::-1,:,:].copy()
 if face in (1,2):return x.copy()
 if face==3:return x[:,:,::-1,:].copy()
 if face==4:return x.transpose(0,1,3,2).copy()
 return np.rot90(x,1,axes=(2,3)).copy()


class Production:
 def __init__(self):
  ctypes.CDLL(str(ROOT/'third_party/deeprwcap/runtime/libdnnsolver.so'),mode=ctypes.RTLD_GLOBAL)
  self.lib=ctypes.CDLL(str(ROOT/'build/cpgr.so'))
  self.parity=self.lib.parity_launch
  self.parity.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];self.parity.restype=ctypes.c_int
  self.selector=self.lib.joint_selector_launch
  self.selector.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p];self.selector.restype=ctypes.c_int
  self.gaps=[];self.ratio_gaps=[];self.compensation_gaps=[]
 def kernel(self,x,q,face):
  xx=torch.from_numpy(np.ascontiguousarray(x)).cuda();qq=q.contiguous()
  ff=torch.full((len(x),),face,dtype=torch.uint8,device='cuda')
  out=torch.empty_like(qq);ratio=torch.empty(len(x),device='cuda')
  rc=self.parity(xx.data_ptr(),qq.data_ptr(),ff.data_ptr(),out.data_ptr(),ratio.data_ptr(),len(x),torch.cuda.current_stream().cuda_stream)
  assert rc==0
  raw=qq.cpu().numpy();got=out.cpu().numpy();rho=ratio.cpu().numpy();mask=masks(x,face>=2)
  expected=project(raw,mask,face>=2);mass=np.abs(expected.astype(float)).sum((1,2));old=np.abs(raw.astype(float)).sum((1,2))
  fallback=expected.copy();fallback[mass==0]=raw[mass==0]
  gap=float(np.max(np.abs(got-fallback)));self.gaps.append(gap);assert gap==0
  er=np.where(mask!=0,mass/old,1.)
  rg=float(np.max(np.abs(rho-er)));self.ratio_gaps.append(rg);assert rg<1e-6
  # Zero projection retains proposal but contributes zero in production.
  effective=got.astype(float)/np.abs(got.astype(float)).sum((1,2),keepdims=True)*rho[:,None,None]
  expected_effective=expected.astype(float)/old[:,None,None]
  cg=float(np.max(np.abs(effective-expected_effective)));self.compensation_gaps.append(cg);assert cg<1e-7
  return raw.astype(float),expected.astype(float),rho.astype(float),mask
 def faces(self,x,fpw):
  xx=torch.from_numpy(np.ascontiguousarray(x)).cuda();out=torch.empty_like(fpw)
  assert self.selector(xx.data_ptr(),fpw.data_ptr(),out.data_ptr(),len(x),torch.cuda.current_stream().cuda_stream)==0
  raw=fpw.cpu().numpy();got=out.cpu().numpy();expected=raw.copy();active=np.zeros(len(x),bool)
  for ax,(a,b) in enumerate(((4,5),(2,3),(0,1))):
   use=np.all(x==np.flip(x,axis=3-ax),axis=(1,2,3));active|=use
   expected[use,a]=expected[use,b]=(raw[use,a]+raw[use,b])*.5
  assert np.array_equal(got,expected) and np.array_equal(raw[:,6],got[:,6])
  return raw.astype(float),got.astype(float),active


@torch.inference_mode()
def infer(net,x):
 t=torch.from_numpy(np.ascontiguousarray(x)).cuda().view(-1,1,1,N,N,N)
 y=net(t).float()
 assert torch.isfinite(y).all()
 return y


def load(relative):return torch.jit.load(str(ROOT/'models'/relative),map_location='cuda').eval()


def probes():
 c=torch.linspace(-1,1,23,device='cuda');yy,xx=torch.meshgrid(c,c,indexing='ij')
 p=torch.stack([xx,yy,xx*yy,torch.cos(torch.pi*xx),torch.cos(torch.pi*yy),torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy)]).reshape(6,529)
 p-=p.mean(1,keepdim=True);p/=p.abs().amax(1,keepdim=True)
 return p.cpu().numpy().astype(float)


def poisson_metrics(p,t,basis):
 return (t*(np.log(np.maximum(t,EPS))-np.log(np.maximum(p,EPS)))).sum((1,2)),np.square((p-t).reshape(-1,529)@basis.T).mean(1)


def evaluate_bpr(ids,out,batch):
 ds=Dataset(DATA/'poisson.bin');p0=load('paper_p0/PoissonPredictor_tensorrt_fp16.jit');bpr=load('bpr/PoissonPredictor_tensorrt_fp16.jit')
 basis=probes();rows=[];maxnorm=0.
 for start in range(0,len(ids),batch):
  ix=ids[start:start+batch];x,exact=ds.read(ix)
  t=np.abs(exact[:,0])+1e-10;t=(t/t.sum((1,2),keepdims=True)).astype(np.float32).astype(float)
  a=infer(p0,x).cpu().numpy().astype(float);b=infer(bpr,x).cpu().numpy().astype(float)
  assert a.shape==b.shape==t.shape and a.min()>=0 and b.min()>=0
  maxnorm=max(maxnorm,float(abs(a.sum((1,2))-1).max()),float(abs(b.sum((1,2))-1).max()))
  # Production torch::multinomial treats engine outputs as nonnegative weights.
  # Report its actual transition probabilities; keep legacy direct-output
  # formulas separately rather than silently calling unnormalised weights KL.
  la,laa=poisson_metrics(a,t,basis);lb,lba=poisson_metrics(b,t,basis)
  sa=a.sum((1,2),keepdims=True);sb=b.sum((1,2),keepdims=True)
  assert (sa>0).all() and (sb>0).all()
  a=a/sa;b=b/sb;t=t/t.sum((1,2),keepdims=True)
  ka,aa=poisson_metrics(a,t,basis);kb,ab=poisson_metrics(b,t,basis);tv=.5*abs(a-b).sum((1,2))
  for j,i in enumerate(ix):rows.append(dict(dataset_index=int(i),raw_kl=float(ka[j]),bpr_kl=float(kb[j]),raw_action_error=float(aa[j]),bpr_action_error=float(ab[j]),bpr_vs_p0_tv=float(tv[j]),raw_output_sum=float(sa[j,0,0]),bpr_output_sum=float(sb[j,0,0]),legacy_direct_raw_kl_formula=float(la[j]),legacy_direct_bpr_kl_formula=float(lb[j]),legacy_direct_raw_action=float(laa[j]),legacy_direct_bpr_action=float(lba[j])))
  if start%1024==0:print('BPR',start+len(ix),'/',len(ids),flush=True)
 write_csv(out/'bpr_samples.csv.gz',rows)
 v={k:stats([r[k] for r in rows]) for k in rows[0] if k!='dataset_index'}
 v.update(kl_relative_change_percent=change(v['raw_kl']['mean'],v['bpr_kl']['mean']),action_relative_change_percent=change(v['raw_action_error']['mean'],v['bpr_action_error']['mean']),normalization_max_error=maxnorm)
 del p0,bpr;torch.cuda.empty_cache();return v


def kernel_rows(ids,face,raw,proj,exact,mask):
 norm=np.linalg.norm(exact.reshape(-1,529),axis=1)
 ev=np.linalg.norm((exact-project(exact,mask,face>=2)).reshape(-1,529),axis=1)/(norm+EPS)
 pv=np.linalg.norm((raw-proj).reshape(-1,529),axis=1)/(np.linalg.norm(raw.reshape(-1,529),axis=1)+EPS)
 ppv=np.linalg.norm((proj-project(proj,mask,face>=2)).reshape(-1,529),axis=1)/(np.linalg.norm(proj.reshape(-1,529),axis=1)+EPS)
 er=np.linalg.norm((raw-exact).reshape(-1,529),axis=1)/(norm+EPS)
 ep=np.linalg.norm((proj-exact).reshape(-1,529),axis=1)/(norm+EPS)
 return [dict(dataset_index=int(i),head='Gradient1' if face==1 else 'Gradient2',face=face,mask=int(mask[j]),active=bool(mask[j]),raw_l2=float(er[j]),cpgr_l2=float(ep[j]),raw_parity=float(pv[j]),cpgr_parity=float(ppv[j]),exact_parity=float(ev[j]),l2_win=bool(ep[j]<er[j])) for j,i in enumerate(ids)]


def summarize_kernel(rows):
 active=[r for r in rows if r['active']]
 assert active
 s=dict(validation_samples=len(rows),activated_samples=len(active),activation_ratio=len(active)/len(rows))
 for k in ('raw_l2','cpgr_l2','raw_parity','cpgr_parity','exact_parity'):s[k]=stats([r[k] for r in active])
 s['l2_relative_change_percent']=change(s['raw_l2']['mean'],s['cpgr_l2']['mean'])
 s['l2_win_rate']=sum(r['l2_win'] for r in active)/len(active)
 s['l2_loss_rate']=sum(r['cpgr_l2']>r['raw_l2'] for r in active)/len(active)
 s['mask_counts']={str(m):sum(r['mask']==m for r in rows) for m in (0,1,2,3)}
 s['all_validation_reference_only']={k:stats([r[k] for r in rows]) for k in ('raw_l2','cpgr_l2')}
 return s


def evaluate_gradient(ids,out,batch):
 ds=Dataset(DATA/'gradient.bin');g1=load('paper_p0/Gradient1Predictor_tensorrt_fp16.jit');g2=load('paper_p0/Gradient2Predictor_tensorrt_fp16.jit');sel=load('paper_p0/GradientSelectorWeight_tensorrt_fp16.jit');prod=Production()
 rows=[];wrows=[];normgap=0.
 for start in range(0,len(ids),batch):
  ix=ids[start:start+batch];x,exact=ds.read(ix)
  masses=abs(exact).sum((2,3));assert (masses>0).all()
  rawf,newf,selector_active=prod.faces(x,infer(sel,x))
  assert (rawf[:,:6]>=0).all() and (rawf[:,6]>0).all()
  f=rawf[:,:6]/rawf[:,:6].sum(1,keepdims=True);fc=newf[:,:6]/newf[:,:6].sum(1,keepdims=True)
  ratios=[];any_active=selector_active.copy()
  for face in range(6):
   xx=transform(x,face);q=infer(g1 if face<2 else g2,xx)
   raw,pg,rho,mask=prod.kernel(xx,q,face)
   normgap=max(normgap,float(abs(abs(raw).sum((1,2))-1).max()));assert normgap<1e-4
   ratios.append(rho);any_active|=mask!=0
   if face in (1,2):
    label=exact[:,face]/masses[:,face,None,None]
    rows.extend(kernel_rows(ix,face,raw,pg,label,mask))
  rho=np.stack(ratios,1)
  mr=rawf[:,6,None]*f;mc=rawf[:,6,None]*fc*rho
  wr=abs(mr-masses).sum(1)/masses.sum(1);wc=abs(mc-masses).sum(1)/masses.sum(1)
  for j,i in enumerate(ix):wrows.append(dict(dataset_index=int(i),active=bool(any_active[j]),selector_active=bool(selector_active[j]),raw_face_mass_nl1=float(wr[j]),cpgr_face_mass_nl1=float(wc[j]),raw_global_weight=float(rawf[j,6]),cpgr_global_weight=float(newf[j,6]),exact_total_mass=float(masses[j].sum())))
  if start%1024==0:print('GRADIENT',start+len(ix),'/',len(ids),flush=True)
 rows.sort(key=lambda r:(r['dataset_index'],r['face']))
 write_csv(out/'gradient_samples.csv.gz',rows);write_csv(out/'weight_samples.csv.gz',wrows)
 models={h:summarize_kernel([r for r in rows if r['head']==h]) for h in ('Gradient1','Gradient2')}
 ws={}
 for scope,rr in [('active',[r for r in wrows if r['active']]),('all_reference_only',wrows)]:
  a=stats([r['raw_face_mass_nl1'] for r in rr]);b=stats([r['cpgr_face_mass_nl1'] for r in rr])
  ws[scope]=dict(samples=len(rr),raw_error=a,cpgr_error=b,relative_change_percent=change(a['mean'],b['mean']))
 ws['metric']='normalized L1 of effective six-face absolute mass: sum|W F_f rho_f - exact_face_mass_f| / exact_total_mass; raw rho=1'
 ws['global_weight_unchanged']=all(r['raw_global_weight']==r['cpgr_global_weight'] for r in wrows)
 ws['activation_ratio']=ws['active']['samples']/len(wrows)
 ws['selector_active_samples']=sum(r['selector_active'] for r in wrows)
 checks=dict(projection_bitwise_matches_source_formula=max(prod.gaps)==0,projection_max_gap=max(prod.gaps),rho_max_gap=max(prod.ratio_gaps),compensated_measure_max_gap=max(prod.compensation_gaps),raw_kernel_l1_max_error=normgap,selector_bitwise_matches_source_formula=True)
 return models,ws,checks


