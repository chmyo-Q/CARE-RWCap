"""Compare CUDA projection/selection and compensation to CPU reference values."""
import ctypes
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, dump
import numpy as np
import torch
import torch_tensorrt

ctypes.CDLL(str(ROOT/'third_party/deeprwcap/runtime/libdnnsolver.so'),mode=ctypes.RTLD_GLOBAL)
lib=ctypes.CDLL(str(ROOT/'build/cpgr.so'))
ptr=ctypes.c_void_p
fn=lib.joint_selector_launch
fn.argtypes=[ptr]*3+[ctypes.c_int,ptr]
fn.restype=ctypes.c_int
rng=np.random.default_rng(9142026)
x=rng.uniform(.2,1,(128,23,23,23)).astype(np.float32)
mask=np.arange(len(x))%8
for axis in range(3):
    ix=(mask&(1<<axis))!=0
    for k in range(11):
        aa=[slice(None)]*4
        bb=aa.copy()
        aa[3-axis]=22-k
        bb[3-axis]=k
        subset=x[ix]
        subset[tuple(aa)]=subset[tuple(bb)]
        x[ix]=subset
fpw=rng.uniform(.01,1,(len(x),7)).astype(np.float32)
fpw[:,:6]/=fpw[:,:6].sum(1,keepdims=True)
expected=fpw.copy()
for axis,(a,b) in enumerate([(4,5),(2,3),(0,1)]):
    ix=(mask&(1<<axis))!=0
    v=(fpw[ix,a]+fpw[ix,b])*.5
    expected[ix,a]=v
    expected[ix,b]=v
xx=torch.from_numpy(x).cuda()
ff=torch.from_numpy(fpw).cuda()
out=torch.empty_like(ff)
stream=torch.cuda.current_stream().cuda_stream
assert fn(xx.data_ptr(),ff.data_ptr(),out.data_ptr(),len(x),stream)==0
torch.cuda.synchronize()
actual=out.cpu().numpy()
assert np.array_equal(actual,expected)
assert np.array_equal(actual[:,6],fpw[:,6])
assert np.array_equal(actual[mask==0],fpw[mask==0])
sum_error=float(np.max(np.abs(actual[:,:6].sum(1)-fpw[:,:6].sum(1))))
assert sum_error<3e-7

project=lib.parity_launch
project.argtypes=[ptr]*5+[ctypes.c_int,ptr]
project.restype=ctypes.c_int
# Cover all eight layout masks and both gradient face types independently.
x=np.concatenate([x,x],axis=0)
mask=np.tile(mask,2)
faces=np.concatenate([np.zeros(len(x)//2,dtype=np.uint8),np.full(len(x)//2,2,dtype=np.uint8)])
g=rng.normal(size=(len(x),23,23)).astype(np.float32)
# Exact zero after odd projection: implementation uses original proposal with zero weight.
zero_index=next(i for i in range(len(x)) if faces[i]>=2 and mask[i]&4)
g[zero_index]=1
projected=np.empty_like(g)
kernel_mask=np.zeros(len(x),dtype=np.uint8)
for i,(q,m,face) in enumerate(zip(g,mask,faces)):
    m1=bool(m&1)
    m2=bool(m&(4 if face>=2 else 2))
    kernel_mask[i]=int(m1)+2*int(m2)
    sign=-1. if face>=2 else 1.
    if m1 and m2: v=((q+q[:,::-1])+sign*(q[::-1,:]+q[::-1,::-1]))*.25
    elif m1: v=(q+q[:,::-1])*.5
    elif m2: v=(q+sign*q[::-1,:])*.5
    else: v=q
    projected[i]=v
l0=np.abs(g).sum((1,2),dtype=np.float64)
l1=np.abs(projected).sum((1,2),dtype=np.float64)
expected_ratio=np.where(kernel_mask!=0,l1/l0,1.)
expected_proposal=np.where((l1>0)[:,None,None],projected,g)
xx=torch.from_numpy(x).cuda()
gg=torch.from_numpy(g).cuda()
faces_gpu=torch.from_numpy(faces).cuda()
out=torch.empty_like(gg)
ratio=torch.empty(len(x),device='cuda')
assert project(xx.data_ptr(),gg.data_ptr(),faces_gpu.data_ptr(),out.data_ptr(),ratio.data_ptr(),len(x),stream)==0
torch.cuda.synchronize()
actual=out.cpu().numpy()
rat=ratio.cpu().numpy()
np.testing.assert_allclose(actual,expected_proposal,rtol=1e-6,atol=1e-7)
np.testing.assert_allclose(rat,expected_ratio,rtol=1e-6,atol=1e-7)
assert rat[zero_index]==0
assert np.array_equal(actual[kernel_mask==0],g[kernel_mask==0])
# Exhaustive expectation, no Monte Carlo noise:
# sampling |Pg| / ||Pg||_1, using sign(Pg)*||Pg||_1/||g||_1,
# must give Pg/||g||_1. This includes the zero-projection fallback.
proposal=np.abs(actual).astype(np.float64)
prob=proposal/proposal.sum((1,2),keepdims=True)
weighted=prob*np.sign(actual)*rat[:,None,None]
np.testing.assert_allclose(weighted,projected/l0[:,None,None],rtol=2e-6,atol=1e-9)
result={'pass':True,'selector_all_8_masks':True,'projection_all_4_masks_both_face_types':True,
    'zero_projection_compensation':True,'sampling_expectation_checked':True,
    'selector_numpy_bitwise_equal':True,'max_selector_sum_error':sum_error,
    'max_ratio_error':float(np.max(np.abs(rat-expected_ratio)))}
dump(Path.cwd()/'KERNEL_CHECK.json',result)
print(result)
