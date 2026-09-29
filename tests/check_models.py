"""Load the actual BPR checkpoint and compare with its archived TorchScript."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, dump
import torch
from bpr.model import ResidualFactorizedPredictor, load_p0, P0Predictor

torch.set_num_threads(1)
torch.manual_seed(2029)
model=ResidualFactorizedPredictor().eval()
model.load_state_dict(torch.load(ROOT/'models/checkpoints/bpr/best.pt',map_location='cpu',weights_only=True),strict=True)
anchor=P0Predictor().eval()
load_p0(anchor,ROOT/'models/checkpoints/paper_p0/PoissonPredictor_best.pt')
for k,v in anchor.state_dict().items():
    assert torch.equal(v,model.anchor.state_dict()[k]),k
# Python's file API also handles non-ASCII checkout paths on Windows.
with (ROOT/'models/checkpoints/bpr/PoissonPredictor_best.jit').open('rb') as stream:
    jit=torch.jit.load(stream,map_location='cpu').eval()
with torch.inference_mode():
    x=torch.rand(8,1,1,23,23,23)+.2
    x=x/x.amax(dim=(-3,-2,-1),keepdim=True)
    y=model(x)
    z=jit(x)
    torch.testing.assert_close(y,z,rtol=1e-5,atol=1e-7)
    assert bool(torch.isfinite(y).all()) and bool((y>=0).all())
    torch.testing.assert_close(y.sum((1,2)),torch.ones(8),rtol=1e-6,atol=1e-6)
result={'pass':True,'anchor_matches_paper_p0':True,'checkpoint_matches_torchscript':True,
        'max_abs_difference':float((y-z).abs().max()),'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
dump(Path.cwd()/'MODEL_CHECK.json',result)
print(result)
