"""Final BPR loss and validation selection rule; no dataset/training driver."""
import math
import torch


def probes_2d(device):
    c = torch.linspace(-1, 1, 23, device=device)
    yy, xx = torch.meshgrid(c, c, indexing='ij')
    probes = torch.stack([xx, yy, xx*yy, torch.cos(torch.pi*xx), torch.cos(torch.pi*yy),
                          torch.cos(torch.pi*xx)*torch.cos(torch.pi*yy)]).reshape(6,529)
    probes = probes-probes.mean(1, keepdim=True)
    return probes/probes.abs().amax(1, keepdim=True).clamp_min(1e-12)


def components(pred, target, p0):
    kl = (target*(target.clamp_min(1e-12).log()-pred.clamp_min(1e-12).log())).sum((1,2))
    action = ((pred-target).flatten(1) @ probes_2d(pred.device).t()).square().mean(1)
    trust = (pred*(pred.clamp_min(1e-12).log()-p0.clamp_min(1e-12).log())).sum((1,2))
    return kl, action, trust


def bpr_loss(pred, target, p0):
    kl, action, trust = components(pred, target, p0)
    primary = kl + 10*action + 2*trust
    k = max(1, math.ceil(primary.numel()*.25))
    return primary.mean() + .5*torch.topk(primary, k).values.mean()


def validation_score(kl, action, trust):
    """Arguments are per-sample metrics for the entire validation set."""
    return kl.mean()+10*action.mean()+2*trust.mean()+.5*(torch.quantile(kl,.95)+10*torch.quantile(action,.95))
