import hashlib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from .representation import segments,flatten_agents,unflatten_agents


def smooth_future(tracks,cut,mode,sigma=2.):
    # C,N,T
    out=np.array(tracks,copy=True)
    valid=np.isfinite(tracks).all(0)&(tracks>=-.1).all(0)
    for agent in range(tracks.shape[1]):
        for a,b in segments(valid[agent]):
            start=a if mode=='joint' else max(a,cut)
            write=max(a,cut)
            if write>=b:continue
            for channel in [0,1]:
                filtered=gaussian_filter1d(tracks[channel,agent,start:b].astype(np.float64),sigma,mode='reflect',truncate=4.)
                out[channel,agent,write:b]=filtered[write-start:]
    return out


class SmoothedDiffusion:
    def __init__(self,diffusion,mode):
        self.diffusion,self.mode=diffusion,mode;self.first_raw_sha256=None
    def sample(self,model,shape,risk,history,mask,initial,**kwargs):
        raw=self.diffusion.sample(model,shape,risk,history,mask,initial,**kwargs)
        tracks=unflatten_agents(raw).cpu().numpy()
        if self.first_raw_sha256 is None:self.first_raw_sha256=hashlib.sha256(tracks.tobytes()).hexdigest()
        cuts=mask[:,0].sum(-1).to(torch.int64).cpu().tolist()
        filtered=np.stack([smooth_future(x,cut,self.mode) for x,cut in zip(tracks,cuts)])
        return flatten_agents(torch.from_numpy(filtered).to(raw.device,dtype=raw.dtype))
