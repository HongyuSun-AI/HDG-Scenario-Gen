import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .cli import load_model
from .congestion_training import real_initial_samples
from .features import extract
from .full_training import fid_report,json_write,torch_write
from .legacy_feature_refit import original_module,FeatureAdapter
from .relabel import sha256


def run(epoch,config_path,feature_dir):
    config=json.loads(Path(config_path).read_text())
    base=Path(config['output'])/'congestion_training';out=Path(feature_dir)
    state_path=out/f'epoch{epoch}_status.json'
    previous=json.loads((out/'epoch21_fid.json').read_text())
    protocol=previous['protocol'];checkpoint=base/f'temporal_epoch_{epoch:04d}.pt'
    feature_path=out/'rnn_feature_extractor.pth'
    module,source=original_module()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    data=torch.load(base/'data.pt',weights_only=True,map_location='cpu')
    model,diffusion,saved=load_model(checkpoint,device)
    if saved['epoch']!=epoch or model.agents!=48: raise ValueError('Wrong checkpoint')
    ids=np.random.default_rng(protocol['evaluation_seed']).choice(len(data['tracks']),256,replace=False)
    checkpoint_sha=sha256(checkpoint);start=time.perf_counter()
    json_write(state_path,dict(state='sampling',epoch=epoch,steps=saved['steps'],samples=256,pid=os.getpid(),training=False))
    print(f'EVALUATING epoch={epoch}, steps={saved["steps"]}, EMA/GRU samples=256',flush=True)
    samples=real_initial_samples(model,diffusion,data,ids,device,protocol['evaluation_seed'],32)
    if not torch.isfinite(samples).all(): raise ValueError('Nonfinite generated values')
    torch_write(out/f'epoch{epoch}_samples.pt',dict(tracks=samples,source_indices=torch.tensor(ids),
        seed=protocol['evaluation_seed'],checkpoint_sha256=checkpoint_sha))
    json_write(state_path,dict(state='computing_fid',epoch=epoch,pid=os.getpid(),training=False))
    encoder=module.RNNFeatureExtractor(input_dim=144,hidden_dim=128,num_layers=2,use_gru=True).to(device)
    encoder.load_state_dict(torch.load(feature_path,weights_only=True,map_location=device))
    encoder.eval().requires_grad_(False)
    real=np.load(out/'real_features.npy');generated=extract(FeatureAdapter(encoder),samples,128)
    report=fid_report(real,generated)
    report.update(epoch=epoch,steps=saved['steps'],feature_epochs=100,feature_checkpoint=str(feature_path),
        feature_sha256=previous['feature_sha256'],trajectory_checkpoint=str(checkpoint),trajectory_checkpoint_sha256=checkpoint_sha,
        evaluation_seed=protocol['evaluation_seed'],reference_subset_seed=42,
        sampling='EMA; 256 starts; unconditional',
        epoch21_fid=previous['fid'],fid_change_from_epoch21=report['fid']-previous['fid'],
        initial_frame_exact=True,all_values_finite=True,elapsed_seconds=time.perf_counter()-start)
    json_write(out/f'epoch{epoch}_fid.json',report)
    json_write(state_path,dict(state='complete',epoch=epoch,fid=report['fid'],pid=os.getpid(),training=False))
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='Evaluate trajectory checkpoint')
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--config',default='configs/hdg_night_experiment.json')
    p.add_argument('--feature-dir',default='outputs/hdg/congestion_feature_legacy_48')
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    with (Path(config['output'])/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(a.epoch,a.config,a.feature_dir)
        except BaseException as exc:
            json_write(Path(a.feature_dir)/f'epoch{a.epoch}_status.json',dict(state='failed',error=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__': main()
