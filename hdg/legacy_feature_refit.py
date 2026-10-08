import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .cli import load_model
from .features import Extractor,extract
from .full_training import seed_all,torch_write,json_write,fid_report
from .congestion_training import real_initial_samples
from .relabel import sha256


def original_module():
    path=Path(__file__).resolve().parents[1]/'data_process/python/Contrastive_Learning.py'
    spec=importlib.util.spec_from_file_location('hdg_original_contrastive',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


class FeatureAdapter(torch.nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
    def forward(self,tracks):
        return self.encoder(tracks.flatten(1,2).transpose(1,2))


def run(config_path,output):
    config=json.loads(Path(config_path).read_text());base=Path(config['output'])/'congestion_training'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    module,source_path=original_module()
    checkpoint=base/'temporal_epoch_0021.pt'
    data=torch.load(base/'data.pt',weights_only=True,map_location='cpu')
    protocol=dict(source=str(source_path),source_sha256=sha256(source_path),
        data_source_sha256=data['metadata']['source_sha256'],trajectory_checkpoint=str(checkpoint),
        trajectory_checkpoint_sha256=sha256(checkpoint),input_dim=144,hidden_dim=128,layers=2,
        bidirectional=True,output_dim=128,epochs=100,batch_size=128,optimizer='Adam',lr=.001,
        temperature=.5,seed=20261006,augmentation='Gaussian augmentation, sigma=0.01, including padding',
        loss='Original contrastive loss',
        input_order='Channel-major, duration-sorted agents',
        evaluation_seed=config['training']['eval_seed'],generated_count=256,reference_subsets=5,
        reference_subset_seed=42,sampling='Epoch21 EMA, real starts, unconditional',
        training_only_real_data=True)
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Refit protocol mismatch')
    json_write(protocol_path,protocol)
    seed_all(protocol['seed'])
    model=module.RNNFeatureExtractor(input_dim=144,hidden_dim=128,num_layers=2,use_gru=True).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    dataset=module.ModifiedDataset(TensorDataset(data['tracks'],data['risk']))
    loader=DataLoader(dataset,batch_size=128,shuffle=True)
    start_epoch=0;last=output/'last.pt'
    if last.exists():
        saved=torch.load(last,weights_only=True,map_location=device)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start_epoch=saved['epoch']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    begin=time.perf_counter()
    for epoch in range(start_epoch+1,101):
        model.train();total_loss=0.;seen=0;start=time.perf_counter()
        for x,augmented in loader:
            x,augmented=x.to(device),augmented.to(device)
            features=model(torch.cat([x,augmented],dim=0))
            loss=module.contrastive_loss(features,.5)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite original-method feature loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            total_loss+=float(loss.detach());seen+=len(x)
        row=dict(epoch=epoch,loss=total_loss/len(loader),seen=seen,seconds=time.perf_counter()-start)
        with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        torch_write(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        json_write(output/'status.json',dict(state='training_features',**row,epochs=100,pid=os.getpid()))
        print('FEATURE '+json.dumps(row),flush=True)
    torch_write(output/'rnn_feature_extractor.pth',model.state_dict())
    adapter=FeatureAdapter(model.eval().requires_grad_(False))
    json_write(output/'status.json',dict(state='extracting_real_features',pid=os.getpid()))
    real=extract(adapter,data['tracks'],128);np.save(output/'real_features.npy',real)
    sample_path=output/'epoch21_samples.pt'
    ids=np.random.default_rng(protocol['evaluation_seed']).choice(len(data['tracks']),256,replace=False)
    if sample_path.exists():
        samples=torch.load(sample_path,weights_only=True,map_location='cpu')['tracks']
    else:
        json_write(output/'status.json',dict(state='sampling_epoch21',samples=256,pid=os.getpid()))
        temporal,diffusion,_=load_model(checkpoint,device)
        samples=real_initial_samples(temporal,diffusion,data,ids,device,protocol['evaluation_seed'],32)
        torch_write(sample_path,dict(tracks=samples,source_indices=torch.tensor(ids),seed=protocol['evaluation_seed'],
            checkpoint_sha256=protocol['trajectory_checkpoint_sha256']))
        del temporal,diffusion
    json_write(output/'status.json',dict(state='computing_fid',pid=os.getpid()))
    generated=extract(adapter,samples,128)
    report=fid_report(real,generated)


    old=Extractor(144,128).to(device)
    old.load_state_dict(torch.load(base/'features.pt',weights_only=True,map_location=device)['model'])
    old_report=fid_report(np.load(base/'real_features.npy'),extract(old,samples,128))
    recorded=json.loads((base/'evaluation_epoch_0021.json').read_text())
    def stats(values):
        return dict(mean_feature_norm=float(np.linalg.norm(values,axis=1).mean()),
                    covariance_trace=float(np.var(values,axis=0,ddof=1).sum()))
    report.update(epoch=21,steps=35448,feature_epochs=100,protocol=protocol,
        feature_checkpoint=str(output/'rnn_feature_extractor.pth'),feature_sha256=sha256(output/'rnn_feature_extractor.pth'),
        old_encoder_same_samples=old_report,old_recorded_fid=recorded['fid'],
        old_fid_reproduction_absolute_error=abs(old_report['fid']-recorded['fid']),
        real_feature_statistics=stats(real),generated_feature_statistics=stats(generated),
        elapsed_seconds=time.perf_counter()-begin)
    json_write(output/'epoch21_fid.json',report)
    json_write(output/'status.json',dict(state='complete',epoch=21,fid=report['fid'],feature_epochs=100,pid=os.getpid()))
    print('RESULT '+json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description='Refit 48-agent GRU')
    parser.add_argument('--config',default='configs/hdg_night_experiment.json')
    parser.add_argument('--output',default='outputs/hdg/congestion_feature_legacy_48')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text())
    with (Path(config['output'])/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(args.config,output)
        except BaseException as exc:
            json_write(output/'status.json',dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                error=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__': main()
