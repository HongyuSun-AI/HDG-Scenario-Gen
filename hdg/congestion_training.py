from copy import deepcopy
import argparse
import fcntl
import os
import itertools
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .models import AxialTemporalDiT
from .diffusion import Diffusion
from .features import Extractor,augment,ntxent,extract
from .full_training import seed_all,torch_write,json_write,fid_report,save_checkpoint
from .representation import flatten_agents,unflatten_agents,history_mask


def frozen_features(data, output, config, device):

    if config.get('feature_method') == 'legacy':
        from .feature_training import train_features
        return train_features(data, output, config, device)
    path=output/'features.pt'; agents=data['tracks'].shape[2]
    model=Extractor(agents*3,128).to(device)
    protocol=dict(agents=agents,hidden=128,epochs=config['feature_epochs'],seed=config['seed'],
        source_sha256=data['metadata']['source_sha256'],input='channel-major; real data only')
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=True)
        if saved['protocol']!=protocol: raise ValueError('Feature protocol changed')
        model.load_state_dict(saved['model'])
    else:
        seed_all(config['seed']); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for epoch in range(config['feature_epochs']):
            json_write(output/'status.json',dict(state='training_feature_extractor',epoch=epoch+1,epochs=config['feature_epochs']))
            loader=DataLoader(data['tracks'],batch_size=128,shuffle=True,drop_last=False,
                generator=torch.Generator().manual_seed(config['seed']+epoch))
            model.train(); start=time.perf_counter(); seen=0
            for x in loader:
                x=x.to(device); opt.zero_grad(set_to_none=True)
                loss=ntxent(model(augment(x)),model(augment(x)))
                if not torch.isfinite(loss): raise ValueError('Nonfinite feature loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); seen+=len(x)
            print(json.dumps(dict(stage='features',epoch=epoch+1,seen=seen,seconds=time.perf_counter()-start)),flush=True)
        torch_write(path,dict(model=model.state_dict(),protocol=protocol))
    return model.eval().requires_grad_(False)


@torch.no_grad()
def real_initial_samples(model,diffusion,data,ids,device,seed,batch=32):
    generator=torch.Generator(device=device).manual_seed(seed); chunks=[]
    for ids_batch in torch.as_tensor(ids).split(batch):
        clean=flatten_agents(data['tracks'][ids_batch].to(device))
        mask=history_mask(torch.ones(len(ids_batch),dtype=torch.long,device=device),model.horizon)
        chunks.append(unflatten_agents(diffusion.sample(model,tuple(clean.shape),
            data['risk'][ids_batch].to(device),clean*mask,mask,data['initial'][ids_batch].to(device),
            generator=generator,unconditional_risk=True)).cpu())
    return torch.cat(chunks)


def same_training_config(previous,current):

    return {k:v for k,v in previous.items() if k!='epochs'}=={k:v for k,v in current.items() if k!='epochs'}


def train(data, output, config, device='cuda'):
    max_epochs=config['epochs']
    device=torch.device(device); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['threads']); seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    n,_,agents,horizon=data['tracks'].shape
    evaluation_epochs=config.get('evaluation_epochs',list(range(1,26)))
    protocol=dict(config=config,source_sha256=data['metadata']['source_sha256'],
        selection=f'Minimum FD; epochs={evaluation_epochs}',
        initialization='256 real starts; unconditional',
        feature=('Frozen original-method GRU' if config.get('feature_method')=='legacy'
                 else 'Frozen 48-agent GRU'),
        reference='Full training data',agents=agents,horizon=horizon)
    protocol_path=output/'training_protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous!=protocol:
            if ({k:v for k,v in previous.items() if k!='config'}!={k:v for k,v in protocol.items() if k!='config'} or
                not same_training_config(previous['config'],config)):
                raise ValueError('Training protocol mismatch')
            with (output/'training_extensions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(previous_epochs=previous['config']['epochs'],epochs=max_epochs,
                    reason='Training extended'))+'\n')
    json_write(protocol_path,protocol)
    feature=frozen_features(data,output,config,device)
    json_write(output/'status.json',dict(state='extracting_real_reference_features',scenarios=n))
    cache=output/'real_features.npy'
    real=np.load(cache) if cache.exists() else extract(feature,data['tracks'],128)
    if not cache.exists(): np.save(cache,real)
    seed_all(config['seed'])
    args=dict(config['temporal'],agents=agents,horizon=horizon,architecture='axial')
    model=AxialTemporalDiT(**args).to(device); ema=deepcopy(model).eval().requires_grad_(False)
    diffusion=Diffusion(**config['diffusion']); opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=0.)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda',init_scale=1024.)
    last=output/'temporal_last.pt'; start_epoch=steps=0
    checkpoint_config=dict(config,temporal=args)
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=True)
        if not same_training_config(saved['config'],checkpoint_config): raise ValueError('Checkpoint configuration differs')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']);start_epoch=saved['epoch'];steps=saved['steps']
        print('RESUME '+json.dumps(dict(epoch=start_epoch,steps=steps,max_epochs=max_epochs,
            model=True,ema=True,optimizer=True,scaler=True)),flush=True)
    ids=np.random.default_rng(config['eval_seed']).choice(n,256,replace=False)
    dataset=TensorDataset(data['tracks'],data['initial'],data['risk'],torch.arange(n))

    def evaluate_epoch(epoch):
        record=output/f'evaluation_epoch_{epoch:04d}.json'
        if record.exists(): report=json.loads(record.read_text())
        else:
            begin=time.perf_counter()
            json_write(output/'status.json',dict(state='evaluating',epoch=epoch,steps=steps,generated_samples=256))
            samples=real_initial_samples(ema,diffusion,data,ids,device,config['eval_seed'],config['eval_batch'])
            report=fid_report(real,extract(feature,samples,128))
            report.update(epoch=epoch,steps=steps,seconds=time.perf_counter()-begin)
            json_write(record,report)
        best=output/'best.json'
        if not best.exists() or report['fid']<json.loads(best.read_text())['fid']:
            torch_write(output/'best_temporal.pt',dict(model=ema.state_dict(),ema=ema.state_dict(),stage='temporal',
                model_args=args,diffusion=config['diffusion'],epoch=epoch,steps=steps,report=report,
                data_metadata=data['metadata'],config=checkpoint_config))
            json_write(best,report)
        print('FID '+json.dumps(report),flush=True)
    if start_epoch in evaluation_epochs: evaluate_epoch(start_epoch)
    epoch_iterator=itertools.count(start_epoch+1) if max_epochs is None else range(start_epoch+1,max_epochs+1)
    for epoch in epoch_iterator:
        schedule=config.get('learning_rate_schedule',{})
        rate=config['learning_rate']
        for boundary,value in sorted(schedule.items(),key=lambda item:int(item[0])):
            if epoch>=int(boundary): rate=value
        for group in opt.param_groups: group['lr']=rate

        seed_all(config['seed']+epoch)
        loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,drop_last=False,num_workers=0,
            generator=torch.Generator().manual_seed(config['seed']+epoch),pin_memory=True)
        seen=torch.zeros(n,dtype=torch.bool); model.train();begin=time.perf_counter();loss_sum=0.;batches=0
        for x,initial,risk,index in loader:
            seen[index]=True;x,initial,risk=x.to(device),initial.to(device),risk.to(device)
            p=torch.randint(horizon,(len(x),),device=device); mode=torch.rand(len(x),device=device)
            p=torch.where(mode<.2,0,torch.where(mode<.4,1,p))
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
            for attempt in range(8):


                torch.set_rng_state(cpu_rng)
                if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    loss=diffusion.loss(model,flatten_agents(x),risk,history_mask(p,horizon),initial)['loss']
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite temporal loss')
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=old_scale: break
            else: raise RuntimeError('AMP retry limit reached')
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,1-config['ema_decay'])
            steps+=1;loss_sum+=float(loss.detach());batches+=1
            if steps%100==0:
                json_write(output/'status.json',dict(state='training',epoch=epoch,steps=steps,seen=int(seen.sum()),total=n))
        record=dict(epoch=epoch,steps=steps,seen=int(seen.sum()),loss=loss_sum/batches,seconds=time.perf_counter()-begin)
        with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        save_checkpoint(last,model,ema,opt,scaler,epoch,steps,checkpoint_config,'temporal')
        if max_epochs is None or epoch in evaluation_epochs:

            import shutil
            shutil.copy2(last,output/f'temporal_epoch_{epoch:04d}.pt')
        if epoch in evaluation_epochs:
            evaluate_epoch(epoch)
    json_write(output/'status.json',dict(state='complete_25_epochs' if max_epochs==25 else 'complete',epochs=max_epochs,steps=steps,
        best=json.loads((output/'best.json').read_text()) if (output/'best.json').exists() else None))
    return output/'best_temporal.pt'


def main():

    from .experiment_data import prepare_congestion
    parser=argparse.ArgumentParser(description='Train temporal model')
    parser.add_argument('--config',default='configs/hdg_night_experiment.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    output=root/'congestion_training';output.mkdir(exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        json_write(output/'launch.json',dict(pid=os.getpid(),mode='training_only',epochs=config['training']['epochs'],
            kappa=config['congestion_kappa'],evaluation_epochs=config['training'].get('evaluation_epochs',list(range(1,26))),closed_loop=False,config=config))
        try:
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
            torch.set_num_threads(config['training']['threads'])
            json_write(output/'status.json',dict(state='preparing_data',kappa=config['congestion_kappa']))
            data=prepare_congestion(config['congestion_source'],output,config['map'],
                config['congestion_kappa'],config['congestion_labels'])
            print(json.dumps(dict(mode='training_only',scenarios=len(data['tracks']),agents=data['tracks'].shape[2],
                horizon=data['tracks'].shape[3],positive_count=int(data['risk'].sum()),epochs=config['training']['epochs'])),flush=True)
            best=train(data,output,config['training'])
            print(f'TRAINING COMPLETE: {best}',flush=True)
        except KeyboardInterrupt:
            state=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
            state.update(state='stopped_by_user',note='Resume completed epoch')
            json_write(output/'status.json',state)
            raise
        except Exception as exc:
            json_write(output/'status.json',dict(state='failed',type=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__': main()
