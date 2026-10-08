import json
from pathlib import Path
import time
import numpy as np
import torch
from .cli import load_model
from .full_training import json_write,torch_write
from .relabel import sha256
from .road import MergeRoad
from .batched_loop import run_batch
from .closed_loop import verify_fixed_replay, evaluation_end
from .experiment_data import select_initializations,initialization
from .metrics import pool_episodes,continuity_features,frechet,minimum_ttc
from .representation import decode


def arrays_to_tensors(value):
    if isinstance(value,np.ndarray): return torch.from_numpy(value.copy())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:arrays_to_tensors(v) for k,v in value.items()}
    if isinstance(value,list): return [arrays_to_tensors(v) for v in value]
    return value


def episode_arrays(value):
    result=dict(value)
    for key in ['tracks','generation_windows']:
        if isinstance(result.get(key),torch.Tensor): result[key]=result[key].numpy()
    return result


def boundary_statistics(episodes,references):
    real=[[],[]]; generated=[[],[]]
    for e,ref in zip(episodes,references):
        terminal=evaluation_end(e,140)-1
        boundaries=[b for b in e['boundaries'] if 1<b<terminal-2]


        for target,features in [(real,continuity_features(ref[None],boundaries)),
                                (generated,continuity_features(e['tracks'][None],boundaries))]:
            for i in range(2): target[i].append(features[i])
    real=[np.concatenate(x) if x else np.empty((0,2)) for x in real]
    generated=[np.concatenate(x) if x else np.empty((0,2)) for x in generated]
    def moments(x):
        return dict(n=len(x),sum=x.sum(0).tolist(),outer=(x.T@x).tolist())
    counts=dict(real_boundary_samples=[len(x) for x in real],generated_boundary_samples=[len(x) for x in generated],
                moments={'real':[moments(x) for x in real],'generated':[moments(x) for x in generated]})
    if any(len(x)<2 for x in real+generated): return dict(kcs=None,reason='Insufficient boundary samples',**counts)
    distance=[frechet(r,g)/max(float(np.var(r,axis=0,ddof=1).sum()),1e-6) for r,g in zip(real,generated)]
    return dict(kcs=float(np.exp(-.5*(.7*distance[0]+.3*distance[1]))),
        acceleration_fd_normalized=distance[0],jerk_fd_normalized=distance[1],**counts,
        convention='Executed trajectories, regeneration-boundary derivatives')


def summarize(episodes,references):
    counts=pool_episodes(episodes); ttc=[]
    for e in episodes:
        end=evaluation_end(e,140)
        s,v,_=decode(e['tracks'][:,:,:end].transpose(1,2,0))
        ttc.append(minimum_ttc(s,v))
    values=[v for v in ttc if v is not None]
    continuity=boundary_statistics(episodes,references)
    return dict(table2={'KCS':continuity['kcs'],'TTC_mean_s':float(np.mean(values)) if values else None,
        'TTC_std_s':float(np.std(values,ddof=0)) if values else None,'Coll_pct':counts['collision_pct'],
        'ACR_pct':counts['acr_pct'],'VARC_pct':counts['varc_pct'],'VARAF_pct':counts['varaf_pct'],'VAFR_pct':counts['vafr_pct']},
        counts=counts,exited_episodes=sum(e.get('exit_time') is not None for e in episodes),episode_counts=[dict(collision=e['collision'],at_fault=e['at_fault'],verified=e['verified']) for e in episodes],
        continuity=continuity,ttc=dict(values=ttc,finite_count=len(values),no_intersection_count=len(ttc)-len(values),
        convention='Episode-minimum TTC, 10s sweep, ddof=0'),
        history_max_error=max(e['history_max_error'] for e in episodes),
        invalid_generated_targets=sum(e['invalid_generated_targets'] for e in episodes))


def run_dataset(data,checkpoint,output,config,device='cuda'):
    from .pdm_planner import PDMClosedAdapter
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    method_files=['batched_loop.py','night_experiment.py','experiment_data.py','closed_loop.py','pdm_planner.py','pdm_map.py','pdm_replay.py','metrics.py','collision_attribution.py','generation_smoothing.py','trajectory_optimization.py','road_projection.py','sv_front_guard.py','sv_lifecycle.py','sv_stable.py','sv_diagnostics.py']
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        method_sha256={name:sha256(Path(__file__).parent/name) for name in method_files},
        data_source_sha256=data['metadata']['source_sha256'],map_sha256=sha256(data['metadata']['map_path']),
        config=config,planner=PDMClosedAdapter(MergeRoad(data['metadata']['map_path'])).describe())
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Experiment protocol changed')
    json_write(path,protocol)
    manifest_path=out/'initializations.json'
    if manifest_path.exists(): manifest=json.loads(manifest_path.read_text())
    else:
        manifest=select_initializations(data,config['groups'],config['seed'],config.get('ego_x_fraction_max'));json_write(manifest_path,manifest)
    model,diffusion,_=load_model(checkpoint,torch.device(device))
    road=MergeRoad(data['metadata']['map_path']); factory=lambda:PDMClosedAdapter(road)
    if config.get('postprocessing') == 'optimized':
        from .trajectory_optimization import OptimizedDiffusion
        diffusion=OptimizedDiffusion(diffusion,'optimized',road)
    episodes=[];references=[];begin=time.perf_counter()
    for item in manifest['groups']:
        initial,first,speeds,ego,goal,reference=initialization(data,item)
        json_write(manifest_path,manifest)
        group=out/f"group_{item['group']:02d}";group.mkdir(exist_ok=True)
        generated_path=group/'generated.pt'
        if generated_path.exists():
            saved=torch.load(generated_path,weights_only=True,map_location='cpu')
            batch=[episode_arrays(x) for x in saved['episodes']]
        else:
            def progress(value):
                json_write(out/'status.json',dict(state='generating',group=item['group'],**value))
            batch,timing=run_batch(model,diffusion,initial,first,speeds,ego,goal,factory,
                batch_size=config['batch_size'],execution=config['execution'],seed=item['seed'],
                risk=config['risk_condition'],cfg=config['cfg'],on_progress=progress,sv_controller=config.get('sv_controller','front_guard'))
            torch_write(generated_path,arrays_to_tensors(dict(episodes=batch,timing=timing)))
        for index,episode in enumerate(batch):
            path=group/f'episode_{index:02d}.pt'
            if path.exists(): episode=episode_arrays(torch.load(path,weights_only=True,map_location='cpu'))
            else:
                verification=verify_fixed_replay(episode,speeds,factory(),goal)
                episode.update(verification=verification,verified=verification['verified'],
                    group=item['group'],episode=index,initialization=item)
                torch_write(path,arrays_to_tensors(episode))
            if episode['preflight_only'] or episode['executed_frames']!=140: raise ValueError('Partial episode in formal report')
            episodes.append(episode);references.append(reference)
            json_write(out/'status.json',dict(state='verifying',completed=len(episodes),
                expected=config['groups']*config['batch_size'],elapsed_seconds=time.perf_counter()-begin))
            print(f"{out.name}: verified {len(episodes)}/{config['groups']*config['batch_size']}",flush=True)
    report=summarize(episodes,references);report.update(protocol=protocol,elapsed_this_run_seconds=time.perf_counter()-begin)
    if config.get('postprocessing') == 'optimized':
        report['optimization_audits']=diffusion.audits
    if config.get('kcs_scales'):
        continuity=report['continuity'];distances=[]
        for i,key in enumerate(['acceleration_fd_normalized','jerk_fd_normalized']):
            if key not in continuity: break
            m=continuity['moments']['real'][i];n=m['n'];s=np.array(m['sum'])
            variance=float(np.trace((np.array(m['outer'])-np.outer(s,s)/n)/(n-1)))
            scale=config['kcs_scales'][['acceleration_fd','jerk_fd'][i]]
            distances.append(continuity[key]*max(variance,1e-6)/scale)
        continuity['kcs']=float(np.exp(-.5*(.7*distances[0]+.3*distances[1]))) if len(distances)==2 else None
        continuity['calibration_scales']=config['kcs_scales']
        report['table2']['KCS']=continuity['kcs']
    json_write(out/'table2.json',report)
    json_write(out/'status.json',dict(state='complete',episodes=len(episodes),table2=report['table2']))
    return report


def combined_table(highway,jam,kcs_scales=None):

    counts=pool_episodes(highway['episode_counts']+jam['episode_counts'])
    distributions={}
    for kind in ['real','generated']:
        values=[]
        for index in range(2):
            pair=[r['continuity']['moments'][kind][index] for r in [highway,jam]]
            n=sum(x['n'] for x in pair);s=sum(np.array(x['sum']) for x in pair)
            outer=sum(np.array(x['outer']) for x in pair)
            values.append(None if n<2 else (s/n,(outer-np.outer(s,s)/n)/(n-1)))
        distributions[kind]=values
    distances=[]
    for index in range(2):
        real,gen=distributions['real'][index],distributions['generated'][index]
        if real is None or gen is None: break
        mr,cr=real;mg,cg=gen
        normalizer=max(float(np.trace(cr)),1e-6);cr=cr+np.eye(2)*1e-6;cg=cg+np.eye(2)*1e-6
        eigen,vectors=np.linalg.eigh(cr);root=(vectors*np.sqrt(np.maximum(eigen,0)))@vectors.T
        fd=max(0.,float(((mr-mg)**2).sum()+np.trace(cr+cg)-2*np.sqrt(np.maximum(np.linalg.eigvalsh(root@cg@root),0)).sum()))
        denominator=kcs_scales[['acceleration_fd','jerk_fd'][index]] if kcs_scales else normalizer
        distances.append(fd/denominator)
    return dict(KCS=float(np.exp(-.5*(.7*distances[0]+.3*distances[1]))) if len(distances)==2 else None,
        TTC_H_C={name:{k:v for k,v in r['table2'].items() if k.startswith('TTC')} for name,r in [('H',highway),('C',jam)]},
        Coll_pct=counts['collision_pct'],ACR_pct=counts['acr_pct'],VARC_pct=counts['varc_pct'],
        VARAF_pct=counts['varaf_pct'],VAFR_pct=counts['vafr_pct'],counts=counts)
