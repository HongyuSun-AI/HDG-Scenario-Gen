import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from . import data
from .models import SpatialDiT, TemporalDiT, AxialTemporalDiT
from .diffusion import Diffusion
from .representation import flatten_agents, unflatten_agents, history_mask, decode, normalize, velocities
from .road import MergeRoad
from .metrics import evaluate, continuity_score, minimum_ttc, pool_episodes
from .risk import assess_risk
from .closed_loop import ReactivePlanner, run_closed_loop, verify_fixed_replay
from .features import Extractor, augment, ntxent, extract, distribution_metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def device_for(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def model_class(stage, model_args):
    if stage == 'spatial':
        return SpatialDiT
    return AxialTemporalDiT if model_args.get('architecture') == 'axial' else TemporalDiT


def train_stage(dataset, config, stage, output, device, steps=None, resume=None):
    seed_all(config['seed'])
    model_args = dict(config[stage], agents=dataset['tracks'].shape[2], horizon=dataset['tracks'].shape[3])
    model = model_class(stage, model_args)(**model_args).to(device)
    diffusion = Diffusion(**config['diffusion'])
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    count = steps or config.get(stage + '_steps', config['train_steps'])
    start_step, log = 0, []
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        if ckpt['model_args'] != model_args or ckpt['diffusion'] != config['diffusion']:
            raise ValueError('Resume model/diffusion configuration mismatch')
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['optimizer'])
        start_step, log = ckpt['step'], ckpt['loss_log']
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type == 'cuda' and ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    model.train()
    begin = time.monotonic()
    for step in range(start_step, start_step + count):
        ids = torch.randint(len(dataset['tracks']), (config['batch_size'],))
        risk = dataset['risk'][ids].to(device)
        initial = dataset['initial'][ids].to(device)
        if stage == 'spatial':
            losses = diffusion.loss(model, initial, risk)
        else:
            clean = flatten_agents(dataset['tracks'][ids].to(device))

            p = torch.randint(clean.shape[-1], (len(ids),), device=device)
            mask = history_mask(p, clean.shape[-1])
            losses = diffusion.loss(model, clean, risk, mask, initial)
        if not torch.isfinite(losses['loss']):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        log.append(dict(step=step + 1, **{k: float(v.detach()) for k, v in losses.items()}))
        if (step + 1) % 100 == 0 or step == start_step + count - 1:
            print(f'{stage} step={step + 1} loss={log[-1]["loss"]:.4f} elapsed={time.monotonic()-begin:.1f}s', flush=True)
    checkpoint = dict(model=model.state_dict(), optimizer=opt.state_dict(), stage=stage,
                      step=start_step + count, model_args=model_args, diffusion=config['diffusion'],
                      loss_log=log, torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                      data_metadata=dataset['metadata'], config=config)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    write_json(str(output) + '.loss.json', log)
    return checkpoint


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = model_class(ckpt['stage'], ckpt['model_args'])(**ckpt['model_args']).to(device)


    model.load_state_dict(ckpt.get('ema', ckpt['model'])); model.eval()
    return model, Diffusion(**ckpt['diffusion']), ckpt


def save_samples(path, tracks, road, metadata, **extras):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(tracks=tracks.cpu(), metadata=metadata, **extras), path)
    report = evaluate(tracks.cpu().numpy(), road)
    scores, ttc = [], []
    for x in tracks.cpu().numpy():
        s, valid, _ = decode(x.transpose(1, 2, 0))
        scores.append(float(assess_risk(s, valid, road, **metadata.get('risk_parameters', {})).max()))
        ttc.append(minimum_ttc(s, valid))
    report['realized_risk'] = scores
    report['minimum_ttc_s'] = ttc
    report['ttc_convention'] = '10s constant-velocity OBB sweep'
    report['metadata'] = metadata
    write_json(str(path) + '.metrics.json', report)
    return report


def sample_temporal(dataset, checkpoint, count, history_length, risk, cfg, output, device, seed=7):
    seed_all(seed)
    model, diffusion, _ = load_model(checkpoint, device)
    ids = torch.arange(count) % len(dataset['tracks'])
    clean = flatten_agents(dataset['tracks'][ids].to(device))
    mask = history_mask(torch.full((count,), history_length, device=device), clean.shape[-1])
    labels = dataset['risk'][ids].to(device) if risk is None else torch.full((count,), risk, device=device)
    result = diffusion.sample(model, tuple(clean.shape), labels, clean * mask, mask, dataset['initial'][ids].to(device), cfg)
    tracks = unflatten_agents(result).cpu()
    error = float(((result - clean) * mask).abs().max())
    return tracks, save_samples(output, tracks, MergeRoad(dataset['metadata']['map_path']),
                                dict(mode='real_initial_condition', history_length=history_length,
                                     history_max_error=error, requested_risk=risk, cfg=cfg, seed=seed,
                                     risk_parameters=dataset['metadata'].get('risk_parameters',
                                         {'kappa': dataset['metadata'].get('kappa', 1.)})))


def sample_two_stage(dataset, spatial_path, temporal_path, count, risk, cfg, output, device, seed=7):
    seed_all(seed)
    spatial, sd, _ = load_model(spatial_path, device)
    temporal, td, _ = load_model(temporal_path, device)
    road = MergeRoad(dataset['metadata']['map_path'])
    labels = torch.full((count,), risk, device=device)
    layouts = sd.sample(spatial, (count, 3, spatial.cells, 3), labels, cfg=cfg)
    history = np.full((count, 3, temporal.agents, temporal.horizon), -1, dtype=np.float32)
    conditions, invalid, overcapacity = [], [], []
    for b, grid in enumerate(layouts.cpu().numpy()):
        states, bad = road.decode_layout(grid)

        states = states[np.argsort(states[:, 0], kind='stable')]
        invalid.append(bad); overcapacity.append(max(0, len(states) - temporal.agents))
        states = states[:temporal.agents]
        history[b, :, :len(states), 0] = normalize(states, np.ones(len(states), bool)).T
        conditions.append(road.layout(states, np.ones(len(states), bool), spatial.cells)[0])
    history = flatten_agents(torch.from_numpy(history).to(device))
    mask = history_mask(torch.ones(count, device=device, dtype=torch.long), temporal.horizon)
    result = td.sample(temporal, tuple(history.shape), labels, history * mask, mask,
                       torch.tensor(np.array(conditions), device=device), cfg)
    tracks = unflatten_agents(result).cpu()
    return tracks, save_samples(output, tracks, road,
                                dict(mode='two_stage_generated_initial', requested_risk=risk, cfg=cfg, seed=seed,
                                     malformed_grid_cells=invalid, truncated_overcapacity_agents=overcapacity,
                                     history_max_error=float((result[..., :1] - history[..., :1]).abs().max())),
                                raw_layouts=layouts.cpu())


def train_features(dataset, output, device, steps=100, hidden=128, batch=32):
    model = Extractor(dataset['tracks'].shape[1] * dataset['tracks'].shape[2], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    losses = []
    for step in range(steps):
        ids = torch.randperm(len(dataset['tracks']))[:batch]
        x = dataset['tracks'][ids].to(device)
        loss = ntxent(model(x), model(augment(x)))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            print(f'features step={step+1} loss={losses[-1]:.4f}', flush=True)
    torch.save(dict(model=model.state_dict(), features=dataset['tracks'].shape[1] * dataset['tracks'].shape[2],
                    hidden=hidden, steps=steps, losses=losses), output)
    return model


def render_results(real, generated, road, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/hdg-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/hdg-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for ax, x, label in zip(axes.ravel(), [real[0], generated[0], real[1], generated[1]],
                             ['Real reference 1', 'HDG raw sample 1', 'Real reference 2', 'HDG raw sample 2']):
        for p in road.lines:
            ax.plot(p[:, 0], p[:, 1], color='gray', lw=.7)
        s, valid, _ = decode(x.transpose(1, 2, 0))
        from .representation import segments
        for i in range(len(s)):
            for a, b in segments(valid[i]):
                ax.plot(s[i, a:b, 0], s[i, a:b, 1], linewidth=1.1, color=f'C{i % 10}')
        ax.set(xlim=(1000, 1140), ylim=(935, 960), xlabel='x (m)', ylabel='y (m)', title=label)
    fig.savefig(output, dpi=140); plt.close(fig)


def export_csv(tracks, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario_id', 'track_id', 'frame_id', 'time_s', 'x', 'y', 'heading', 'valid', 'malformed'])
        for b, x in enumerate(np.asarray(tracks)):
            s, valid, bad = decode(x.transpose(1, 2, 0))
            for i, t in zip(*np.nonzero(valid | bad)):
                writer.writerow([b, i, t, round(t * .1, 3), *s[i, t], bool(valid[i, t]), bool(bad[i, t])])


def export_interaction(tracks, directory):

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for b, x in enumerate(np.asarray(tracks)):
        states, valid, bad = decode(x.transpose(1, 2, 0))
        velocity = velocities(states, valid)
        velocity[bad] = np.nan
        with open(directory / f'scenario_{b:03d}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width', 'valid', 'malformed'])
            for i, t in zip(*np.nonzero(valid | bad)):
                writer.writerow([i, t, t * 100, 'car', *states[i, t, :2], *velocity[i, t],
                                 states[i, t, 2], 4.5, 1.8, bool(valid[i, t]), bool(bad[i, t])])


def loop(dataset, checkpoint, output, device, count=2, execution=10, risk=1., cfg=1., planner_name='reactive'):
    model, diffusion, _ = load_model(checkpoint, device)
    road = MergeRoad(dataset['metadata']['map_path'])
    planner_class = ReactivePlanner
    if planner_name == 'pdm-closed':
        from .pdm_planner import PDMClosedAdapter
        planner_class = PDMClosedAdapter
    elif planner_name != 'reactive':
        raise ValueError(f'Unknown planner: {planner_name}')
    episodes, tracks, refs = [], [], []
    for i in range(min(count, len(dataset['tracks']))):
        raw = dataset['tracks'][i].numpy().transpose(1, 2, 0)
        s, valid, _ = decode(raw)
        scores = dataset['agent_risk'][i].numpy().copy()
        scores[~valid[:, 0]] = -1
        ego = int(scores.argmax())
        destination = s[ego, np.flatnonzero(valid[ego])[-1]]
        speeds = np.linalg.norm(velocities(s, valid)[:, 0], axis=-1)
        planner = planner_class(road)
        result = run_closed_loop(model, diffusion, dataset['initial'][i:i+1], raw[:, 0], speeds,
                                 risk, ego, planner, destination, model.horizon, execution, cfg)
        verification = verify_fixed_replay(result, speeds, planner_class(road), destination)
        tracks.append(torch.tensor(result.pop('tracks')))
        refs.append(dataset['tracks'][i])
        result['verification'] = verification
        result['verified'] = verification['verified']
        episodes.append(result)
    stacked = torch.stack(tracks)
    risk_parameters = dataset['metadata'].get('risk_parameters', {'kappa': dataset['metadata'].get('kappa', 1.)})
    report = save_samples(output, stacked, road, dict(mode='closed_loop', planner=planner_class.name,
        cfg=cfg, requested_risk=risk, risk_parameters=risk_parameters))
    report['episodes'] = episodes
    report['continuity'] = continuity_score(torch.stack(refs).numpy(), stacked.numpy(), sorted(set(b for e in episodes for b in e['boundaries'])))
    report['closed_loop_metrics'] = pool_episodes(episodes)
    report['attribution_scope'] = 'nuPlan attribution, OSM adapter'
    report['verification_scope'] = f'Fixed replay: {planner_class.name}'
    write_json(str(output) + '.metrics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description='HDG commands')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', default='data_process/python/processed_data/Track_dataset_smooth.pth')
    p.add_argument('--output', default='outputs/hdg/data.pt')
    p.add_argument('--limit', type=int, default=512); p.add_argument('--horizon', type=int, default=140)
    p.add_argument('--seed', type=int, default=7); p.add_argument('--map', default='data_process/maps/merge.osm')
    p = sub.add_parser('train')
    p.add_argument('--data', required=True); p.add_argument('--config', default='configs/hdg.json')
    p.add_argument('--stage', choices=['spatial', 'temporal'], required=True)
    p.add_argument('--output', required=True); p.add_argument('--steps', type=int); p.add_argument('--resume')
    p = sub.add_parser('sample')
    p.add_argument('--data', required=True); p.add_argument('--temporal', required=True); p.add_argument('--spatial')
    p.add_argument('--output', default='outputs/hdg/samples.pt'); p.add_argument('--count', type=int, default=8)
    p.add_argument('--history', type=int, default=1); p.add_argument('--risk', type=float); p.add_argument('--cfg', type=float, default=1.)
    p.add_argument('--seed', type=int, default=7)
    p = sub.add_parser('closed-loop')
    p.add_argument('--data', required=True); p.add_argument('--temporal', required=True)
    p.add_argument('--output', default='outputs/hdg/closed_loop.pt'); p.add_argument('--count', type=int, default=2)
    p.add_argument('--planner', choices=['reactive','pdm-closed'], default='reactive')
    p.add_argument('--execution', type=int, default=10); p.add_argument('--risk', type=float, default=1.); p.add_argument('--cfg', type=float, default=1.)
    p = sub.add_parser('features')
    p.add_argument('--data', required=True); p.add_argument('--output', default='outputs/hdg/features.pt')
    p.add_argument('--steps', type=int, default=100)
    p = sub.add_parser('evaluate')
    p.add_argument('--data', required=True); p.add_argument('--samples', required=True); p.add_argument('--features')
    p.add_argument('--output', default='outputs/hdg/evaluation.json')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    seed_all(7)
    if getattr(args, 'risk', None) is not None and not 0 <= args.risk <= 1:
        parser.error('--risk must be in [0,1]')
    if args.command == 'prepare':
        d = data.prepare(args.source, args.output, args.horizon, args.limit, args.seed, args.map)
        print(json.dumps(dict(scenarios=len(d['tracks']), metadata=d['metadata']), indent=2)); return
    d = data.load(args.data)
    if args.command == 'train':
        train_stage(d, json.loads(Path(args.config).read_text()), args.stage, args.output, device, args.steps, args.resume)
    elif args.command == 'sample':
        if args.spatial:
            sample_two_stage(d, args.spatial, args.temporal, args.count, args.risk if args.risk is not None else .95,
                             args.cfg, args.output, device, args.seed)
        else:
            sample_temporal(d, args.temporal, args.count, args.history, args.risk, args.cfg, args.output, device, args.seed)
    elif args.command == 'closed-loop':
        loop(d, args.temporal, args.output, device, args.count, args.execution, args.risk, args.cfg, args.planner)
    elif args.command == 'features':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        train_features(d, args.output, device, args.steps)
    elif args.command == 'evaluate':
        generated = torch.load(args.samples, map_location='cpu', weights_only=True)['tracks']
        result = evaluate(generated.numpy(), MergeRoad(d['metadata']['map_path']))
        if args.features:
            ckpt = torch.load(args.features, map_location=device, weights_only=True)
            model = Extractor(ckpt['features'], ckpt['hidden']).to(device)
            model.load_state_dict(ckpt['model'])
            result['feature_metrics'] = distribution_metrics(extract(model, d['tracks']), extract(model, generated))
        write_json(args.output, result)


if __name__ == '__main__':
    main()
