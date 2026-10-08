import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .models import SpatialDiT, AxialTemporalDiT
from .diffusion import Diffusion
from .features import Extractor, extract
from .metrics import frechet
from .representation import decode, normalize, flatten_agents, unflatten_agents, history_mask
from .road import MergeRoad
from .relabel import sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_process/python/processed_data/Track_dataset_smooth.pth'
LABELS = ROOT / 'outputs/hdg/risk_smooth_full140_calibrated/risk_labels.pt'
FEATURES = ROOT / 'data_process/python/Contrastive_Learning_model/rnn_feature_extractor.pth'
OUT = ROOT / 'outputs/hdg/full_binary_kappa16'


def json_write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def torch_write(path, value):
    tmp = Path(str(path) + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def legacy_load(path):
    sys.path.insert(0, str(ROOT / 'model_2'))
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        sys.path.pop(0)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = torch.load(LABELS, weights_only=True)
    n = len(labels['risk'])

    classes = (labels['risk'] >= .9).to(torch.int32)
    backup = SOURCE.with_name(SOURCE.stem + '.before_binary_kappa16.pth')
    current_sha = sha256(SOURCE)
    original_sha = labels['metadata']['source_sha256']
    old = legacy_load(SOURCE)
    if current_sha == original_sha:
        if not backup.exists():
            shutil.copy2(SOURCE, backup)
        original_conditions = [item[0] for item in old.labels]
        old.labels = [(condition, classes[i].clone()) for i, condition in enumerate(original_conditions)]
        old.risk_scores = labels['risk'].clone()
        old.label_metadata = dict(kappa=1.6, threshold=.9, positive_rule='risk >= 0.9',
                                  boundary_rule='risk >= 0.9 -> 1',
                                  original_sha256=original_sha, risk_sidecar=str(LABELS),
                                  negative_count=int((classes == 0).sum()), positive_count=int(classes.sum()))
        torch_write(SOURCE, old)
    if (OUT / 'data.pt').exists():
        return
    tracks = old.data.float()
    valid = (tracks >= 0).all(1)
    order = torch.argsort(valid.sum(-1), dim=1, descending=True, stable=True)
    tracks = tracks.gather(2, order[:, None, :, None].expand(-1, 3, -1, 140))
    road = MergeRoad(str(ROOT / 'data_process/maps/merge.osm'))
    initial = np.empty((n, 3, 28, 3), np.float32)
    conflicts = 0
    for i in range(n):
        states, present, malformed = decode(tracks[i, :, :, 0].T.numpy())
        initial[i], collisions = road.layout(states, present)
        conflicts += collisions
        if (i + 1) % 5000 == 0:
            print(f'prepare {i+1}/{n}', flush=True)
    metadata = dict(source=str(SOURCE), source_sha256=sha256(SOURCE), original_sha256=original_sha,
                    backup=str(backup), scenarios=n, agents=12, horizon=140, dt=.1,
                    kappa=1.6, label_rule='risk>=0.9:1; risk<0.9:0',
                    equal_to_threshold_count=int((labels['risk'] == .9).sum()),
                    class_counts={'0':int((classes == 0).sum()), '1':int(classes.sum())},
                    coverage='Full source, 140 frames',
                    grid_conflicts=conflicts, map_path=str(ROOT / 'data_process/maps/merge.osm'))
    torch_write(OUT / 'data.pt', dict(tracks=tracks, initial=torch.from_numpy(initial), risk=classes.float(),
                                    continuous_risk=labels['risk'], agent_risk=labels['agent_risk'].gather(1, order),
                                    source_indices=torch.arange(n), order=order, metadata=metadata))
    json_write(OUT / 'data_manifest.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def feature_model(device):
    saved = torch.load(FEATURES, map_location='cpu', weights_only=True)
    mapped = {k.replace('rnn.', 'gru.').replace('fc.', 'projection.'): v for k, v in saved.items()}
    model = Extractor(36, 128).to(device)
    model.load_state_dict(mapped)
    model.eval().requires_grad_(False)
    return model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_models(config, device):
    return SpatialDiT(**config['spatial']).to(device), AxialTemporalDiT(**config['temporal']).to(device)


@torch.no_grad()
def unguided_samples(spatial, temporal, diffusion, road, count, batch, device, seed):

    generator = torch.Generator(device=device).manual_seed(seed)
    results, audits = [], []
    spatial.eval(); temporal.eval()
    for start in range(0, count, batch):
        b = min(batch, count - start)
        risk = torch.zeros(b, device=device)

        grid = diffusion.sample(spatial, (b, 3, 28, 3), risk, generator=generator, unconditional_risk=True)
        history = np.full((b, 3, 12, 140), -1., np.float32)
        conditions = []
        for i, layout in enumerate(grid.cpu().numpy()):
            states, malformed = road.decode_layout(layout)
            states = states[np.argsort(states[:, 0], kind='stable')]
            overflow = max(0, len(states) - 12)
            states = states[:12]
            history[i, :, :len(states), 0] = normalize(states, np.ones(len(states), bool)).T
            conditions.append(road.layout(states, np.ones(len(states), bool))[0])
            audits.append(dict(malformed_grid_cells=malformed, overcapacity=overflow, initial_agents=len(states)))
        history = flatten_agents(torch.tensor(history, device=device))
        mask = history_mask(torch.ones(b, dtype=torch.long, device=device), 140)
        result = diffusion.sample(temporal, tuple(history.shape), risk, history, mask,
                                  torch.tensor(np.array(conditions), device=device), generator=generator,
                                  unconditional_risk=True)
        results.append(unflatten_agents(result).cpu())
        if (start + b) % 64 == 0:
            print(f'evaluation sampling {start+b}/{count}', flush=True)
    return torch.cat(results), audits


def fid_report(real, generated):
    rng = np.random.default_rng(42)
    ids = [rng.choice(len(real), 256, replace=False) for _ in range(5)]
    scores = [frechet(real[index], generated) for index in ids]
    return dict(fid=float(np.mean(scores)), fid_repetitions=scores,
                fid_all_reference=frechet(real, generated), generated_count=len(generated),
                real_reference_count=len(real), real_subset_size=256, repetitions=5)


def save_checkpoint(path, model, ema, optimizer, scaler, epoch, steps, config, stage):
    torch_write(path, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                           scaler=scaler.state_dict(), epoch=epoch, steps=steps, config=config, stage=stage,
                           model_args=dict(config[stage], **({'architecture':'axial'} if stage == 'temporal' else {})),
                           diffusion=config['diffusion'],
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))


def run(config, benchmark=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda')
    torch.set_num_threads(config['threads'])
    seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    d = torch.load(OUT / 'data.pt', weights_only=True)
    n = len(d['tracks'])
    dataset = TensorDataset(d['tracks'], d['initial'], d['risk'], d['source_indices'])
    spatial, temporal = make_models(config, device)
    diffusion = Diffusion(**config['diffusion'])
    if benchmark:
        model = temporal.train()
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scaler = torch.cuda.amp.GradScaler()
        batch = config['batch_size']
        x = flatten_agents(d['tracks'][:batch].to(device))
        init = d['initial'][:batch].to(device); risk = d['risk'][:batch].to(device)
        start = time.monotonic()
        for _ in range(10):
            mask = history_mask(torch.randint(140, (batch,), device=device), 140)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = diffusion.loss(model, x, risk, mask, init)['loss']
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        print(dict(batch=batch, seconds_per_step=(time.monotonic()-start)/10,
                   peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9, loss=float(loss)), flush=True)
        return
    protocol = dict(feature_checkpoint=str(FEATURES), feature_sha256=sha256(FEATURES),
                    preprocessing='Channel-major, normalized raw output',
                    samples=config['eval_samples'], seed=config['eval_seed'], confirmation_seed=config['eval_seed']+1,
                    reference='37263 scenes; five 256-scene subsets',
                    target=config['target_fid'], stopping_rule='Repeated FD target confirmation',
                    guidance='Unconditional spatial/temporal sampling',
                    initial_condition='Spatial DDPM initialization',
                    label_rule=d['metadata']['label_rule'])
    if not (OUT / 'evaluation_protocol.json').exists():
        json_write(OUT / 'evaluation_protocol.json', protocol)
    feature = feature_model(device)
    cache = OUT / 'real_features.npy'
    if cache.exists():
        real = np.load(cache)
    else:
        real = extract(feature, d['tracks'], 128)
        np.save(cache, real)
    road = MergeRoad(d['metadata']['map_path'])
    emas = {}
    history_file = OUT / 'training.jsonl'
    for stage, model in [('spatial', spatial), ('temporal', temporal)]:
        ema = deepcopy(model).eval().requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
        scaler = torch.cuda.amp.GradScaler()
        epoch, steps = 0, 0
        checkpoint = OUT / f'{stage}_last.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            previous = saved['config']
            changed = {k for k in set(previous) | set(config) if previous.get(k) != config.get(k)}


            if changed:
                with open(OUT / 'training_extensions.jsonl', 'a') as stream:
                    stream.write(json.dumps(dict(stage=stage, resumed_epoch=saved['epoch'],
                        changes={k:dict(previous=previous.get(k), current=config.get(k)) for k in changed})) + '\n')
            model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema']); opt.load_state_dict(saved['optimizer'])
            for group in opt.param_groups:
                group['lr'] = config['learning_rate']
            scaler.load_state_dict(saved['scaler']); epoch, steps = saved['epoch'], saved['steps']
            torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        emas[stage] = ema
        while stage == 'temporal' or epoch < config['spatial_epochs']:
            epoch += 1
            rng = torch.Generator().manual_seed(config['seed'] + epoch + (0 if stage == 'spatial' else 100000))
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, generator=rng,
                                drop_last=False, num_workers=0, pin_memory=True)
            model.train()
            seen = torch.zeros(n, dtype=torch.bool)
            begin, total, updates = time.monotonic(), 0., 0
            for x, init, risk, indices in loader:
                seen[indices] = True
                x, init, risk = x.to(device, non_blocking=True), init.to(device, non_blocking=True), risk.to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    if stage == 'spatial':
                        loss = diffusion.loss(model, init, risk)['loss']
                    else:

                        p = torch.randint(140, (len(x),), device=device)
                        mode = torch.rand(len(x), device=device)
                        p = torch.where(mode < .2, 0, torch.where(mode < .4, 1, p))
                        loss = diffusion.loss(model, flatten_agents(x), risk, history_mask(p, 140), init)['loss']
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {stage}: epoch={epoch}, step={steps}')
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt); scaler.update()
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), model.parameters()):
                        a.lerp_(b, 1 - config['ema_decay'])
                steps += 1; updates += 1; total += float(loss.detach())
                if steps % 100 == 0:
                    status = dict(state='training', stage=stage, epoch=epoch, steps=steps,
                                  epoch_seen=int(seen.sum()), dataset_count=n, mean_loss=total/updates,
                                  elapsed_epoch_seconds=round(time.monotonic()-begin, 1))
                    json_write(OUT / 'status.json', status)
                    print(json.dumps(status), flush=True)
                if steps % 1000 == 0:


                    save_checkpoint(checkpoint, model, ema, opt, scaler, epoch - 1, steps, config, stage)
            record = dict(stage=stage, epoch=epoch, steps=steps, unique_scenarios=int(seen.sum()), mean_loss=total/updates,
                          seconds=time.monotonic()-begin)
            with open(history_file, 'a') as stream:
                stream.write(json.dumps(record) + '\n')
            save_checkpoint(checkpoint, model, ema, opt, scaler, epoch, steps, config, stage)
            shutil.copy2(checkpoint, OUT / f'{stage}_epoch_{epoch:04d}.pt')
            print('EPOCH ' + json.dumps(record), flush=True)
            if stage == 'temporal':
                json_write(OUT / 'status.json', dict(state='evaluating', stage=stage, epoch=epoch, steps=steps))
                samples, audit = unguided_samples(emas['spatial'], ema, diffusion, road, config['eval_samples'],
                                                  config['eval_batch'], device, config['eval_seed'])
                report = fid_report(real, extract(feature, samples, 128))
                report.update(epoch=epoch, steps=steps, spatial_epochs=config['spatial_epochs'], protocol=protocol)
                torch_write(OUT / f'samples_epoch_{epoch:04d}.pt', dict(tracks=samples, spatial_audit=audit, report=report))
                json_write(OUT / f'evaluation_epoch_{epoch:04d}.json', report)
                best_file = OUT / 'best.json'
                if not best_file.exists() or report['fid'] < json.loads(best_file.read_text())['fid']:
                    torch_write(OUT / 'best.pt', dict(spatial=emas['spatial'].state_dict(), temporal=ema.state_dict(), config=config, report=report))
                    json_write(best_file, report)
                print('FID ' + json.dumps({k:v for k,v in report.items() if k != 'protocol'}), flush=True)
                json_write(OUT / 'status.json', dict(state='evaluated', **report))
                if max(report['fid'], report['fid_all_reference']) <= config['target_fid']:
                    confirm, confirm_audit = unguided_samples(emas['spatial'], ema, diffusion, road, config['eval_samples'],
                                                            config['eval_batch'], device, config['eval_seed'] + 1)
                    confirmation = fid_report(real, extract(feature, confirm, 128))
                    report['confirmation'] = confirmation
                    torch_write(OUT / 'confirmation_samples.pt', dict(tracks=confirm, spatial_audit=confirm_audit, report=confirmation))
                    if max(confirmation['fid'], confirmation['fid_all_reference']) <= config['target_fid']:
                        torch_write(OUT / 'achieved.pt', dict(spatial=emas['spatial'].state_dict(), temporal=ema.state_dict(), config=config, report=report))
                        for name, trained in [('spatial', emas['spatial']), ('temporal', ema)]:
                            torch_write(OUT / f'{name}.pt', dict(model=trained.state_dict(), stage=name,
                                model_args=dict(config[name], **({'architecture':'axial'} if name == 'temporal' else {})),
                                diffusion=config['diffusion'], config=config, report=report, data_metadata=d['metadata']))
                        json_write(OUT / 'achieved.json', report)
                        json_write(OUT / 'status.json', dict(state='target_achieved', **report))
                        print('TARGET ACHIEVED ' + json.dumps(report), flush=True)
                        return


def main():
    parser = argparse.ArgumentParser(description='Train binary-risk HDG')
    parser.add_argument('command', choices=['prepare', 'run', 'benchmark'])
    parser.add_argument('--config', default='configs/hdg_full_binary.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        prepare()
    else:
        config = json.loads(Path(args.config).read_text())
        try:
            run(config, args.command == 'benchmark')
        except Exception as error:
            json_write(OUT / 'error.json', dict(error=repr(error)))
            raise


if __name__ == '__main__':
    main()
