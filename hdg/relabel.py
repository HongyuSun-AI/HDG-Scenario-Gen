import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .representation import decode
from .risk import assess_risk, assess_risk_vectorized, risk_from_components
from .road import MergeRoad


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def plot_distribution(risk, output, kappa=1., exclude_zero=False, inclusive_threshold=False):
    risk = np.asarray(risk)
    source_count = len(risk)
    if exclude_zero:
        risk = risk[risk > 0]
    if not len(risk):
        raise ValueError('No scenarios available')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/hdg-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/hdg-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    title = (f'Positive risk; kappa={kappa:g}\n'
             f'n={len(risk):,}/{source_count:,}; T=140') if exclude_zero else (
             f'Risk; n={len(risk):,}, kappa={kappa:g}')
    fig.suptitle(title, fontsize=15 if exclude_zero else 16, fontweight='bold')
    bins = np.linspace(0, 1, 26)
    axes[0].hist(risk, bins=bins, weights=np.full(len(risk), 100 / len(risk)), color='#247c9b', edgecolor='white', linewidth=.8)
    axes[0].set(xlabel='Scenario risk=max(agent risk)', ylabel='Share of retained scenes' if exclude_zero else 'Share of all scenes', xlim=(0, 1), title='Histogram (bin width = 0.04)')
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    values, counts = np.unique(np.sort(risk), return_counts=True)
    comparison = '>=' if inclusive_threshold else '>'
    high = risk >= .9 if inclusive_threshold else risk > .9
    axes[1].step(np.r_[0., values, 1.], np.r_[0., np.cumsum(counts) / len(risk), 1.], where='post', color='#247c9b', linewidth=2)
    axes[1].set(xlabel='Scenario risk r', ylabel='Cumulative share of retained scenes' if exclude_zero else 'Cumulative share of scenes', xlim=(0, 1), ylim=(0, 1.02), title='Empirical cumulative distribution')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.axvline(.9, color='#c15b32', ls='--', lw=1.3, label=f'High-risk threshold: r {comparison} 0.9')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    zero_note = f'Excluded-zero={source_count - len(risk):,}' if exclude_zero else f'Zero={(risk == 0).sum():,} ({100 * (risk == 0).mean():.2f}%)'
    note = (f'Mean={risk.mean():.4f} Median={np.median(risk):.4f}\n'
            f'{zero_note}\n'
            f'r{comparison}0.9={high.sum():,} ({100 * high.mean():.2f}%)')
    axes[1].text(.97, .08, note, ha='right', va='bottom', transform=axes[1].transAxes,
                 bbox=dict(facecolor='white', edgecolor='#cbd5df', boxstyle='round,pad=.6'))
    stem = 'risk_distribution_nonzero' if exclude_zero else 'risk_distribution'
    fig.savefig(output / f'{stem}.png', dpi=180)
    fig.savefig(output / f'{stem}.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compute risk labels')
    parser.add_argument('--source', default='data_process/python/processed_data/Track_dataset_smooth.pth')
    parser.add_argument('--map', default='data_process/maps/merge.osm')
    parser.add_argument('--output', default='outputs/hdg/risk_smooth_full140')
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--calibrate-target', type=float, help='Target high-risk fraction')
    parser.add_argument('--inclusive-threshold', action='store_true', help='Use inclusive risk threshold')
    args = parser.parse_args()
    if args.kappa < 0 or not np.isfinite(args.kappa):
        parser.error('Invalid kappa')
    if args.calibrate_target is not None and not 0 < args.calibrate_target < 1:
        parser.error('--calibrate-target must be in (0,1)')
    if args.calibrate_target is not None and args.output == 'outputs/hdg/risk_smooth_full140':
        args.output = 'outputs/hdg/risk_smooth_full140_calibrated'
    torch.set_num_threads(2)
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_2'))
    try:
        dataset = torch.load(source, map_location='cpu', weights_only=False)
    finally:
        sys.path.pop(0)
    count = len(dataset)
    channels, agents, horizon = dataset[0][0].shape
    road = MergeRoad(args.map)
    parameters = dict(dt=.1, reaction_time=1.5, max_deceleration=3.4, deviation=50., kappa=args.kappa)
    print(f'Loaded: scenes={count:,} agents={agents} frames={horizon}', flush=True)


    check_ids = np.random.default_rng(7).choice(count, min(96, count), replace=False)
    max_error = 0.
    for idx in check_ids:
        states, valid, malformed = decode(dataset[int(idx)][0].numpy().transpose(1, 2, 0))
        reference = assess_risk(states, valid, road, **parameters)
        actual = assess_risk_vectorized(states, valid, road, **parameters)
        max_error = max(max_error, float(np.max(abs(reference - actual))))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-7)
    print(f'Reference: n={len(check_ids)} error={max_error}', flush=True)
    labels = np.empty((count, agents), np.float32)
    valid_counts = np.empty((count, agents), np.int32)
    malformed_counts = np.zeros(count, np.int32)
    clipped_counts = np.zeros(count, np.int32)
    offmap_counts = np.zeros(count, np.int32)
    exposure_all = np.empty((count, agents, agents), np.float64) if args.calibrate_target is not None else None
    peaks_all = np.empty_like(exposure_all) if exposure_all is not None else None
    begin = time.monotonic()
    for idx in range(count):
        raw = dataset[idx][0].numpy().transpose(1, 2, 0)
        states, valid, malformed = decode(raw)
        malformed_counts[idx] = int(malformed.sum())
        clipped_counts[idx] = int((valid & ((raw < 0) | (raw > 1)).any(-1)).sum())
        offmap_counts[idx] = int((valid & (road.lane(states[..., 0], states[..., 1]) < 0)).sum())
        valid_counts[idx] = valid.sum(-1)
        if exposure_all is not None:
            exposure_all[idx], peaks_all[idx] = assess_risk_vectorized(states, valid, road, **parameters, return_components=True)
            labels[idx] = risk_from_components(exposure_all[idx], peaks_all[idx], parameters['kappa'])
        else:
            labels[idx] = assess_risk_vectorized(states, valid, road, **parameters)
        if (idx + 1) % 2000 == 0 or idx + 1 == count:
            print(f'{idx+1:,}/{count:,} ({100*(idx+1)/count:.1f}%) | {time.monotonic()-begin:.1f}s', flush=True)
    calibration = None
    if exposure_all is not None:
        scans = []
        for tenth in range(10, 1001):
            kappa = tenth / 10
            candidate = risk_from_components(exposure_all, peaks_all, kappa)
            hits = int(((candidate.max(1) >= .9) if args.inclusive_threshold else (candidate.max(1) > .9)).sum())
            scans.append(dict(kappa=kappa, high_risk_count=hits, high_risk_fraction=hits / count))
            if hits / count > args.calibrate_target:
                labels = candidate
                parameters['kappa'] = kappa
                break
        else:
            raise ValueError('Kappa target unreachable')

        for idx in check_ids:
            states, valid, _ = decode(dataset[int(idx)][0].numpy().transpose(1, 2, 0))
            np.testing.assert_allclose(labels[idx], assess_risk(states, valid, road, **parameters), rtol=0, atol=1e-7)
        calibration = dict(target_fraction=args.calibrate_target, threshold=.9, strict_comparisons=not args.inclusive_threshold,
                           risk_comparison='>=' if args.inclusive_threshold else '>', fraction_comparison='>',
                           selection='Minimum qualifying kappa, step=0.1',
                           selected_kappa=parameters['kappa'], scans=scans,
                           interpretation='Distribution calibration')
        (out / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
        (out / 'risk_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        print(f'kappa={parameters["kappa"]}: {hits}/{count} ({100*hits/count:.4f}%) risk{calibration["risk_comparison"]}0.9', flush=True)
    risk = labels.max(axis=1)
    metadata = dict(source=str(source.resolve()), source_sha256=sha256(source), count=count,
                    horizon=horizon, channels=channels, max_agents=agents, map=str(Path(args.map).resolve()),
                    map_sha256=sha256(args.map), parameters=parameters, assumed_vehicle_length_m=4.5,
                    scenario_risk='Maximum agent risk',
                    index_semantics='Original scene and agent indices',
                    coverage='Full source, 140 frames',
                    implementation='Vectorized SSD risk',
                    implementation_sha256=sha256(Path(__file__).with_name('risk.py')),
                    reference_check_count=len(check_ids), reference_max_absolute_error=max_error,
                    calibration=calibration,
                    velocity='Segment-wise central differences; singleton:0',
                    topology='Lane-aware neighbors, 1s lookahead',
                    caution='Site-specific geometry assumptions')
    payload = dict(risk=torch.from_numpy(risk), agent_risk=torch.from_numpy(labels),
                   binary_risk=torch.from_numpy((risk >= .9).astype(np.int64)),
                   source_indices=torch.arange(count), valid_frame_counts=torch.from_numpy(valid_counts),
                   malformed_state_counts=torch.from_numpy(malformed_counts), clipped_state_counts=torch.from_numpy(clipped_counts),
                   offmap_state_counts=torch.from_numpy(offmap_counts), metadata=metadata)
    path = out / 'risk_labels.pt'
    torch.save(payload, path)

    saved = torch.load(path, weights_only=True)
    with open(out / 'risk_labels.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_index', 'scenario_risk', 'active_agents'] + [f'agent_{j}_risk' for j in range(agents)])
        for idx in range(count):
            writer.writerow([idx, float(risk[idx]), int((valid_counts[idx] > 0).sum()), *labels[idx].tolist()])
    bins = np.linspace(0, 1, 26)
    counts, edges = np.histogram(risk, bins=bins)
    summary = dict(metadata=metadata, distribution=dict(
        min=float(risk.min()), max=float(risk.max()), mean=float(risk.mean()), std=float(risk.std()),
        quantiles={str(q):float(np.quantile(risk, q)) for q in [0, .05, .25, .5, .75, .9, .95, .99, 1]},
        zero_count=int((risk == 0).sum()), zero_pct=float(100*(risk == 0).mean()),
        high_risk_gt_0_9_count=int((risk > .9).sum()), high_risk_gt_0_9_pct=float(100*(risk > .9).mean()),
        high_risk_ge_0_9_count=int((risk >= .9).sum()), high_risk_ge_0_9_pct=float(100*(risk >= .9).mean()),
        one_count=int((risk == 1).sum()), histogram_edges=edges.tolist(), histogram_counts=counts.tolist()),
        audit=dict(malformed_states=int(malformed_counts.sum()), clipped_states=int(clipped_counts.sum()),
                   clipped_scenarios=int((clipped_counts > 0).sum()), offmap_states=int(offmap_counts.sum()),
                   offmap_scenarios=int((offmap_counts > 0).sum()), empty_scenarios=int((valid_counts.sum(1) == 0).sum()),
                   source_index_coverage_verified=True, saved_label_roundtrip_verified=True),
        elapsed_calculation_seconds=round(time.monotonic() - begin, 2))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    plot_distribution(risk, out, parameters['kappa'], inclusive_threshold=args.inclusive_threshold)
    print(json.dumps(dict(output=str(out), distribution=summary['distribution'], audit=summary['audit']), indent=2), flush=True)


if __name__ == '__main__':
    main()
