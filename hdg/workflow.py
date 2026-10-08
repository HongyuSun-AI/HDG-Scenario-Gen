import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from .cli import device_for
from .data import load
from .experiment_data import prepare_congestion
from .full_training import json_write
from .relabel import sha256


def parser():
    root = argparse.ArgumentParser(description='Prepare, train, validate')
    commands = root.add_subparsers(dest='command', required=True)
    for name in ['prepare', 'train', 'validate']:
        command = commands.add_parser(name)
        command.add_argument('--dataset', required=True, choices=['highway', 'congestion'])
        command.add_argument('--config', default='configs/hdg_experiment.json')
        command.add_argument('--output', help='Dataset working directory')
        if name == 'prepare':
            command.add_argument('--source')
        else:
            command.add_argument('--data')
            command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        if name == 'validate':
            command.add_argument('--checkpoint')
            command.add_argument('--initial-scenes', type=int)
            command.add_argument('--batch-size', type=int)
            command.add_argument('--results')
    return root


def validate(data, checkpoint, output, config, device):
    from .night_experiment import run_dataset
    output = Path(output)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if config.get('initialization') == 'merge_intent':
        from .congestion_intent_experiment import select_initializations
    else:
        from .experiment_data import select_initializations
    selection = (select_initializations(data, config['groups'], config['seed'])
                 if config.get('initialization') == 'merge_intent' else
                 select_initializations(data, config['groups'], config['seed'], config.get('ego_x_fraction_max')))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'initializations.json'
    if not manifest.exists():
        json_write(manifest, selection)
    else:
        saved = json.loads(manifest.read_text())
    report = run_dataset(data, checkpoint, output, config, device=str(device))
    fields = ['KCS', 'TTC_mean_s', 'TTC_std_s', 'Coll_pct', 'ACR_pct', 'VARC_pct', 'VARAF_pct', 'VAFR_pct']
    rows = ['| Metric | Value |', '| --- | --- |']
    for field in fields:
        value = report['table2'][field]
        rows.append('| {} | {} |'.format(field, 'N/A' if value is None else '{:.6f}'.format(value)))
    (output / 'metrics.md').write_text('\n'.join(rows) + '\n')
    print(json.dumps(report['table2'], indent=2), flush=True)
    from .collision_demo import render_saved
    print('Demos:', render_saved(output), flush=True)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    settings = json.loads(Path(args.config).read_text())
    config = settings['datasets'][args.dataset]
    output = Path(args.output or config['output'])
    torch.set_num_threads(config['training']['threads'])
    if args.command == 'prepare':
        print(json.dumps(dict(dataset=args.dataset, source=args.source or config['source'],
                              kappa=config['kappa'], output=str(output / 'data.pt')), indent=2), flush=True)
        prepared = prepare_congestion(args.source or config['source'], output,
                                      settings['map'], kappa=config['kappa'])
        print(json.dumps(dict(dataset=args.dataset, data=str(output / 'data.pt'),
                              scenes=len(prepared['tracks']), agents=prepared['tracks'].shape[2]), indent=2))
        return
    data_path = Path(args.data or output / 'data.pt')
    prepared = load(data_path)
    if prepared['metadata']['kappa'] != config['kappa']:
        raise ValueError('Prepared dataset configuration differs')
    device = device_for(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if args.command == 'train':
        from .congestion_training import train
        training = deepcopy(config['training'])
        training['prepared_data_sha256'] = sha256(data_path)
        checkpoint = train(prepared, output / 'model', training, device=str(device))
        print('Checkpoint:', checkpoint, flush=True)
    else:
        closed_loop = deepcopy(config['closed_loop'])
        if args.initial_scenes is not None: closed_loop['groups'] = args.initial_scenes
        if args.batch_size is not None: closed_loop['batch_size'] = args.batch_size
        closed_loop['prepared_data_sha256'] = sha256(data_path)
        validate(prepared, args.checkpoint or output / 'model/best_temporal.pt',
                 args.results or output / 'closed_loop', closed_loop, device)


if __name__ == '__main__':
    main()
