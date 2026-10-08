import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .full_training import seed_all, torch_write
from .legacy_feature_refit import original_module, FeatureAdapter
from .relabel import sha256


def train_features(data, output, config, device):
    output = Path(output)
    module, source = original_module()
    seed = config.get('feature_seed', config['seed'])
    seed_all(seed)
    protocol = dict(source_sha256=data['metadata']['source_sha256'],
                    prepared_data_sha256=config.get('prepared_data_sha256'),
                    code_sha256=sha256(source), agents=data['tracks'].shape[2],
                    epochs=config['feature_epochs'], seed=seed)
    model = module.RNNFeatureExtractor(input_dim=data['tracks'].shape[2] * 3,
                                      hidden_dim=128, num_layers=2, use_gru=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    final = output / 'features.pt'
    if final.exists():
        saved = torch.load(final, map_location=device, weights_only=True)
        if saved['protocol'] != protocol:
            raise ValueError('Feature protocol changed')
        model.load_state_dict(saved['model'])
        return FeatureAdapter(model.eval().requires_grad_(False))
    last = output / 'features_last.pt'
    start = 0
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        if saved['protocol'] != protocol:
            raise ValueError('Feature protocol changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch']
    dataset = module.ModifiedDataset(TensorDataset(data['tracks'], data['risk']))
    for epoch in range(start + 1, config['feature_epochs'] + 1):
        seed_all(seed + epoch)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
        model.train()
        seen, loss_sum = 0, 0.
        for x, augmented in loader:
            features = model(torch.cat([x, augmented], dim=0).to(device))
            loss = module.contrastive_loss(features, .5)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite feature loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seen += len(x)
            loss_sum += float(loss.detach())
        record = dict(epoch=epoch, seen=seen, loss=loss_sum / len(loader))
        with (output / 'features_training.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        torch_write(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              epoch=epoch, protocol=protocol))
        print('FEATURE', record, flush=True)
    torch_write(final, dict(model=model.state_dict(), protocol=protocol))
    return FeatureAdapter(model.eval().requires_grad_(False))
