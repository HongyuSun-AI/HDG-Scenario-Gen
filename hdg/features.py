import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .metrics import frechet


class Extractor(nn.Module):
    def __init__(self, features, hidden=128):
        super().__init__()
        self.gru = nn.GRU(features, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden * 2, hidden)

    def forward(self, tracks):

        seq = tracks.flatten(1, 2).transpose(1, 2)
        return self.projection(self.gru(seq)[0][:, -1])


def augment(tracks, sigma=.01):
    return torch.where(tracks == -1, tracks, (tracks + torch.randn_like(tracks) * sigma).clamp(0, 1))


def ntxent(first, second, temperature=.5):
    n = len(first)
    z = F.normalize(torch.cat([first, second]), dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def extract(model, tracks, batch=128):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(x.to(device)).cpu() for x in tracks.split(batch)]).numpy().astype(np.float64)


def distribution_metrics(real, generated, seed=42):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    if min(len(real), len(generated)) < 2:
        return dict(fd=None, nll=None, reason='At least two scenarios required')
    rng = np.random.default_rng(seed)
    fd = [frechet(real[rng.choice(len(real), min(256, len(real)), replace=False)], generated) for _ in range(5)]
    pca = PCA(n_components=min(16, real.shape[1], len(real)))
    real_z = pca.fit_transform(real)
    gmm = GaussianMixture(n_components=min(10, len(real)), covariance_type='full', reg_covar=1e-4, random_state=42)
    gmm.fit(real_z)
    return dict(fd=float(np.mean(fd)), fd_repetitions=fd,
                nll=float(-gmm.score(pca.transform(generated))), real_nll_in_sample=float(-gmm.score(real_z)),
                gmm_components=min(10, len(real)), pca_dimensions=pca.n_components_,
                note='Feature-space reference density')
