import numpy as np
import torch

LOW = np.array([1000., 935., -3.14])
HIGH = np.array([1140., 955., 3.14])


def flatten_agents(x):
    # B,C,H,W -> B,H*C,W
    b, c, h, w = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, h * c, w)


def unflatten_agents(x, channels=3):
    b, f, w = x.shape
    return x.reshape(b, f // channels, channels, w).permute(0, 2, 1, 3)


def history_mask(lengths, horizon):
    lengths = torch.as_tensor(lengths)
    return (torch.arange(horizon, device=lengths.device)[None, None] < lengths[:, None, None]).float()


def sort_by_duration(x):
    # C,H,W
    valid = (x >= 0).all(dim=0)
    order = torch.argsort(valid.sum(-1), descending=True, stable=True)
    return x[:, order], order


def decode(x):
    # (..., 3)
    x = np.asarray(x)
    pad = (x < -.9).all(-1)
    valid = np.isfinite(x).all(-1) & (x >= -.1).all(-1)
    malformed = ~(pad | valid)
    physical = np.clip(x, 0, 1) * (HIGH - LOW) + LOW
    physical[~valid] = np.nan
    return physical, valid, malformed


def normalize(states, valid):
    return np.where(valid[..., None], (states - LOW) / (HIGH - LOW), -1).astype(np.float32)


def segments(valid):
    edges = np.diff(np.r_[False, valid, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def velocities(states, valid, dt=.1):
    v = np.zeros(states.shape[:-1] + (2,))
    for i in range(len(states)):
        for a, b in segments(valid[i]):
            if b - a >= 2:
                v[i, a:b] = np.gradient(states[i, a:b, :2], dt, axis=0)
    return v
