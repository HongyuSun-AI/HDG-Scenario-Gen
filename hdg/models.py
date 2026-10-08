import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal(t, dim):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t.float()[..., None] * f
    return torch.cat([a.cos(), a.sin()], -1)


class Attention(nn.Module):
    def __init__(self, dim, heads, features=None, channels=3):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.structured = features is not None
        if self.structured:
            agent = torch.arange(features) // channels
            same = agent[:, None] == agent[None]
            self.register_buffer('intra', (same & ~torch.eye(features, dtype=torch.bool)).float())
            self.register_buffer('inter', (~same).float())
            self.alpha = nn.Parameter(torch.zeros(heads))
            self.beta = nn.Parameter(torch.zeros(heads))

    def bias(self):
        if not self.structured:
            return None
        return self.alpha[:, None, None] * self.intra + self.beta[:, None, None] * self.inter

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        b, l, d = x.shape
        def split(z):
            return z.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        bias = self.bias()
        if mask is not None:
            bias = mask if bias is None else bias + mask
        out = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(context)), split(self.v(context)), attn_mask=bias)
        return self.proj(out.transpose(1, 2).reshape(b, l, d))


class Block(nn.Module):
    def __init__(self, dim, heads, features=None, cross=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, features)
        self.cross = Attention(dim, heads) if cross else None
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, c, context=None):
        shift, scale, gate, shift2, scale2, gate2 = self.modulation(c).chunk(6, -1)
        x = x + gate[:, None] * self.attn(self.norm1(x) * (1 + scale[:, None]) + shift[:, None])
        if self.cross is not None:
            x = x + self.cross(self.norm1(x), context)
        return x + gate2[:, None] * self.mlp(self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None])


class Conditions(nn.Module):
    def __init__(self, dim, drop=.1):
        super().__init__()
        self.dim, self.drop = dim, drop
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.risk = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.randn(dim) * .02)

    def forward(self, t, risk, drop_risk=False):
        r = self.risk(risk.reshape(-1, 1))
        drop = torch.full((len(t),), bool(drop_risk), device=t.device)
        if self.training and not drop_risk:
            drop = torch.rand(len(t), device=t.device) < self.drop
        return self.time(sinusoidal(t, self.dim)) + torch.where(drop[:, None], self.null, r)


class SpatialDiT(nn.Module):
    def __init__(self, cells=28, dim=32, depth=2, heads=4, learn_sigma=True, **unused):
        super().__init__()
        self.cells, self.learn_sigma = cells, learn_sigma
        self.input = nn.Linear(3, dim)
        self.register_buffer('position', sinusoidal(torch.arange(3 * cells), dim)[None])
        self.conditions = Conditions(dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6 if learn_sigma else 3))

    def forward(self, x, t, risk, drop_risk=False, **unused):
        b = len(x)
        h = self.input(x.reshape(b, -1, 3)) + self.position
        c = self.conditions(t, risk, drop_risk)
        for block in self.blocks:
            h = block(h, c)
        h = self.out(h).reshape(b, 3, self.cells, -1)
        return h.chunk(2, -1) if self.learn_sigma else (h, None)


class TemporalDiT(nn.Module):
    def __init__(self, agents=12, horizon=140, cells=28, dim=64, depth=3, heads=4,
                 learn_sigma=True, structural_bias=True, **unused):
        super().__init__()
        self.agents, self.horizon, self.learn_sigma = agents, horizon, learn_sigma
        features = agents * 3
        # [future(F,W), history(F,W), mask(1,W)]

        self.input = nn.Linear(3 * horizon, dim)
        self.register_buffer('position', sinusoidal(torch.arange(features), dim)[None])
        self.register_buffer('grid_position', sinusoidal(torch.arange(3 * cells), dim)[None])
        self.grid_input = nn.Linear(3, dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0, batch_first=True), 1)
        self.conditions = Conditions(dim)
        self.blocks = nn.ModuleList([Block(dim, heads, features if structural_bias else None, cross=True) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon * (2 if learn_sigma else 1)))

    def forward(self, x, t, risk, history, mask, initial, drop_risk=False):
        h = self.input(torch.cat([x, history, mask.expand_as(x)], -1)) + self.position
        context = self.encoder(self.grid_input(initial.flatten(1, 2)) + self.grid_position)
        c = self.conditions(t, risk, drop_risk)
        for block in self.blocks:
            h = block(h, c, context)
        h = self.out(h)
        return h.chunk(2, -1) if self.learn_sigma else (h, None)


class AxialBlock(nn.Module):

    def __init__(self, dim, heads, features, structural_bias=True):
        super().__init__()
        self.feature = Attention(dim, heads, features if structural_bias else None)
        self.temporal = Attention(dim, heads)
        self.cross = Attention(dim, heads)
        self.norm = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(4)])
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))

    def forward(self, h, c, context, mask):
        b, f, w, d = h.shape
        shift, scale, gate = self.modulation(c).chunk(3, -1)
        def norm(index, x):
            return self.norm[index](x) * (1 + scale[:, None, None]) + shift[:, None, None]
        x = norm(0, h).transpose(1, 2).reshape(b * w, f, d)
        h = h + gate[:, None, None].tanh() * self.feature(x).reshape(b, w, f, d).transpose(1, 2)


        observed = mask[:, 0].bool()
        forbidden = observed[:, :, None] & ~observed[:, None, :]
        temporal_mask = torch.zeros(b, w, w, device=h.device, dtype=h.dtype).masked_fill(forbidden, -torch.inf)
        temporal_mask = temporal_mask[:, None].expand(b, f, w, w).reshape(b * f, 1, w, w)
        x = norm(1, h).reshape(b * f, w, d)
        h = h + self.temporal(x, mask=temporal_mask).reshape(b, f, w, d)
        x = norm(2, h).reshape(b, f * w, d)
        h = h + self.cross(x, context).reshape(b, f, w, d)
        return h + self.mlp(norm(3, h))


class AxialTemporalDiT(nn.Module):

    def __init__(self, agents=12, horizon=140, cells=28, dim=32, depth=2, heads=4,
                 learn_sigma=True, structural_bias=True, **unused):
        super().__init__()
        self.agents, self.horizon, self.learn_sigma = agents, horizon, learn_sigma
        features = agents * 3
        self.input = nn.Linear(3, dim)
        self.register_buffer('feature_position', sinusoidal(torch.arange(features), dim)[None, :, None])
        self.register_buffer('time_position', sinusoidal(torch.arange(horizon), dim)[None, None])
        self.register_buffer('grid_position', sinusoidal(torch.arange(3 * cells), dim)[None])
        self.grid_input = nn.Linear(3, dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0, batch_first=True), 1)
        self.conditions = Conditions(dim)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads, features, structural_bias) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 if learn_sigma else 1))

    def forward(self, x, t, risk, history, mask, initial, drop_risk=False):
        h = self.input(torch.stack([x, history, mask.expand_as(x)], -1))
        h = h + self.feature_position + self.time_position
        c = self.conditions(t, risk, drop_risk)
        context = self.encoder(self.grid_input(initial.flatten(1, 2)) + self.grid_position)
        for block in self.blocks:
            h = block(h, c, context, mask)
        result = self.out(h)
        return result[..., 0], result[..., 1] if self.learn_sigma else None
