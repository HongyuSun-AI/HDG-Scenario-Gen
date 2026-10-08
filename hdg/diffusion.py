import torch


class Diffusion:
    def __init__(self, steps=100, beta_start=None, beta_end=None, vlb_weight=.001):
        self.steps, self.vlb_weight = steps, vlb_weight

        self.beta = torch.linspace(beta_start or .0001 * 1000 / steps,
                                   beta_end or min(.02 * 1000 / steps, .999), steps)
        self.alpha = 1 - self.beta
        self.abar = self.alpha.cumprod(0)
        prev = torch.cat([torch.ones(1), self.abar[:-1]])
        self.posterior_var = self.beta * (1 - prev) / (1 - self.abar)
        self.posterior_logvar = torch.cat([self.posterior_var[1:2], self.posterior_var[1:]]).log()
        self.c1 = self.beta * prev.sqrt() / (1 - self.abar)
        self.c2 = (1 - prev) * self.alpha.sqrt() / (1 - self.abar)

    @staticmethod
    def extract(values, t, x):
        return values.to(x.device)[t].reshape(-1, *([1] * (x.ndim - 1)))

    def corrupt(self, clean, t, mask, noise):
        a = self.extract(self.abar, t, clean)
        return (1 - mask) * (a.sqrt() * clean + (1 - a).sqrt() * noise)

    def mean_var(self, x, t, eps, variance, clip=False):
        a = self.extract(self.abar, t, x)
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        if clip:
            x0 = x0.clamp(-1, 1)
        mean = self.extract(self.c1, t, x) * x0 + self.extract(self.c2, t, x) * x
        lo = self.extract(self.posterior_logvar, t, x)
        if variance is None:
            return mean, lo
        hi = self.extract(self.beta.log(), t, x)
        frac = (variance.tanh() + 1) / 2
        return mean, frac * hi + (1 - frac) * lo

    def loss(self, model, clean, risk, mask=None, initial=None, t=None, noise=None):
        b = len(clean)
        t = torch.randint(self.steps, (b,), device=clean.device) if t is None else t
        mask = torch.zeros_like(clean) if mask is None else mask
        noise = torch.randn_like(clean) if noise is None else noise
        x = self.corrupt(clean, t, mask, noise)
        kwargs = {} if initial is None else dict(history=clean * mask, mask=mask, initial=initial)
        eps, variance = model(x, t, risk=risk, **kwargs)
        unknown = (1 - mask).expand_as(clean)
        denom = unknown.flatten(1).sum(-1).clamp_min(1)
        def reduce(value):
            return (value * unknown).flatten(1).sum(-1) / denom
        mse = reduce((eps - noise).square())
        vb = torch.zeros_like(mse)
        if variance is not None:

            mean, logvar = self.mean_var(x, t, eps.detach(), variance)
            true_mean = self.extract(self.c1, t, x) * clean + self.extract(self.c2, t, x) * x
            true_logvar = self.extract(self.posterior_logvar, t, x)
            kl = .5 * (logvar - true_logvar - 1 + (true_logvar - logvar).exp() + (true_mean - mean).square() * (-logvar).exp())

            vb = reduce(kl) * (t > 0) * self.steps
        return {'loss': (mse + self.vlb_weight * vb).mean(), 'mse': mse.mean(), 'vlb': vb.mean()}

    @torch.no_grad()
    def sample(self, model, shape, risk, history=None, mask=None, initial=None, cfg=1., generator=None,
               unconditional_risk=False):
        device = next(model.parameters()).device
        history = torch.zeros(shape, device=device) if history is None else history.to(device)
        mask = torch.zeros_like(history) if mask is None else mask.to(device)
        if torch.all(mask == 1):
            return history.clone()
        was_training = model.training
        model.eval()
        try:
            x = torch.randn(shape, device=device, generator=generator) * (1 - mask)
            kwargs = {} if initial is None else dict(history=history * mask, mask=mask, initial=initial)
            for k in reversed(range(self.steps)):
                t = torch.full((shape[0],), k, device=device, dtype=torch.long)
                eps, var = model(x, t, risk=risk, drop_risk=unconditional_risk, **kwargs)
                if cfg != 1 and not unconditional_risk:
                    null, _ = model(x, t, risk=risk, drop_risk=True, **kwargs)
                    eps = null + cfg * (eps - null)
                mean, logvar = self.mean_var(x, t, eps, var, clip=True)
                x = mean
                if k:
                    x = x + (.5 * logvar).exp() * torch.randn(shape, device=device, generator=generator)
                x = x * (1 - mask)
            return torch.where(mask.bool().expand_as(x), history, x)
        finally:
            model.train(was_training)
