import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hdg.representation import flatten_agents, unflatten_agents, history_mask, decode, normalize
from hdg.models import Attention, TemporalDiT, SpatialDiT, AxialTemporalDiT
from hdg.diffusion import Diffusion
from hdg.metrics import kinematics, overlap, pool_episodes, frechet, pair_ttc, evaluate
from hdg.risk import assess_risk, assess_risk_vectorized
from hdg.road import MergeRoad
from hdg.features import augment, ntxent
from hdg.closed_loop import run_closed_loop, verify_fixed_replay, ReactivePlanner


class EchoDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, t, risk, history=None, mask=None, initial=None, drop_risk=False):
        self.calls.append((x.clone(), history.clone(), mask.clone(), initial.clone(), drop_risk))
        return torch.zeros_like(x) + self.anchor, None


class HDGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_agent_major_roundtrip(self):
        x = torch.arange(2*3*4*5).reshape(2, 3, 4, 5)
        flat = flatten_agents(x)
        self.assertTrue(torch.equal(flat[0, :3], x[0, :, 0]))
        self.assertTrue(torch.equal(unflatten_agents(flat), x))

    def test_structural_bias(self):
        attn = Attention(16, 4, features=6)
        self.assertEqual(attn.bias().abs().sum(), 0)
        with torch.no_grad():
            attn.alpha.fill_(2); attn.beta.fill_(-3)
        bias = attn.bias()
        self.assertEqual(bias[0, 0, 1], 2)
        self.assertEqual(bias[0, 0, 3], -3)
        self.assertTrue(torch.equal(bias.diagonal(dim1=-2, dim2=-1), torch.zeros(4, 6)))
        attn(torch.randn(2, 6, 16)).square().sum().backward()
        self.assertGreater(attn.alpha.grad.abs().sum(), 0)
        self.assertGreater(attn.beta.grad.abs().sum(), 0)

    def test_future_corruption_and_loss(self):
        diffusion = Diffusion(10)
        model = EchoDenoiser()
        x = torch.randn(2, 6, 8)
        mask = history_mask(torch.tensor([2, 6]), 8)
        eps = torch.ones_like(x)
        eps[mask.expand_as(eps).bool()] = 999
        t = torch.tensor([1, 5])
        result = diffusion.loss(model, x, torch.zeros(2), mask, torch.randn(2, 3, 28, 3), t, eps)
        self.assertAlmostEqual(result['mse'].item(), 1.)
        noisy, history, _, _, _ = model.calls[0]
        self.assertEqual((noisy * mask).abs().sum(), 0)
        self.assertTrue(torch.equal(history, x * mask))

    def test_history_exact_at_every_step_and_cfg(self):
        model = EchoDenoiser()
        diffusion = Diffusion(6)
        x = torch.randn(2, 6, 8)
        initial = torch.randn(2, 3, 28, 3)
        mask = history_mask(torch.tensor([3, 8]), 8)
        out = diffusion.sample(model, tuple(x.shape), torch.ones(2), x, mask, initial, cfg=3.)
        self.assertTrue(torch.equal(out[mask.expand_as(x).bool()], x[mask.expand_as(x).bool()]))
        self.assertEqual(len(model.calls), 12)
        for noisy, history, m, condition, _ in model.calls:
            self.assertEqual((noisy * mask).abs().sum(), 0)
            self.assertTrue(torch.equal(history, x * mask))
            self.assertTrue(torch.equal(condition, initial))
            self.assertTrue(torch.equal(mask, m))
        before = len(model.calls)
        out = diffusion.sample(model, tuple(x.shape), torch.ones(2), x, torch.ones(2, 1, 8), initial)
        self.assertTrue(torch.equal(out, x)); self.assertEqual(len(model.calls), before)

    def test_zero_history_and_all_history_losses(self):
        model = TemporalDiT(agents=2, horizon=8, dim=16, depth=1, heads=4)
        d = Diffusion(10)
        x = torch.randn(2, 6, 8)
        initial = torch.randn(2, 3, 28, 3)
        for p in [0, 3, 8]:
            loss = d.loss(model, x, torch.ones(2), history_mask(torch.tensor([p, p]), 8), initial)['loss']
            self.assertTrue(torch.isfinite(loss))
            if p == 8:
                self.assertEqual(loss.item(), 0)
            else:
                loss.backward()

    def test_unconditional_sampling_drops_only_risk(self):
        model = EchoDenoiser()
        clean = torch.rand(2, 6, 8)
        initial = torch.rand(2, 3, 28, 3)
        mask = history_mask(torch.tensor([1, 3]), 8)
        Diffusion(6).sample(model, tuple(clean.shape), torch.ones(2), clean, mask, initial,
                            cfg=4., unconditional_risk=True)
        self.assertEqual(len(model.calls), 6)
        for _, history, observed, context, risk_dropped in model.calls:
            self.assertTrue(risk_dropped)
            self.assertTrue(torch.equal(history, clean * mask))
            self.assertTrue(torch.equal(context, initial))

    def test_dimensions_48_agents(self):
        model = TemporalDiT(agents=48, horizon=6, dim=16, depth=1, heads=4)
        x = torch.zeros(1, 144, 6)
        eps, var = model(x, torch.zeros(1), torch.ones(1), x, torch.zeros(1, 1, 6), torch.zeros(1, 3, 28, 3))
        self.assertEqual(eps.shape, x.shape); self.assertEqual(var.shape, x.shape)
        spatial = SpatialDiT(dim=16, depth=1)
        s = torch.zeros(2, 3, 28, 3)
        self.assertEqual(spatial(s, torch.zeros(2), torch.ones(2))[0].shape, s.shape)

    def test_axial_mask_and_full_horizon(self):
        model = AxialTemporalDiT(agents=2, horizon=140, dim=16, depth=1, heads=4)
        clean = torch.rand(2, 6, 140)
        mask = history_mask(torch.tensor([1, 140]), 140)
        initial = torch.randn(2, 3, 28, 3)
        losses = Diffusion(6).loss(model, clean, torch.ones(2), mask, initial)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertGreater(model.blocks[0].feature.alpha.grad.abs().sum(), 0)
        self.assertGreater(model.blocks[0].feature.beta.grad.abs().sum(), 0)
        result = Diffusion(6).sample(model, tuple(clean.shape), torch.ones(2), clean, mask, initial, cfg=2.)
        self.assertTrue(torch.equal(result[1], clean[1]))
        self.assertTrue(torch.equal(result[0, :, 0], clean[0, :, 0]))
        self.assertTrue(torch.isfinite(result).all())

    def test_decoding_invalid_band(self):
        x = np.array([[-1., -1, -1], [-.5, .2, .3], [-.05, 1.2, .5], [-1, .3, .4], [np.nan, 0, 0]])
        s, valid, bad = decode(x)
        np.testing.assert_array_equal(valid, [False, False, True, False, False])
        np.testing.assert_array_equal(bad, [False, True, False, True, True])
        self.assertEqual(s[2, 0], 1000); self.assertEqual(s[2, 1], 955)

    def test_kinematics_and_padding_gaps(self):
        s = np.zeros((1, 8, 3)); s[0, :, 0] = np.arange(8)
        v = np.ones((1, 8), bool)
        self.assertFalse(kinematics(s, v)['violation'][0])

        v[:, 3:5] = False; s[:, 5:, 0] += 100
        self.assertFalse(kinematics(s, v)['violation'][0])
        v[:] = True
        self.assertTrue(kinematics(s, v)['violation'][0])
        v[:] = False; v[0, :2] = True
        self.assertTrue(kinematics(s, v)['failure'][0])

    def test_stationary_and_nonfinite_kinematics(self):
        s = np.zeros((1, 6, 3)); s[0, :, 2] = np.arange(6)
        valid = np.ones((1, 6), bool)
        self.assertFalse(kinematics(s, valid)['violation'][0])
        s[0, 2, 0] = np.nan
        self.assertTrue(kinematics(s, valid)['failure'][0])

    def test_geometry_and_pooled_rates(self):
        self.assertTrue(overlap(np.array([0., 0., 0.]), np.array([1., 0., .2])))
        self.assertFalse(overlap(np.array([0., 0., 0.]), np.array([10., 0., .2])))
        e = [dict(collision=True, at_fault=True, verified=True), dict(collision=True, at_fault=False, verified=False)]
        e += [dict(collision=False, at_fault=False, verified=False)] * 8
        m = pool_episodes(e)
        self.assertEqual(m['vafr_pct'], 10); self.assertEqual(m['varaf_pct'], 100)
        self.assertIsNone(pool_episodes(e[2:])['varc_pct'])

    def test_ttc_and_nonfinite_generation_failure(self):
        a, b = np.array([0., 0., 0.]), np.array([10., 0., 0.])
        self.assertAlmostEqual(pair_ttc(a, b, np.array([2., 0.]), np.zeros(2)), 2.75)
        self.assertIsNone(pair_ttc(a, b, np.zeros(2), np.array([2., 0.])))
        x = np.zeros((1, 3, 1, 8), dtype=np.float32)
        x[0, 0, 0] = .5 + np.arange(8) * .001
        x[0, 1:, 0] = .5
        x[0, 0, 0, 3] = np.nan
        result = evaluate(x, MergeRoad())
        self.assertIsNone(result['kv_pct'])
        self.assertEqual(result['unscorable_nonempty_trajectories'], 1)

    def test_layout_roundtrip_and_risk(self):
        road = MergeRoad()
        s = np.array([[1010., road.center(0, 1010.), 0.], [1040., road.center(1, 1040.), .02]])
        grid, conflicts = road.layout(s, np.ones(2, bool))
        restored, bad = road.decode_layout(grid)
        self.assertEqual(conflicts, 0); self.assertEqual(bad, 0)
        np.testing.assert_allclose(restored, s, atol=1e-5)
        states = np.zeros((2, 8, 3))
        states[0, :, 0] = 1010 + np.arange(8)
        states[1, :, 0] = 1016 + np.arange(8) * .1
        for i in range(2):
            states[i, :, 1] = road.center(0, states[i, :, 0])
        risk = assess_risk(states, np.ones((2, 8), bool), road)
        self.assertGreater(risk[0], 0); self.assertEqual(risk[1], 0)

    def test_mask_aware_contrastive(self):
        x = torch.tensor([[[-1., .2, .5]]])
        self.assertEqual(augment(x)[0, 0, 0], -1)
        features = torch.eye(8)
        self.assertLess(ntxent(features, features), ntxent(features, features.roll(1, 0)))
        a = np.random.default_rng(0).normal(size=(30, 4))
        self.assertAlmostEqual(frechet(a, a), 0., places=7)

    def test_vectorized_risk_matches_reference(self):
        road = MergeRoad()
        rng = np.random.default_rng(21)
        for n, w in [(1, 1), (4, 8), (12, 140)]:
            s = np.zeros((n, w, 3))
            s[..., 0] = rng.uniform(1005, 1110, (n, 1)) + np.arange(w)[None] * rng.uniform(.05, .4, (n, 1))
            lanes = rng.integers(0, 3, (n, w))
            for i in range(n):
                for t in range(w):
                    s[i, t, 1] = road.center(lanes[i, t], s[i, t, 0])
            valid = rng.random((n, w)) > .15
            s[~valid] = np.nan
            lengths = rng.uniform(3.5, 5.5, n)
            previous = np.zeros(n)
            for kappa in (0., 1., 1.5, 3.):
                reference = assess_risk(s, valid, road, lengths=lengths, kappa=kappa)
                fast = assess_risk_vectorized(s, valid, road, lengths=lengths, kappa=kappa)
                np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-7)
                self.assertTrue((fast >= previous).all())
                previous = fast

    def test_closed_loop_feedback_and_fixed_replay(self):
        road = MergeRoad()
        class StubSampler:
            def __init__(self):
                self.histories = []
            def sample(self, model, shape, risk, history, mask, initial, cfg):
                self.histories.append(history.clone())
                p = int(mask.sum())
                out = history.clone()
                for t in range(p, shape[-1]):
                    out[..., t] = history[..., p - 1]
                return out
        model, sampler = EchoDenoiser(), StubSampler()
        s = np.array([[1010., road.center(0, 1010.), 0.]])
        initial, _ = road.layout(s, np.ones(1, bool))
        result = run_closed_loop(model, sampler, torch.tensor(initial)[None], normalize(s, np.ones(1, bool)),
                                 [2.], .5, 0, ReactivePlanner(road), np.array([1100., 945., 0.]), 8, execution=3)
        self.assertEqual(result['history_max_error'], 0.)
        self.assertFalse(result['at_fault'])
        self.assertEqual(result['attribution']['events'], [])
        self.assertEqual(result['boundaries'], [1, 4, 7])
        expected = flatten_agents(torch.tensor(result['tracks'])[None])
        self.assertTrue(torch.equal(sampler.histories[1][..., :4], expected[..., :4]))
        verification = verify_fixed_replay(result, [2.], ReactivePlanner(road), np.array([1100., 945., 0.]))
        self.assertTrue(verification['complete'])

    def test_initial_collision_attribution_excludes_postimpact_replay(self):
        road = MergeRoad()
        class RepeatSampler:
            def sample(self, model, shape, risk, history, mask, initial, cfg):
                return history[..., :1].expand(shape).clone()
        y = road.center(0, 1010.)
        states = np.array([[1010., y, 0.], [1014., y, 0.], [1004., y, 0.]])
        initial, _ = road.layout(states, np.ones(3, bool))
        result = run_closed_loop(EchoDenoiser(), RepeatSampler(), torch.tensor(initial)[None],
            normalize(states, np.ones(3, bool)), [0., 0., 20.], 1., 0,
            ReactivePlanner(road), np.array([1100., y, 0.]), 8, execution=3)
        self.assertEqual(result['collision_time'], 0)
        self.assertFalse(result['at_fault'])
        self.assertEqual(len(result['attribution']['events']), 1)
        self.assertEqual(result['attribution']['events'][0]['collision_type'], 'STOPPED_EGO_COLLISION')

    def test_interaction_export_legacy_reader(self):
        from hdg.cli import export_interaction
        from data_process.visualisation.utils.dataset_reader import read_tracks
        x = np.full((1, 3, 2, 5), -1., dtype=np.float32)
        x[0, :, 0] = .5
        x[0, 0, 0] += np.arange(5) * .001
        with tempfile.TemporaryDirectory() as folder:
            export_interaction(x, folder)
            records = read_tracks(Path(folder) / 'scenario_000.csv')
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].motion_states), 5)
            self.assertAlmostEqual(records[0].motion_states[0].x, 1070.)


if __name__ == '__main__':
    unittest.main()
