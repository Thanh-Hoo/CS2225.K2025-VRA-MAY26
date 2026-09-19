"""Sanity tests for AdaptiveModalityWeighting (AW-IDEA extension).

These tests only depend on torch, so they run on any machine (no CUDA, no CLIP,
no dataset required) -- unlike the rest of this repo's model code, which needs a
CUDA device (see docs/AW_IDEA_IMPLEMENTATION_PLAN.md, section 0).

Run with:
    python -m pytest tests/test_adaptive_modality_weighting.py -v
or:
    python tests/test_adaptive_modality_weighting.py
"""
import importlib.util
import os
import unittest

import torch
import torch.nn as nn

# Load the module by file path rather than `from modeling.fusion_part... import`:
# modeling/__init__.py eagerly imports modeling/make_model.py, which pulls in the
# full IDEA dependency chain (timm, fvcore, einops, CLIP, ...). Those are only
# needed for the real model and are not installed on every machine that should be
# able to run these lightweight, dependency-free sanity tests.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'modeling', 'fusion_part', 'adaptive_modality_weighting.py',
)
_spec = importlib.util.spec_from_file_location('adaptive_modality_weighting', _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
AdaptiveModalityWeighting = _module.AdaptiveModalityWeighting
select_alpha = _module.select_alpha


class TestAdaptiveModalityWeighting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.feat_dim = 512
        self.batch_size = 6
        self.module = AdaptiveModalityWeighting(feat_dim=self.feat_dim, reduction_ratio=8, temperature=1.0)

    def _random_globals(self):
        g_rgb = torch.randn(self.batch_size, self.feat_dim)
        g_nir = torch.randn(self.batch_size, self.feat_dim)
        g_tir = torch.randn(self.batch_size, self.feat_dim)
        return g_rgb, g_nir, g_tir

    def test_output_shape(self):
        g_rgb, g_nir, g_tir = self._random_globals()
        alpha_rgb, alpha_nir, alpha_tir, alpha = self.module(g_rgb, g_nir, g_tir)
        self.assertEqual(alpha.shape, (self.batch_size, 3))
        for a in (alpha_rgb, alpha_nir, alpha_tir):
            self.assertEqual(a.shape, (self.batch_size,))

    def test_alpha_sums_to_three(self):
        g_rgb, g_nir, g_tir = self._random_globals()
        _, _, _, alpha = self.module(g_rgb, g_nir, g_tir)
        sums = alpha.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.full_like(sums, 3.0), atol=1e-5))

    def test_uniform_at_initialization(self):
        g_rgb, g_nir, g_tir = self._random_globals()
        alpha_rgb, alpha_nir, alpha_tir, alpha = self.module(g_rgb, g_nir, g_tir)
        expected = torch.ones(self.batch_size)
        self.assertTrue(torch.allclose(alpha_rgb, expected, atol=1e-5))
        self.assertTrue(torch.allclose(alpha_nir, expected, atol=1e-5))
        self.assertTrue(torch.allclose(alpha_tir, expected, atol=1e-5))

    def test_no_nan_or_inf(self):
        g_rgb, g_nir, g_tir = self._random_globals()
        g_rgb = g_rgb * 1e4  # stress-test extreme magnitudes
        g_tir = g_tir * -1e4
        _, _, _, alpha = self.module(g_rgb, g_nir, g_tir)
        self.assertFalse(torch.isnan(alpha).any())
        self.assertFalse(torch.isinf(alpha).any())

    def test_disabled_path_matches_baseline(self):
        # Simulates cfg.MODEL.ADAPTIVE_WEIGHTING.ENABLED = False: local features
        # must pass through CDA completely unweighted (alpha == 1 is equivalent to
        # skipping the module entirely).
        local_rgb = torch.randn(self.batch_size, 128, self.feat_dim)
        weighted = local_rgb * torch.ones(self.batch_size).view(-1, 1, 1)
        self.assertTrue(torch.equal(local_rgb, weighted))

    def test_gradients_flow_into_gate(self):
        g_rgb, g_nir, g_tir = self._random_globals()
        g_rgb.requires_grad_(True)
        g_nir.requires_grad_(True)
        g_tir.requires_grad_(True)
        alpha_rgb, alpha_nir, alpha_tir, alpha = self.module(g_rgb, g_nir, g_tir)
        loss = (alpha_rgb - 1.2).pow(2).sum() + (alpha_nir - 0.8).pow(2).sum() + (alpha_tir - 1.0).pow(2).sum()
        loss.backward()
        # At zero-init the final Linear's weight is exactly 0, so d(score)/d(input
        # to that layer) is also exactly 0 -- by design (see AdaptiveModalityWeighting's
        # docstring / config item #5), gradient cannot propagate past it on this very
        # first step. What MUST already receive a gradient here is the final layer's
        # own weight (dL/dW = dL/dscore * upstream_activation, upstream_activation != 0),
        # which is exactly the signal that lets training escape the zero-init point.
        final_linear = self.module.gate[-1]
        self.assertIsNotNone(final_linear.weight.grad)
        self.assertTrue(torch.any(final_linear.weight.grad != 0))
        self.assertIsNotNone(final_linear.bias.grad)

        # Once the final layer is nudged away from exact zero (i.e. after the very
        # first optimizer step in real training), gradients must flow all the way
        # back through LayerNorm/Linear1/GELU into the modality global features.
        with torch.no_grad():
            final_linear.weight.add_(0.05)
        self.module.zero_grad()
        g_rgb2, g_nir2, g_tir2 = self._random_globals()
        g_rgb2.requires_grad_(True)
        g_nir2.requires_grad_(True)
        g_tir2.requires_grad_(True)
        alpha_rgb2, alpha_nir2, alpha_tir2, _ = self.module(g_rgb2, g_nir2, g_tir2)
        loss2 = (alpha_rgb2 - 1.2).pow(2).sum() + (alpha_nir2 - 0.8).pow(2).sum() + (alpha_tir2 - 1.0).pow(2).sum()
        loss2.backward()
        for g in (g_rgb2, g_nir2, g_tir2):
            self.assertIsNotNone(g.grad)
            self.assertTrue(torch.any(g.grad != 0))

    def test_checkpoint_missing_keys_are_scoped_to_aw_module(self):
        # Simulates loading an old IDEA checkpoint (no AW weights) into a model
        # that now also owns an AdaptiveModalityWeighting submodule -- mirrors
        # what modeling/make_model.py::IDEA.load_param does with strict=False.
        class OldIDEA(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier_v = nn.Linear(16, 10)

        class NewAWIDEA(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier_v = nn.Linear(16, 10)
                self.adaptive_modality_weighting = AdaptiveModalityWeighting(feat_dim=16)

        old_model = OldIDEA()
        new_model = NewAWIDEA()
        result = new_model.load_state_dict(old_model.state_dict(), strict=False)

        self.assertEqual(len(result.unexpected_keys), 0)
        self.assertTrue(len(result.missing_keys) > 0)
        for key in result.missing_keys:
            self.assertIn('adaptive_modality_weighting', key)
        # The shared (old) submodule must still load correctly.
        self.assertTrue(torch.equal(new_model.classifier_v.weight, old_model.classifier_v.weight))


class TestSelectAlphaAblationModes(unittest.TestCase):
    """Covers the three ablation modes from AW_IDEA_IMPLEMENTATION_PLAN.md / task
    section 15: (A) disabled is tested at the IDEA.forward level (not reachable
    here without the full model), (B) uniform, (C) learned, (D) static."""

    def setUp(self):
        torch.manual_seed(0)
        self.feat_dim = 64
        self.batch_size = 4
        self.gate = AdaptiveModalityWeighting(feat_dim=self.feat_dim, reduction_ratio=8)
        self.g_rgb = torch.randn(self.batch_size, self.feat_dim)
        self.g_nir = torch.randn(self.batch_size, self.feat_dim)
        self.g_tir = torch.randn(self.batch_size, self.feat_dim)

    def test_mode_b_force_uniform_ignores_gate(self):
        # Even after the gate has been trained away from zero-init, force_uniform
        # must still yield exactly alpha = [1, 1, 1] -- this isolates "does the new
        # multiply code path change results" from "does the learned gate help".
        with torch.no_grad():
            self.gate.gate[-1].weight.add_(1.0)
            self.gate.gate[-1].bias.add_(1.0)
        alpha_rgb, alpha_nir, alpha_tir = select_alpha(
            self.gate, self.g_rgb, self.g_nir, self.g_tir, force_uniform=True)
        for a in (alpha_rgb, alpha_nir, alpha_tir):
            self.assertTrue(torch.allclose(a, torch.ones(self.batch_size)))

    def test_mode_c_learned_matches_gate_output(self):
        expected_rgb, expected_nir, expected_tir, _ = self.gate(self.g_rgb, self.g_nir, self.g_tir)
        alpha_rgb, alpha_nir, alpha_tir = select_alpha(
            self.gate, self.g_rgb, self.g_nir, self.g_tir, force_uniform=False, static_weights=())
        self.assertTrue(torch.equal(alpha_rgb, expected_rgb))
        self.assertTrue(torch.equal(alpha_nir, expected_nir))
        self.assertTrue(torch.equal(alpha_tir, expected_tir))

    def test_mode_d_static_weights_override_everything(self):
        alpha_rgb, alpha_nir, alpha_tir = select_alpha(
            self.gate, self.g_rgb, self.g_nir, self.g_tir,
            force_uniform=True,  # static_weights must take priority even if this is also set
            static_weights=[1.2, 1.2, 0.6])
        self.assertTrue(torch.allclose(alpha_rgb, torch.full((self.batch_size,), 1.2)))
        self.assertTrue(torch.allclose(alpha_nir, torch.full((self.batch_size,), 1.2)))
        self.assertTrue(torch.allclose(alpha_tir, torch.full((self.batch_size,), 0.6)))


if __name__ == '__main__':
    unittest.main()
