"""
Unit tests for Step 7-9 additions: generative augmenters, temporal change models,
Siamese ResNet change detector (incl. inference loading) and the land-use module.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from src.advanced import generative as G
from src.advanced import temporal as TM
from src.landuse import build_model, landuse_transitions, load_synthetic, stratified_split
from src.resnet_change import SiameseResNetCD


class TestGenerative(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(12, 3, 16, 16)
        self.y = torch.arange(4).repeat(3)

    def test_cvae_shapes_and_loss(self):
        model = G.CVAE(num_classes=4, latent_dim=6)
        recon, mu, logvar = model(self.x, self.y)
        self.assertEqual(recon.shape, self.x.shape)
        self.assertEqual(mu.shape, (12, 6))
        losses = G.CVAE.loss(recon, self.x, mu, logvar)
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertGreaterEqual(losses['kl'].item(), 0.0)
        s = model.eval().sample(self.y)
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_unconditional_vae_and_training_reduces_loss(self):
        model, hist = G.train_cvae(self.x, self.y, num_classes=0, epochs=40, batch_size=6)
        self.assertLess(hist[-1]['loss'], hist[0]['loss'])
        self.assertEqual(model.encode(self.x, self.y)[0].shape[1], model.latent_dim)

    def test_kl_zero_for_standard_normal_posterior(self):
        mu, logvar = torch.zeros(3, 4), torch.zeros(3, 4)
        x = torch.full((3, 3, 4, 4), 0.5)
        self.assertAlmostEqual(G.CVAE.loss(x, x, mu, logvar)['kl'].item(), 0.0, places=6)

    def test_cgan_shapes_and_training(self):
        gen, hist = G.train_cgan(self.x, self.y, steps=20, batch_size=6)
        out = gen.sample(self.y)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(all(np.isfinite(h['d_loss']) and np.isfinite(h['g_loss']) for h in hist))
        disc = G.CondDiscriminator()
        self.assertEqual(disc(self.x, self.y).shape, (12,))

    def test_dihedral_preserves_pixels(self):
        out = G.dihedral_torch(self.x)
        for a, b in zip(self.x, out):
            torch.testing.assert_close(a.flatten().sort().values, b.flatten().sort().values)

    def test_classifier_metrics_and_diversity(self):
        class Oracle(nn.Module):
            def forward(self, x):
                return torch.nn.functional.one_hot(torch.arange(4).repeat(3), 4).float()
        self.assertEqual(G.evaluate_classifier(Oracle(), self.x, self.y), {'accuracy': 1.0, 'macro_f1': 1.0})
        self.assertEqual(G.diversity(self.x[:1].repeat(5, 1, 1, 1)), 0.0)
        self.assertGreater(G.diversity(self.x), 0.0)


class TestTemporal(unittest.TestCase):
    def test_sequence_generator(self):
        frames, dates, clouds = TM.make_sequence(size=40, steps=5, seed=1, force_cloudy=[4])
        self.assertEqual(frames.shape, (5, 40, 40, 3))
        self.assertTrue(0 <= frames.min() and frames.max() <= 1)
        self.assertTrue(set(np.unique(dates)) <= set(range(5)))
        self.assertGreater(dates.max(), 0)
        self.assertGreater(clouds[4].max(), 0)
        self.assertEqual(clouds[0].max(), 0)          # first date is never cloudy

    def test_models_output_shapes(self):
        data = TM.make_sequence_dataset(2, seed=0, size=32, steps=4)
        self.assertEqual(tuple(data['x'].shape), (2, 4, 3, 32, 32))
        for name, builder in TM.MODEL_BUILDERS.items():
            out = builder(4)(data['x'])
            self.assertEqual(tuple(out.shape), (2, 4, 32, 32), msg=name)

    def test_attention_and_hard_gates(self):
        x = TM.make_sequence_dataset(1, seed=0, size=32, steps=4)['x']
        tr = TM.TemporalTransformerCD(4)
        tr.attn.keep_attention = True
        tr(x)
        att = tr.attn.last_attention
        self.assertEqual(tuple(att.shape), (32 * 32, 2, 4, 4))
        torch.testing.assert_close(att.sum(-1), torch.ones(32 * 32, 2, 4))
        lstm = TM.ConvLSTMCD(4)
        lstm.cell.hard_gates = True
        self.assertEqual(tuple(lstm(x).shape), (1, 4, 32, 32))

    def test_metrics_hand_computed(self):
        y = torch.tensor([[[0, 1], [2, 0]]])
        pred = torch.tensor([[[0, 1], [3, 1]]])
        m = TM.temporal_metrics(pred, y)
        self.assertAlmostEqual(m['f1'], 2 * 2 / (2 * 2 + 1 + 0))
        self.assertAlmostEqual(m['date_accuracy'], 0.5)
        self.assertAlmostEqual(m['change_recall'], 1.0)

    def test_training_smoke(self):
        tr = TM.make_sequence_dataset(4, seed=1, size=32, steps=4)
        va = TM.make_sequence_dataset(2, seed=2, size=32, steps=4)
        info = TM.train_temporal(TM.Conv3DCD(4), tr, va, 4, epochs=2, batch_size=2)
        self.assertEqual(len(info['history']), 2)


class TestResNetChange(unittest.TestCase):
    def test_forward_both_fusions(self):
        t1, t2 = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
        for fusion in ('diff', 'concat'):
            model = SiameseResNetCD(fusion=fusion)
            out = model(t1, t2)
            self.assertEqual(tuple(out.shape), (2, 1, 64, 64))
            out.mean().backward()
            self.assertIsNotNone(model.stem[0].weight.grad)

    def test_diff_fusion_is_symmetric(self):
        model = SiameseResNetCD(fusion='diff').eval()
        t1, t2 = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(model(t1, t2), model(t2, t1))

    def test_validation(self):
        model = SiameseResNetCD()
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 60, 64), torch.rand(1, 3, 60, 64))
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))
        with self.assertRaises(ValueError):
            SiameseResNetCD(fusion='sum')

    def test_inference_engine_loads_resnet_checkpoint(self):
        from src.inference import ChangeDetectorInference
        model = SiameseResNetCD(fusion='concat')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'rcd.pt')
            torch.save({'model_state_dict': model.state_dict(), 'architecture_class': 'SiameseResNetCD',
                        'backbone': 'resnet18', 'fusion': 'concat', 'optimal_threshold': 0.3}, path)
            engine = ChangeDetectorInference(checkpoint_path=path, device='cpu')
            img = (np.random.default_rng(0).random((96, 96, 3)) * 255).astype(np.uint8)
            res = engine.predict(img, img)
        self.assertEqual(res.prob_map.shape, (96, 96))
        self.assertEqual(engine.threshold, 0.3)
        self.assertIn('SiameseResNetCD', res.model_name)


class TestLanduse(unittest.TestCase):
    def test_stratified_split(self):
        files = {c: [f'{c}_{i}.jpg' for i in range(20)] for c in ('a', 'b')}
        split = stratified_split(files, seed=0)
        allf = [f for v in split.values() for f, _ in v]
        self.assertEqual(len(allf), 40)
        self.assertEqual(len(set(allf)), 40)                     # no file in two splits
        self.assertEqual(sum(1 for _, y in split['train'] if y == 0), 14)
        self.assertEqual(stratified_split(files, seed=0), split)  # reproducible

    def test_models_build(self):
        x = torch.rand(2, 3, 64, 64)
        for name in ('resnet18', 'googlenet'):
            model = build_model(name, 5).eval()
            self.assertEqual(tuple(model(x).shape), (2, 5), msg=name)
        with self.assertRaises(ValueError):
            build_model('vgg', 3)

    def test_synthetic_loader(self):
        data, classes = load_synthetic(8, 4, 4, size=64)
        self.assertEqual(tuple(data['train'][0].shape), (8, 3, 64, 64))
        self.assertEqual(len(classes), 4)

    def test_transitions_with_brightness_classifier(self):
        class Bright(nn.Module):
            def forward(self, x):
                b = x.mean(dim=(1, 2, 3))
                return torch.stack([-b, b], dim=1)   # class 1 if bright
        t1 = np.zeros((128, 192, 3), np.float32)
        t2 = t1.copy()
        t2[:64, :64] = 1.0                           # one tile turns bright
        res = landuse_transitions(Bright(), t1, t2, ['dark', 'bright'], tile=64, input_size=32)
        self.assertEqual(np.array(res['grid_t1']).shape, (2, 3))
        self.assertEqual(res['transition_matrix'], [[5, 1], [0, 0]])
        self.assertEqual(res['changed_tiles'][0]['to'], 'bright')
        with self.assertRaises(ValueError):
            landuse_transitions(Bright(), t1[:32, :32], t2[:32, :32], ['d', 'b'], tile=64)


if __name__ == '__main__':
    unittest.main()
