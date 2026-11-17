"""
Unit tests for Siamese U-Net Architecture and Loss Functions.

Tests:
1. Model initialization & parameter counting.
2. Correct output shape and data type.
3. Shared encoder weight usage.
4. Correct feature difference operation (symmetry & zero-difference identity).
5. Soft Dice Loss and Composite Loss computation.
6. Finite loss values (no NaN or Inf).
7. Gradient propagation across all trainable parameters.
8. Robust input validation and error raising.
9. Real LEVIR-CD sample batch forward pass.
10. Apple Silicon MPS execution verification.
"""

import os
import unittest
import torch

from src.model import SiameseUNet
from src.losses import SoftDiceLoss, CompositeChangeDetectionLoss
from src.dataset import LEVIRCDDataset, get_dataloader
from src.config import CONFIG


class TestSiameseUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_cpu = torch.device('cpu')
        cls.mps_available = torch.backends.mps.is_available()
        cls.device_mps = torch.device('mps') if cls.mps_available else None

    def setUp(self):
        # Use lightweight base_channels=16 for fast, memory-safe testing
        self.model = SiameseUNet(in_channels=3, out_channels=1, base_channels=16, bilinear=True)
        self.model.eval()

    def test_model_initialization_and_params(self):
        """Test parameter counting and memory footprint estimation."""
        params = self.model.count_parameters()
        self.assertGreater(params, 1_000_000, "Lightweight model should have >1M parameters.")
        self.assertLess(params, 5_000_000, "Lightweight model with base_channels=16 should have <5M parameters.")
        size_mb = self.model.model_size_mb()
        self.assertGreater(size_mb, 4.0)
        self.assertLess(size_mb, 20.0)

    def test_output_shape_and_dtype(self):
        """Test forward pass returns [B, 1, H, W] float32 raw logits."""
        b, c, h, w = 2, 3, 256, 256
        t1 = torch.randn(b, c, h, w)
        t2 = torch.randn(b, c, h, w)

        with torch.no_grad():
            logits = self.model(t1, t2)

        self.assertEqual(logits.shape, (b, 1, h, w))
        self.assertEqual(logits.dtype, torch.float32)

    def test_shared_encoder_weight_usage(self):
        """Test that the encoder weights are strictly shared across both streams."""
        # The model should only possess one encoder submodule
        self.assertTrue(hasattr(self.model, 'encoder'))
        self.assertFalse(hasattr(self.model, 'encoder_t1'))
        self.assertFalse(hasattr(self.model, 'encoder_t2'))

        # Feature extraction on the same input through the shared encoder must be identical
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            f1 = self.model.extract_features(x)
            f2 = self.model.extract_features(x)

        for tensor_a, tensor_b in zip(f1, f2):
            self.assertTrue(torch.equal(tensor_a, tensor_b), "Encoder must output identical features for identical inputs.")

    def test_feature_difference_operation(self):
        """Test absolute difference behavior: symmetry and zero-difference on identical inputs."""
        t1 = torch.randn(1, 3, 128, 128)
        t2 = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            f1 = self.model.extract_features(t1)
            f2 = self.model.extract_features(t2)

            # 1. Symmetry: |F_T1 - F_T2| == |F_T2 - F_T1|
            for feat1, feat2 in zip(f1, f2):
                diff_12 = torch.abs(feat1 - feat2)
                diff_21 = torch.abs(feat2 - feat1)
                self.assertTrue(torch.allclose(diff_12, diff_21, atol=1e-6))
                self.assertTrue(torch.all(diff_12 >= 0.0), "Absolute difference must be non-negative.")

            # 2. Zero-difference: when T1 == T2, all differential skips are identically 0
            f_same1 = self.model.extract_features(t1)
            f_same2 = self.model.extract_features(t1)
            for feat_a, feat_b in zip(f_same1, f_same2):
                diff_zero = torch.abs(feat_a - feat_b)
                self.assertTrue(torch.allclose(diff_zero, torch.zeros_like(diff_zero)))

    def test_loss_calculation_and_finiteness(self):
        """Test SoftDiceLoss and CompositeChangeDetectionLoss produce finite positive scalars."""
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)
        targets = (torch.rand(2, 1, 64, 64) > 0.8).to(torch.float32)

        # 1. Soft Dice Loss
        dice_fn = SoftDiceLoss()
        dice_val = dice_fn(logits, targets)
        self.assertTrue(torch.isfinite(dice_val))
        self.assertGreaterEqual(dice_val.item(), 0.0)
        self.assertLessEqual(dice_val.item(), 1.0)

        # 2. Composite Loss
        composite_fn = CompositeChangeDetectionLoss(bce_weight=0.5, dice_weight=0.5)
        total_loss, components = composite_fn(logits, targets, return_components=True)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn('bce_loss', components)
        self.assertIn('dice_loss', components)

    def test_backpropagation_and_gradients(self):
        """Test that backpropagation generates valid non-zero gradients on all trainable weights."""
        self.model.train()
        self.model.zero_grad()

        t1 = torch.randn(1, 3, 64, 64)
        t2 = torch.randn(1, 3, 64, 64)
        target = torch.ones(1, 1, 64, 64)

        loss_fn = CompositeChangeDetectionLoss()
        logits = self.model(t1, t2)
        loss = loss_fn(logits, target)
        loss.backward()

        # Check gradients exist and are non-zero across encoder and decoder layers
        has_grad_count = 0
        total_param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_param_count += 1
                self.assertIsNotNone(param.grad, f"Parameter {name} should have a gradient.")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN.")
                if torch.abs(param.grad).sum() > 0:
                    has_grad_count += 1

        self.assertEqual(has_grad_count, total_param_count, "All parameters must receive non-zero gradients.")

    def test_invalid_input_handling(self):
        """Test that invalid input dimensions or types raise descriptive exceptions."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.model("invalid_input", torch.randn(1, 3, 64, 64))

        # 3D tensor instead of 4D
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 64, 64), torch.randn(3, 64, 64))

        # Mismatched spatial sizes
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 32, 32))

        # Wrong channel count (e.g. 1 channel instead of 3)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64))

    def test_actual_levir_cd_batch_forward_pass(self):
        """Test forward pass using an actual batch from the LEVIR-CD validation split."""
        data_dir = os.path.abspath('data')
        val_dir = os.path.join(data_dir, 'val', 'A')
        if not os.path.isdir(val_dir):
            self.skipTest("Validation data directory not found.")

        loader = get_dataloader(
            root_dir=data_dir,
            split='val',
            batch_size=2,
            target_size=(256, 256),
            shuffle=False
        )
        b_t1, b_t2, b_mask = next(iter(loader))

        with torch.no_grad():
            logits = self.model(b_t1, b_t2)

        self.assertEqual(logits.shape, (2, 1, 256, 256))
        self.assertEqual(logits.dtype, torch.float32)

        # Verify loss can be evaluated on real batch
        loss_fn = CompositeChangeDetectionLoss()
        loss = loss_fn(logits, b_mask)
        self.assertTrue(torch.isfinite(loss))

    def test_apple_silicon_mps_execution(self):
        """Test that the model runs on Apple Silicon MPS with consistent outputs."""
        if not self.mps_available:
            self.skipTest("Apple Silicon MPS not available on this machine.")

        model_mps = SiameseUNet(in_channels=3, out_channels=1, base_channels=16).to(self.device_mps)
        model_mps.eval()

        t1_mps = torch.randn(1, 3, 128, 128, device=self.device_mps)
        t2_mps = torch.randn(1, 3, 128, 128, device=self.device_mps)

        with torch.no_grad():
            logits_mps = model_mps(t1_mps, t2_mps)

        self.assertEqual(logits_mps.device.type, 'mps')
        self.assertEqual(logits_mps.shape, (1, 1, 128, 128))
        self.assertTrue(torch.isfinite(logits_mps).all())


if __name__ == '__main__':
    unittest.main()

