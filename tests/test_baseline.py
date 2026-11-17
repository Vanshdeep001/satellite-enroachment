"""
Unit tests for the traditional pixel-difference baseline in src/baseline.py.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from src.baseline import (
    compute_difference_map,
    threshold_difference_map,
    detect_change,
    to_numpy_image
)
from src.dataset import LEVIRCDDataset


class TestBaseline(unittest.TestCase):
    def test_identical_images_produce_zero_change(self):
        # 256x256 image with arbitrary pattern
        np.random.seed(42)
        img1 = np.random.rand(256, 256, 3).astype(np.float32)
        img2 = img1.copy()

        diff_map = compute_difference_map(img1, img2, method='euclidean')
        self.assertEqual(diff_map.shape, (256, 256))
        self.assertAlmostEqual(float(diff_map.max()), 0.0, places=6)
        self.assertAlmostEqual(float(diff_map.min()), 0.0, places=6)

        pred_mask = threshold_difference_map(diff_map, threshold=0.1)
        self.assertEqual(int(pred_mask.sum()), 0, "Identical images should predict 0 changed pixels.")

    def test_known_pixel_difference(self):
        # Image 1 is completely black [0, 0, 0]
        # Image 2 has a 10x10 pure white block [1, 1, 1] at top-left
        img1 = np.zeros((50, 50, 3), dtype=np.float32)
        img2 = np.zeros((50, 50, 3), dtype=np.float32)
        img2[0:10, 0:10, :] = 1.0

        # In RGB Euclidean, sqrt((1^2 + 1^2 + 1^2)) / sqrt(3) = 1.0
        diff_map = compute_difference_map(img1, img2, method='euclidean')
        self.assertEqual(diff_map.shape, (50, 50))
        self.assertAlmostEqual(float(diff_map[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(diff_map[20, 20]), 0.0, places=5)

        # Threshold at 0.5
        pred_mask = threshold_difference_map(diff_map, threshold=0.5)
        self.assertEqual(int(pred_mask.sum()), 100, "10x10 patch should produce exactly 100 changed pixels.")
        self.assertTrue(np.all(pred_mask[0:10, 0:10] == 1.0))
        self.assertTrue(np.all(pred_mask[10:, 10:] == 0.0))

    def test_difference_methods_bounds(self):
        img1 = np.ones((20, 20, 3), dtype=np.float32)
        img2 = np.zeros((20, 20, 3), dtype=np.float32)

        for method in ['euclidean', 'mad', 'grayscale']:
            diff = compute_difference_map(img1, img2, method=method)
            self.assertEqual(diff.shape, (20, 20))
            self.assertGreaterEqual(float(diff.min()), 0.0)
            self.assertLessEqual(float(diff.max()), 1.0)
            self.assertAlmostEqual(float(diff.max()), 1.0, places=4)

    def test_threshold_validation_errors(self):
        diff = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            threshold_difference_map(diff, threshold=-0.1)
        with self.assertRaises(ValueError):
            threshold_difference_map(diff, threshold=1.5)

    def test_input_conversion_with_torch_tensors(self):
        t1 = torch.zeros(3, 32, 32)
        t2 = torch.ones(3, 32, 32)
        diff, pred = detect_change(t1, t2, threshold=0.5)
        self.assertEqual(diff.shape, (32, 32))
        self.assertEqual(pred.shape, (32, 32))
        self.assertEqual(int(pred.sum()), 32 * 32)

    def test_with_actual_levir_cd_sample(self):
        data_dir = os.path.abspath('data')
        val_dir = os.path.join(data_dir, 'val', 'A')
        if not os.path.isdir(val_dir):
            self.skipTest("Validation dataset not available on disk.")

        dataset = LEVIRCDDataset(root_dir=data_dir, split='val', target_size=(256, 256))
        t1, t2, mask = dataset[0]

        diff_map, pred_mask = detect_change(t1, t2, threshold=0.20)
        self.assertEqual(diff_map.shape, (256, 256))
        self.assertEqual(pred_mask.shape, (256, 256))
        self.assertGreaterEqual(float(diff_map.min()), 0.0)
        self.assertLessEqual(float(diff_map.max()), 1.0)
        # Check binary
        unique_pred = set(np.unique(pred_mask).tolist())
        self.assertTrue(unique_pred.issubset({0.0, 1.0}))


if __name__ == '__main__':
    unittest.main()

