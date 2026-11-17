"""
Unit tests for Phase 7 Experiment Models and Augmentations.
"""

import unittest
from PIL import Image
import numpy as np
import torch

from src.experiments.models import SiameseUNetConc, SiameseUNetNoSkip
from src.experiments.augmentations import SynchronousBitemporalAugmentation
from src.model import SiameseUNet


class TestExperimentModules(unittest.TestCase):
    def test_concatenation_model_forward(self):
        model = SiameseUNetConc(in_channels=3, out_channels=1, base_channels=16)
        t1 = torch.randn(2, 3, 64, 64)
        t2 = torch.randn(2, 3, 64, 64)
        logits = model(t1, t2)
        self.assertEqual(logits.shape, (2, 1, 64, 64))
        self.assertGreater(model.count_parameters(), 1_500_000)

    def test_no_skip_model_forward(self):
        model = SiameseUNetNoSkip(in_channels=3, out_channels=1, base_channels=16)
        t1 = torch.randn(2, 3, 64, 64)
        t2 = torch.randn(2, 3, 64, 64)
        logits = model(t1, t2)
        self.assertEqual(logits.shape, (2, 1, 64, 64))
        self.assertLess(model.count_parameters(), 1_964_097)

    def test_synchronous_augmentation_alignment(self):
        aug = SynchronousBitemporalAugmentation(hflip_prob=1.0, vflip_prob=0.0, rotate_prob=0.0)
        # Create asymmetric PIL images
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[0:10, 0:10, :] = 255  # White square at top-left
        im1 = Image.fromarray(arr)
        im2 = Image.fromarray(arr)
        mask = Image.fromarray(arr[:, :, 0])

        im1_aug, im2_aug, mask_aug = aug(im1, im2, mask)

        # After horizontal flip, the square must be at top-right [0:10, 22:32]
        arr1_aug = np.array(im1_aug)
        arr2_aug = np.array(im2_aug)
        mask_arr = np.array(mask_aug)

        self.assertTrue(np.all(arr1_aug[0:10, 22:32, :] == 255))
        self.assertTrue(np.all(arr2_aug[0:10, 22:32, :] == 255))
        self.assertTrue(np.all(mask_arr[0:10, 22:32] == 255))
        self.assertEqual(arr1_aug[0, 0, 0], 0)


if __name__ == '__main__':
    unittest.main()

