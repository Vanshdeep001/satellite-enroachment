"""
Unit tests for the LEVIRCDDataset and DataLoader in src/dataset.py.
"""

import os
import unittest
import torch
from src.dataset import LEVIRCDDataset, get_dataloader


class TestLEVIRCDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path relative to project root
        cls.root_dir = os.path.abspath('data')
        cls.val_exists = os.path.isdir(os.path.join(cls.root_dir, 'val', 'A'))

    def test_dataset_initialization(self):
        if not self.val_exists:
            self.skipTest("Validation data not yet extracted.")
        ds = LEVIRCDDataset(root_dir=self.root_dir, split='val', target_size=(256, 256))
        self.assertGreater(len(ds), 0, "Dataset should have at least 1 sample.")
        self.assertEqual(len(ds), 64, "Validation split should have exactly 64 pairs.")

    def test_sample_shapes_and_types(self):
        if not self.val_exists:
            self.skipTest("Validation data not yet extracted.")
        ds = LEVIRCDDataset(root_dir=self.root_dir, split='val', target_size=(256, 256))
        t1, t2, mask = ds[0]

        # Check tensor types
        self.assertIsInstance(t1, torch.Tensor)
        self.assertIsInstance(t2, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

        # Check shapes: [3, 256, 256] and [1, 256, 256]
        self.assertEqual(t1.shape, (3, 256, 256))
        self.assertEqual(t2.shape, (3, 256, 256))
        self.assertEqual(mask.shape, (1, 256, 256))

        # Check value ranges
        self.assertGreaterEqual(t1.min().item(), 0.0)
        self.assertLessEqual(t1.max().item(), 1.0)
        self.assertGreaterEqual(t2.min().item(), 0.0)
        self.assertLessEqual(t2.max().item(), 1.0)

        # Check mask binary constraint {0.0, 1.0}
        unique_vals = set(torch.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}), f"Mask must be binary, got: {unique_vals}")

    def test_dataloader_batching(self):
        if not self.val_exists:
            self.skipTest("Validation data not yet extracted.")
        loader = get_dataloader(
            root_dir=self.root_dir,
            split='val',
            batch_size=4,
            target_size=(256, 256),
            shuffle=False
        )
        b_t1, b_t2, b_mask = next(iter(loader))
        self.assertEqual(b_t1.shape, (4, 3, 256, 256))
        self.assertEqual(b_t2.shape, (4, 3, 256, 256))
        self.assertEqual(b_mask.shape, (4, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()

