"""
Unit tests for the independent Change Detection Inference Pipeline (Phase 8).
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import ChangeDetectorInference, ChangeDetectionResult


class TestChangeDetectorInference(unittest.TestCase):
    """Test suite for the standalone inference module."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints', 'best_model.pt')
        if not os.path.isfile(cls.ckpt_path):
            raise unittest.SkipTest("best_model.pt not available for inference testing")

        cls.engine = ChangeDetectorInference(checkpoint_path=cls.ckpt_path, device='cpu', default_threshold=0.20)

        # Create dummy synthetic images for testing
        cls.temp_dir = os.path.join(PROJECT_ROOT, 'outputs', 'test_tmp')
        os.makedirs(cls.temp_dir, exist_ok=True)

        cls.img_t1_path = os.path.join(cls.temp_dir, 'dummy_t1.png')
        cls.img_t2_path = os.path.join(cls.temp_dir, 'dummy_t2.png')

        # Dummy 512x512 images
        np.random.seed(42)
        arr1 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        arr2 = arr1.copy()
        # Add a synthetic change block in T2
        arr2[100:200, 100:200, :] = 255

        Image.fromarray(arr1).save(cls.img_t1_path)
        Image.fromarray(arr2).save(cls.img_t2_path)

    @classmethod
    def tearDownClass(cls):
        for f in (cls.img_t1_path, cls.img_t2_path):
            if os.path.isfile(f):
                os.remove(f)
        overlay_out = os.path.join(cls.temp_dir, 'test_overlay.png')
        if os.path.isfile(overlay_out):
            os.remove(overlay_out)
        if os.path.isdir(cls.temp_dir):
            try:
                os.rmdir(cls.temp_dir)
            except Exception:
                pass

    def test_01_initialization(self):
        """Test engine initialization and threshold assignment."""
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(self.engine.threshold, 0.20)
        self.assertEqual(self.engine.device.type, 'cpu')

        # Test default initialization with final_selected_model.pt
        final_ckpt = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints', 'final_selected_model.pt')
        if os.path.isfile(final_ckpt):
            final_engine = ChangeDetectorInference(checkpoint_path=final_ckpt, device='cpu')
            self.assertEqual(final_engine.model.__class__.__name__, 'SiameseUNetConc')
            self.assertEqual(final_engine.threshold, 0.20)

    def test_02_preprocessing_types(self):
        """Test preprocessing across file paths, PIL Images, and NumPy arrays."""
        # 1. Path
        t_path, size1 = self.engine.preprocess_image(self.img_t1_path)
        self.assertEqual(t_path.shape, (1, 3, 256, 256))
        self.assertEqual(size1, (512, 512))
        self.assertTrue(0.0 <= t_path.min() and t_path.max() <= 1.0)

        # 2. PIL Image
        pil_img = Image.open(self.img_t1_path)
        t_pil, size2 = self.engine.preprocess_image(pil_img)
        self.assertEqual(t_pil.shape, (1, 3, 256, 256))
        self.assertEqual(size2, (512, 512))

        # 3. NumPy Array
        np_arr = np.array(pil_img)
        t_np, size3 = self.engine.preprocess_image(np_arr)
        self.assertEqual(t_np.shape, (1, 3, 256, 256))
        self.assertEqual(size3, (512, 512))

    def test_03_prediction_output(self):
        """Test end-to-end prediction outputs and datatypes."""
        result = self.engine.predict(self.img_t1_path, self.img_t2_path, threshold=0.20, resize_mask_to_original=True)

        self.assertIsInstance(result, ChangeDetectionResult)
        self.assertEqual(result.prob_map.shape, (512, 512))
        self.assertEqual(result.binary_mask.shape, (512, 512))
        self.assertTrue(np.all((result.binary_mask == 0) | (result.binary_mask == 255)))
        self.assertTrue(0.0 <= result.prob_map.min() and result.prob_map.max() <= 1.0)
        self.assertEqual(result.total_pixels, 512 * 512)
        self.assertTrue(0 <= result.changed_pixels <= result.total_pixels)
        self.assertTrue(0.0 <= result.change_percentage <= 100.0)

    def test_04_threshold_behavior(self):
        """Test that stricter (higher) thresholds yield equal or fewer change pixels."""
        res_loose = self.engine.predict(self.img_t1_path, self.img_t2_path, threshold=0.10)
        res_strict = self.engine.predict(self.img_t1_path, self.img_t2_path, threshold=0.90)

        self.assertGreaterEqual(res_loose.changed_pixels, res_strict.changed_pixels)

    def test_05_visualization_generation(self):
        """Test saving 4-panel visual report."""
        res = self.engine.predict(self.img_t1_path, self.img_t2_path)
        out_path = os.path.join(self.temp_dir, 'test_overlay.png')
        saved = self.engine.save_visualization(self.img_t1_path, self.img_t2_path, res, out_path)

        self.assertTrue(os.path.isfile(saved))
        self.assertGreater(os.path.getsize(saved), 1000)

    def test_06_error_handling(self):
        """Test robust error handling for missing files, invalid thresholds, and corrupt inputs."""
        # Non-existent image
        with self.assertRaises(FileNotFoundError):
            self.engine.preprocess_image("non_existent_file_path_xyz.png")

        # Non-existent checkpoint
        with self.assertRaises(FileNotFoundError):
            ChangeDetectorInference(checkpoint_path="non_existent_model.pt")

        # Invalid threshold
        with self.assertRaises(ValueError):
            self.engine.predict(self.img_t1_path, self.img_t2_path, threshold=1.5)

        with self.assertRaises(ValueError):
            self.engine.predict(self.img_t1_path, self.img_t2_path, threshold=-0.1)


if __name__ == '__main__':
    unittest.main()

