"""
Unit tests for evaluation metrics in src/metrics.py.
"""

import unittest
import numpy as np
import torch
from src.metrics import compute_confusion_matrix, compute_metrics_from_counts, MetricTracker


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix_perfect_prediction(self):
        pred = np.array([[1, 0], [0, 1]])
        target = np.array([[1, 0], [0, 1]])
        counts = compute_confusion_matrix(pred, target)
        self.assertEqual(counts['tp'], 2)
        self.assertEqual(counts['fp'], 0)
        self.assertEqual(counts['fn'], 0)
        self.assertEqual(counts['tn'], 2)

    def test_confusion_matrix_all_zeros(self):
        pred = np.zeros((4, 4))
        target = np.zeros((4, 4))
        counts = compute_confusion_matrix(pred, target)
        self.assertEqual(counts['tp'], 0)
        self.assertEqual(counts['fp'], 0)
        self.assertEqual(counts['fn'], 0)
        self.assertEqual(counts['tn'], 16)

    def test_confusion_matrix_with_tensors(self):
        pred = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        counts = compute_confusion_matrix(pred, target, threshold=0.5)
        self.assertEqual(counts['tp'], 2)
        self.assertEqual(counts['fp'], 0)
        self.assertEqual(counts['fn'], 0)
        self.assertEqual(counts['tn'], 2)

    def test_metrics_calculation_hand_crafted(self):
        # 10 TP, 5 FP, 5 FN, 80 TN -> total 100
        # Precision = 10 / (10 + 5) = 10 / 15 = 0.6667
        # Recall = 10 / (10 + 5) = 10 / 15 = 0.6667
        # F1 = 0.6667
        # IoU = 10 / (10 + 5 + 5) = 10 / 20 = 0.50
        # Accuracy = 90 / 100 = 0.90
        metrics = compute_metrics_from_counts(tp=10, fp=5, fn=5, tn=80)
        self.assertAlmostEqual(metrics['precision'], 10 / 15, places=4)
        self.assertAlmostEqual(metrics['recall'], 10 / 15, places=4)
        self.assertAlmostEqual(metrics['f1_score'], 10 / 15, places=4)
        self.assertAlmostEqual(metrics['iou'], 0.50, places=4)
        self.assertAlmostEqual(metrics['accuracy'], 0.90, places=4)

    def test_zero_division_safety(self):
        # No positive predictions and no ground truth changes
        metrics = compute_metrics_from_counts(tp=0, fp=0, fn=0, tn=100)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1_score'], 0.0)
        self.assertEqual(metrics['iou'], 0.0)
        self.assertEqual(metrics['accuracy'], 1.0)

    def test_metric_tracker_accumulation(self):
        tracker = MetricTracker()
        
        # Sample 1: 2 TP, 1 FP, 0 FN, 5 TN
        p1 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        t1 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        tracker.update(p1, t1)

        # Sample 2: 1 TP, 0 FP, 1 FN, 6 TN
        p2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        t2 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        tracker.update(p2, t2)

        metrics = tracker.compute()
        # Cumulative:
        # TP = 2 + 1 = 3
        # FP = 1 + 0 = 1
        # FN = 0 + 1 = 1
        # TN = 5 + 6 = 11
        # Total = 16
        self.assertEqual(metrics['tp'], 3)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['tn'], 11)
        self.assertEqual(metrics['num_samples'], 2)
        self.assertAlmostEqual(metrics['precision'], 3 / 4, places=4)
        self.assertAlmostEqual(metrics['recall'], 3 / 4, places=4)
        self.assertAlmostEqual(metrics['f1_score'], 3 / 4, places=4)
        self.assertAlmostEqual(metrics['iou'], 3 / 5, places=4)


if __name__ == '__main__':
    unittest.main()

