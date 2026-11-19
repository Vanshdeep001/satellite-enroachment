"""
Unit tests for the product layer (src/product): instances, encroachment, evaluation,
robustness, synthetic scenes and the command-line app.
"""

import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.product.encroachment import (
    GeoTransform, RestrictedZone, assess_encroachment, rasterize_zone, zones_from_geojson,
)
from src.product.evaluation import binary_change_report, multiclass_report
from src.product.instances import label_instances, match_instances
from src.product.robustness import DEFAULT_SUITE, apply_perturbation, run_robustness
from src.product.synthetic_scenes import demo_zones_geojson, make_dataset, make_scene_pair


def _square(points):
    return [np.array(points, dtype=float)]


class TestInstances(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((20, 20), bool)
        self.mask[2:6, 2:6] = True        # 16 px
        self.mask[10:15, 12:18] = True    # 30 px
        self.mask[18, 0] = True           # 1 px speck

    def test_labels_areas_bboxes(self):
        labels, inst = label_instances(self.mask, pixel_area_m2=0.25, min_area_px=2)
        self.assertEqual(len(inst), 2)
        self.assertEqual(labels.max(), 2)
        self.assertEqual(labels[18, 0], 0)  # speck removed
        self.assertEqual(sorted(i.area_px for i in inst), [16, 30])
        first = inst[0]
        self.assertEqual(first.bbox, (2, 2, 6, 6))
        self.assertAlmostEqual(first.area_m2, 4.0)
        self.assertEqual(first.centroid, (3.5, 3.5))

    def test_connectivity(self):
        diag = np.eye(4, dtype=bool)
        self.assertEqual(len(label_instances(diag, connectivity=8)[1]), 1)
        self.assertEqual(len(label_instances(diag, connectivity=4)[1]), 4)

    def test_mean_probability(self):
        prob = np.where(self.mask, 0.8, 0.1)
        _, inst = label_instances(self.mask, min_area_px=2, prob_map=prob)
        self.assertTrue(all(abs(i.mean_probability - 0.8) < 1e-9 for i in inst))

    def test_match_perfect_and_partial(self):
        gt, _ = label_instances(self.mask, min_area_px=2)
        self.assertEqual(match_instances(gt, gt)['f1'], 1.0)
        shifted = np.zeros_like(self.mask)
        shifted[2:6, 3:7] = True          # IoU with the 4x4 square = 12/20 = 0.6
        pl, _ = label_instances(shifted)
        m = match_instances(pl, gt, iou_threshold=0.5)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 0, 1))
        self.assertAlmostEqual(m['pairs'][0]['iou'], 0.6)
        self.assertEqual(match_instances(pl, gt, iou_threshold=0.7)['tp'], 0)

    def test_match_empty(self):
        empty = np.zeros((5, 5), np.int32)
        m = match_instances(empty, empty)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['f1']), (0, 0, 0, 0.0))


class TestEncroachment(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((40, 40), bool)
        self.mask[5:15, 5:15] = True      # instance inside zone
        self.mask[30:35, 30:35] = True    # instance outside
        self.labels, self.inst = label_instances(self.mask)
        self.zone_poly = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]  # pixel x, y

    def test_rasterize_square_and_hole(self):
        zone = RestrictedZone('z', [_square(self.zone_poly)], True, 'test')
        self.assertEqual(rasterize_zone(zone, (40, 40)).sum(), 400)
        hole = np.array([[5, 5], [10, 5], [10, 10], [5, 10], [5, 5]], float)
        zone_h = RestrictedZone('z', [[np.array(self.zone_poly, float), hole]], True, 'test')
        self.assertEqual(rasterize_zone(zone_h, (40, 40)).sum(), 400 - 25)

    def test_no_zones_makes_no_statement(self):
        rep = assess_encroachment(self.labels, self.inst, [], 0.25)
        self.assertEqual(rep['status'], 'no_boundaries_supplied')
        self.assertEqual(rep['alerts'], [])

    def test_unverified_zone_never_alerts(self):
        zone = RestrictedZone('draft', [_square(self.zone_poly)], verified=False)
        rep = assess_encroachment(self.labels, self.inst, [zone], 0.25)
        self.assertEqual(rep['alerts'], [])
        self.assertEqual(rep['status'], 'no_verified_boundaries')
        self.assertEqual(rep['zones'][0]['instances_touching'], 1)

    def test_verified_zone_alerts_with_overlap(self):
        zone = RestrictedZone('reserve', [_square(self.zone_poly)], verified=True, source='record 42')
        rep = assess_encroachment(self.labels, self.inst, [zone], 0.25)
        self.assertEqual(rep['status'], 'alerts_raised')
        self.assertEqual(len(rep['alerts']), 1)
        alert = rep['alerts'][0]
        self.assertEqual(alert['overlap_px'], 100)
        self.assertAlmostEqual(alert['overlap_m2'], 25.0)
        self.assertAlmostEqual(alert['fraction_of_change_inside_zone'], 1.0)
        self.assertIn('advisory', alert['status'])

    def test_min_overlap_fraction_suppresses_edge_touch(self):
        poly = [[0, 0], [6, 0], [6, 40], [0, 40], [0, 0]]  # covers 1 of 10 columns of instance 1
        zone = RestrictedZone('strip', [_square(poly)], verified=True, source='s')
        self.assertEqual(len(assess_encroachment(self.labels, self.inst, [zone], 1.0)['alerts']), 1)
        self.assertEqual(len(assess_encroachment(self.labels, self.inst, [zone], 1.0,
                                                 min_overlap_fraction=0.5)['alerts']), 0)

    def test_geojson_parsing_and_verification_rule(self):
        gj = {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {'name': 'a', 'verified': True},  # no source -> not verified
             'geometry': {'type': 'Polygon', 'coordinates': [self.zone_poly]}},
            {'type': 'Feature', 'properties': {'name': 'b', 'verified': True, 'source': 'gazette'},
             'geometry': {'type': 'MultiPolygon', 'coordinates': [[self.zone_poly], [self.zone_poly]]}},
        ]}
        zones = zones_from_geojson(gj)
        self.assertFalse(zones[0].verified)
        self.assertTrue(zones[1].verified)
        self.assertEqual(len(zones[1].rings), 2)
        with self.assertRaises(ValueError):
            zones_from_geojson({'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': [0, 0]}})

    def test_geotransform(self):
        gt = GeoTransform(x0=1000.0, pixel_width=0.5, y0=2000.0, pixel_height=-0.5)
        np.testing.assert_allclose(gt.to_pixel(np.array([[1005.0, 1990.0]])), [[10.0, 20.0]])


class TestEvaluation(unittest.TestCase):
    def test_binary_report_hand_computed(self):
        gt = np.zeros((10, 10), bool)
        gt[0:4, 0:5] = True                      # 20 px
        pred = np.zeros_like(gt)
        pred[0:4, 0:4] = True                    # 16 px, all TP
        pred[8:10, 8:10] = True                  # 4 px FP
        rep = binary_change_report([pred], [gt], pixel_area_m2=0.25)
        p = rep['pixel']
        self.assertEqual(rep['confusion_matrix']['counts'], [[76, 4], [4, 16]])
        self.assertAlmostEqual(p['precision'], 0.8)
        self.assertAlmostEqual(p['recall'], 0.8)
        self.assertAlmostEqual(p['f1_no_change'], 2 * 76 / (2 * 76 + 8))
        self.assertAlmostEqual(p['macro_f1'], (0.8 + 2 * 76 / 160) / 2)
        self.assertAlmostEqual(rep['area']['relative_area_error'], 0.0)   # 20 px predicted, 20 px true
        self.assertEqual((rep['object']['tp'], rep['object']['fp'], rep['object']['fn']), (1, 1, 0))

    def test_area_error_sign(self):
        gt = np.zeros((4, 4), bool)
        gt[:2] = True
        rep = binary_change_report([np.ones((4, 4), bool)], [gt], pixel_area_m2=1.0)
        self.assertAlmostEqual(rep['area']['relative_area_error'], 1.0)
        self.assertAlmostEqual(rep['area']['area_error_m2'], 8.0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            binary_change_report([], [])

    def test_multiclass(self):
        rep = multiclass_report([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], ['a', 'b', 'c'])
        self.assertEqual(rep['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(rep['accuracy'], 4 / 6)
        self.assertAlmostEqual(rep['per_class']['b']['precision'], 2 / 3)
        self.assertAlmostEqual(rep['macro_f1'], np.mean([0.5, 0.8, 2 / 3]))


class TestRobustnessAndScenes(unittest.TestCase):
    def test_scene_pair(self):
        t1, t2, change, info = make_scene_pair(size=96, seed=3, n_existing=3, n_new=2, max_building=14)
        self.assertEqual(t1.shape, (96, 96, 3))
        self.assertTrue(0 <= t1.min() and t2.max() <= 1)
        expected = sum((r1 - r0) * (c1 - c0) for r0, c0, r1, c1 in info['new'])
        self.assertEqual(change.sum(), expected)
        a, b, c, _ = make_scene_pair(size=96, seed=3, n_existing=3, n_new=2, max_building=14)
        np.testing.assert_array_equal(a, t1)

    def test_demo_zones(self):
        _, _, _, info = make_scene_pair(size=128, seed=0, n_existing=3, n_new=2)
        zones = zones_from_geojson(demo_zones_geojson(info, 128))
        self.assertEqual([z.verified for z in zones], [True, False])

    def test_perturbations(self):
        rng = np.random.default_rng(0)
        t1 = rng.random((32, 32, 3))
        t2 = rng.random((32, 32, 3))
        for name, levels in DEFAULT_SUITE.items():
            a, b = apply_perturbation(t1, t2, name, levels[-1], rng)
            self.assertEqual(b.shape, t2.shape)
            self.assertTrue(0 <= b.min() and b.max() <= 1, msg=name)
        a, b = apply_perturbation(t1, t2, 'brightness', 0.5, rng)
        np.testing.assert_array_equal(a, t1)                      # only T2 changes
        _, shifted = apply_perturbation(t1, t2, 'misregistration', 2, rng)
        np.testing.assert_allclose(shifted[5:, 5:], t2[3:-2, 3:-2])
        with self.assertRaises(ValueError):
            apply_perturbation(t1, t2, 'earthquake', 1, rng)

    def test_run_robustness_rows(self):
        samples = make_dataset(2, size=64, seed=1, n_existing=2, n_new=2, max_building=12)
        oracle = {id(s[0]): s[2] for s in samples}
        # Oracle detector on clean data; perturbed inputs are new arrays, so fall back to the diff.
        fn = lambda a, b: oracle.get(id(a), np.abs(a - b).mean(axis=2) > 0.1)  # noqa: E731
        res = run_robustness(fn, samples, suite={'blur': [1.0]})
        self.assertEqual([r['perturbation'] for r in res['rows']], ['clean', 'blur'])
        self.assertAlmostEqual(res['clean_f1'], 1.0)
        self.assertLess(res['rows'][1]['f1'], 1.0)


class TestApp(unittest.TestCase):
    def test_demo_command(self):
        from src.product import app
        with tempfile.TemporaryDirectory() as tmp:
            report = app.main(['demo', '--method', 'baseline', '--out', tmp])
            for name in ['demo_report.json', 'demo_report.png', 'demo_t1.png', 'demo_zones.geojson']:
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)), msg=name)
        self.assertGreater(report['num_instances'], 0)
        self.assertGreater(report['evaluation_vs_synthetic_truth']['pixel']['f1_change'], 0.5)
        self.assertIn('ADVISORY', report['responsible_use'])

    def test_detect_and_evaluate_on_levir_layout(self):
        from src.product import app
        with tempfile.TemporaryDirectory() as tmp:
            for sub in ('A', 'B', 'label'):
                os.makedirs(os.path.join(tmp, 'data', 'test', sub))
            for i, (t1, t2, gt) in enumerate(make_dataset(2, size=96, seed=5, n_existing=2, n_new=2, max_building=14)):
                app.save_rgb(t1, os.path.join(tmp, 'data', 'test', 'A', f'test_{i}.png'))
                app.save_rgb(t2, os.path.join(tmp, 'data', 'test', 'B', f'test_{i}.png'))
                Image.fromarray(gt.astype(np.uint8) * 255).save(os.path.join(tmp, 'data', 'test', 'label', f'test_{i}.png'))
            rep = app.main(['evaluate', '--method', 'baseline', '--threshold', '0.12',
                            '--data-root', os.path.join(tmp, 'data'), '--out', os.path.join(tmp, 'eval')])
            self.assertEqual(rep['num_images'], 2)
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'eval', 'confusion_test.png')))

            zones = os.path.join(tmp, 'zones.geojson')
            with open(zones, 'w') as f:
                json.dump({'type': 'FeatureCollection', 'features': [
                    {'type': 'Feature', 'properties': {'name': 'all', 'verified': True, 'source': 'test'},
                     'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [96, 0], [96, 96], [0, 96], [0, 0]]]}}]}, f)
            det = app.main(['detect', '--method', 'baseline', '--threshold', '0.12',
                            '--t1', os.path.join(tmp, 'data', 'test', 'A', 'test_0.png'),
                            '--t2', os.path.join(tmp, 'data', 'test', 'B', 'test_0.png'),
                            '--zones', zones, '--min-area-m2', '5', '--out', os.path.join(tmp, 'det')])
            self.assertEqual(det['encroachment']['status'], 'alerts_raised')
            self.assertEqual(len(det['encroachment']['alerts']), det['num_instances'])

    def test_evaluate_missing_data_message(self):
        from src.product import app
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                app.main(['evaluate', '--method', 'baseline', '--data-root', tmp, '--out', tmp])


if __name__ == '__main__':
    unittest.main()
