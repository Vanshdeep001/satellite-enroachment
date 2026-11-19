"""
Satellite Land Use Change & Encroachment Detector: command-line application.

Commands
    detect      T1 + T2 image -> change mask, change instances, optional encroachment
                assessment against restricted-zone GeoJSON, JSON report + visual.
    evaluate    Run on a LEVIR-CD style split (data/<split>/{A,B,label}) and write the full
                evaluation package (pixel, macro F1, area error, object level, confusion matrix).
    robustness  Perturbation suite on a LEVIR-CD split or on synthetic scenes.
    demo        End-to-end run on a synthetic scene with demo zones (no dataset needed).

Detection backends
    model     trained Siamese U-Net checkpoint via src/inference.py (default when a checkpoint exists)
    baseline  pixel-difference baseline (src/baseline.py); always available

Examples
    python -m src.product.app demo
    python -m src.product.app detect --t1 a.png --t2 b.png --zones zones.geojson --gsd 0.5
    python -m src.product.app evaluate --data-root data --split test
    python -m src.product.app robustness --synthetic 12
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.baseline import compute_difference_map  # noqa: E402
from src.product import synthetic_scenes as S  # noqa: E402
from src.product.encroachment import (  # noqa: E402
    DISCLAIMER, GeoTransform, assess_encroachment, load_zones_geojson, zones_from_geojson,
)
from src.product.evaluation import binary_change_report, plot_confusion_matrix  # noqa: E402
from src.product.instances import label_instances  # noqa: E402
from src.product.robustness import plot_robustness, run_robustness  # noqa: E402

DEFAULT_CHECKPOINTS = [os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints', name)
                       for name in ('final_selected_model.pt', 'best_model.pt')]


# ---------------------------------------------------------------------------
# Detection backends
# ---------------------------------------------------------------------------

class Detector:
    """Uniform interface: detect(t1, t2) -> (score map in [0, 1], binary mask)."""

    name = 'detector'

    def detect(self, t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def mask_fn(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return lambda a, b: self.detect(a, b)[1]


class BaselineDetector(Detector):
    def __init__(self, threshold: float = 0.40, method: str = 'euclidean'):
        self.threshold, self.method = threshold, method
        self.name = f'pixel-difference baseline ({method}, tau={threshold:.2f})'

    def detect(self, t1, t2):
        diff = compute_difference_map(np.asarray(t1, np.float32), np.asarray(t2, np.float32), method=self.method)
        return diff, diff >= self.threshold


class ModelDetector(Detector):
    def __init__(self, checkpoint: Optional[str] = None, threshold: Optional[float] = None, device: Optional[str] = None):
        from src.inference import ChangeDetectorInference  # torch import deferred
        self.engine = ChangeDetectorInference(checkpoint_path=checkpoint, device=device, default_threshold=threshold)
        self.name = f"{self.engine.model_metadata.get('resolved_model_name', 'model')} (tau={self.engine.threshold:.2f})"

    def detect(self, t1, t2):
        to_u8 = lambda a: (np.clip(np.asarray(a, np.float64), 0, 1) * 255).round().astype(np.uint8)  # noqa: E731
        res = self.engine.predict(to_u8(t1), to_u8(t2), resize_mask_to_original=True)
        return res.prob_map, res.binary_mask > 0


def tune_baseline_threshold(samples, grid=None, method: str = 'euclidean') -> float:
    """Pick the baseline threshold with the best pixel F1 on calibration samples (never the eval set)."""
    grid = grid if grid is not None else np.round(np.arange(0.04, 0.61, 0.02), 2)
    diffs = [compute_difference_map(t1.astype(np.float32), t2.astype(np.float32), method=method) for t1, t2, _ in samples]
    best, best_f1 = float(grid[0]), -1.0
    for tau in grid:
        rep = binary_change_report([d >= tau for d in diffs], [g for _, _, g in samples])
        if rep['pixel']['f1_change'] > best_f1:
            best, best_f1 = float(tau), rep['pixel']['f1_change']
    return best


def build_detector(args) -> Detector:
    checkpoint = getattr(args, 'checkpoint', None) or next((p for p in DEFAULT_CHECKPOINTS if os.path.isfile(p)), None)
    method = getattr(args, 'method', 'auto')
    if method == 'model' or (method == 'auto' and checkpoint):
        if not checkpoint:
            raise SystemExit('No model checkpoint found. Pass --checkpoint or use --method baseline.')
        return ModelDetector(checkpoint, threshold=getattr(args, 'threshold', None), device=getattr(args, 'device', None))
    return BaselineDetector(threshold=getattr(args, 'threshold', None) or 0.40)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_rgb(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise SystemExit(f'Image not found: {path}')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0


def read_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L')) > 127


def save_rgb(img: np.ndarray, path: str) -> None:
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path)


def load_split(data_root: str, split: str, limit: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a_dir = os.path.join(data_root, split, 'A')
    files = sorted(glob.glob(os.path.join(a_dir, '*.png')))
    if not files:
        raise SystemExit(f"No images in {a_dir}. Expected LEVIR-CD layout: {data_root}/{split}/{{A,B,label}}/*.png")
    files = files[:limit] if limit else files
    out = []
    for fa in files:
        name = os.path.basename(fa)
        out.append((read_rgb(fa), read_rgb(os.path.join(data_root, split, 'B', name)),
                    read_mask(os.path.join(data_root, split, 'label', name))))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

def run_detection(
    detector: Detector,
    t1: np.ndarray,
    t2: np.ndarray,
    gsd: float,
    zones_geojson: Optional[Dict[str, Any]] = None,
    zones_path: Optional[str] = None,
    transform: Optional[GeoTransform] = None,
    min_area_m2: float = 10.0,
    min_overlap_fraction: float = 0.1
) -> Dict[str, Any]:
    """Full product pipeline for one image pair. Returns report dict + arrays for plotting."""
    if t1.shape != t2.shape:
        raise SystemExit(f'T1 and T2 must have the same size, got {t1.shape} and {t2.shape}')
    pixel_area = gsd * gsd
    start = time.perf_counter()
    score, mask = detector.detect(t1, t2)
    det_ms = 1000 * (time.perf_counter() - start)

    min_px = max(1, int(round(min_area_m2 / pixel_area)))
    labels, instances = label_instances(mask, pixel_area_m2=pixel_area, min_area_px=min_px, prob_map=score)
    if zones_path:
        zones = load_zones_geojson(zones_path, transform)
    elif zones_geojson:
        zones = zones_from_geojson(zones_geojson, transform)
    else:
        zones = []
    encroachment = assess_encroachment(labels, instances, zones, pixel_area,
                                       min_overlap_fraction=min_overlap_fraction)
    total_ms = 1000 * (time.perf_counter() - start)
    report = {
        'detector': detector.name,
        'image_size': list(t1.shape[:2]),
        'gsd_m': gsd,
        'changed_area_m2': float(sum(i.area_m2 for i in instances)),
        'changed_fraction': float((labels > 0).mean()),
        'min_instance_area_m2': min_area_m2,
        'num_instances': len(instances),
        'instances': [i.to_dict() for i in instances],
        'encroachment': encroachment,
        'runtime_ms': {'detection': round(det_ms, 2), 'total': round(total_ms, 2)},
        'responsible_use': DISCLAIMER,
    }
    return {'report': report, 'score': score, 'labels': labels, 'zones': zones}


def plot_detection(t1, t2, result: Dict[str, Any], path: str, title: str = '') -> str:
    from src.product.encroachment import rasterize_zone

    report, labels = result['report'], result['labels']
    alert_ids = {a['instance_id'] for a in report['encroachment']['alerts']}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(t1)
    axes[0].set_title('T1 (before)')
    axes[1].imshow(t2)
    axes[1].set_title('T2 (after)')
    im = axes[2].imshow(result['score'], cmap='magma', vmin=0, vmax=max(1e-6, float(np.max(result['score']))))
    axes[2].set_title('change score')
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    overlay = np.asarray(t2, dtype=np.float64).copy()
    changed = labels > 0
    overlay[changed] = 0.45 * overlay[changed] + 0.55 * np.array([1.0, 0.85, 0.0])
    axes[3].imshow(overlay)
    for zone in result['zones']:
        zmask = rasterize_zone(zone, labels.shape)
        axes[3].contour(zmask, levels=[0.5], colors=['red' if zone.verified else 'white'],
                        linestyles='solid' if zone.verified else 'dashed', linewidths=1.8)
    for inst in report['instances']:
        r0, c0, r1, c1 = inst['bbox']
        color = 'red' if inst['id'] in alert_ids else 'yellow'
        axes[3].add_patch(mpatches.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec=color, lw=1.5))
        axes[3].text(c0, r0 - 2, f"#{inst['id']} {inst['area_m2']:.0f} m²", color=color, fontsize=7,
                     bbox=dict(facecolor='black', alpha=0.5, pad=1, lw=0))
    status = report['encroachment']['status'].replace('_', ' ')
    axes[3].set_title(f"{report['num_instances']} change instances, {report['changed_area_m2']:.0f} m²\n"
                      f"encroachment: {status} ({len(report['encroachment']['alerts'])} advisory alerts)", fontsize=10)
    handles = [mpatches.Patch(color='yellow', label='change instance'),
               mpatches.Patch(color='red', label='instance in verified zone (advisory)'),
               plt.Line2D([], [], color='red', label='verified boundary'),
               plt.Line2D([], [], color='white', ls='--', label='unverified boundary (no alerts)')]
    axes[3].legend(handles=handles, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2,
                   facecolor='lightgrey', framealpha=0.9)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title or f"Change & encroachment report: {report['detector']}", fontsize=12)
    fig.text(0.5, 0.01, DISCLAIMER, ha='center', fontsize=7, wrap=True, style='italic')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def cmd_detect(args) -> Dict[str, Any]:
    os.makedirs(args.out, exist_ok=True)
    t1, t2 = read_rgb(args.t1), read_rgb(args.t2)
    transform = GeoTransform(*args.geotransform) if args.geotransform else None
    result = run_detection(build_detector(args), t1, t2, args.gsd, zones_path=args.zones, transform=transform,
                           min_area_m2=args.min_area_m2, min_overlap_fraction=args.min_overlap)
    write_json(result['report'], os.path.join(args.out, 'report.json'))
    plot_detection(t1, t2, result, os.path.join(args.out, 'report.png'))
    Image.fromarray((result['labels'] > 0).astype(np.uint8) * 255).save(os.path.join(args.out, 'change_mask.png'))
    _print_summary(result['report'], args.out)
    return result['report']


def _print_summary(report: Dict[str, Any], out: str) -> None:
    enc = report['encroachment']
    print(f"Detector          : {report['detector']}")
    print(f"Change instances  : {report['num_instances']}  ({report['changed_area_m2']:.1f} m² total)")
    print(f"Encroachment      : {enc['status']}  -> {len(enc['alerts'])} advisory alert(s)")
    for a in enc['alerts']:
        print(f"   - zone '{a['zone']}': instance #{a['instance_id']}, {a['overlap_m2']:.1f} m² inside "
              f"({a['fraction_of_change_inside_zone']:.0%} of that change)")
    print(f"Runtime           : {report['runtime_ms']['total']:.1f} ms")
    print(f"Outputs           : {out}")
    print(f"NOTE: {report['responsible_use']}")


# ---------------------------------------------------------------------------
# evaluate / robustness / demo
# ---------------------------------------------------------------------------

def cmd_evaluate(args) -> Dict[str, Any]:
    os.makedirs(args.out, exist_ok=True)
    samples = load_split(args.data_root, args.split, args.limit)
    detector = build_detector(args)
    preds, start = [], time.perf_counter()
    for t1, t2, _ in samples:
        preds.append(detector.detect(t1, t2)[1])
    elapsed = time.perf_counter() - start
    rep = binary_change_report(preds, [g for _, _, g in samples], pixel_area_m2=args.gsd ** 2,
                               min_instance_px=max(1, int(round(args.min_area_m2 / args.gsd ** 2))))
    rep.update({'detector': detector.name, 'split': args.split,
                'runtime': {'total_s': elapsed, 'ms_per_pair': 1000 * elapsed / len(samples)}})
    write_json(rep, os.path.join(args.out, f'evaluation_{args.split}.json'))
    plot_confusion_matrix(np.array(rep['confusion_matrix']['counts']), ['no change', 'change'],
                          os.path.join(args.out, f'confusion_{args.split}.png'), f'{detector.name} on {args.split}')
    p = rep['pixel']
    print(f"{detector.name} on {args.split} ({len(samples)} pairs)")
    print(f"  F1 {p['f1_change']:.4f} | IoU {p['iou']:.4f} | change recall {p['recall']:.4f} | macro F1 {p['macro_f1']:.4f}")
    print(f"  relative area error {rep['area']['relative_area_error']:+.3f} | object F1 {rep['object']['f1']:.4f} | "
          f"{rep['runtime']['ms_per_pair']:.1f} ms/pair")
    return rep


def _synthetic_with_calibration(n: int, seed: int):
    calibration = S.make_dataset(8, seed=seed + 1)   # disjoint seeds: threshold never sees eval scenes
    evaluation = S.make_dataset(n, seed=seed + 2)
    return calibration, evaluation


def cmd_robustness(args) -> Dict[str, Any]:
    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        calibration, samples = _synthetic_with_calibration(args.synthetic, args.seed)
        gsd = 0.5
        if args.method != 'model' and args.threshold is None:
            args.threshold = tune_baseline_threshold(calibration)
            print(f"Baseline threshold tuned on 8 calibration scenes: {args.threshold:.2f}")
    else:
        samples, gsd = load_split(args.data_root, args.split, args.limit), args.gsd
    detector = build_detector(args)
    result = run_robustness(detector.mask_fn(), samples, pixel_area_m2=gsd ** 2, seed=args.seed,
                            min_instance_px=max(1, int(round(args.min_area_m2 / gsd ** 2))))
    result.update({'detector': detector.name, 'num_pairs': len(samples),
                   'data': 'synthetic' if args.synthetic else f'{args.data_root}/{args.split}'})
    write_json(result, os.path.join(args.out, 'robustness.json'))
    plot_robustness(result, os.path.join(args.out, 'robustness.png'), f"Robustness: {detector.name}")
    print(f"{'condition':<24}{'F1':>8}{'dF1':>8}{'recall':>8}{'area err':>10}{'obj F1':>8}")
    for r in result['rows']:
        area = r['relative_area_error']
        print(f"{r['perturbation'] + ' ' + (format(r['level'], 'g') if r['perturbation'] != 'clean' else ''):<24}"
              f"{r['f1']:8.3f}{r['delta_f1']:+8.3f}{r['change_recall']:8.3f}"
              f"{(area if area is not None else float('nan')):+10.3f}{r['object_f1']:8.3f}")
    return result


def cmd_demo(args) -> Dict[str, Any]:
    os.makedirs(args.out, exist_ok=True)
    t1, t2, gt, info = S.make_scene_pair(size=256, seed=args.seed)
    zones = S.demo_zones_geojson(info, 256)
    save_rgb(t1, os.path.join(args.out, 'demo_t1.png'))
    save_rgb(t2, os.path.join(args.out, 'demo_t2.png'))
    write_json(zones, os.path.join(args.out, 'demo_zones.geojson'))

    if args.method != 'model' and args.threshold is None:
        args.threshold = tune_baseline_threshold(S.make_dataset(8, seed=args.seed + 1))
    detector = build_detector(args)
    result = run_detection(detector, t1, t2, gsd=0.5, zones_geojson=zones, min_area_m2=args.min_area_m2)
    report = result['report']
    pred = result['labels'] > 0
    report['evaluation_vs_synthetic_truth'] = binary_change_report([pred], [gt], pixel_area_m2=0.25)
    write_json(report, os.path.join(args.out, 'demo_report.json'))
    plot_detection(t1, t2, result, os.path.join(args.out, 'demo_report.png'),
                   f"DEMO on a synthetic scene: {detector.name}")
    _print_summary(report, args.out)
    ev = report['evaluation_vs_synthetic_truth']
    print(f"Vs synthetic truth: F1 {ev['pixel']['f1_change']:.3f}, object F1 {ev['object']['f1']:.3f}, "
          f"area error {ev['area']['relative_area_error']:+.1%}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m src.product.app',
                                     description='Satellite land-use change & encroachment detector')
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p, out_default):
        p.add_argument('--method', choices=['auto', 'model', 'baseline'], default='auto',
                       help='auto = model if a checkpoint exists, else baseline')
        p.add_argument('--checkpoint', default=None)
        p.add_argument('--threshold', type=float, default=None)
        p.add_argument('--device', default=None)
        p.add_argument('--min-area-m2', type=float, default=10.0, help='ignore change blobs smaller than this')
        p.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'outputs', 'product', out_default))
        p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('detect', help='detect change (and encroachment) for one image pair')
    common(p, 'detect')
    p.add_argument('--t1', required=True)
    p.add_argument('--t2', required=True)
    p.add_argument('--gsd', type=float, default=0.5, help='ground sampling distance, metres per pixel')
    p.add_argument('--zones', default=None, help='restricted-zone GeoJSON (pixel coords unless --geotransform)')
    p.add_argument('--geotransform', type=float, nargs=4, metavar=('X0', 'PX_W', 'Y0', 'PX_H'), default=None)
    p.add_argument('--min-overlap', type=float, default=0.1,
                   help='minimum fraction of a change inside a verified zone to raise an alert')
    p.set_defaults(func=cmd_detect)

    for name, func, help_text in [('evaluate', cmd_evaluate, 'evaluation package on a LEVIR-CD split'),
                                  ('robustness', cmd_robustness, 'perturbation robustness suite')]:
        p = sub.add_parser(name, help=help_text)
        common(p, name)
        p.add_argument('--data-root', default=os.path.join(PROJECT_ROOT, 'data'))
        p.add_argument('--split', default='test')
        p.add_argument('--limit', type=int, default=None)
        p.add_argument('--gsd', type=float, default=0.5)
        if name == 'robustness':
            p.add_argument('--synthetic', type=int, default=0, help='use N synthetic scenes instead of a data split')
        p.set_defaults(func=func)

    p = sub.add_parser('demo', help='end-to-end demo on a synthetic scene (no dataset needed)')
    common(p, 'demo')
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    main()
