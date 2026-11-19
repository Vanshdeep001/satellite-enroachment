"""
Land-use classification, ResNet vs GoogLeNet benchmark, and T1 -> T2 land-use transitions
(Project Steps 7 and 8).

Data sources
    --data-dir DIR   ImageFolder layout DIR/<class>/*.jpg|png, e.g. EuroSAT RGB (10 classes,
                     64x64, CC-BY-4.0). A stratified, seeded 70/15/15 split is written to
                     split_manifest.json so the partition is reproducible and leakage-safe
                     (no file is in two splits).
    --synthetic      synthetic 4-class land-cover patches (no download; pipeline check only).

Models (torchvision, random init by default; --pretrained downloads ImageNet weights once)
    resnet18   residual blocks, 11.2 M parameters
    googlenet  Inception modules (parallel 1x1 / 3x3 / 5x5 / pool branches), auxiliary heads off

Commands
    python src/landuse.py benchmark --synthetic
    python src/landuse.py benchmark --data-dir data/EuroSAT_RGB --epochs 10
    python src/landuse.py transitions --t1 a.png --t2 b.png --checkpoint outputs/landuse/.../resnet18.pt
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJECT_ROOT, '.mpl_config'))

from src.product.evaluation import multiclass_report, plot_confusion_matrix  # noqa: E402

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def stratified_split(files_by_class: Dict[str, List[str]], seed: int = 0,
                     fractions: Tuple[float, float] = (0.70, 0.15)) -> Dict[str, List[Tuple[str, int]]]:
    rng = np.random.default_rng(seed)
    split = {'train': [], 'val': [], 'test': []}
    for label, cls in enumerate(sorted(files_by_class)):
        files = sorted(files_by_class[cls])
        rng.shuffle(files)
        n_tr = int(round(fractions[0] * len(files)))
        n_va = int(round(fractions[1] * len(files)))
        split['train'] += [(f, label) for f in files[:n_tr]]
        split['val'] += [(f, label) for f in files[n_tr:n_tr + n_va]]
        split['test'] += [(f, label) for f in files[n_tr + n_va:]]
    return split


def load_image_folder(data_dir: str, size: int, seed: int, out_dir: str):
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if not classes:
        raise SystemExit(f"No class sub-folders in {data_dir}")
    files = {c: [f for f in glob.glob(os.path.join(data_dir, c, '*')) if f.lower().endswith(IMAGE_EXTS)]
             for c in classes}
    split = stratified_split(files, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'split_manifest.json'), 'w') as f:
        json.dump({'classes': classes, 'seed': seed,
                   **{k: [[os.path.relpath(p, data_dir), y] for p, y in v] for k, v in split.items()}}, f)

    def load(items):
        x = np.stack([np.asarray(Image.open(p).convert('RGB').resize((size, size), Image.BILINEAR),
                                 dtype=np.float32) / 255.0 for p, _ in items])
        return torch.tensor(x).permute(0, 3, 1, 2), torch.tensor([y for _, y in items])

    return {k: load(v) for k, v in split.items()}, classes


def load_synthetic(n_train: int, n_val: int, n_test: int, size: int):
    from src.fundamentals.landcover_toy import CLASS_NAMES, make_landcover_patches

    def load(n, seed):
        x, y = make_landcover_patches(n, patch=32, seed=seed)
        x = F.interpolate(torch.tensor(x, dtype=torch.float32), size=(size, size), mode='bilinear', align_corners=False)
        return x, torch.tensor(y)

    return {'train': load(n_train, 501), 'val': load(n_val, 502), 'test': load(n_test, 503)}, list(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Models and training
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'googlenet':
        if pretrained:
            model = torchvision.models.googlenet(weights='DEFAULT')
            model.aux_logits, model.aux1, model.aux2 = False, None, None
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=True,
                                                 num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model '{name}' (resnet18 | googlenet)")
    return model


class Normalised(nn.Module):
    """Wraps a classifier so it takes [0, 1] RGB and applies ImageNet normalisation itself."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer('mean', IMAGENET_MEAN.clone())
        self.register_buffer('std', IMAGENET_STD.clone())

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def _dihedral(x: torch.Tensor) -> torch.Tensor:
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(1).item() < 0.5 else x


@torch.no_grad()
def predict_classes(model: nn.Module, x: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])


def train_classifier(model: nn.Module, data, classes, epochs: int, batch_size: int = 32, lr: float = 1e-3,
                     seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x_tr, y_tr = data['train']
    best, best_state, history = -1.0, None, []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(_dihedral(x_tr[idx])), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val = multiclass_report(data['val'][1].tolist(), predict_classes(model, data['val'][0]).tolist(), classes)
        history.append({'epoch': epoch + 1, 'train_loss': total / len(x_tr), 'val_accuracy': val['accuracy'],
                        'val_macro_f1': val['macro_f1']})
        print(f"  epoch {epoch + 1:>2}: loss {total / len(x_tr):.4f}  val acc {val['accuracy']:.4f}  "
              f"macro F1 {val['macro_f1']:.4f}")
        if val['macro_f1'] > best:
            best, best_state = val['macro_f1'], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {'history': history, 'train_seconds': time.perf_counter() - start, 'best_val_macro_f1': best}


@torch.no_grad()
def latency_ms(model: nn.Module, size: int, repeats: int = 20) -> float:
    model.eval()
    x = torch.rand(1, 3, size, size)
    for _ in range(3):
        model(x)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    return 1000 * (time.perf_counter() - start) / repeats


# ---------------------------------------------------------------------------
# Land-use transitions between two dates
# ---------------------------------------------------------------------------

@torch.no_grad()
def landuse_transitions(model: nn.Module, t1: np.ndarray, t2: np.ndarray, classes: Sequence[str],
                        tile: int = 64, input_size: int = 64) -> Dict[str, Any]:
    """
    Classify non-overlapping tiles of T1 and T2 and count class transitions.

    Returns per-tile class grids for both dates, the k x k transition count matrix
    (rows = T1 class, columns = T2 class) and a list of changed tiles.
    """
    if t1.shape != t2.shape:
        raise ValueError("T1 and T2 must have the same shape")
    h, w = t1.shape[:2]
    rows, cols = h // tile, w // tile
    if rows == 0 or cols == 0:
        raise ValueError(f"Image {h}x{w} is smaller than one {tile}px tile")

    def tiles(img):
        arr = torch.tensor(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
        t = arr[:, :rows * tile, :cols * tile].unfold(1, tile, tile).unfold(2, tile, tile)  # [3, R, C, tile, tile]
        t = t.permute(1, 2, 0, 3, 4).reshape(-1, 3, tile, tile)
        return F.interpolate(t, size=(input_size, input_size), mode='bilinear', align_corners=False)

    g1 = predict_classes(model, tiles(t1)).view(rows, cols).numpy()
    g2 = predict_classes(model, tiles(t2)).view(rows, cols).numpy()
    k = len(classes)
    matrix = np.bincount(g1.ravel() * k + g2.ravel(), minlength=k * k).reshape(k, k)
    changed = [{'row': int(r), 'col': int(c), 'from': classes[g1[r, c]], 'to': classes[g2[r, c]],
                'bbox': [int(r * tile), int(c * tile), int((r + 1) * tile), int((c + 1) * tile)]}
               for r, c in zip(*np.nonzero(g1 != g2))]
    return {'classes': list(classes), 'grid_t1': g1.tolist(), 'grid_t2': g2.tolist(),
            'transition_matrix': matrix.tolist(), 'changed_tiles': changed, 'tile_px': tile}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_benchmark(args) -> Dict[str, Any]:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tag = 'synthetic' if args.synthetic else os.path.basename(os.path.normpath(args.data_dir))
    out = args.out or os.path.join(PROJECT_ROOT, 'outputs', 'landuse', tag)
    os.makedirs(out, exist_ok=True)
    if args.synthetic:
        data, classes = load_synthetic(args.n_train, args.n_val, args.n_test, args.size)
    elif args.data_dir:
        data, classes = load_image_folder(args.data_dir, args.size, args.seed, out)
    else:
        raise SystemExit('Pass --data-dir (e.g. EuroSAT RGB) or --synthetic')
    print(f"{tag}: {len(classes)} classes, train/val/test = "
          f"{len(data['train'][1])}/{len(data['val'][1])}/{len(data['test'][1])}")

    results = {}
    for name in args.models:
        print(f"== {name}")
        model = Normalised(build_model(name, len(classes), args.pretrained))
        info = train_classifier(model, data, classes, args.epochs, lr=args.lr, seed=args.seed)
        rep = multiclass_report(data['test'][1].tolist(), predict_classes(model, data['test'][0]).tolist(), classes)
        params = sum(p.numel() for p in model.parameters())
        results[name] = {'test_accuracy': rep['accuracy'], 'test_macro_f1': rep['macro_f1'],
                         'per_class': rep['per_class'], 'confusion_matrix': rep['confusion_matrix'],
                         'parameters': params, 'latency_ms_per_image_cpu': latency_ms(model, args.size),
                         **info}
        plot_confusion_matrix(np.array(rep['confusion_matrix']), classes, os.path.join(out, f'confusion_{name}.png'),
                              f'{name} on {tag} test')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': name, 'classes': classes,
                    'input_size': args.size, 'synthetic': args.synthetic}, os.path.join(out, f'{name}.pt'))
        print(f"   test acc {rep['accuracy']:.4f}  macro F1 {rep['macro_f1']:.4f}  params {params:,}  "
              f"latency {results[name]['latency_ms_per_image_cpu']:.1f} ms")

    summary = {'dataset': tag, 'classes': classes, 'size': args.size, 'epochs': args.epochs,
               'pretrained': args.pretrained, 'models': results}
    with open(os.path.join(out, 'benchmark.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def load_classifier(checkpoint: str) -> Tuple[nn.Module, List[str], int]:
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Normalised(build_model(ckpt['model_name'], len(ckpt['classes'])))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes'], ckpt['input_size']


def cmd_transitions(args) -> Dict[str, Any]:
    model, classes, size = load_classifier(args.checkpoint)
    read = lambda p: np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0  # noqa: E731
    res = landuse_transitions(model, read(args.t1), read(args.t2), classes, tile=args.tile, input_size=size)
    print(json.dumps({'transition_matrix': res['transition_matrix'], 'changed_tiles': len(res['changed_tiles'])}))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description='Land-use classification benchmark and transitions')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('benchmark')
    b.add_argument('--data-dir', default=None)
    b.add_argument('--synthetic', action='store_true')
    b.add_argument('--models', nargs='+', default=['resnet18', 'googlenet'])
    b.add_argument('--pretrained', action='store_true')
    b.add_argument('--epochs', type=int, default=10)
    b.add_argument('--lr', type=float, default=1e-3)
    b.add_argument('--size', type=int, default=64)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--n-train', type=int, default=800)
    b.add_argument('--n-val', type=int, default=200)
    b.add_argument('--n-test', type=int, default=800)
    b.add_argument('--out', default=None)
    b.set_defaults(func=cmd_benchmark)
    t = sub.add_parser('transitions')
    t.add_argument('--t1', required=True)
    t.add_argument('--t2', required=True)
    t.add_argument('--checkpoint', required=True)
    t.add_argument('--tile', type=int, default=64)
    t.add_argument('--out', default=None)
    t.set_defaults(func=cmd_transitions)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    main()
