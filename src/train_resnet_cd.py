"""
Train the Siamese ResNet change detector (Step 7).

Real data (LEVIR-CD in data/{train,val}/{A,B,label}):
    python src/train_resnet_cd.py --epochs 15 --fusion concat
Synthetic smoke run (no dataset needed; proves the pipeline end to end on CPU):
    python src/train_resnet_cd.py --synthetic --epochs 6

Same recipe as the Phase 8 U-Net candidates so the comparison is fair: AdamW (lr 1e-3, wd 1e-4),
cosine schedule, 0.5 BCE + 0.5 Dice, checkpoint on best validation F1, threshold tuned on
validation. The checkpoint stores architecture metadata so src/inference.py (and the product
CLI) can load it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.losses import CompositeChangeDetectionLoss  # noqa: E402
from src.resnet_change import SiameseResNetCD  # noqa: E402
from src.train import evaluate_validation, find_best_threshold, select_device, set_seed, train_one_epoch  # noqa: E402


def synthetic_loaders(n_train: int, n_val: int, size: int, batch_size: int, seed: int):
    from src.product.synthetic_scenes import make_dataset

    def to_tensors(samples):
        t1 = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32).permute(0, 3, 1, 2)
        t2 = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32).permute(0, 3, 1, 2)
        m = torch.tensor(np.stack([s[2] for s in samples]), dtype=torch.float32)[:, None]
        return TensorDataset(t1, t2, m)

    small = size < 128
    kw = dict(size=size, n_existing=2 if small else 4, n_new=2 if small else 3,
              min_building=max(4, size // 32), max_building=max(8, size // 16))
    train = to_tensors(make_dataset(n_train, seed=seed + 11, **kw))
    val = to_tensors(make_dataset(n_val, seed=seed + 12, **kw))
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False))


def levir_loaders(batch_size: int, size: int):
    from src.dataset import get_dataloader
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    return (get_dataloader(root_dir=data_dir, split='train', batch_size=batch_size, target_size=(size, size),
                           shuffle=True, num_workers=0),
            get_dataloader(root_dir=data_dir, split='val', batch_size=batch_size, target_size=(size, size),
                           shuffle=False, num_workers=0))


def main(argv=None):
    p = argparse.ArgumentParser(description='Train the Siamese ResNet change detector')
    p.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
    p.add_argument('--fusion', default='concat', choices=['concat', 'diff'])
    p.add_argument('--pretrained', action='store_true', help='ImageNet weights (one-off torchvision download)')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--synthetic', action='store_true', help='train on synthetic scenes (no dataset needed)')
    p.add_argument('--n-train', type=int, default=48)
    p.add_argument('--n-val', type=int, default=12)
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = select_device()
    tag = f"resnet_cd_{args.backbone}_{args.fusion}" + ('_synthetic' if args.synthetic else '')
    out = args.out or os.path.join(PROJECT_ROOT, 'outputs', 'resnet_cd', tag)
    os.makedirs(out, exist_ok=True)

    if args.synthetic:
        train_loader, val_loader = synthetic_loaders(args.n_train, args.n_val, args.size, args.batch_size, args.seed)
    else:
        train_loader, val_loader = levir_loaders(args.batch_size, args.size)

    model = SiameseResNetCD(args.backbone, args.fusion, args.pretrained).to(device)
    print(f"{tag}: {model.count_parameters():,} parameters ({model.model_size_mb()} MB) on {device}")
    loss_fn = CompositeChangeDetectionLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-6)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    history, best_f1, best_epoch = [], -1.0, 0
    ckpt_path = os.path.join(out, 'best_model.pt')
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t_loss, _, _ = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val = evaluate_validation(model, val_loader, loss_fn, device, threshold=0.5)
        sched.step()
        history.append({'epoch': epoch, 'train_loss': t_loss, 'val_loss': val['val_loss'],
                        'val_f1': val['f1_score'], 'val_iou': val['iou']})
        print(f"epoch {epoch:>2}: train {t_loss:.4f}  val loss {val['val_loss']:.4f}  "
              f"F1 {val['f1_score']:.4f}  IoU {val['iou']:.4f}")
        if val['f1_score'] > best_f1:
            best_f1, best_epoch = val['f1_score'], epoch
            torch.save({'model_state_dict': model.state_dict(), 'architecture_class': 'SiameseResNetCD',
                        'backbone': args.backbone, 'fusion': args.fusion, 'epoch': epoch,
                        'val_f1': best_f1, 'synthetic': args.synthetic}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    tuned = find_best_threshold(model, val_loader, device, [0.2, 0.3, 0.4, 0.5, 0.6])
    ckpt['optimal_threshold'] = tuned['best_threshold']
    torch.save(ckpt, ckpt_path)

    summary = {'tag': tag, 'parameters': model.count_parameters(), 'best_epoch': best_epoch,
               'best_val_f1_at_0.5': best_f1, 'tuned_threshold': tuned['best_threshold'],
               'tuned_val_f1': tuned['best_f1'], 'train_seconds': time.perf_counter() - start,
               'synthetic': args.synthetic, 'history': history}
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"best epoch {best_epoch}, val F1 {best_f1:.4f} (tau 0.5) -> {tuned['best_f1']:.4f} "
          f"(tau {tuned['best_threshold']}); checkpoint {ckpt_path}")
    return summary


if __name__ == '__main__':
    main()
