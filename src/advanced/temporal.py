"""
Multi-date (temporal) change models (Project Step 9; 3-D convolution for Step 7).

Task: given T co-registered images of a site, predict for every pixel the date at which a new
structure appeared (class k in 1..T-1) or "no change" (class 0). Binary change = class > 0.

Why more than two dates? The bitemporal formulation compares only the first and last image,
so a cloud, haze or seasonal effect in either one corrupts the result and the timing of the
change is unknown. A sequence model can use the clean dates and localise the change in time.

Models (all share the same per-frame 2-D encoder width so the comparison is about temporal
reasoning, not capacity):
    BitemporalCD      first and last frame only: head on [f_0, f_T, |f_0 - f_T|]
    ConvLSTMCD        convolutional LSTM over the T frames; gates i, f, o (sigmoid), g (tanh)
    TemporalTransformerCD  per-pixel tokens across time + multi-head self-attention
    Conv3DCD          3-D convolutions over the (time, height, width) volume

Synthetic sequences (make_sequence) include illumination changes, a seasonal vegetation
cycle, sensor noise and random clouds.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


# ---------------------------------------------------------------------------
# Synthetic sequences
# ---------------------------------------------------------------------------

def make_sequence(
    size: int = 48,
    steps: int = 6,
    seed: int = 0,
    n_existing: int = 3,
    n_new: int = 3,
    cloud_prob: float = 0.3,
    force_cloudy: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        frames [T, H, W, 3] float32 in [0, 1]
        date_map [H, W] int64 (0 = no change, k = structure appeared at date k)
        cloud_masks [T, H, W] float32 cloud opacity per date
    """
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.normal(size=(size, size)), sigma=size / 10)
    field = (field - field.min()) / (np.ptp(field) + 1e-12)
    veg, soil = np.array([0.30, 0.45, 0.22]), np.array([0.55, 0.47, 0.35])
    base = field[..., None] * veg + (1 - field[..., None]) * soil

    occupied = np.zeros((size, size), bool)
    buildings = []  # (r0, c0, r1, c1, colour, appear_date)

    def place(appear: int) -> None:
        for _ in range(200):
            h, w = rng.integers(5, 11, size=2)
            r, c = rng.integers(1, size - h - 2), rng.integers(1, size - w - 2)
            if not occupied[max(r - 2, 0):r + h + 2, max(c - 2, 0):c + w + 2].any():
                occupied[r:r + h + 1, c:c + w] = True
                colour = [np.array([0.75, 0.72, 0.68]), np.array([0.62, 0.30, 0.25]),
                          np.array([0.55, 0.58, 0.62])][rng.integers(3)]
                buildings.append((r, c, r + h, c + w, colour, appear))
                return

    for _ in range(n_existing):
        place(0)
    for _ in range(n_new):
        place(int(rng.integers(1, steps)))

    date_map = np.zeros((size, size), np.int64)
    frames = np.empty((steps, size, size, 3), np.float32)
    clouds = np.zeros((steps, size, size), np.float32)
    force = set(force_cloudy or [])
    for t in range(steps):
        img = base.copy()
        season = math.sin(2 * math.pi * t / steps)
        img[..., 1] += 0.06 * season * field
        img[..., 0] -= 0.03 * season * field
        for r0, c0, r1, c1, colour, appear in buildings:
            if t >= appear:
                img[r0:r1, c0:c1] = colour
                img[r1:r1 + 1, c0:c1] *= 0.5  # shadow
        img *= rng.uniform(0.85, 1.15)
        img += rng.normal(0, 0.02, img.shape)
        if t > 0 and (t in force or rng.random() < cloud_prob):
            blob = ndimage.gaussian_filter(rng.normal(size=(size, size)), sigma=size / 8)
            blob = (blob - blob.mean()) / (blob.std() + 1e-12)
            opacity = np.clip((blob - rng.uniform(-0.5, 0.5)) * 1.2, 0, 0.9)
            img = img * (1 - opacity[..., None]) + 0.92 * opacity[..., None]
            clouds[t] = opacity
        frames[t] = np.clip(img, 0, 1)
    for r0, c0, r1, c1, _, appear in buildings:
        if appear > 0:
            date_map[r0:r1, c0:c1] = appear
    return frames, date_map, clouds


def make_sequence_dataset(n: int, seed: int, **kwargs) -> Dict[str, torch.Tensor]:
    seqs = [make_sequence(seed=seed * 100_000 + i, **kwargs) for i in range(n)]
    x = torch.tensor(np.stack([s[0] for s in seqs])).permute(0, 1, 4, 2, 3).contiguous()  # [N, T, 3, H, W]
    return {'x': x, 'y': torch.tensor(np.stack([s[1] for s in seqs])),
            'clouds': torch.tensor(np.stack([s[2] for s in seqs]))}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.ReLU())

    def forward(self, x):  # [B, 3, H, W]
        return self.net(x)


class BitemporalCD(nn.Module):
    def __init__(self, steps: int, width: int = 16):
        super().__init__()
        self.encoder = FrameEncoder(width)
        self.head = nn.Sequential(nn.Conv2d(3 * width, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, steps, 1))

    def forward(self, x):
        f0, f1 = self.encoder(x[:, 0]), self.encoder(x[:, -1])
        return self.head(torch.cat([f0, f1, (f0 - f1).abs()], dim=1))


class ConvLSTMCell(nn.Module):
    """
    i, f, o = sigmoid(...), g = tanh(...); c' = f*c + i*g; h' = o * tanh(c').

    hard_gates=True swaps the smooth gates for their hard limits at inference
    (sigmoid -> step, tanh -> sign) to show what the smooth activations contribute.
    """

    def __init__(self, cin: int, hidden: int, kernel: int = 3):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(cin + hidden, 4 * hidden, kernel, padding=kernel // 2)
        self.record_gates = False
        self.hard_gates = False
        self.gates: List[Dict[str, torch.Tensor]] = []

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], dim=1)), self.hidden, dim=1)
        if self.hard_gates:
            i, f, o, g = (i > 0).float(), (f > 0).float(), (o > 0).float(), torch.sign(g)
            c = f * c + i * g
            h = o * torch.sign(c)
        else:
            i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
        if self.record_gates:
            self.gates.append({'input': i.detach(), 'forget': f.detach(), 'output': o.detach(), 'candidate': g.detach()})
        return h, c


class ConvLSTMCD(nn.Module):
    def __init__(self, steps: int, width: int = 16, hidden: int = 16):
        super().__init__()
        self.encoder = FrameEncoder(width)
        self.cell = ConvLSTMCell(width, hidden)
        self.head = nn.Sequential(nn.Conv2d(steps * hidden, 32, 1), nn.ReLU(), nn.Conv2d(32, steps, 1))

    def forward(self, x):
        b, t, _, hgt, wid = x.shape
        h = x.new_zeros(b, self.cell.hidden, hgt, wid)
        c = torch.zeros_like(h)
        states = []
        for k in range(t):
            h, c = self.cell(self.encoder(x[:, k]), (h, c))
            states.append(h)
        return self.head(torch.cat(states, dim=1))


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 2):
        super().__init__()
        self.heads, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.last_attention: Optional[torch.Tensor] = None
        self.keep_attention = False

    def forward(self, x):  # [N, T, D]
        n, t, d = x.shape
        q, k, v = self.qkv(x).view(n, t, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1)  # [N, heads, T, T]
        if self.keep_attention:
            self.last_attention = attn.detach()
        return self.out((attn @ v).transpose(1, 2).reshape(n, t, d))


class TemporalTransformerCD(nn.Module):
    """Per-pixel temporal tokens: x_t(pixel) = encoder features + learned date embedding."""

    def __init__(self, steps: int, width: int = 16, heads: int = 2):
        super().__init__()
        self.encoder = FrameEncoder(width)
        self.pos = nn.Parameter(torch.randn(steps, width) * 0.1)
        self.attn = TemporalSelfAttention(width, heads)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.head = nn.Sequential(nn.Linear(steps * width, 32), nn.ReLU(), nn.Linear(32, steps))

    def forward(self, x):
        b, t, _, h, w = x.shape
        feats = self.encoder(x.reshape(b * t, *x.shape[2:])).view(b, t, -1, h, w)
        tokens = feats.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, -1) + self.pos
        tokens = self.norm1(tokens + self.attn(tokens))
        tokens = self.norm2(tokens + self.ffn(tokens))
        logits = self.head(tokens.reshape(b * h * w, -1))
        return logits.view(b, h, w, t).permute(0, 3, 1, 2)


class Conv3DCD(nn.Module):
    """3-D kernels (time x height x width) see spatio-temporal patterns directly."""

    def __init__(self, steps: int, width: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(3, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(steps * width, 32, 1), nn.ReLU(), nn.Conv2d(32, steps, 1))

    def features(self, x):
        v = x.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
        return F.relu(self.conv2(F.relu(self.conv1(v))))

    def forward(self, x):
        f = self.features(x)
        b, c, t, h, w = f.shape
        return self.head(f.reshape(b, c * t, h, w))


MODEL_BUILDERS = {
    'Bitemporal (first vs last)': BitemporalCD,
    'ConvLSTM': ConvLSTMCD,
    'Temporal transformer': TemporalTransformerCD,
    '3-D CNN': Conv3DCD,
}


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).argmax(dim=1) for i in range(0, len(x), batch_size)])


def temporal_metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pb, yb = pred > 0, y > 0
    tp = (pb & yb).sum().item()
    fp = (pb & ~yb).sum().item()
    fn = (~pb & yb).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    changed = yb
    date_acc = (pred[changed] == y[changed]).float().mean().item() if changed.any() else float('nan')
    detected = pb & yb
    date_acc_detected = (pred[detected] == y[detected]).float().mean().item() if detected.any() else float('nan')
    return {'f1': f1, 'iou': iou, 'change_recall': tp / (tp + fn) if tp + fn else 0.0,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'date_accuracy': date_acc, 'date_accuracy_on_detected': date_acc_detected}


def train_temporal(model: nn.Module, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor], steps: int,
                   epochs: int = 20, batch_size: int = 8, lr: float = 2e-3, change_weight: float = 5.0,
                   seed: int = 0) -> Dict[str, Any]:
    """Adam + class-weighted cross-entropy; keeps the epoch with the best validation F1."""
    torch.manual_seed(seed)
    weights = torch.tensor([1.0] + [change_weight] * (steps - 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train['x'])
    history, best_f1, best_state = [], -1.0, None
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(train['x'][idx]), train['y'][idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        m = temporal_metrics(predict(model, val['x']), val['y'])
        history.append({'epoch': epoch + 1, 'train_loss': total / n, **{f'val_{k}': v for k, v in m.items()}})
        if m['f1'] > best_f1:
            best_f1, best_state = m['f1'], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {'history': history, 'train_seconds': time.perf_counter() - start, 'best_val_f1': best_f1}
