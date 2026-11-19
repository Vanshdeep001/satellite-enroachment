"""
Generative augmenters (Project Step 9): conditional VAE and conditional GAN.

Both learn p(x | class) for small RGB land-cover patches so that extra labelled samples can
be generated for scarce classes.

CVAE (Kingma & Welling 2014; Sohn et al. 2015)
    q(z | x, y) = N(mu, diag(sigma^2)),  p(x | z, y) = Bernoulli(decoder(z, y))
    loss = reconstruction BCE + beta * KL(q(z | x, y) || N(0, I))   (negative ELBO)
    Reparameterisation z = mu + sigma * eps keeps sampling differentiable.
    num_classes = 0 gives an unconditional VAE (used to visualise latent organisation).

Conditional GAN (Mirza & Osindero 2014, DCGAN-style)
    G(z, y) -> image;  D(x, y) -> real / fake logit, with spectral normalisation for stability.
    Non-saturating loss: D maximises log D(x) + log(1 - D(G(z))), G maximises log D(G(z)).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 0:
        return torch.zeros(y.shape[0], 0, device=y.device)
    return F.one_hot(y.long(), num_classes).float()


class CVAE(nn.Module):
    def __init__(self, num_classes: int = 4, latent_dim: int = 8, channels: int = 3, size: int = 16,
                 width: int = 32):
        super().__init__()
        if size % 4:
            raise ValueError("size must be divisible by 4")
        self.num_classes, self.latent_dim, self.size = num_classes, latent_dim, size
        self.width = width
        feat = 2 * width * (size // 4) ** 2
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(width, 2 * width, 4, 2, 1), nn.BatchNorm2d(2 * width), nn.LeakyReLU(0.2),
            nn.Flatten())
        self.fc_mu = nn.Linear(feat + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(feat + num_classes, latent_dim)
        self.fc_dec = nn.Linear(latent_dim + num_classes, feat)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2 * width, width, 4, 2, 1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.ConvTranspose2d(width, channels, 4, 2, 1), nn.Sigmoid())

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        h = torch.cat([self.encoder(x), one_hot(y, self.num_classes)], dim=1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(torch.cat([z, one_hot(y, self.num_classes)], dim=1))
        return self.decoder(h.view(-1, 2 * self.width, self.size // 4, self.size // 4))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y), mu, logvar

    @staticmethod
    def loss(recon, x, mu, logvar, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        rec = F.binary_cross_entropy(recon, x, reduction='sum') / b
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        return {'loss': rec + beta * kl, 'reconstruction': rec, 'kl': kl}

    @torch.no_grad()
    def sample(self, y: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        z = torch.randn(y.shape[0], self.latent_dim, device=y.device) * temperature
        return self.decode(z, y)


class CondGenerator(nn.Module):
    def __init__(self, num_classes: int = 4, noise_dim: int = 16, channels: int = 3, size: int = 16, width: int = 32):
        super().__init__()
        self.num_classes, self.noise_dim, self.size, self.width = num_classes, noise_dim, size, width
        self.fc = nn.Sequential(nn.Linear(noise_dim + num_classes, 2 * width * (size // 4) ** 2), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * width, width, 4, 2, 1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.ConvTranspose2d(width, channels, 4, 2, 1), nn.Sigmoid())

    def forward(self, z, y):
        h = self.fc(torch.cat([z, one_hot(y, self.num_classes)], dim=1))
        return self.net(h.view(-1, 2 * self.width, self.size // 4, self.size // 4))

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        return self.forward(torch.randn(y.shape[0], self.noise_dim, device=y.device), y)


class CondDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 3, size: int = 16, width: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels + num_classes, width, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(width, 2 * width, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.Flatten(), spectral_norm(nn.Linear(2 * width * (size // 4) ** 2, 1)))

    def forward(self, x, y):
        maps = one_hot(y, self.num_classes)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, maps], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_cvae(x: torch.Tensor, y: torch.Tensor, num_classes: int = 4, latent_dim: int = 8, epochs: int = 300,
               batch_size: int = 32, lr: float = 2e-3, beta: float = 1.0, seed: int = 0) -> (CVAE, List[Dict[str, float]]):
    torch.manual_seed(seed)
    model = CVAE(num_classes=num_classes, latent_dim=latent_dim, size=x.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = {'loss': 0.0, 'reconstruction': 0.0, 'kl': 0.0}
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.numel() < 2:
                continue
            recon, mu, logvar = model(x[idx], y[idx])
            losses = CVAE.loss(recon, x[idx], mu, logvar, beta)
            opt.zero_grad()
            losses['loss'].backward()
            opt.step()
            for k in tot:
                tot[k] += losses[k].item() * idx.numel()
        history.append({k: v / n for k, v in tot.items()})
    model.eval()
    return model, history


def train_cgan(x: torch.Tensor, y: torch.Tensor, num_classes: int = 4, steps: int = 3000, batch_size: int = 32,
               lr: float = 2e-4, seed: int = 0) -> (CondGenerator, List[Dict[str, float]]):
    torch.manual_seed(seed)
    gen = CondGenerator(num_classes=num_classes, size=x.shape[-1])
    disc = CondDiscriminator(num_classes=num_classes, size=x.shape[-1])
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    n = x.shape[0]
    history = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        real, labels = x[idx], y[idx]
        fake = gen(torch.randn(batch_size, gen.noise_dim), labels)

        d_loss = bce(disc(real, labels), torch.full((batch_size,), 0.9)) + \
            bce(disc(fake.detach(), labels), torch.zeros(batch_size))   # one-sided label smoothing
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        g_loss = bce(disc(fake, labels), torch.ones(batch_size))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        if step % 50 == 0:
            history.append({'step': step, 'd_loss': d_loss.item(), 'g_loss': g_loss.item()})
    gen.eval()
    return gen, history


# ---------------------------------------------------------------------------
# Small CNN classifier used to judge augmentation value and sample quality
# ---------------------------------------------------------------------------

class PatchCNN(nn.Module):
    def __init__(self, num_classes: int = 4, size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * (size // 4) ** 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)


def dihedral_torch(x: torch.Tensor) -> torch.Tensor:
    """Random rot90 + flip per sample (label-preserving for overhead imagery)."""
    out = torch.empty_like(x)
    ks = torch.randint(0, 4, (x.shape[0],))
    flips = torch.rand(x.shape[0]) < 0.5
    for i in range(x.shape[0]):
        img = torch.rot90(x[i], int(ks[i]), dims=(1, 2))
        out[i] = torch.flip(img, dims=(2,)) if flips[i] else img
    return out


def train_classifier(x_tr, y_tr, x_val, y_val, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
                     augment: bool = False, seed: int = 0, num_classes: int = 4) -> PatchCNN:
    """Adam training; keeps the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    model = PatchCNN(num_classes=num_classes, size=x_tr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc, best_state = -1.0, None
    n = x_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = dihedral_torch(x_tr[idx]) if augment else x_tr[idx]
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate_classifier(model, x_val, y_val)['accuracy']
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    model.eval()
    pred = model(x).argmax(dim=1)
    acc = (pred == y).float().mean().item()
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {'accuracy': acc, 'macro_f1': float(np.mean(f1s))}


def diversity(samples: torch.Tensor) -> float:
    """Mean pairwise L2 distance between flattened samples (higher = more varied)."""
    flat = samples.reshape(samples.shape[0], -1)
    d = torch.cdist(flat, flat)
    n = flat.shape[0]
    return float(d.sum() / (n * (n - 1))) if n > 1 else 0.0
