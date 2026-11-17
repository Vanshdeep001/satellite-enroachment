"""
PyTorch Dataset and DataLoader Utilities for LEVIR-CD Change Detection.

Provides:
- LEVIRCDDataset: PyTorch Dataset loading registered bitemporal image pairs
  (T1 and T2) and corresponding binary change masks.
- Transformations: Resizing with Bilinear interpolation for images,
  Nearest-Neighbor for masks, normalization to [0.0, 1.0], and tensor conversion.
"""

import os
from typing import Tuple, Optional, Callable, Dict, Any, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class LEVIRCDDataset(Dataset):
    """
    Dataset class for Bitemporal Remote Sensing Building Change Detection (LEVIR-CD).

    Each sample consists of:
    - image_t1: Image at time T1 (pre-change) [3, H, W], float32 in [0, 1]
    - image_t2: Image at time T2 (post-change) [3, H, W], float32 in [0, 1]
    - mask: Ground-truth change mask [1, H, W], float32 in {0.0, 1.0}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'val',
        target_size: Optional[Tuple[int, int]] = (256, 256),
        transform: Optional[Callable] = None,
        manifest_file: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory of extracted LEVIR-CD data (containing train/, val/, test/).
            split: 'train', 'val', or 'test'.
            target_size: (height, width) to resize images/masks (default: (256, 256)).
                         If None, original resolution (1024x1024) is retained.
            transform: Optional custom augmentation callable for training.
            manifest_file: Optional path to a pre-generated CSV manifest.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split.lower()
        self.target_size = target_size
        self.transform = transform

        if self.split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split '{self.split}'. Expected 'train', 'val', or 'test'.")

        self.samples: List[Dict[str, str]] = []

        if manifest_file and os.path.isfile(manifest_file):
            self._load_from_manifest(manifest_file)
        else:
            self._discover_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No valid image pairs found for split '{self.split}' in '{self.root_dir}'. "
                f"Please ensure directories A/, B/, and label/ exist under '{self.root_dir}/{self.split}'."
            )

    def _discover_samples(self):
        """Discover aligned (T1, T2, mask) triplets by scanning directory structure."""
        split_dir = os.path.join(self.root_dir, self.split)
        dir_t1 = os.path.join(split_dir, 'A')
        dir_t2 = os.path.join(split_dir, 'B')
        dir_mask = os.path.join(split_dir, 'label')

        if not (os.path.isdir(dir_t1) and os.path.isdir(dir_t2) and os.path.isdir(dir_mask)):
            # Check alternative flattened layout (e.g., data/val/A or data/raw/val/A)
            return

        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        filenames = sorted([
            f for f in os.listdir(dir_t1)
            if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith('.')
        ])

        for fname in filenames:
            path_t1 = os.path.join(dir_t1, fname)
            path_t2 = os.path.join(dir_t2, fname)
            path_mask = os.path.join(dir_mask, fname)

            if not os.path.isfile(path_t2):
                raise FileNotFoundError(f"Missing corresponding T2 image for {fname}: '{path_t2}'")
            if not os.path.isfile(path_mask):
                raise FileNotFoundError(f"Missing corresponding label mask for {fname}: '{path_mask}'")

            self.samples.append({
                'name': fname,
                'path_t1': path_t1,
                'path_t2': path_t2,
                'path_mask': path_mask,
                'split': self.split
            })

    def _load_from_manifest(self, manifest_file: str):
        """Load file mappings from a pre-generated CSV manifest."""
        import csv
        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('split', '').lower() == self.split:
                    self.samples.append({
                        'name': row.get('filename', os.path.basename(row['path_t1'])),
                        'path_t1': os.path.abspath(row['path_t1']),
                        'path_t2': os.path.abspath(row['path_t2']),
                        'path_mask': os.path.abspath(row['path_mask']),
                        'split': self.split
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the (T1, T2, mask) triplet at index `idx`.

        Returns:
            t1_tensor:   torch.FloatTensor of shape [3, H, W], normalized in [0.0, 1.0]
            t2_tensor:   torch.FloatTensor of shape [3, H, W], normalized in [0.0, 1.0]
            mask_tensor: torch.FloatTensor of shape [1, H, W], binary values in {0.0, 1.0}
        """
        sample_meta = self.samples[idx]

        try:
            # 1. Load images using Pillow
            img_t1 = Image.open(sample_meta['path_t1']).convert('RGB')
            img_t2 = Image.open(sample_meta['path_t2']).convert('RGB')
            mask = Image.open(sample_meta['path_mask']).convert('L')
        except Exception as e:
            raise IOError(f"Error loading image pair for '{sample_meta['name']}': {e}")

        # 2. Consistent Spatial Resizing
        # IMPORTANT: Use Bilinear/Bicubic for RGB images, Nearest-Neighbor for binary masks!
        if self.target_size is not None:
            img_t1 = TF.resize(img_t1, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            img_t2 = TF.resize(img_t2, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Custom Augmentations (applied synchronously across T1, T2, and Mask)
        if self.transform is not None:
            img_t1, img_t2, mask = self.transform(img_t1, img_t2, mask)

        # 4. Conversion to Tensors
        # TF.to_tensor scales uint8 [0, 255] to float32 [0.0, 1.0]
        t1_tensor = TF.to_tensor(img_t1)
        t2_tensor = TF.to_tensor(img_t2)

        # Masks: Convert to tensor and threshold strictly to {0.0, 1.0}
        # In LEVIR-CD, changed pixels are typically 255 (or > 0).
        mask_np = np.array(mask, dtype=np.uint8)
        mask_bin = (mask_np > 127).astype(np.float32)  # threshold at midpoint
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # Shape: [1, H, W]

        return t1_tensor, t2_tensor, mask_tensor

    def get_sample_metadata(self, idx: int) -> Dict[str, str]:
        """Return metadata paths for a specific index."""
        return self.samples[idx]


def get_dataloader(
    root_dir: str,
    split: str = 'val',
    batch_size: int = 16,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    shuffle: bool = False,
    num_workers: int = 0,
    manifest_file: Optional[str] = None
) -> DataLoader:
    """Convenience helper to instantiate a DataLoader for a given dataset split."""
    dataset = LEVIRCDDataset(
        root_dir=root_dir,
        split=split,
        target_size=target_size,
        manifest_file=manifest_file
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Note: pin_memory is not supported on MPS
    )
