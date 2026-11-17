"""
Synchronous Bitemporal Data Augmentations for Remote Sensing Change Detection.

Applies identical spatial transformations simultaneously across:
- Pre-change Image (T1)
- Post-change Image (T2)
- Ground-Truth Binary Mask

Transformations include:
- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Orthogonal Rotation (0, 90, 180, 270 degrees)
"""

import random
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF


class SynchronousBitemporalAugmentation:
    """
    Applies identical geometric augmentations to (img_t1, img_t2, mask) synchronously.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob

    def __call__(
        self,
        img_t1: Image.Image,
        img_t2: Image.Image,
        mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        # 1. Random Horizontal Flip
        if random.random() < self.hflip_prob:
            img_t1 = TF.hflip(img_t1)
            img_t2 = TF.hflip(img_t2)
            mask = TF.hflip(mask)

        # 2. Random Vertical Flip
        if random.random() < self.vflip_prob:
            img_t1 = TF.vflip(img_t1)
            img_t2 = TF.vflip(img_t2)
            mask = TF.vflip(mask)

        # 3. Random Orthogonal Rotation (0, 90, 180, 270 degrees)
        if random.random() < self.rotate_prob:
            angle = random.choice([90, 180, 270])
            img_t1 = TF.rotate(img_t1, angle)
            img_t2 = TF.rotate(img_t2, angle)
            mask = TF.rotate(mask, angle)

        return img_t1, img_t2, mask

