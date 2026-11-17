"""
LEVIR-CD Dataset Extraction, Audit, Manifest & Sample Visualization Utility.

Tasks:
1. Extract train.zip, val.zip, test.zip into data/train/, data/val/, data/test/.
2. Audit all splits programmatically (dimensions, channels, label values, corruption check).
3. Generate data/dataset_manifest.csv with metadata for every single image pair.
4. Generate visual comparison grid (T1, T2, Change Mask) saved under outputs/sample_pairs_preview.png.
"""

import os
import sys
import zipfile
import csv
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_split(zip_path: str, target_dir: str) -> str:
    """Safely extract split archive into split directory without collisions."""
    split_name = os.path.splitext(os.path.basename(zip_path))[0]
    dest_dir = os.path.join(target_dir, split_name)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"[*] Extracting {os.path.basename(zip_path)} into {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print(f"[+] Successfully extracted {split_name}.")
    return dest_dir


def audit_split(split_dir: str) -> Dict[str, Any]:
    """Audit an extracted split directory."""
    split_name = os.path.basename(split_dir)
    dir_a = os.path.join(split_dir, 'A')
    dir_b = os.path.join(split_dir, 'B')
    dir_label = os.path.join(split_dir, 'label')

    for d in [dir_a, dir_b, dir_label]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing expected directory: {d}")

    files_a = set(f for f in os.listdir(dir_a) if f.lower().endswith('.png'))
    files_b = set(f for f in os.listdir(dir_b) if f.lower().endswith('.png'))
    files_label = set(f for f in os.listdir(dir_label) if f.lower().endswith('.png'))

    missing_b = files_a - files_b
    missing_label = files_a - files_label

    matched_pairs = sorted(list(files_a & files_b & files_label))

    dims = set()
    channels_a = set()
    channels_b = set()
    label_values = set()
    corrupted = []
    total_change_pixels = 0
    total_pixels = 0

    records = []

    for fname in matched_pairs:
        path_a = os.path.join(dir_a, fname)
        path_b = os.path.join(dir_b, fname)
        path_lbl = os.path.join(dir_label, fname)

        try:
            with Image.open(path_a) as img_a:
                dims.add(img_a.size)
                channels_a.add(img_a.mode)
                w, h = img_a.size

            with Image.open(path_b) as img_b:
                dims.add(img_b.size)
                channels_b.add(img_b.mode)

            with Image.open(path_lbl) as img_lbl:
                lbl_arr = np.array(img_lbl)
                label_values.update(np.unique(lbl_arr))
                change_count = int(np.sum(lbl_arr > 127))
                total_change_pixels += change_count
                total_pixels += int(lbl_arr.size)
                pair_change_pct = (change_count / lbl_arr.size) * 100.0

            records.append({
                'split': split_name,
                'filename': fname,
                'path_t1': os.path.relpath(path_a),
                'path_t2': os.path.relpath(path_b),
                'path_mask': os.path.relpath(path_lbl),
                'width': w,
                'height': h,
                'change_pixels': change_count,
                'change_percentage': f"{pair_change_pct:.3f}"
            })

        except Exception as e:
            corrupted.append((fname, str(e)))

    change_ratio = (total_change_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    return {
        'split': split_name,
        'num_pairs': len(matched_pairs),
        'missing_b': list(missing_b),
        'missing_label': list(missing_label),
        'corrupted': corrupted,
        'dimensions': list(dims),
        'channels_t1': list(channels_a),
        'channels_t2': list(channels_b),
        'unique_mask_values': sorted(list(label_values)),
        'total_change_pixels': total_change_pixels,
        'total_pixels': total_pixels,
        'change_pixel_percentage': change_ratio,
        'records': records
    }


def generate_visual_samples(split_dir: str, output_path: str, num_samples: int = 4):
    """Generate side-by-side visualization of T1, T2, and ground-truth masks."""
    dir_a = os.path.join(split_dir, 'A')
    dir_b = os.path.join(split_dir, 'B')
    dir_label = os.path.join(split_dir, 'label')

    files = sorted([f for f in os.listdir(dir_a) if f.lower().endswith('.png')])
    selected_files = []
    for f in files:
        mask_path = os.path.join(dir_label, f)
        if os.path.exists(mask_path):
            arr = np.array(Image.open(mask_path))
            if np.sum(arr > 127) > 1000:
                selected_files.append(f)
            if len(selected_files) >= num_samples:
                break

    if len(selected_files) < num_samples:
        selected_files = files[:num_samples]

    fig, axes = plt.subplots(len(selected_files), 3, figsize=(12, 4 * len(selected_files)))
    if len(selected_files) == 1:
        axes = np.expand_dims(axes, 0)

    for i, fname in enumerate(selected_files):
        img_t1 = Image.open(os.path.join(dir_a, fname)).convert('RGB')
        img_t2 = Image.open(os.path.join(dir_b, fname)).convert('RGB')
        mask = Image.open(os.path.join(dir_label, fname)).convert('L')

        axes[i, 0].imshow(img_t1)
        axes[i, 0].set_title(f"T1 (Pre-Change): {fname}", fontsize=11)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_t2)
        axes[i, 1].set_title(f"T2 (Post-Change): {fname}", fontsize=11)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title(f"Ground-Truth Change Mask", fontsize=11)
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Visual comparison sample grid saved to: {output_path}")


def write_manifest(all_records: List[Dict[str, Any]], manifest_path: str):
    """Write comprehensive dataset manifest CSV."""
    if not all_records:
        return
    fieldnames = list(all_records[0].keys())
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[+] Dataset manifest written: {manifest_path} ({len(all_records)} entries)")
