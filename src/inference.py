"""
Inference Pipeline for Bitemporal Satellite Land Use Change & Encroachment Detection.

Provides:
- Independent, production-ready inference pipeline decoupled from training loops.
- Support for multiple image inputs: file paths, PIL Images, or NumPy arrays.
- Automatic architecture detection (SiameseUNet vs SiameseUNetConc) from checkpoint metadata.
- Exact training-matched preprocessing: Bilinear resize to 256x256, [0.0, 1.0] scaling.
- Probability map generation and validation-tuned threshold binarization.
- Rich 4-panel overlay visualizer (T1, T2, Probability Heatmap, Change Overlay on T2).
- Robust error handling for missing files, corrupt imagery, and parameter mismatches.
"""

import os
import sys
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.mpl_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.experiments.models import SiameseUNetConc


@dataclass
class ChangeDetectionResult:
    """Encapsulates the full output of a change detection inference run."""
    prob_map: np.ndarray             # Float32 array in [0.0, 1.0], shape [H, W]
    binary_mask: np.ndarray          # Uint8 array in {0, 255}, shape [H, W]
    threshold_used: float            # Decision threshold applied
    changed_pixels: int              # Total count of changed pixels
    total_pixels: int                # Total image pixels
    change_percentage: float         # Percentage of surface area changed
    original_size: Tuple[int, int]   # Original (width, height) of inputs
    model_name: str                  # Name of the model used
    device: str                      # Compute device used


class ChangeDetectorInference:
    """
    Production inference engine for registered bitemporal satellite image pairs.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        default_threshold: Optional[float] = None
    ):
        """
        Initialize the inference engine with a serialized model checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file. Defaults to outputs/checkpoints/final_selected_model.pt.
            device: Compute device ('mps', 'cuda', 'cpu'). Auto-selects if None.
            default_threshold: Override probability threshold. If None, uses threshold saved in checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints', 'final_selected_model.pt')
            if not os.path.isfile(checkpoint_path):
                # Fallback to Phase 5 best_model.pt if final_selected_model not yet generated
                checkpoint_path = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints', 'best_model.pt')

        self.checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: '{self.checkpoint_path}'")

        self.device = self._resolve_device(device)
        self.model, self.model_metadata = self._load_model(self.checkpoint_path, self.device)
        self.model.eval()

        # Resolve probability decision threshold
        if default_threshold is not None:
            if not (0.0 < default_threshold < 1.0):
                raise ValueError(f"Threshold must be strictly between 0.0 and 1.0, got: {default_threshold}")
            self.threshold = float(default_threshold)
        else:
            self.threshold = float(self.model_metadata.get('optimal_threshold', 0.20))

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.backends.mps.is_available():
            try:
                t = torch.zeros(1, device='mps')
                del t
                return torch.device('mps')
            except Exception:
                return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _load_model(self, checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
        """Safely load model weights and resolve architecture dynamically."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Detect whether checkpoint is raw state_dict or structured dictionary
        if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
            metadata = {k: v for k, v in checkpoint.items() if k not in ('state_dict', 'model_state_dict')}
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        else:
            metadata = {}
            state_dict = checkpoint

        arch_class = metadata.get('architecture_class', '')
        base_channels = int(metadata.get('base_channels', 16))

        # Determine architecture
        if arch_class == 'SiameseResNetCD':
            from src.resnet_change import SiameseResNetCD
            model = SiameseResNetCD(backbone=metadata.get('backbone', 'resnet18'),
                                    fusion=metadata.get('fusion', 'concat'), pretrained=False)
            model_name = f"SiameseResNetCD-{metadata.get('backbone', 'resnet18')}-{metadata.get('fusion', 'concat')}"
        elif 'Conc' in arch_class or 'Concatenation' in str(metadata.get('candidate_name', '')):
            model = SiameseUNetConc(in_channels=3, out_channels=1, base_channels=base_channels, bilinear=True)
            model_name = "SiameseUNetConc"
        elif 'SiameseUNet' in arch_class or 'Diff' in str(metadata.get('candidate_name', '')):
            model = SiameseUNet(in_channels=3, out_channels=1, base_channels=base_channels, bilinear=True)
            model_name = "SiameseUNet"
        else:
            # Check state_dict layer signatures: bottleneck_conv is unique to SiameseUNetConc
            is_conc = any('bottleneck_conv' in k for k in state_dict.keys())
            if is_conc:
                model = SiameseUNetConc(in_channels=3, out_channels=1, base_channels=base_channels, bilinear=True)
                model_name = "SiameseUNetConc"
            else:
                model = SiameseUNet(in_channels=3, out_channels=1, base_channels=base_channels, bilinear=True)
                model_name = "SiameseUNet"

        model.load_state_dict(state_dict)
        model.to(device)
        metadata['resolved_model_name'] = model_name
        return model, metadata

    def preprocess_image(self, img_input: Union[str, Image.Image, np.ndarray], target_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert input image (path, PIL, or ndarray) into preprocessed torch.Tensor [1, 3, H, W].

        Returns:
            tensor: Preprocessed tensor ready for model input.
            original_size: (width, height) of the original image before resizing.
        """
        if isinstance(img_input, str):
            if not os.path.isfile(img_input):
                raise FileNotFoundError(f"Input image not found: '{img_input}'")
            try:
                pil_img = Image.open(img_input).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to open image file '{img_input}': {e}")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert('RGB')
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                pil_img = Image.fromarray(img_input).convert('RGB')
            elif img_input.ndim == 3:
                if img_input.shape[2] == 4:
                    pil_img = Image.fromarray(img_input).convert('RGB')
                elif img_input.shape[2] == 3:
                    pil_img = Image.fromarray(img_input)
                else:
                    raise ValueError(f"Unsupported array shape: {img_input.shape}")
            else:
                raise ValueError(f"Array must be 2D or 3D, got ndim={img_input.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        orig_w, orig_h = pil_img.size

        # Exact training preprocessing: Bilinear resize to 256x256, [0.0, 1.0] scaling
        resized = TF.resize(pil_img, target_size, interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized).unsqueeze(0)  # Shape [1, 3, H, W]

        return tensor, (orig_w, orig_h)

    def predict(
        self,
        img_t1: Union[str, Image.Image, np.ndarray],
        img_t2: Union[str, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
        resize_mask_to_original: bool = True
    ) -> ChangeDetectionResult:
        """
        Run end-to-end inference on a pair of bitemporal satellite images.

        Args:
            img_t1: Pre-change satellite image (path, PIL, or ndarray).
            img_t2: Post-change satellite image (path, PIL, or ndarray).
            threshold: Probability threshold in (0, 1). If None, uses engine default.
            resize_mask_to_original: If True, interpolates output mask and probability map
                                     back to the original input resolution.

        Returns:
            ChangeDetectionResult dataclass.
        """
        th = self.threshold if threshold is None else float(threshold)
        if not (0.0 < th < 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {th}")

        # Preprocess both images
        t1_tensor, size1 = self.preprocess_image(img_t1)
        t2_tensor, size2 = self.preprocess_image(img_t2)

        if size1 != size2:
            print(f"[Warning] Image dimension mismatch: T1 is {size1}, T2 is {size2}. Rescaling both to 256x256.")

        t1_tensor = t1_tensor.to(self.device)
        t2_tensor = t2_tensor.to(self.device)

        # Forward pass in no_grad mode
        with torch.no_grad():
            logits = self.model(t1_tensor, t2_tensor)
            probs = torch.sigmoid(logits)  # [1, 1, 256, 256]

        probs_np = probs.squeeze().cpu().numpy()  # [256, 256]

        if resize_mask_to_original and size1 != (256, 256):
            # Bilinear resize probability map to original resolution
            probs_pil = Image.fromarray(probs_np)
            probs_pil = probs_pil.resize(size1, resample=Image.BILINEAR)
            probs_np = np.array(probs_pil, dtype=np.float32)

        binary_mask = (probs_np >= th).astype(np.uint8) * 255
        changed_count = int(np.count_nonzero(binary_mask))
        total_pixels = int(binary_mask.size)
        change_pct = (changed_count / total_pixels) * 100.0

        return ChangeDetectionResult(
            prob_map=probs_np,
            binary_mask=binary_mask,
            threshold_used=th,
            changed_pixels=changed_count,
            total_pixels=total_pixels,
            change_percentage=round(change_pct, 3),
            original_size=size1,
            model_name=self.model_metadata.get('resolved_model_name', self.model.__class__.__name__),
            device=self.device.type
        )

    def save_visualization(
        self,
        img_t1: Union[str, Image.Image, np.ndarray],
        img_t2: Union[str, Image.Image, np.ndarray],
        result: ChangeDetectionResult,
        save_path: str,
        title: Optional[str] = None
    ) -> str:
        """
        Generate and save a 4-panel visual report:
        [Image T1] | [Image T2] | [Probability Heatmap] | [Change Overlay on T2]
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Convert inputs to PIL for display
        if isinstance(img_t1, str):
            p1 = Image.open(img_t1).convert('RGB')
        elif isinstance(img_t1, np.ndarray):
            p1 = Image.fromarray(img_t1).convert('RGB')
        else:
            p1 = img_t1.convert('RGB')

        if isinstance(img_t2, str):
            p2 = Image.open(img_t2).convert('RGB')
        elif isinstance(img_t2, np.ndarray):
            p2 = Image.fromarray(img_t2).convert('RGB')
        else:
            p2 = img_t2.convert('RGB')

        # Match sizes
        p1 = p1.resize(result.original_size, resample=Image.BILINEAR)
        p2 = p2.resize(result.original_size, resample=Image.BILINEAR)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#ffffff')

        # 1. Image T1
        axes[0].imshow(p1)
        axes[0].set_title("Pre-Change (T1)", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 2. Image T2
        axes[1].imshow(p2)
        axes[1].set_title("Post-Change (T2)", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 3. Probability Heatmap
        im_heat = axes[2].imshow(result.prob_map, cmap='magma', vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Change Probability (τ*={result.threshold_used:.2f})", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        cbar = fig.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # 4. Red Mask Overlay on Image T2
        img_t2_np = np.array(p2).astype(np.float32) / 255.0
        mask_bool = result.binary_mask > 0

        overlay = img_t2_np.copy()
        # Red tint over detected building changes (RGB: [1.0, 0.1, 0.1])
        alpha = 0.55
        overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * np.array([1.0, 0.1, 0.1])

        axes[3].imshow(np.clip(overlay, 0.0, 1.0))
        axes[3].set_title(
            f"Detected Changes ({result.change_percentage:.2f}% | {result.changed_pixels:,} px)",
            fontsize=12,
            fontweight='bold',
            color='#b30000' if result.changed_pixels > 0 else '#333333'
        )
        axes[3].axis('off')

        sup_title = title or f"Satellite Change Detection Result ({result.model_name} on {result.device.upper()})"
        plt.suptitle(sup_title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        return os.path.abspath(save_path)


def main():
    """Command-line interface for running inference."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Satellite Change Detection Inference")
    parser.add_argument('--t1', type=str, required=True, help="Path to pre-change image (T1)")
    parser.add_argument('--t2', type=str, required=True, help="Path to post-change image (T2)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument('--output', type=str, default="outputs/inference_result.png", help="Path to save overlay visual")
    parser.add_argument('--threshold', type=float, default=None, help="Probability threshold (e.g. 0.20)")
    parser.add_argument('--device', type=str, default=None, help="Compute device (mps, cuda, cpu)")

    args = parser.parse_args()

    engine = ChangeDetectorInference(checkpoint_path=args.checkpoint, device=args.device, default_threshold=args.threshold)
    print(f"Loaded {engine.model_metadata.get('resolved_model_name')} on {engine.device} with threshold {engine.threshold:.2f}")

    result = engine.predict(img_t1=args.t1, img_t2=args.t2)
    print(f"\nInference Complete:")
    print(f"  Changed Area: {result.change_percentage:.2f}% ({result.changed_pixels:,} of {result.total_pixels:,} pixels)")
    print(f"  Applied Threshold: {result.threshold_used:.2f}")

    saved_path = engine.save_visualization(img_t1=args.t1, img_t2=args.t2, result=result, save_path=args.output)
    print(f"  Overlay visual saved to: {saved_path}")


if __name__ == '__main__':
    main()
