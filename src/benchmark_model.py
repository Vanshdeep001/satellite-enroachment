"""
Benchmark and Tensor Dimensions Audit for Siamese U-Net.

Measures:
1. Exact parameter counts and memory footprint for lightweight and standard configs.
2. Tensor dimensions at every stage of the encoder and decoder.
3. Latency benchmarks on CPU and Apple Silicon MPS.
4. Generates an architectural diagram saved to outputs/architecture_diagram.png.
"""

import os
import sys
import time
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import SiameseUNet
from src.config import CONFIG


def profile_tensor_shapes(model: SiameseUNet, img_size: int = 256):
    """Trace tensor shapes through every stage of Siamese U-Net."""
    print("=" * 70)
    print("SIAMESE U-NET TENSOR SHAPE TRACE")
    print("=" * 70)

    b = 1
    t1 = torch.zeros(b, 3, img_size, img_size)
    t2 = torch.zeros(b, 3, img_size, img_size)

    print(f"Input T1 Shape: {list(t1.shape)}")
    print(f"Input T2 Shape: {list(t2.shape)}")
    print("-" * 50)

    # Encoder trace
    f1_1, f1_2, f1_3, f1_4, f1_bot = model.encoder(t1)
    f2_1, f2_2, f2_3, f2_4, f2_bot = model.encoder(t2)

    print("SHARED ENCODER STAGES (T1 and T2):")
    print(f"  Level 1 (ConvBlock 1):   {list(f1_1.shape)} (Resolution: {f1_1.shape[2]}x{f1_1.shape[3]})")
    print(f"  Level 2 (ConvBlock 2):   {list(f1_2.shape)} (Resolution: {f1_2.shape[2]}x{f1_2.shape[3]})")
    print(f"  Level 3 (ConvBlock 3):   {list(f1_3.shape)} (Resolution: {f1_3.shape[2]}x{f1_3.shape[3]})")
    print(f"  Level 4 (ConvBlock 4):   {list(f1_4.shape)} (Resolution: {f1_4.shape[2]}x{f1_4.shape[3]})")
    print(f"  Bottleneck:              {list(f1_bot.shape)} (Resolution: {f1_bot.shape[2]}x{f1_bot.shape[3]})")
    print("-" * 50)

    # Differential features
    diff_1 = torch.abs(f1_1 - f2_1)
    diff_2 = torch.abs(f1_2 - f2_2)
    diff_3 = torch.abs(f1_3 - f2_3)
    diff_4 = torch.abs(f1_4 - f2_4)
    diff_bot = torch.abs(f1_bot - f2_bot)

    print("MULTI-SCALE DIFFERENTIAL SKIPS (|F_T1 - F_T2|):")
    print(f"  Diff 1 Skip:             {list(diff_1.shape)}")
    print(f"  Diff 2 Skip:             {list(diff_2.shape)}")
    print(f"  Diff 3 Skip:             {list(diff_3.shape)}")
    print(f"  Diff 4 Skip:             {list(diff_4.shape)}")
    print(f"  Bottleneck Diff:         {list(diff_bot.shape)}")
    print("-" * 50)

    # Decoder trace
    c = model.base_channels
    d4 = model.decoder.up4(diff_bot, diff_4)
    d3 = model.decoder.up3(d4, diff_3)
    d2 = model.decoder.up2(d3, diff_2)
    d1 = model.decoder.up1(d2, diff_1)
    logits = model.decoder.out_conv(d1)

    print("U-NET DECODER STAGES:")
    print(f"  Decoder Stage 4 (Up + Diff 4 Concat): {list(d4.shape)}")
    print(f"  Decoder Stage 3 (Up + Diff 3 Concat): {list(d3.shape)}")
    print(f"  Decoder Stage 2 (Up + Diff 2 Concat): {list(d2.shape)}")
    print(f"  Decoder Stage 1 (Up + Diff 1 Concat): {list(d1.shape)}")
    print(f"  Final Logits (1x1 Conv):              {list(logits.shape)}")
    print("=" * 70)

    return {
        'input': list(t1.shape),
        'enc_level1': list(f1_1.shape),
        'enc_level2': list(f1_2.shape),
        'enc_level3': list(f1_3.shape),
        'enc_level4': list(f1_4.shape),
        'bottleneck': list(f1_bot.shape),
        'dec_stage4': list(d4.shape),
        'dec_stage3': list(d3.shape),
        'dec_stage2': list(d2.shape),
        'dec_stage1': list(d1.shape),
        'logits': list(logits.shape)
    }


def benchmark_speed(model: SiameseUNet, device: torch.device, num_warmup: int = 5, num_runs: int = 20):
    """Measure average forward pass latency and FPS on specified device."""
    model = model.to(device)
    model.eval()

    b, c, h, w = 1, 3, 256, 256
    t1 = torch.randn(b, c, h, w, device=device)
    t2 = torch.randn(b, c, h, w, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(t1, t2)
            if device.type == 'mps':
                torch.mps.synchronize()

    # Timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(t1, t2)
            if device.type == 'mps':
                torch.mps.synchronize()
    total_time = time.perf_counter() - start

    avg_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    return round(avg_ms, 2), round(fps, 2)


def render_architecture_diagram(save_path: str):
    """Draw a clean visual architecture diagram and save as image."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.axis('off')

    # Color scheme
    c_input = '#e3f2fd'
    c_encoder = '#bbdefb'
    c_diff = '#ffe0b2'
    c_decoder = '#c8e6c9'
    c_output = '#ffcdd2'
    border_color = '#37474f'

    # Title
    ax.text(0.5, 0.96, "Siamese U-Net Architecture for Remote Sensing Building Change Detection",
            ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Box drawer helper
    def draw_box(x, y, w, h, label, subtext, color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            facecolor=color, edgecolor=border_color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center', fontsize=9.5, fontweight='bold')
        ax.text(x + w/2, y + h*0.28, subtext, ha='center', va='center', fontsize=8, color='#455a64')

    # 1. Inputs
    draw_box(0.04, 0.78, 0.18, 0.10, "Image T1", "[B, 3, 256, 256]", c_input)
    draw_box(0.04, 0.12, 0.18, 0.10, "Image T2", "[B, 3, 256, 256]", c_input)

    # 2. Shared Encoders
    draw_box(0.28, 0.78, 0.20, 0.10, "Encoder Stream T1", "Shared Weights W_E", c_encoder)
    draw_box(0.28, 0.12, 0.20, 0.10, "Encoder Stream T2", "Shared Weights W_E", c_encoder)

    # 3. Differential Skips
    draw_box(0.53, 0.78, 0.18, 0.10, "Diff 1 Skip", "|F1_1 - F2_1| [16, 256, 256]", c_diff)
    draw_box(0.53, 0.62, 0.18, 0.10, "Diff 2 Skip", "|F1_2 - F2_2| [32, 128, 128]", c_diff)
    draw_box(0.53, 0.46, 0.18, 0.10, "Diff 3 Skip", "|F1_3 - F2_3| [64, 64, 64]", c_diff)
    draw_box(0.53, 0.30, 0.18, 0.10, "Diff 4 Skip", "|F1_4 - F2_4| [128, 32, 32]", c_diff)
    draw_box(0.53, 0.12, 0.18, 0.10, "Bottleneck Diff", "|F1_b - F2_b| [256, 16, 16]", c_diff)

    # 4. Decoder
    draw_box(0.76, 0.45, 0.20, 0.20, "U-Net Decoder", "Upsample + Skip Concat\nProgressive Spatial Recovery", c_decoder)

    # 5. Output
    draw_box(0.76, 0.12, 0.20, 0.12, "Change Logits Head", "1x1 Conv\n[B, 1, 256, 256]", c_output)

    # Arrows
    arrow_style = dict(arrowstyle="->", color='#37474f', lw=1.8, mutation_scale=15)
    ax.annotate("", xy=(0.28, 0.83), xytext=(0.22, 0.83), arrowprops=arrow_style)
    ax.annotate("", xy=(0.28, 0.17), xytext=(0.22, 0.17), arrowprops=arrow_style)

    ax.annotate("", xy=(0.53, 0.83), xytext=(0.48, 0.83), arrowprops=arrow_style)
    ax.annotate("", xy=(0.53, 0.17), xytext=(0.48, 0.17), arrowprops=arrow_style)

    ax.annotate("", xy=(0.76, 0.55), xytext=(0.71, 0.55), arrowprops=arrow_style)
    ax.annotate("", xy=(0.86, 0.24), xytext=(0.86, 0.45), arrowprops=arrow_style)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved architecture diagram to: {save_path}")


def main():
    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Profile Model Configurations
    print("PROFILING ARCHITECTURES:")
    model_light = SiameseUNet(in_channels=3, out_channels=1, base_channels=16)
    params_light = model_light.count_parameters()
    size_light_mb = model_light.model_size_mb()

    model_std = SiameseUNet(in_channels=3, out_channels=1, base_channels=32)
    params_std = model_std.count_parameters()
    size_std_mb = model_std.model_size_mb()

    print(f"  Lightweight Config (base_channels=16): {params_light:,} parameters ({size_light_mb} MB in float32)")
    print(f"  Standard Config    (base_channels=32): {params_std:,} parameters ({size_std_mb} MB in float32)")

    # 2. Trace Tensor Shapes
    shapes_info = profile_tensor_shapes(model_light, img_size=256)

    # 3. Hardware Speed Benchmarks
    print("\nHARDWARE LATENCY BENCHMARKS (Lightweight Config, 256x256 pair):")
    cpu_latency, cpu_fps = benchmark_speed(model_light, torch.device('cpu'))
    print(f"  CPU Latency: {cpu_latency} ms/pair ({cpu_fps} FPS)")

    mps_available = torch.backends.mps.is_available()
    mps_latency, mps_fps = None, None
    if mps_available:
        mps_latency, mps_fps = benchmark_speed(model_light, torch.device('mps'))
        print(f"  Apple Silicon MPS Latency: {mps_latency} ms/pair ({mps_fps} FPS)")
        speedup = round(cpu_latency / max(mps_latency, 1e-4), 2)
        print(f"  MPS Speedup: {speedup}x over CPU")
    else:
        print("  Apple Silicon MPS: Not Available")

    # 4. Generate Architecture Diagram
    diagram_path = os.path.join(outputs_dir, 'architecture_diagram.png')
    render_architecture_diagram(diagram_path)

    # 5. Export JSON summary
    summary = {
        'lightweight_model': {
            'base_channels': 16,
            'trainable_parameters': params_light,
            'model_size_mb': size_light_mb
        },
        'standard_model': {
            'base_channels': 32,
            'trainable_parameters': params_std,
            'model_size_mb': size_std_mb
        },
        'tensor_shapes': shapes_info,
        'benchmark': {
            'cpu_latency_ms': cpu_latency,
            'cpu_fps': cpu_fps,
            'mps_available': mps_available,
            'mps_latency_ms': mps_latency,
            'mps_fps': mps_fps
        }
    }

    summary_path = os.path.join(metrics_dir, 'model_benchmark_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model benchmark summary to: {summary_path}")


if __name__ == '__main__':
    main()
