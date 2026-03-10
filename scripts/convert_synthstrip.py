#!/usr/bin/env python3
"""
Convert SynthStrip's PyTorch 3D UNet model to ONNX format.

SynthStrip is a skull-stripping tool from FreeSurfer that outputs a signed
distance transform (SDT). Threshold at 0 to get the brain mask.

Usage (from project root):
    python scripts/convert_synthstrip.py --checkpoint synthstrip.1.pt
    python scripts/convert_synthstrip.py --checkpoint synthstrip.1.pt --quantize
    python scripts/convert_synthstrip.py --checkpoint synthstrip.1.pt --output web/models/synthstrip.onnx

Requires:
    pip install torch onnx onnxruntime

Reference:
    Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M.
    SynthStrip: Skull-Stripping for Any Brain Image. NeuroImage. 2022;260:119474.

Architecture:
    VoxelMorph-style 3D UNet (no BatchNorm). 6 encoder levels with features
    capped at max_features=64, 6 decoder levels with skip-size-targeted
    upsampling, plus "remaining" output layers.
    State dict keys: encoder.{0-5}.{0,1}.conv, decoder.{0-5}.{0,1}.conv,
    remaining.{0,1,2}.conv.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== SynthStrip Architecture (VoxelMorph UNet) ====================
# Matches the checkpoint state dict keys exactly:
#   encoder.{level}.{0,1}.conv.{weight,bias}
#   decoder.{level}.{0,1}.conv.{weight,bias}
#   remaining.{0,1,2}.conv.{weight,bias}


class ConvLayer(nn.Module):
    """Single 3D conv + LeakyReLU (no BatchNorm)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.conv(x))


class ConvLayerNoActivation(nn.Module):
    """Single 3D conv without activation (for final output)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class SynthStripUNet(nn.Module):
    """
    VoxelMorph-style 3D UNet for SynthStrip skull-stripping.

    Architecture (from checkpoint weights):
      - encoder: 6 levels, features [16, 32, 64, 64, 64, 64], 2 convs each + maxpool
      - decoder.0: bottleneck (64ch in, no skip concat), 2 convs
      - decoder.1-5: upsample-to-skip-size + skip concat + 2 convs
      - remaining: upsample-to-skip-size + skip concat (48ch) + 3 convs -> 1ch SDT output

    Input:  [B, 1, D, H, W] normalized to [0, 1]
    Output: [B, 1, D, H, W] signed distance transform
    """
    def __init__(self, in_channels=1, out_channels=1, nb_features=16,
                 nb_levels=7, feat_mult=2, max_features=64):
        super().__init__()

        # Encoder feature counts: [16, 32, 64, 64, 64, 64] (6 levels, capped at 64)
        num_enc_levels = nb_levels - 1  # 6
        enc_features = []
        for i in range(num_enc_levels):
            nf = min(nb_features * (feat_mult ** i), max_features)
            enc_features.append(nf)

        # Encoder: 6 levels, each with 2 conv layers + maxpool
        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for nf in enc_features:
            level = nn.ModuleList([
                ConvLayer(prev_ch, nf),
                ConvLayer(nf, nf),
            ])
            self.encoder.append(level)
            prev_ch = nf

        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(num_enc_levels)])

        # Decoder: 6 levels
        self.decoder = nn.ModuleList()

        # Level 0: bottleneck - takes encoder.5 pooled output (64ch), no skip concat
        self.decoder.append(nn.ModuleList([
            ConvLayer(enc_features[-1], enc_features[-1]),
            ConvLayer(enc_features[-1], enc_features[-1]),
        ]))

        # Levels 1-5: upsample-to-skip-size + skip concat
        # decoder.1: up(64) + skip_enc5(64) = 128 -> 64
        # decoder.2: up(64) + skip_enc4(64) = 128 -> 64
        # decoder.3: up(64) + skip_enc3(64) = 128 -> 64
        # decoder.4: up(64) + skip_enc2(64) = 128 -> 64
        # decoder.5: up(64) + skip_enc1(32) = 96 -> 32
        prev_dec = enc_features[-1]  # 64
        for i in range(1, num_enc_levels):
            skip_ch = enc_features[num_enc_levels - i]
            in_ch = prev_dec + skip_ch
            out_ch = skip_ch if i == num_enc_levels - 1 else enc_features[-1]
            self.decoder.append(nn.ModuleList([
                ConvLayer(in_ch, out_ch),
                ConvLayer(out_ch, out_ch),
            ]))
            prev_dec = out_ch

        # Remaining: upsample decoder.5 (32ch) + skip_enc0 (16ch) = 48ch -> 16 -> 16 -> 1
        last_dec_ch = 32
        skip0_ch = enc_features[0]  # 16
        remaining_in = last_dec_ch + skip0_ch  # 48
        self.remaining = nn.ModuleList([
            ConvLayer(remaining_in, nb_features),
            ConvLayer(nb_features, nb_features),
            ConvLayerNoActivation(nb_features, out_channels),
        ])

    def forward(self, x):
        # Encoder
        skips = []
        for enc_level, pool in zip(self.encoder, self.pools):
            for conv in enc_level:
                x = conv(x)
            skips.append(x)
            x = pool(x)

        # Decoder level 0: bottleneck (no skip concat)
        for conv in self.decoder[0]:
            x = conv(x)

        # Decoder levels 1-5: upsample to skip spatial size + concat
        for i in range(1, len(self.decoder)):
            skip = skips[len(skips) - i]
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            for conv in self.decoder[i]:
                x = conv(x)

        # Remaining: upsample to skip0 size + concat + 3 convs
        skip0 = skips[0]
        x = F.interpolate(x, size=skip0.shape[2:], mode='nearest')
        x = torch.cat([x, skip0], dim=1)
        for conv in self.remaining:
            x = conv(x)

        return x


# ==================== Configuration ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "web", "models")


# ==================== Conversion ====================

def load_model(checkpoint_path):
    """Load a SynthStrip model from a PyTorch checkpoint."""
    model = SynthStripUNet(
        in_channels=1,
        out_channels=1,
        nb_features=16,
        nb_levels=7,
        feat_mult=2,
        max_features=64
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(model, output_path, patch_size=64, opset_version=17):
    """Export model to ONNX format.

    Uses 64^3 dummy input for export (smaller memory footprint) with dynamic
    spatial axes, so the model accepts any input size at runtime (e.g. 96^3).
    """
    dummy_input = torch.randn(1, 1, patch_size, patch_size, patch_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
        },
        dynamo=False,
    )
    print(f"  Exported ONNX: {output_path}")


def quantize_model(input_path, output_path):
    """Apply UINT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"  Quantized: {output_path}")


def verify_model(onnx_path, pytorch_model=None, patch_size=96):
    """Verify an ONNX model runs correctly and outputs SDT."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, patch_size, patch_size, patch_size).astype(np.float32)
    # Clamp to [0,1] to match expected input range
    dummy = np.clip(dummy * 0.2 + 0.5, 0, 1).astype(np.float32)

    result = session.run(None, {"input": dummy})
    output = result[0]
    print(f"  Verified: output shape {output.shape}, "
          f"range [{output.min():.3f}, {output.max():.3f}]")

    expected_shape = (1, 1, patch_size, patch_size, patch_size)
    if output.shape != expected_shape:
        print(f"  WARNING: expected shape {expected_shape}, got {output.shape}")
        return False

    # SDT should have both positive and negative values
    has_positive = output.max() > 0
    has_negative = output.min() < 0
    if has_positive and has_negative:
        print("  SDT output contains both positive and negative values (expected)")
    else:
        print("  WARNING: SDT output may not contain expected signed distance values")

    if pytorch_model is not None:
        with torch.no_grad():
            pt_output = pytorch_model(torch.from_numpy(dummy)).numpy()
        diff = np.abs(pt_output - output).mean()
        print(f"  Mean absolute difference vs PyTorch: {diff:.6f}")
        if diff > 0.01:
            print("  WARNING: large difference between PyTorch and ONNX outputs")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SynthStrip model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to SynthStrip .pt checkpoint (e.g. synthstrip.1.pt)")
    parser.add_argument("--output", default=None, help="Output ONNX path (default: web/models/synthstrip.onnx)")
    parser.add_argument("--quantize", action="store_true", help="Apply UINT8 dynamic quantization")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size for ONNX export dummy input (default: 64). Model accepts any size at runtime.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, "synthstrip.onnx")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_path}")
    print(f"Quantize: {args.quantize}")
    print(f"Architecture: SynthStrip VoxelMorph 3D UNet, features=16, max=64, patch_size={args.patch_size}")

    print("\nLoading PyTorch model...")
    model = load_model(args.checkpoint)

    if args.quantize:
        fp32_path = output_path.replace(".onnx", "-fp32.onnx")
        print("Exporting to ONNX (FP32)...")
        export_to_onnx(model, fp32_path, patch_size=args.patch_size)

        print("Quantizing to UINT8...")
        quantize_model(fp32_path, output_path)

        os.remove(fp32_path)
        data_file = fp32_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
    else:
        print("Exporting to ONNX (FP32)...")
        export_to_onnx(model, output_path, patch_size=args.patch_size)

    print("Verifying model...")
    ok = verify_model(output_path, model, patch_size=args.patch_size)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSize: {size_mb:.1f} MB")
    if ok:
        print("SUCCESS")
    else:
        print("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
