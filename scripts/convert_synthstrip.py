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

Architecture source:
    https://github.com/freesurfer/freesurfer (mri_synthstrip)
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn


# ==================== SynthStrip 3D UNet Architecture ====================
# 7-level 3D UNet matching FreeSurfer's SynthStrip implementation.
# Input: [B, 1, D, H, W] normalized to [0,1]
# Output: [B, 1, D, H, W] signed distance transform


class ConvBlock(nn.Module):
    """Two consecutive 3D conv-batchnorm-leakyrelu blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        return x


class SynthStripUNet(nn.Module):
    """
    7-level 3D UNet for SynthStrip skull-stripping.

    Encoder: 7 levels with max-pooling (min input: 2^6 = 64)
    Decoder: 7 levels with trilinear upsampling + skip connections
    Output: signed distance transform (1 channel)
    """
    def __init__(self, in_channels=1, out_channels=1, nb_features=16, nb_levels=7, feat_mult=2):
        super().__init__()

        # Compute feature counts per level
        enc_features = [nb_features * (feat_mult ** i) for i in range(nb_levels)]
        dec_features = enc_features[::-1]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for nf in enc_features:
            self.encoders.append(ConvBlock(prev_channels, nf))
            self.pools.append(nn.MaxPool3d(2))
            prev_channels = nf

        # Bottleneck
        self.bottleneck = ConvBlock(enc_features[-1], enc_features[-1])

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_channels = enc_features[-1]
        for i, nf in enumerate(dec_features):
            self.upsamplers.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            self.decoders.append(ConvBlock(prev_channels + nf, nf))
            prev_channels = nf

        # Output
        self.output_conv = nn.Conv3d(dec_features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upsampler, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skip_connections)):
            x = upsampler(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.output_conv(x)


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
        feat_mult=2
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


def export_to_onnx(model, output_path, patch_size=96, opset_version=17):
    """Export model to ONNX format."""
    dummy_input = torch.randn(1, 1, patch_size, patch_size, patch_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
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
    parser.add_argument("--patch-size", type=int, default=96, help="Patch size for export (default: 96)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, "synthstrip.onnx")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_path}")
    print(f"Quantize: {args.quantize}")
    print(f"Architecture: SynthStrip 7-level 3D UNet, features=16, patch_size={args.patch_size}")

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
