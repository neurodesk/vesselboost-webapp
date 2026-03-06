#!/usr/bin/env python3
"""
Convert VesselBoost's PyTorch 3D UNet model to ONNX format.

Usage (from project root):
    python scripts/convert_model.py --checkpoint /path/to/model.pth
    python scripts/convert_model.py --checkpoint /path/to/model.pth --quantize
    python scripts/convert_model.py --checkpoint /path/to/model.pth --output web/models/vesselboost.onnx

Requires:
    pip install torch onnx onnxruntime

Input:  PyTorch .pth checkpoint (VesselBoost 3D UNet)
Output: ONNX model in web/models/
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn


# ==================== 3D UNet Architecture ====================

class DoubleConv3D(nn.Module):
    """Two consecutive 3D conv-batchnorm-relu blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D UNet for binary vessel segmentation.

    Architecture: 4 encoder/decoder levels with base 16 filters.
    Encoder: 16 -> 32 -> 64 -> 128
    Bottleneck: 256
    Decoder: 128 -> 64 -> 32 -> 16
    Output: 1 channel (sigmoid for binary segmentation)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, f)
        self.enc2 = DoubleConv3D(f, f * 2)
        self.enc3 = DoubleConv3D(f * 2, f * 4)
        self.enc4 = DoubleConv3D(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = DoubleConv3D(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv3D(f * 16, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv3D(f * 2, f)

        # Output
        self.out_conv = nn.Conv3d(f, out_channels, 1)

        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ==================== Configuration ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "web", "models")


# ==================== Conversion ====================

def load_model(checkpoint_path, in_channels=1, out_channels=1, base_filters=16):
    """Load a VesselBoost 3D UNet model from a PyTorch checkpoint."""
    model = UNet3D(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
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
    """Export a PyTorch model to ONNX format."""
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


def verify_model(onnx_path, pytorch_model=None, patch_size=64):
    """Verify an ONNX model runs correctly."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, patch_size, patch_size, patch_size).astype(np.float32)
    result = session.run(None, {"input": dummy})
    output = result[0]
    print(f"  Verified: output shape {output.shape}, "
          f"range [{output.min():.3f}, {output.max():.3f}]")

    expected_shape = (1, 1, patch_size, patch_size, patch_size)
    if output.shape != expected_shape:
        print(f"  WARNING: expected shape {expected_shape}, got {output.shape}")
        return False

    if pytorch_model is not None:
        with torch.no_grad():
            pt_output = pytorch_model(torch.from_numpy(dummy)).numpy()
        diff = np.abs(pt_output - output).mean()
        print(f"  Mean absolute difference vs PyTorch: {diff:.6f}")
        if diff > 0.01:
            print("  WARNING: large difference between PyTorch and ONNX outputs")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert VesselBoost model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--output", default=None, help="Output ONNX path (default: web/models/vesselboost.onnx)")
    parser.add_argument("--quantize", action="store_true", help="Apply UINT8 dynamic quantization")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size (default: 64)")
    parser.add_argument("--base-filters", type=int, default=16, help="Base filter count (default: 16)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, "vesselboost.onnx")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_path}")
    print(f"Quantize: {args.quantize}")
    print(f"Architecture: 3D UNet, base_filters={args.base_filters}, patch_size={args.patch_size}")

    print("\nLoading PyTorch model...")
    model = load_model(args.checkpoint, base_filters=args.base_filters)

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
