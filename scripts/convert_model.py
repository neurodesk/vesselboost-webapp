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


# ==================== VesselBoost 3D UNet Architecture ====================
# Exact copy from https://github.com/KMarshallX/VesselBoost/blob/master/models/unet_3d.py
# State dict keys must match: EncB1.convb.convb.0.weight, DecB1.upsample.weight, etc.

class ConvBlock(nn.Module):
    """Two consecutive 3D conv-batchnorm-relu blocks."""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convb(x)


class EncBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool."""
    def __init__(self, in_chan, num_filter):
        super().__init__()
        self.convb = ConvBlock(in_chan, num_filter)
        self.maxp = nn.MaxPool3d(2)

    def forward(self, x):
        xx = self.convb(x)
        p = self.maxp(xx)
        return xx, p


class DecBlock(nn.Module):
    """Decoder block: ConvTranspose (same channels) + cat + ConvBlock."""
    def __init__(self, in_chan, feat_chan, num_filter):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_chan, in_chan, kernel_size=(2, 2, 2), stride=2)
        self.convb = ConvBlock(feat_chan, num_filter)

    def forward(self, x, cat_block):
        xx = self.upsample(x)
        cated_block = torch.cat([cat_block, xx], dim=1)
        out = self.convb(cated_block)
        return out


class Unet(nn.Module):
    """VesselBoost 3D UNet."""
    def __init__(self, in_chan, out_chan, filter_num):
        super().__init__()
        self.EncB1 = EncBlock(in_chan, filter_num)
        self.EncB2 = EncBlock(filter_num, filter_num * 2)
        self.EncB3 = EncBlock(filter_num * 2, filter_num * 4)
        self.EncB4 = EncBlock(filter_num * 4, filter_num * 8)

        self.bridge = ConvBlock(filter_num * 8, filter_num * 16)

        self.DecB1 = DecBlock(filter_num * 16, filter_num * 24, filter_num * 8)
        self.DecB2 = DecBlock(filter_num * 8, filter_num * 12, filter_num * 4)
        self.DecB3 = DecBlock(filter_num * 4, filter_num * 6, filter_num * 2)
        self.DecB4 = DecBlock(filter_num * 2, filter_num * 3, filter_num)

        self.out = nn.Conv3d(filter_num, out_chan, kernel_size=1)

    def forward(self, x):
        xx1, p1 = self.EncB1(x)
        xx2, p2 = self.EncB2(p1)
        xx3, p3 = self.EncB3(p2)
        xx4, p4 = self.EncB4(p3)

        p5 = self.bridge(p4)

        p6 = self.DecB1(p5, xx4)
        p7 = self.DecB2(p6, xx3)
        p8 = self.DecB3(p7, xx2)
        p9 = self.DecB4(p8, xx1)

        return self.out(p9)


# ==================== Configuration ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "web", "models")


# ==================== Conversion ====================

def load_model(checkpoint_path, in_chan=1, out_chan=1, filter_num=16):
    """Load a VesselBoost 3D UNet model from a PyTorch checkpoint."""
    model = Unet(in_chan=in_chan, out_chan=out_chan, filter_num=filter_num)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

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
    parser.add_argument("--filters", type=int, default=16, help="Base filter count (default: 16)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, "vesselboost.onnx")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_path}")
    print(f"Quantize: {args.quantize}")
    print(f"Architecture: VesselBoost 3D UNet, filters={args.filters}, patch_size={args.patch_size}")

    print("\nLoading PyTorch model...")
    model = load_model(args.checkpoint, filter_num=args.filters)

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
