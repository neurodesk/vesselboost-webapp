#!/usr/bin/env python3
"""PyTorch-only test (no ONNX to avoid OMP conflict)."""
import numpy as np
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(__file__))

import torch
from convert_synthstrip import load_model

PT_PATH = "/Users/uqsbollm/github-repos/vesselboost-webapp/.tmp_weights/synthstrip.1.pt"
model = load_model(PT_PATH)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Uniform inputs
print("\nUniform 96^3:")
for val in [0.0, 0.5, 1.0]:
    t = torch.full((1, 1, 96, 96, 96), val)
    with torch.no_grad():
        o = model(t).squeeze().numpy()
    print(f"  val={val}: [{o.min():.3f}, {o.max():.3f}] mean={o.mean():.3f}")

# Sphere test
print("\nSphere in 192^3:")
vol = np.zeros((192, 192, 192), dtype=np.float32)
for x in range(192):
    for y in range(192):
        for z in range(192):
            if ((x-96)**2 + (y-96)**2 + (z-96)**2) < 70**2:
                vol[x, y, z] = 0.5
t = torch.from_numpy(vol[np.newaxis, np.newaxis])
with torch.no_grad():
    sdt = model(t).squeeze().numpy()

print(f"  SDT range: [{sdt.min():.3f}, {sdt.max():.3f}]")
print(f"  center [96,96,96]: {sdt[96,96,96]:.3f}")
print(f"  boundary [26,96,96]: {sdt[26,96,96]:.3f}")
print(f"  outside [10,96,96]: {sdt[10,96,96]:.3f}")
print(f"  SDT < 0: {(sdt<0).mean()*100:.1f}%, SDT < 1: {(sdt<1).mean()*100:.1f}%")

print(f"\n  Along X at [_,96,96]:")
for x in range(0, 192, 8):
    dist = np.sqrt((x-96)**2)
    inside = "IN" if dist < 70 else "OUT"
    print(f"    x={x:3d} ({inside:3s}, d={dist:5.1f}): SDT={sdt[x,96,96]:7.3f}")

# Real brain 192^3 center crop
print("\nReal TOF center crop (192^3):")
import nibabel as nib
img = nib.load("/Users/uqsbollm/Downloads/testdata/tof_input.nii")
ras = nib.as_closest_canonical(img)
data = ras.get_fdata(dtype=np.float32)
vmin, p99 = data.min(), np.percentile(data, 99)
norm = np.clip((data - vmin) / (p99 - vmin + 1e-8), 0, 1).astype(np.float32)
cx, cy, cz = [d//2 for d in data.shape]
crop = np.zeros((192, 192, 192), dtype=np.float32)
c = norm[cx-96:cx+96, cy-96:cy+96, :]
crop[:c.shape[0], :c.shape[1], :c.shape[2]] = c

t = torch.from_numpy(crop[np.newaxis, np.newaxis])
with torch.no_grad():
    sdt_r = model(t).squeeze().numpy()
print(f"  SDT range: [{sdt_r.min():.3f}, {sdt_r.max():.3f}], mean={sdt_r.mean():.3f}")
print(f"  SDT < 0: {(sdt_r<0).mean()*100:.1f}%, SDT < 1: {(sdt_r<1).mean()*100:.1f}%")
mask = (sdt_r < 1).astype(np.uint8)
print(f"  mask (SDT<1): {mask.sum()} ({mask.mean()*100:.1f}%)")
