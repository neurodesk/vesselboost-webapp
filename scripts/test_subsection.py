#!/usr/bin/env python3
"""Test how the ONNX model handles thin subsections (matching JS webapp behavior)."""

import numpy as np
import nibabel as nib
import onnxruntime as ort
import scipy.ndimage as scind
import sys


def standardiser(x):
    mean_x, std_x = np.mean(x), np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    return (x - mean_x) / std_x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_subsection(data, session, input_name, output_name, start_z, end_z, label):
    """Test inference on a subsection of slices."""
    subsection = data[:, :, start_z:end_z].copy()
    nz_sub = end_z - start_z
    nx, ny = subsection.shape[:2]
    print(f"\n=== {label}: slices {start_z}-{end_z} ({nz_sub} slices) ===")
    print(f"  Subsection shape: {subsection.shape}")

    # Approach A: Stretch to patch multiples (matching old Python _resize_image)
    ps = 64
    target_x = int(np.ceil(nx / ps)) * ps if nx > ps else ps
    target_y = int(np.ceil(ny / ps)) * ps if ny > ps else ps
    target_z = int(np.ceil(nz_sub / ps)) * ps if nz_sub > ps else ps
    target_size = (target_x, target_y, target_z)
    zoom_factors = tuple(t / o for t, o in zip(target_size, subsection.shape))
    resized = scind.zoom(subsection, zoom_factors, order=0, mode='nearest')
    standardized = standardiser(resized)
    print(f"  Stretched approach: {subsection.shape} -> {resized.shape}, zoom={[f'{z:.2f}' for z in zoom_factors]}")

    # Run patches
    gi, gj, gk = target_x // ps, target_y // ps, target_z // ps
    total_vessel = 0
    for i in range(gi):
        for j in range(gj):
            for k in range(gk):
                patch = standardized[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps].astype(np.float32)
                result = session.run([output_name], {input_name: patch[np.newaxis, np.newaxis]})[0]
                prob = sigmoid(result[0, 0])
                total_vessel += np.sum(prob >= 0.1)
    print(f"  Stretched result: {total_vessel} vessel voxels (in padded space)")

    # Resize back and count
    # Build prediction map
    prediction = np.zeros(target_size, dtype=np.float32)
    for i in range(gi):
        for j in range(gj):
            for k in range(gk):
                patch = standardized[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps].astype(np.float32)
                result = session.run([output_name], {input_name: patch[np.newaxis, np.newaxis]})[0]
                prediction[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps] = sigmoid(result[0, 0])
    pred_back = scind.zoom(prediction, tuple(o / t for o, t in zip(subsection.shape, target_size)), order=0, mode='nearest')
    binary_stretch = (pred_back >= 0.1).astype(int).sum()
    print(f"  Stretched result (original space): {binary_stretch} vessel voxels")

    # Approach B: Zero-pad (don't stretch thin dims)
    pad_x = target_x - nx if nx < ps else (target_x - nx)
    pad_y = target_y - ny if ny < ps else (target_y - ny)
    pad_z = target_z - nz_sub if nz_sub < ps else (target_z - nz_sub)

    # Normalize on original data first, then pad with zeros (=mean after standardization)
    std_sub = standardiser(subsection)

    if nz_sub < ps:
        # Zero-pad z
        padded = np.zeros(target_size, dtype=std_sub.dtype)
        # For x,y that need zoom:
        if nx != target_x or ny != target_y:
            zoom_xy = (target_x / nx, target_y / ny, 1.0)
            zoomed = scind.zoom(std_sub, zoom_xy, order=0, mode='nearest')
            padded[:, :, :nz_sub] = zoomed
        else:
            padded[:nx, :ny, :nz_sub] = std_sub
        standardized_zp = padded
    else:
        standardized_zp = scind.zoom(std_sub, zoom_factors, order=0, mode='nearest')

    print(f"  Zero-pad approach: {std_sub.shape} -> {standardized_zp.shape}")

    total_vessel_zp = 0
    for i in range(gi):
        for j in range(gj):
            for k in range(gk):
                patch = standardized_zp[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps].astype(np.float32)
                result = session.run([output_name], {input_name: patch[np.newaxis, np.newaxis]})[0]
                prob = sigmoid(result[0, 0])
                total_vessel_zp += np.sum(prob >= 0.1)
    print(f"  Zero-pad result: {total_vessel_zp} vessel voxels (in padded space)")


def main():
    nifti_path = sys.argv[1]
    model_path = sys.argv[2]

    img = nib.load(nifti_path)
    data = img.get_fdata()
    print(f"Volume: {data.shape}, orientation: {nib.aff2axcodes(img.affine)}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Test various subsection sizes
    nz = data.shape[2]
    mid = nz // 2

    test_subsection(data, session, input_name, output_name, mid-7, mid+7, "14 slices (thin)")
    test_subsection(data, session, input_name, output_name, mid-32, mid+32, "64 slices (1 patch)")
    test_subsection(data, session, input_name, output_name, 0, nz, "Full volume")


if __name__ == "__main__":
    main()
