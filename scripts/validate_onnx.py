#!/usr/bin/env python3
"""
Validate ONNX model against Python VesselBoost pipeline.

Runs the exact same steps as the Python ImagePredictor to verify
the ONNX model produces correct segmentation, and dumps intermediate
values for comparison with the JS implementation.
"""

import numpy as np
import nibabel as nib
import onnxruntime as ort
import scipy.ndimage as scind
import sys
import os


def standardiser(x):
    """Z-score normalization over ALL voxels (matching Python VesselBoost)."""
    mean_x, std_x = np.mean(x), np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    return (x - mean_x) / std_x


def calculate_patch_dimensions(original_size, patch_size=64):
    """Pad dimensions to nearest multiple of patch_size."""
    new_dims = []
    for dim in original_size:
        if dim > patch_size and dim % patch_size != 0:
            new_dim = int(np.ceil(dim / patch_size)) * patch_size
        elif dim < patch_size:
            new_dim = patch_size
        else:
            new_dim = dim
        new_dims.append(new_dim)
    return tuple(new_dims)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_onnx.py <nifti_path> <onnx_model_path>")
        sys.exit(1)

    nifti_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load NIfTI
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    print(f"=== NIfTI Info ===")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Voxel size: {header.get_zooms()[:3]}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Orientation: {nib.aff2axcodes(affine)}")
    print(f"Affine diagonal signs: {np.sign(np.diag(affine)[:3])}")
    print()

    # Take a subsection (matching JS: slices 65-79 from the screenshot)
    # But first, let's check what the full volume looks like
    original_size = data.shape
    print(f"=== Full volume ===")
    print(f"Original size: {original_size}")

    # Pad to patch multiples (matching Python ImagePredictor)
    target_size = calculate_patch_dimensions(original_size)
    print(f"Padded size: {target_size}")

    # Resize using nearest-neighbor (matching Python)
    zoom_factors = tuple(t / o for t, o in zip(target_size, original_size))
    print(f"Zoom factors: {zoom_factors}")
    resized = scind.zoom(data, zoom_factors, order=0, mode='nearest')
    print(f"Resized shape: {resized.shape}")
    print(f"Resized range: [{resized.min():.2f}, {resized.max():.2f}]")
    print()

    # Standardize
    standardized = standardiser(resized)
    print(f"=== After standardization ===")
    print(f"Range: [{standardized.min():.4f}, {standardized.max():.4f}]")
    print(f"Mean: {standardized.mean():.6f}")
    print(f"Std: {standardized.std():.6f}")
    print()

    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"=== ONNX Model ===")
    print(f"Input: {input_name}, shape={session.get_inputs()[0].shape}")
    print(f"Output: {output_name}, shape={session.get_outputs()[0].shape}")
    print()

    # Run inference on patches (no overlap, matching Python default)
    ps = 64
    ni, nj, nk = target_size
    gi, gj, gk = ni // ps, nj // ps, nk // ps
    print(f"=== Patches ===")
    print(f"Patch grid: ({gi}, {gj}, {gk})")
    print(f"Total patches: {gi * gj * gk}")
    print()

    # Process each patch
    total_above_thresh = 0
    patch_results = []
    prediction_map = np.zeros(target_size, dtype=np.float32)

    for i in range(gi):
        for j in range(gj):
            for k in range(gk):
                patch = standardized[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps].copy()
                patch_tensor = patch.astype(np.float32)[np.newaxis, np.newaxis, :, :, :]

                result = session.run([output_name], {input_name: patch_tensor})
                output = result[0][0, 0, :, :, :]

                prob = sigmoid(output)
                n_above = int(np.sum(prob >= 0.1))
                total_above_thresh += n_above

                patch_results.append({
                    'pos': (i, j, k),
                    'input_range': (float(patch.min()), float(patch.max())),
                    'input_mean': float(patch.mean()),
                    'output_range': (float(output.min()), float(output.max())),
                    'prob_range': (float(prob.min()), float(prob.max())),
                    'prob_mean': float(prob.mean()),
                    'n_above_0.1': n_above,
                })

                prediction_map[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps] = prob

    print(f"=== Patch-level results (showing first 20 + any with vessels) ===")
    for idx, r in enumerate(patch_results):
        if idx < 20 or r['n_above_0.1'] > 0:
            print(f"  Patch {r['pos']}: "
                  f"in=[{r['input_range'][0]:.3f},{r['input_range'][1]:.3f}] mean={r['input_mean']:.3f}, "
                  f"logit=[{r['output_range'][0]:.3f},{r['output_range'][1]:.3f}], "
                  f"prob=[{r['prob_range'][0]:.4f},{r['prob_range'][1]:.4f}] mean={r['prob_mean']:.4f}, "
                  f"n>0.1={r['n_above_0.1']}")
    print()

    # Resize back to original
    prediction_map = scind.zoom(prediction_map, tuple(o / t for o, t in zip(original_size, target_size)), order=0, mode='nearest')

    binary = (prediction_map >= 0.1).astype(np.int32)
    print(f"=== Final result ===")
    print(f"Total voxels >= 0.1 threshold: {total_above_thresh}")
    print(f"Binary mask sum: {binary.sum()}")
    print(f"Binary mask shape: {binary.shape}")

    # Save output for visual inspection
    out_path = os.path.splitext(nifti_path)[0] + "_python_seg.nii"
    if out_path.endswith('.nii_python_seg.nii'):
        out_path = nifti_path.replace('.nii', '_python_seg.nii')
    out_img = nib.Nifti1Image(binary.astype(np.uint8), affine, header)
    nib.save(out_img, out_path)
    print(f"Saved Python segmentation to: {out_path}")


if __name__ == "__main__":
    main()
