#!/usr/bin/env python3
"""Save Python segmentation for a thin subsection to compare with JS output."""

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


def main():
    nifti_path = sys.argv[1]
    model_path = sys.argv[2]

    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Extract subsection matching JS: slices 65-79
    start_z, end_z = 65, 79
    subsection = data[:, :, start_z:end_z].copy()
    original_sub_shape = subsection.shape
    print(f"Subsection: {original_sub_shape}")

    # Match Python pipeline: pad to multiples of 64
    ps = 64
    nx, ny, nz = original_sub_shape
    tnx = int(np.ceil(nx / ps)) * ps if nx > ps else ps
    tny = int(np.ceil(ny / ps)) * ps if ny > ps else ps
    tnz = int(np.ceil(nz / ps)) * ps if nz > ps else ps
    target = (tnx, tny, tnz)
    zoom_factors = tuple(t / o for t, o in zip(target, original_sub_shape))
    resized = scind.zoom(subsection, zoom_factors, order=0, mode='nearest')
    standardized = standardiser(resized)
    print(f"Resized: {resized.shape}, zoom={[f'{z:.2f}' for z in zoom_factors]}")

    # Inference
    gi, gj, gk = tnx // ps, tny // ps, tnz // ps
    prediction = np.zeros(target, dtype=np.float32)
    for i in range(gi):
        for j in range(gj):
            for k in range(gk):
                patch = standardized[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps].astype(np.float32)
                result = session.run([output_name], {input_name: patch[np.newaxis, np.newaxis]})[0]
                prediction[i*ps:(i+1)*ps, j*ps:(j+1)*ps, k*ps:(k+1)*ps] = sigmoid(result[0, 0])

    # Resize back
    pred_back = scind.zoom(prediction, tuple(o / t for o, t in zip(original_sub_shape, target)), order=0, mode='nearest')
    binary = (pred_back >= 0.1).astype(np.uint8)
    print(f"Vessel voxels: {binary.sum()}")

    # Embed into full volume
    full_seg = np.zeros(data.shape, dtype=np.uint8)
    full_seg[:, :, start_z:end_z] = binary

    out_path = nifti_path.replace('.nii', '_sub14_seg.nii')
    out_img = nib.Nifti1Image(full_seg, affine, header)
    nib.save(out_img, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
