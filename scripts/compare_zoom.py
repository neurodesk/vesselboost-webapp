#!/usr/bin/env python3
"""Compare nearest-neighbor zoom between scipy and JS implementations."""

import numpy as np
import nibabel as nib
import scipy.ndimage as scind
import sys


def js_zoom_floor(data, target_size):
    """Old JS implementation using Math.floor."""
    nx, ny, nz = data.shape
    nnx, nny, nnz = target_size
    result = np.zeros(target_size, dtype=data.dtype)
    rx, ry, rz = nx / nnx, ny / nny, nz / nnz
    for z in range(nnz):
        sz = min(int(np.floor(z * rz)), nz - 1)
        for y in range(nny):
            sy = min(int(np.floor(y * ry)), ny - 1)
            for x in range(nnx):
                sx = min(int(np.floor(x * rx)), nx - 1)
                result[x, y, z] = data[sx, sy, sz]
    return result


def js_math_round(x):
    """Match JavaScript Math.round() - round half up (not banker's rounding)."""
    return int(np.floor(x + 0.5))


def js_zoom_round(data, target_size):
    """New JS implementation using half-pixel center mapping (matching scipy)."""
    nx, ny, nz = data.shape
    nnx, nny, nnz = target_size
    result = np.zeros(target_size, dtype=data.dtype)
    for z in range(nnz):
        sz = min(max(0, int(np.floor((z + 0.5) * nz / nnz))), nz - 1)
        for y in range(nny):
            sy = min(max(0, int(np.floor((y + 0.5) * ny / nny))), ny - 1)
            for x in range(nnx):
                sx = min(max(0, int(np.floor((x + 0.5) * nx / nnx))), nx - 1)
                result[x, y, z] = data[sx, sy, sz]
    return result


def main():
    nifti_path = sys.argv[1]
    img = nib.load(nifti_path)
    data = img.get_fdata()
    print(f"Original shape: {data.shape}")

    # Calculate target size
    ps = 64
    target = tuple(
        int(np.ceil(d / ps)) * ps if d > ps and d % ps != 0
        else ps if d < ps
        else d
        for d in data.shape
    )
    print(f"Target shape: {target}")

    zoom_factors = tuple(t / o for t, o in zip(target, data.shape))
    print(f"Zoom factors: {zoom_factors}")

    # scipy zoom
    scipy_result = scind.zoom(data, zoom_factors, order=0, mode='nearest')
    print(f"\nscipy zoom shape: {scipy_result.shape}")

    # Old JS (floor)
    floor_result = js_zoom_floor(data, target)
    print(f"JS floor zoom shape: {floor_result.shape}")

    # New JS (round)
    round_result = js_zoom_round(data, target)
    print(f"JS round zoom shape: {round_result.shape}")

    # Compare
    # Note: scipy returns C-order, our JS emulation uses numpy so both are comparable
    diff_floor = np.sum(scipy_result != floor_result)
    diff_round = np.sum(scipy_result != round_result)
    total = np.prod(target)

    print(f"\nDifferences vs scipy:")
    print(f"  floor: {diff_floor}/{total} voxels differ ({100*diff_floor/total:.2f}%)")
    print(f"  round: {diff_round}/{total} voxels differ ({100*diff_round/total:.2f}%)")

    # Check per-axis mapping differences
    for axis_name, orig_size, tgt_size in zip(['X', 'Y', 'Z'], data.shape, target):
        if orig_size == tgt_size:
            print(f"\n  {axis_name}: no zoom needed ({orig_size})")
            continue
        ratio = orig_size / tgt_size
        diffs = 0
        for i in range(tgt_size):
            scipy_src = min(max(0, int(np.floor(i / (tgt_size / orig_size) + 0.5))), orig_size - 1)
            floor_src = min(int(np.floor(i * ratio)), orig_size - 1)
            round_src = min(max(0, round(i * ratio)), orig_size - 1)
            if floor_src != scipy_src:
                diffs += 1
        print(f"\n  {axis_name}: {orig_size}->{tgt_size}, floor differs in {diffs}/{tgt_size} coords")
        diffs = 0
        for i in range(tgt_size):
            scipy_src = min(max(0, int(np.floor(i / (tgt_size / orig_size) + 0.5))), orig_size - 1)
            round_src = min(max(0, round(i * ratio)), orig_size - 1)
            if round_src != scipy_src:
                diffs += 1
        print(f"  {axis_name}: {orig_size}->{tgt_size}, round differs in {diffs}/{tgt_size} coords")


if __name__ == "__main__":
    main()
