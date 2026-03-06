/// Get voxel index in a 3D volume (Fortran/column-major order).
#[inline]
pub fn idx3(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    x + y * nx + z * nx * ny
}

/// Compute mean and standard deviation of nonzero values.
#[allow(dead_code)]
pub fn nonzero_stats(data: &[f32]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for &v in data {
        if v != 0.0 {
            sum += v as f64;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 1.0);
    }
    let mean = sum / count as f64;
    let mut sum_sq = 0.0f64;
    for &v in data {
        if v != 0.0 {
            let d = v as f64 - mean;
            sum_sq += d * d;
        }
    }
    let std = (sum_sq / count as f64).sqrt();
    (mean as f32, if std > 0.0 { std as f32 } else { 1.0 })
}

/// Estimate noise standard deviation using median absolute deviation (MAD)
/// of the smallest 25% of nonzero voxels.
pub fn estimate_noise_std(data: &[f32]) -> f32 {
    let mut values: Vec<f32> = data.iter().copied().filter(|&v| v > 0.0).collect();
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use lowest quartile
    let n = values.len() / 4;
    if n < 2 {
        return 1.0;
    }
    let subset = &values[..n];

    let median = subset[n / 2];
    let mut abs_devs: Vec<f32> = subset.iter().map(|&v| (v - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = abs_devs[abs_devs.len() / 2];

    // MAD to std: sigma = 1.4826 * MAD
    let sigma = 1.4826 * mad;
    if sigma > 0.0 {
        sigma
    } else {
        1.0
    }
}

/// Downsample a 3D volume by an integer factor using mean pooling.
pub fn downsample_volume(
    data: &[f32],
    dims: [usize; 3],
    factor: usize,
) -> (Vec<f32>, [usize; 3]) {
    let [nx, ny, nz] = dims;
    let snx = (nx + factor - 1) / factor;
    let sny = (ny + factor - 1) / factor;
    let snz = (nz + factor - 1) / factor;
    let mut result = vec![0.0f32; snx * sny * snz];

    for sz in 0..snz {
        for sy in 0..sny {
            for sx in 0..snx {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dz in 0..factor {
                    let z = sz * factor + dz;
                    if z >= nz {
                        continue;
                    }
                    for dy in 0..factor {
                        let y = sy * factor + dy;
                        if y >= ny {
                            continue;
                        }
                        for dx in 0..factor {
                            let x = sx * factor + dx;
                            if x >= nx {
                                continue;
                            }
                            sum += data[idx3(x, y, z, nx, ny)];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    result[idx3(sx, sy, sz, snx, sny)] = sum / count as f32;
                }
            }
        }
    }

    (result, [snx, sny, snz])
}

/// Upsample a 3D volume by an integer factor using nearest-neighbor.
pub fn upsample_volume(
    data: &[f32],
    dims: [usize; 3],
    target_dims: [usize; 3],
) -> Vec<f32> {
    let [snx, sny, snz] = dims;
    let [nx, ny, nz] = target_dims;
    let mut result = vec![0.0f32; nx * ny * nz];

    let sx = snx as f32 / nx as f32;
    let sy = sny as f32 / ny as f32;
    let sz = snz as f32 / nz as f32;

    for z in 0..nz {
        let iz = ((z as f32 * sz) as usize).min(snz - 1);
        for y in 0..ny {
            let iy = ((y as f32 * sy) as usize).min(sny - 1);
            for x in 0..nx {
                let ix = ((x as f32 * sx) as usize).min(snx - 1);
                result[idx3(x, y, z, nx, ny)] = data[idx3(ix, iy, iz, snx, sny)];
            }
        }
    }

    result
}
