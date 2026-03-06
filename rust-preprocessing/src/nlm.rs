/// Non-local means denoising for 3D MRI volumes.
///
/// Optimized blockwise implementation:
/// 1. For each voxel, search within a local window for similar patches
/// 2. Weight neighbors by patch similarity (Gaussian kernel)
/// 3. Weighted average gives denoised value

use crate::utils;

pub fn nlm_denoise_impl(
    data: &[f32],
    dims: [usize; 3],
    search_radius: usize,
    patch_radius: usize,
    h: f32,
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;

    // Auto-estimate h if 0
    let h_val = if h <= 0.0 {
        let sigma = utils::estimate_noise_std(data);
        // h is typically slightly larger than noise std
        sigma * 1.2
    } else {
        h
    };

    let h_sq = h_val * h_val;
    if h_sq < 1e-10 {
        return data.to_vec();
    }

    let patch_size = 2 * patch_radius + 1;
    let patch_vol = patch_size * patch_size * patch_size;
    let norm_factor = 1.0 / (patch_vol as f32 * h_sq);

    let mut result = vec![0.0f32; n];

    // Block distance for blockwise speedup
    let block_dist: usize = 2;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let center_idx = utils::idx3(x, y, z, nx, ny);
                let center_val = data[center_idx];

                // Skip zero voxels
                if center_val == 0.0 {
                    continue;
                }

                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                // Search window bounds
                let sz_start = z.saturating_sub(search_radius);
                let sz_end = (z + search_radius + 1).min(nz);
                let sy_start = y.saturating_sub(search_radius);
                let sy_end = (y + search_radius + 1).min(ny);
                let sx_start = x.saturating_sub(search_radius);
                let sx_end = (x + search_radius + 1).min(nx);

                for sz in (sz_start..sz_end).step_by(block_dist.max(1)) {
                    for sy in (sy_start..sy_end).step_by(block_dist.max(1)) {
                        for sx in (sx_start..sx_end).step_by(block_dist.max(1)) {
                            let search_idx = utils::idx3(sx, sy, sz, nx, ny);
                            let search_val = data[search_idx];

                            // Skip zero voxels
                            if search_val == 0.0 {
                                continue;
                            }

                            // Quick mean-based preselection
                            if (center_val - search_val).abs() > 3.0 * h_val {
                                continue;
                            }

                            // Compute squared Euclidean distance between patches
                            let dist = patch_distance(
                                data, dims, x, y, z, sx, sy, sz, patch_radius,
                            );

                            // Weight = exp(-dist / h^2)
                            let weight = (-dist * norm_factor).exp();

                            weighted_sum += search_val * weight;
                            weight_total += weight;
                        }
                    }
                }

                if weight_total > 0.0 {
                    result[center_idx] = weighted_sum / weight_total;
                } else {
                    result[center_idx] = center_val;
                }
            }
        }
    }

    result
}

/// Compute squared Euclidean distance between two 3D patches.
#[inline]
fn patch_distance(
    data: &[f32],
    dims: [usize; 3],
    x1: usize,
    y1: usize,
    z1: usize,
    x2: usize,
    y2: usize,
    z2: usize,
    patch_radius: usize,
) -> f32 {
    let [nx, ny, nz] = dims;
    let mut dist = 0.0f32;
    let pr = patch_radius as isize;

    for dz in -pr..=pr {
        let pz1 = z1 as isize + dz;
        let pz2 = z2 as isize + dz;
        if pz1 < 0 || pz1 >= nz as isize || pz2 < 0 || pz2 >= nz as isize {
            continue;
        }

        for dy in -pr..=pr {
            let py1 = y1 as isize + dy;
            let py2 = y2 as isize + dy;
            if py1 < 0 || py1 >= ny as isize || py2 < 0 || py2 >= ny as isize {
                continue;
            }

            for dx in -pr..=pr {
                let px1 = x1 as isize + dx;
                let px2 = x2 as isize + dx;
                if px1 < 0 || px1 >= nx as isize || px2 < 0 || px2 >= nx as isize {
                    continue;
                }

                let v1 = data[utils::idx3(px1 as usize, py1 as usize, pz1 as usize, nx, ny)];
                let v2 = data[utils::idx3(px2 as usize, py2 as usize, pz2 as usize, nx, ny)];
                let d = v1 - v2;
                dist += d * d;
            }
        }
    }

    dist
}
