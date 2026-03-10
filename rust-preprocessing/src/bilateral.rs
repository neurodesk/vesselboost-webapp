/// 3D bilateral filter for volumetric denoising.
///
/// Edge-preserving denoising that is significantly faster than NLM.
/// Combines spatial Gaussian weighting with intensity-based weighting
/// to smooth homogeneous regions while preserving edges.

use crate::utils;

pub fn bilateral_filter_impl(
    data: &[f32],
    dims: [usize; 3],
    spatial_radius: usize,
    sigma_spatial: f32,
    sigma_intensity: f32,
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;

    // Auto-estimate sigma_intensity from noise if 0
    let sigma_i = if sigma_intensity <= 0.0 {
        let sigma = utils::estimate_noise_std(data);
        // Use 2x noise std for intensity range — preserves edges while smoothing noise
        (sigma * 2.0).max(1e-6)
    } else {
        sigma_intensity
    };

    let inv_2_sigma_s_sq = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    let inv_2_sigma_i_sq = 1.0 / (2.0 * sigma_i * sigma_i);

    let mut result = vec![0.0f32; n];

    // Precompute spatial Gaussian weights for the kernel
    let kernel_size = 2 * spatial_radius + 1;
    let mut spatial_weights = vec![0.0f32; kernel_size * kernel_size * kernel_size];
    for dz in 0..kernel_size {
        for dy in 0..kernel_size {
            for dx in 0..kernel_size {
                let fx = dx as f32 - spatial_radius as f32;
                let fy = dy as f32 - spatial_radius as f32;
                let fz = dz as f32 - spatial_radius as f32;
                let dist_sq = fx * fx + fy * fy + fz * fz;
                spatial_weights[dx + dy * kernel_size + dz * kernel_size * kernel_size] =
                    (-dist_sq * inv_2_sigma_s_sq).exp();
            }
        }
    }

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let center_idx = utils::idx3(x, y, z, nx, ny);
                let center_val = data[center_idx];

                // Skip zero voxels (background)
                if center_val == 0.0 {
                    continue;
                }

                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                let z_start = z.saturating_sub(spatial_radius);
                let z_end = (z + spatial_radius + 1).min(nz);
                let y_start = y.saturating_sub(spatial_radius);
                let y_end = (y + spatial_radius + 1).min(ny);
                let x_start = x.saturating_sub(spatial_radius);
                let x_end = (x + spatial_radius + 1).min(nx);

                for nz_idx in z_start..z_end {
                    let dz = nz_idx + spatial_radius - z;
                    for ny_idx in y_start..y_end {
                        let dy = ny_idx + spatial_radius - y;
                        for nx_idx in x_start..x_end {
                            let dx = nx_idx + spatial_radius - x;

                            let neighbor_idx = utils::idx3(nx_idx, ny_idx, nz_idx, nx, ny);
                            let neighbor_val = data[neighbor_idx];

                            // Skip zero neighbors
                            if neighbor_val == 0.0 {
                                continue;
                            }

                            // Spatial weight (precomputed)
                            let w_spatial = spatial_weights
                                [dx + dy * kernel_size + dz * kernel_size * kernel_size];

                            // Intensity weight
                            let diff = center_val - neighbor_val;
                            let w_intensity = (-diff * diff * inv_2_sigma_i_sq).exp();

                            let weight = w_spatial * w_intensity;
                            weighted_sum += neighbor_val * weight;
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
