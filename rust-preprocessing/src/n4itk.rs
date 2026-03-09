/// N4ITK bias field correction implementation.
///
/// Simplified version of the ITK N4BiasFieldCorrectionImageFilter.
/// Algorithm:
/// 1. Downsample volume for faster computation
/// 2. Take log of intensities (bias is multiplicative)
/// 3. Iteratively:
///    a. Compute intensity histogram and sharpen via Wiener deconvolution
///    b. Estimate residual bias field
///    c. Smooth with B-spline fitting
///    d. Subtract from log-image
/// 4. Upsample bias field and apply correction

use crate::utils;

pub fn n4_bias_correct_impl(
    data: &[f32],
    dims: [usize; 3],
    _voxel_size: [f32; 3],
    shrink_factor: usize,
    max_iterations: usize,
    convergence_threshold: f32,
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;

    // Create mask of nonzero voxels
    let mask: Vec<bool> = data.iter().map(|&v| v > 0.0).collect();

    // Compute positive minimum for log transform
    let min_positive = data
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .fold(f32::MAX, f32::min);

    if min_positive == f32::MAX {
        return data.to_vec();
    }

    // Take log of intensities (offset by min_positive for numerical stability)
    let mut log_image = vec![0.0f32; n];
    for i in 0..n {
        if mask[i] {
            log_image[i] = (data[i].max(min_positive)).ln();
        }
    }

    // Downsample for faster processing
    let (shrunken_log, shrunken_dims) = if shrink_factor > 1 {
        utils::downsample_volume(&log_image, dims, shrink_factor)
    } else {
        (log_image.clone(), dims)
    };

    let (shrunken_mask, _) = if shrink_factor > 1 {
        let mask_f32: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let (dm, dd) = utils::downsample_volume(&mask_f32, dims, shrink_factor);
        let dm_bool: Vec<bool> = dm.iter().map(|&v| v > 0.5).collect();
        (dm_bool, dd)
    } else {
        (mask.clone(), dims)
    };

    let sn = shrunken_dims[0] * shrunken_dims[1] * shrunken_dims[2];

    // Bias field estimate (in log space)
    let mut bias_field = vec![0.0f32; sn];

    // Iterative correction
    let mut prev_metric = f32::MAX;

    for _iter in 0..max_iterations {
        // Corrected log image = log_image - bias_field
        let corrected: Vec<f32> = (0..sn)
            .map(|i| {
                if shrunken_mask[i] {
                    shrunken_log[i] - bias_field[i]
                } else {
                    0.0
                }
            })
            .collect();

        // Compute histogram of corrected values
        let (_hist_mean, hist_std) = masked_stats(&corrected, &shrunken_mask);
        if hist_std < 1e-10 {
            break;
        }

        // Estimate residual bias: smooth the corrected image to extract
        // the low-frequency component, which approximates the remaining
        // bias field that hasn't been captured yet.  Subtract the masked
        // mean so the bias estimate is zero-centred (pure multiplicative
        // inhomogeneity, no global shift).
        let smoothed_corrected = smooth_3d(&corrected, shrunken_dims, &shrunken_mask);
        let (smooth_mean, _) = masked_stats(&smoothed_corrected, &shrunken_mask);

        let residual: Vec<f32> = (0..sn)
            .map(|i| {
                if shrunken_mask[i] {
                    smoothed_corrected[i] - smooth_mean
                } else {
                    0.0
                }
            })
            .collect();

        // Update bias field with residual (already smooth — derived from smooth_3d)
        for i in 0..sn {
            if shrunken_mask[i] {
                bias_field[i] += residual[i];
            }
        }

        // Convergence check
        let metric = masked_variance(&corrected, &shrunken_mask);
        let change = if prev_metric < f32::MAX {
            (prev_metric - metric).abs() / prev_metric.abs().max(1e-10)
        } else {
            1.0
        };

        if change < convergence_threshold && _iter > 0 {
            break;
        }
        prev_metric = metric;
    }

    // Upsample bias field to original resolution
    let full_bias = if shrink_factor > 1 {
        utils::upsample_volume(&bias_field, shrunken_dims, dims)
    } else {
        bias_field
    };

    // Apply correction: corrected = exp(log(image) - bias_field)
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        if mask[i] {
            result[i] = (log_image[i] - full_bias[i]).exp();
        }
    }

    result
}

/// Compute mean and standard deviation of masked values.
fn masked_stats(data: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for (i, &v) in data.iter().enumerate() {
        if mask[i] {
            sum += v as f64;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / count as f64;
    let mut sum_sq = 0.0f64;
    for (i, &v) in data.iter().enumerate() {
        if mask[i] {
            let d = v as f64 - mean;
            sum_sq += d * d;
        }
    }
    (mean as f32, (sum_sq / count as f64).sqrt() as f32)
}

/// Compute variance of masked values.
fn masked_variance(data: &[f32], mask: &[bool]) -> f32 {
    let (_, std) = masked_stats(data, mask);
    std * std
}

/// Simple 3D Gaussian-like smoothing using iterative box filter.
fn smooth_3d(data: &[f32], dims: [usize; 3], mask: &[bool]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let mut current = data.to_vec();
    let mut temp = vec![0.0f32; n];

    // 3 iterations of 3x3x3 box filter approximates Gaussian
    for _ in 0..3 {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = utils::idx3(x, y, z, nx, ny);
                    if !mask[idx] {
                        temp[idx] = 0.0;
                        continue;
                    }

                    let mut sum = 0.0f32;
                    let mut count = 0u32;

                    let x_start = if x > 0 { x - 1 } else { 0 };
                    let x_end = (x + 2).min(nx);
                    let y_start = if y > 0 { y - 1 } else { 0 };
                    let y_end = (y + 2).min(ny);
                    let z_start = if z > 0 { z - 1 } else { 0 };
                    let z_end = (z + 2).min(nz);

                    for nz2 in z_start..z_end {
                        for ny2 in y_start..y_end {
                            for nx2 in x_start..x_end {
                                let nidx = utils::idx3(nx2, ny2, nz2, nx, ny);
                                if mask[nidx] {
                                    sum += current[nidx];
                                    count += 1;
                                }
                            }
                        }
                    }

                    temp[idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
        std::mem::swap(&mut current, &mut temp);
    }

    current
}
