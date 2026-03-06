use wasm_bindgen::prelude::*;

mod n4itk;
mod nlm;
mod utils;

/// N4ITK bias field correction for 3D MRI volumes.
///
/// # Arguments
/// * `data` - Flattened Float32 volume data
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `vox_x`, `vox_y`, `vox_z` - Voxel sizes in mm
/// * `shrink_factor` - Downsampling factor for speed (default: 4)
/// * `max_iterations` - Maximum iterations per level (default: 50)
/// * `convergence_threshold` - Convergence threshold (default: 0.001)
#[wasm_bindgen]
pub fn n4_bias_correct(
    data: &[f32],
    nx: u32,
    ny: u32,
    nz: u32,
    vox_x: f32,
    vox_y: f32,
    vox_z: f32,
    shrink_factor: u32,
    max_iterations: u32,
    convergence_threshold: f32,
) -> Vec<f32> {
    n4itk::n4_bias_correct_impl(
        data,
        [nx as usize, ny as usize, nz as usize],
        [vox_x, vox_y, vox_z],
        shrink_factor as usize,
        max_iterations as usize,
        convergence_threshold,
    )
}

/// Non-local means denoising for 3D MRI volumes.
///
/// # Arguments
/// * `data` - Flattened Float32 volume data
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `search_radius` - Search window half-size (default: 5)
/// * `patch_radius` - Patch half-size (default: 1, gives 3x3x3 patches)
/// * `h` - Smoothing parameter (0.0 = auto-estimate from noise)
#[wasm_bindgen]
pub fn nlm_denoise(
    data: &[f32],
    nx: u32,
    ny: u32,
    nz: u32,
    search_radius: u32,
    patch_radius: u32,
    h: f32,
) -> Vec<f32> {
    nlm::nlm_denoise_impl(
        data,
        [nx as usize, ny as usize, nz as usize],
        search_radius as usize,
        patch_radius as usize,
        h,
    )
}
