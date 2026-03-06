use wasm_bindgen::prelude::*;
use js_sys;

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

/// BET brain extraction (FSL-BET2 algorithm via qsm-core).
///
/// Accepts Float32 input and converts to Float64 internally to avoid
/// doubling memory usage on the JS side.
///
/// # Arguments
/// * `data` - Flattened Float32 magnitude volume data
/// * `nx`, `ny`, `nz` - Volume dimensions
/// * `vsx`, `vsy`, `vsz` - Voxel sizes in mm
/// * `fractional_intensity` - Intensity threshold (0.0-1.0, smaller = larger brain, default: 0.5)
/// * `progress_callback` - JS function(current, total) for progress updates
///
/// # Returns
/// Binary mask as Uint8Array (1 = brain, 0 = background)
#[wasm_bindgen]
pub fn bet_brain_extract(
    data: &[f32],
    nx: u32,
    ny: u32,
    nz: u32,
    vsx: f32,
    vsy: f32,
    vsz: f32,
    fractional_intensity: f32,
    progress_callback: &js_sys::Function,
) -> Vec<u8> {
    // Convert f32 -> f64 inside WASM to avoid JS-side Float64Array allocation
    let f64_data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let callback = progress_callback.clone();
    let mask = qsm_core::bet::run_bet_with_progress(
        &f64_data,
        nx as usize, ny as usize, nz as usize,
        vsx as f64, vsy as f64, vsz as f64,
        fractional_intensity as f64,
        1.0,  // smoothness_factor (FSL default)
        0.0,  // gradient_threshold (FSL default)
        1000, // iterations
        4,    // subdivisions (2562 vertices)
        |current, total| {
            let this = JsValue::null();
            let _ = callback.call2(
                &this,
                &JsValue::from(current as u32),
                &JsValue::from(total as u32),
            );
        },
    );
    mask
}
