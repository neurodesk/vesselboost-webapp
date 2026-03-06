/**
 * VesselBoost label definitions.
 * Binary segmentation: background (0) and vessel (1).
 */

export const LABELS = [
  { index: 0, name: 'Background', color: [0, 0, 0, 0] },
  { index: 1, name: 'Vessel', color: [255, 50, 50, 255] },
];

/**
 * Generate a NiiVue-compatible discrete colormap LUT.
 * Returns an object { R, G, B, A, min, max } for nv.addColormap().
 */
export function generateNiivueColormap() {
  const size = 256;
  const R = new Array(size).fill(0);
  const G = new Array(size).fill(0);
  const B = new Array(size).fill(0);
  const A = new Array(size).fill(0);

  // NiiVue maps voxel values through the LUT linearly:
  // voxel 0 (cal_min) -> index 0, voxel 1 (cal_max) -> index 255
  // So vessel (value=1) maps to the last index (255)
  R[255] = 255;
  G[255] = 50;
  B[255] = 50;
  A[255] = 255;

  return { R, G, B, A, min: 0, max: 1 };
}

/**
 * Get label name by index.
 */
export function getLabelName(index) {
  return LABELS[index]?.name || `Label ${index}`;
}

/**
 * Get label color as [R, G, B, A] (0-255).
 */
export function getLabelColor(index) {
  return LABELS[index]?.color || [128, 128, 128, 255];
}
