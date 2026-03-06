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

  // Background: transparent
  // Vessel (index 1): red
  R[1] = 255;
  G[1] = 50;
  B[1] = 50;
  A[1] = 255;

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
