/**
 * Auto-contrast windowing for MRA volumes.
 * Uses mean ± k*std of non-zero voxels, which handles the
 * heavily skewed intensity distribution of angiography data
 * better than fixed percentiles.
 */

/**
 * Compute auto-contrast window using mean/std of non-zero voxels.
 *
 * @param {TypedArray} img - Volume image data
 * @param {number} [globalMin] - Data minimum (clamps lower bound)
 * @returns {{ low: number, high: number }}
 */
export function computeAutoWindow(img, globalMin) {
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v === 0) continue;
    sum += v;
    sumSq += v * v;
    count++;
  }

  if (count === 0) return { low: 0, high: 1 };

  const mean = sum / count;
  const variance = sumSq / count - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  const low = Math.max(globalMin ?? 0, mean - 0.5 * std);
  const high = mean + 2 * std;

  return { low, high };
}
