/**
 * Auto-contrast windowing for MRA volumes.
 * Uses histogram-based percentiles of non-background voxels,
 * which is robust to the heavily skewed intensity distribution
 * of angiography data and numerically stable for large volumes.
 */

/**
 * Compute auto-contrast window using histogram-based percentiles.
 *
 * @param {TypedArray} img - Volume image data
 * @returns {{ low: number, high: number }}
 */
export function computeAutoWindow(img) {
  const THRESHOLD = 1e-6;
  const LOW_PCT = 0.02;
  const HIGH_PCT = 0.998;
  const N_BINS = 1024;

  // Pass 1: find data range of non-background voxels
  let min = Infinity;
  let max = -Infinity;
  let nonZeroCount = 0;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (Math.abs(v) <= THRESHOLD) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    nonZeroCount++;
  }

  if (nonZeroCount === 0) return { low: 0, high: 1 };
  if (max <= min) return { low: min, high: min + 1 };

  // Pass 2: build histogram
  const bins = new Uint32Array(N_BINS);
  const scale = (N_BINS - 1) / (max - min);

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (Math.abs(v) <= THRESHOLD) continue;
    bins[Math.round((v - min) * scale)]++;
  }

  // Compute percentiles from cumulative histogram
  const lowTarget = Math.floor(nonZeroCount * LOW_PCT);
  const highTarget = Math.floor(nonZeroCount * HIGH_PCT);
  let cumulative = 0;
  let low = min;
  let high = max;

  for (let i = 0; i < N_BINS; i++) {
    cumulative += bins[i];
    if (low === min && cumulative >= lowTarget) {
      low = min + i / scale;
    }
    if (cumulative >= highTarget) {
      high = min + i / scale;
      break;
    }
  }

  return { low, high };
}
