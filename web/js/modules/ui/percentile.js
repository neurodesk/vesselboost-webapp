/**
 * Auto-contrast windowing for MRA volumes.
 * Computes histogram-based percentiles on all voxels.
 * Returns raw-space values; caller must convert to scaled space
 * if vol.img uses a different space than vol.cal_min/cal_max.
 */

/**
 * Compute auto-contrast window using histogram-based percentiles.
 *
 * @param {TypedArray} img - Volume image data (may be raw or scaled)
 * @returns {{ low: number, high: number, min: number, max: number }}
 */
export function computeAutoWindow(img) {
  const LOW_PCT = 0.02;
  const HIGH_PCT = 0.998;
  const N_BINS = 4096;

  if (!img || img.length === 0) return { low: 0, high: 1, min: 0, max: 1 };

  // Pass 1: find full data range
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (!isFinite(min) || !isFinite(max)) return { low: 0, high: 1, min: 0, max: 1 };
  if (max <= min) return { low: min, high: min + 1, min, max: min + 1 };

  // Pass 2: build histogram of ALL data
  const bins = new Uint32Array(N_BINS);
  const scale = (N_BINS - 1) / (max - min);

  for (let i = 0; i < img.length; i++) {
    bins[Math.round((img[i] - min) * scale)]++;
  }

  // Pass 3: compute percentiles from cumulative histogram
  const lowTarget = Math.floor(img.length * LOW_PCT);
  const highTarget = Math.floor(img.length * HIGH_PCT);
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

  return { low, high, min, max };
}
