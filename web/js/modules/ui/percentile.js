/**
 * Histogram-based percentile computation for auto-contrast windowing.
 * O(n) time, fixed ~16KB memory (4096 bins).
 */

/**
 * Compute percentile values from a volume image using a histogram approach.
 * Skips zero voxels (background).
 *
 * @param {TypedArray} img - Volume image data
 * @param {number} pLow - Lower percentile (default 2)
 * @param {number} pHigh - Upper percentile (default 98)
 * @param {number} numBins - Number of histogram bins (default 4096)
 * @param {number} [dataMin] - Data minimum (computed if not provided)
 * @param {number} [dataMax] - Data maximum (computed if not provided)
 * @returns {{ low: number, high: number }} Percentile values
 */
export function computePercentiles(img, pLow = 2, pHigh = 98, numBins = 4096, dataMin, dataMax) {
  // Single pass: find min/max of non-zero voxels and count them
  let min = dataMin ?? Infinity;
  let max = dataMax ?? -Infinity;
  let nonZeroCount = 0;

  const needMinMax = (dataMin === undefined || dataMax === undefined);

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v === 0) continue;
    nonZeroCount++;
    if (needMinMax) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  // Edge case: all zeros
  if (nonZeroCount === 0) return { low: 0, high: 1 };

  // Edge case: all same value
  if (min >= max) return { low: min, high: min };

  // Build histogram
  const histogram = new Uint32Array(numBins);
  const range = max - min;
  const scale = (numBins - 1) / range;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v === 0) continue;
    const bin = Math.min(numBins - 1, Math.floor((v - min) * scale));
    histogram[bin]++;
  }

  // Walk histogram to find percentile edges
  const targetLow = Math.floor((pLow / 100) * nonZeroCount);
  const targetHigh = Math.floor((pHigh / 100) * nonZeroCount);

  let cumulative = 0;
  let lowBin = 0;
  let highBin = numBins - 1;

  for (let i = 0; i < numBins; i++) {
    cumulative += histogram[i];
    if (cumulative >= targetLow) {
      lowBin = i;
      break;
    }
  }

  cumulative = 0;
  for (let i = 0; i < numBins; i++) {
    cumulative += histogram[i];
    if (cumulative >= targetHigh) {
      highBin = i;
      break;
    }
  }

  // Convert bin indices back to values
  const low = min + (lowBin / (numBins - 1)) * range;
  const high = min + (highBin / (numBins - 1)) * range;

  return { low, high };
}
