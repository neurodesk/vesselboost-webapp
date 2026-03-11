/**
 * Auto-contrast windowing for MRA volumes.
 * Uses a two-stage approach: first identifies the background level
 * (dominant peak in histogram), then computes percentiles on
 * foreground voxels only. Works regardless of data offset/scaling.
 */

/**
 * Compute auto-contrast window using histogram-based percentiles.
 * Automatically detects background level rather than assuming zero.
 *
 * @param {TypedArray} img - Volume image data
 * @returns {{ low: number, high: number }}
 */
export function computeAutoWindow(img) {
  const LOW_PCT = 0.02;
  const HIGH_PCT = 0.998;
  const N_BINS = 1024;

  if (!img || img.length === 0) return { low: 0, high: 1 };

  // Pass 1: find full data range
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (!isFinite(min) || !isFinite(max)) return { low: 0, high: 1 };
  if (max <= min) return { low: min, high: min + 1 };

  // Pass 2: build histogram of ALL data
  const bins = new Uint32Array(N_BINS);
  const scale = (N_BINS - 1) / (max - min);

  for (let i = 0; i < img.length; i++) {
    bins[Math.round((img[i] - min) * scale)]++;
  }

  // Find background: the dominant peak in the lower half of the histogram.
  // In MRA, background (air/empty space) is the most frequent intensity
  // and sits at the low end regardless of data offset.
  let bgBin = 0;
  let bgCount = 0;
  const halfBins = Math.floor(N_BINS / 2);
  for (let i = 0; i < halfBins; i++) {
    if (bins[i] > bgCount) {
      bgCount = bins[i];
      bgBin = i;
    }
  }

  // Threshold: exclude voxels in/near the background peak.
  // Walk right from peak until counts drop below 1% of peak height.
  let cutoffBin = bgBin;
  const peakThreshold = Math.max(1, bgCount * 0.01);
  for (let i = bgBin + 1; i < N_BINS; i++) {
    if (bins[i] < peakThreshold) {
      cutoffBin = i;
      break;
    }
  }
  const bgCutoff = min + cutoffBin / scale;

  // Pass 3: compute percentiles on foreground voxels (above background)
  let fgMin = Infinity;
  let fgMax = -Infinity;
  let fgCount = 0;

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v <= bgCutoff) continue;
    if (v < fgMin) fgMin = v;
    if (v > fgMax) fgMax = v;
    fgCount++;
  }

  // Fallback: if no foreground found, use full-range percentiles
  if (fgCount === 0 || fgMax <= fgMin) {
    const lowTarget = Math.floor(img.length * LOW_PCT);
    const highTarget = Math.floor(img.length * HIGH_PCT);
    let cumulative = 0;
    let low = min;
    let high = max;
    for (let i = 0; i < N_BINS; i++) {
      cumulative += bins[i];
      if (low === min && cumulative >= lowTarget) low = min + i / scale;
      if (cumulative >= highTarget) { high = min + i / scale; break; }
    }
    return { low, high };
  }

  // Build foreground histogram
  const fgBins = new Uint32Array(N_BINS);
  const fgScale = (N_BINS - 1) / (fgMax - fgMin);

  for (let i = 0; i < img.length; i++) {
    const v = img[i];
    if (v <= bgCutoff) continue;
    fgBins[Math.round((v - fgMin) * fgScale)]++;
  }

  // Compute percentiles from foreground histogram
  const lowTarget = Math.floor(fgCount * LOW_PCT);
  const highTarget = Math.floor(fgCount * HIGH_PCT);
  let cumulative = 0;
  let low = fgMin;
  let high = fgMax;

  for (let i = 0; i < N_BINS; i++) {
    cumulative += fgBins[i];
    if (low === fgMin && cumulative >= lowTarget) {
      low = fgMin + i / fgScale;
    }
    if (cumulative >= highTarget) {
      high = fgMin + i / fgScale;
      break;
    }
  }

  return { low, high };
}
