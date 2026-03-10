/**
 * VesselBoost Inference Worker
 *
 * Runs ONNX model inference for 3D patch-based vessel segmentation.
 * Pipeline is split into interactive steps:
 *   1. Load (NIfTI parse + orient to RAS)
 *   2. N4 bias field correction (optional)
 *   3. BET brain extraction
 *   4. Slice range selection
 *   5. NLM denoising (optional)
 *   6. Inference (resample → normalize → crop → sliding window → threshold → CC → inverse)
 */

/* global importScripts, ort, localforage, nifti, wasm_bindgen */

importScripts('../wasm/ort.webgpu.min.js');
importScripts('https://cdn.jsdelivr.net/npm/localforage@1.10.0/dist/localforage.min.js');
importScripts('../nifti-js/index.js');

// Preprocessing WASM (optional - loaded if available)
let wasmPreprocessingAvailable = false;
try {
  importScripts('../preprocessing-wasm/preprocessing.js');
  wasmPreprocessingAvailable = true;
} catch (e) {
  console.warn('Preprocessing WASM import failed:', e);
}

const FIXED_TARGET_SPACING = [0.3, 0.3, 0.3];
const MAX_PROCESSING_VOXELS = 100 * 1024 * 1024;

// ==================== Shared Worker State ====================

let workerState = {
  headerBytes: null,
  origDims: null,
  affine: null,
  perm: null,
  flip: null,
  isIdentity: null,
  rasData: null,
  rasDims: null,
  rasSpacing: null,
  brainMask: null,
  sliceStartZ: null,
  sliceEndZ: null,
  subsectionData: null,
  subsectionDims: null,
  denoisedData: null
};

function resetState() {
  workerState = {
    headerBytes: null,
    origDims: null,
    affine: null,
    perm: null,
    flip: null,
    isIdentity: null,
    rasData: null,
    rasDims: null,
    rasSpacing: null,
    brainMask: null,
    sliceStartZ: null,
    sliceEndZ: null,
    subsectionData: null,
    subsectionDims: null,
    denoisedData: null
  };
}

// ==================== Message Helpers ====================

function postProgress(value, text) {
  self.postMessage({ type: 'progress', value, text });
}

function postLog(message) {
  self.postMessage({ type: 'log', message });
}

function postError(message) {
  self.postMessage({ type: 'error', message });
}

function postComplete() {
  self.postMessage({ type: 'complete' });
}

function postStageData(stage, niftiData, description) {
  self.postMessage(
    { type: 'stageData', stage, niftiData, description },
    [niftiData]
  );
}

function postStepComplete(step) {
  self.postMessage({ type: 'step-complete', step });
}

function postVolumeInfo(info) {
  self.postMessage({ type: 'volume-info', ...info });
}

// ==================== NIfTI Parsing ====================

function decompressIfNeeded(data) {
  const bytes = new Uint8Array(data);
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof nifti !== 'undefined' && nifti.isCompressed) {
      if (nifti.isCompressed(bytes.buffer)) {
        return new Uint8Array(nifti.decompress(bytes.buffer));
      }
    }
    throw new Error('Gzipped NIfTI detected but decompression not available');
  }
  return bytes;
}

function parseNiftiInput(arrayBuffer) {
  const data = decompressIfNeeded(arrayBuffer);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

  const dims = [];
  for (let i = 0; i < 8; i++) dims.push(view.getInt16(40 + i * 2, true));
  const nx = dims[1], ny = dims[2], nz = dims[3];

  const pixDims = [];
  for (let i = 0; i < 8; i++) pixDims.push(view.getFloat32(76 + i * 4, true));

  const datatype = view.getInt16(70, true);
  const voxOffset = view.getFloat32(108, true);
  const sclSlope = view.getFloat32(112, true) || 1;
  const sclInter = view.getFloat32(116, true) || 0;
  const dataStart = Math.ceil(voxOffset);
  const nTotal = nx * ny * nz;

  const imageData = new Float32Array(nTotal);
  switch (datatype) {
    case 2:
      for (let i = 0; i < nTotal; i++) imageData[i] = data[dataStart + i] * sclSlope + sclInter;
      break;
    case 4:
      for (let i = 0; i < nTotal; i++) imageData[i] = view.getInt16(dataStart + i * 2, true) * sclSlope + sclInter;
      break;
    case 8:
      for (let i = 0; i < nTotal; i++) imageData[i] = view.getInt32(dataStart + i * 4, true) * sclSlope + sclInter;
      break;
    case 16:
      for (let i = 0; i < nTotal; i++) imageData[i] = view.getFloat32(dataStart + i * 4, true) * sclSlope + sclInter;
      break;
    case 64:
      for (let i = 0; i < nTotal; i++) imageData[i] = view.getFloat64(dataStart + i * 8, true) * sclSlope + sclInter;
      break;
    case 512:
      for (let i = 0; i < nTotal; i++) imageData[i] = view.getUint16(dataStart + i * 2, true) * sclSlope + sclInter;
      break;
    default:
      throw new Error(`Unsupported NIfTI datatype: ${datatype}`);
  }

  const affine = extractAffine(view);

  const headerSize = dataStart;
  const headerBytes = new ArrayBuffer(headerSize);
  new Uint8Array(headerBytes).set(data.slice(0, headerSize));

  return {
    imageData,
    dims: [nx, ny, nz],
    voxelSize: [Math.abs(pixDims[1]) || 1, Math.abs(pixDims[2]) || 1, Math.abs(pixDims[3]) || 1],
    headerBytes,
    affine
  };
}

function extractAffine(view) {
  const sformCode = view.getInt16(254, true);
  const qformCode = view.getInt16(252, true);

  if (sformCode > 0) {
    const affine = [new Float64Array(4), new Float64Array(4), new Float64Array(4), new Float64Array([0, 0, 0, 1])];
    for (let i = 0; i < 4; i++) {
      affine[0][i] = view.getFloat32(280 + i * 4, true);
      affine[1][i] = view.getFloat32(296 + i * 4, true);
      affine[2][i] = view.getFloat32(312 + i * 4, true);
    }
    return affine;
  }

  if (qformCode > 0) {
    const pixDims = [];
    for (let i = 0; i < 4; i++) pixDims.push(view.getFloat32(76 + i * 4, true));
    const qb = view.getFloat32(256, true);
    const qc = view.getFloat32(260, true);
    const qd = view.getFloat32(264, true);
    const qx = view.getFloat32(268, true);
    const qy = view.getFloat32(272, true);
    const qz = view.getFloat32(276, true);
    const sqr = qb * qb + qc * qc + qd * qd;
    const qa = sqr > 1.0 ? 0.0 : Math.sqrt(1.0 - sqr);
    const R = [
      [qa*qa+qb*qb-qc*qc-qd*qd, 2*(qb*qc-qa*qd), 2*(qb*qd+qa*qc)],
      [2*(qb*qc+qa*qd), qa*qa+qc*qc-qb*qb-qd*qd, 2*(qc*qd-qa*qb)],
      [2*(qb*qd-qa*qc), 2*(qc*qd+qa*qb), qa*qa+qd*qd-qb*qb-qc*qc]
    ];
    const qfac = pixDims[0] < 0 ? -1 : 1;
    return [
      new Float64Array([R[0][0]*pixDims[1], R[0][1]*pixDims[2], R[0][2]*pixDims[3]*qfac, qx]),
      new Float64Array([R[1][0]*pixDims[1], R[1][1]*pixDims[2], R[1][2]*pixDims[3]*qfac, qy]),
      new Float64Array([R[2][0]*pixDims[1], R[2][1]*pixDims[2], R[2][2]*pixDims[3]*qfac, qz]),
      new Float64Array([0, 0, 0, 1])
    ];
  }

  const pixDims = [];
  for (let i = 0; i < 4; i++) pixDims.push(view.getFloat32(76 + i * 4, true));
  return [
    new Float64Array([pixDims[1] || 1, 0, 0, 0]),
    new Float64Array([0, pixDims[2] || 1, 0, 0]),
    new Float64Array([0, 0, pixDims[3] || 1, 0]),
    new Float64Array([0, 0, 0, 1])
  ];
}

// ==================== NIfTI Output ====================

function createOutputNifti(uint8Data, sourceHeader, dims) {
  const srcView = new DataView(sourceHeader);
  const voxOffset = srcView.getFloat32(108, true);
  const headerSize = Math.ceil(voxOffset);

  const buffer = new ArrayBuffer(headerSize + uint8Data.length);
  const destBytes = new Uint8Array(buffer);
  const destView = new DataView(buffer);

  destBytes.set(new Uint8Array(sourceHeader).slice(0, headerSize));

  // Set datatype to UINT8
  destView.setInt16(70, 2, true);
  destView.setInt16(72, 8, true);

  // Update dims if provided
  if (dims) {
    destView.setInt16(40, 3, true);
    destView.setInt16(42, dims[0], true);
    destView.setInt16(44, dims[1], true);
    destView.setInt16(46, dims[2], true);
    destView.setInt16(48, 1, true);
  }

  destView.setFloat32(112, 1, true);  // scl_slope
  destView.setFloat32(116, 0, true);  // scl_inter

  // cal_min/cal_max for binary mask
  destView.setFloat32(124, 1, true);   // cal_max
  destView.setFloat32(128, 0, true);   // cal_min

  new Uint8Array(buffer, headerSize).set(uint8Data);
  return buffer;
}

function createFloat32Nifti(float32Data, sourceHeader, dims, spacing) {
  const srcView = new DataView(sourceHeader);
  const voxOffset = srcView.getFloat32(108, true);
  const headerSize = Math.ceil(voxOffset);

  const dataBytes = float32Data.length * 4;
  const buffer = new ArrayBuffer(headerSize + dataBytes);
  const destBytes = new Uint8Array(buffer);
  const destView = new DataView(buffer);

  destBytes.set(new Uint8Array(sourceHeader).slice(0, headerSize));

  // Set datatype to FLOAT32
  destView.setInt16(70, 16, true);
  destView.setInt16(72, 32, true);

  if (dims) {
    destView.setInt16(40, 3, true);
    destView.setInt16(42, dims[0], true);
    destView.setInt16(44, dims[1], true);
    destView.setInt16(46, dims[2], true);
    destView.setInt16(48, 1, true);
  }

  if (spacing) {
    destView.setFloat32(80, spacing[0], true);  // pixdim[1]
    destView.setFloat32(84, spacing[1], true);  // pixdim[2]
    destView.setFloat32(88, spacing[2], true);  // pixdim[3]
  }

  destView.setFloat32(112, 1, true);  // scl_slope
  destView.setFloat32(116, 0, true);  // scl_inter

  // cal_min/cal_max: auto range
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < float32Data.length; i++) {
    if (float32Data[i] < minVal) minVal = float32Data[i];
    if (float32Data[i] > maxVal) maxVal = float32Data[i];
  }
  destView.setFloat32(124, maxVal, true);  // cal_max
  destView.setFloat32(128, minVal, true);  // cal_min

  new Uint8Array(buffer, headerSize).set(new Uint8Array(float32Data.buffer, float32Data.byteOffset, dataBytes));
  return buffer;
}

// ==================== Preprocessing ====================

function getOrientationTransform(affine) {
  const mat = [
    [affine[0][0], affine[0][1], affine[0][2]],
    [affine[1][0], affine[1][1], affine[1][2]],
    [affine[2][0], affine[2][1], affine[2][2]]
  ];

  const perm = [0, 0, 0];
  const flip = [false, false, false];
  const used = [false, false, false];

  for (let outAxis = 0; outAxis < 3; outAxis++) {
    let bestAxis = -1;
    let bestVal = -1;
    for (let inAxis = 0; inAxis < 3; inAxis++) {
      if (used[inAxis]) continue;
      const val = Math.abs(mat[outAxis][inAxis]);
      if (val > bestVal) {
        bestVal = val;
        bestAxis = inAxis;
      }
    }
    perm[outAxis] = bestAxis;
    flip[outAxis] = mat[outAxis][bestAxis] < 0;
    used[bestAxis] = true;
  }

  return { perm, flip };
}

function orientToRAS(data, dims, perm, flip) {
  const [nx, ny, nz] = dims;
  const srcDims = [nx, ny, nz];
  const dstDims = [srcDims[perm[0]], srcDims[perm[1]], srcDims[perm[2]]];
  const [dx, dy, dz] = dstDims;
  const result = new Float32Array(dx * dy * dz);

  for (let oz = 0; oz < dz; oz++) {
    for (let oy = 0; oy < dy; oy++) {
      for (let ox = 0; ox < dx; ox++) {
        const coords = [ox, oy, oz];
        const src = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
          src[perm[i]] = flip[i] ? (dstDims[i] - 1 - coords[i]) : coords[i];
        }
        const srcIdx = src[0] + src[1] * nx + src[2] * nx * ny;
        const dstIdx = ox + oy * dx + oz * dx * dy;
        result[dstIdx] = data[srcIdx];
      }
    }
  }

  return { data: result, dims: dstDims };
}

function computeResampledDims(dims, srcSpacing, tgtSpacing) {
  return [
    Math.max(1, Math.round(dims[0] * srcSpacing[0] / tgtSpacing[0])),
    Math.max(1, Math.round(dims[1] * srcSpacing[1] / tgtSpacing[1])),
    Math.max(1, Math.round(dims[2] * srcSpacing[2] / tgtSpacing[2]))
  ];
}

function resampleVolume(data, dims, srcSpacing, tgtSpacing) {
  const [nx, ny, nz] = dims;
  const newDims = computeResampledDims(dims, srcSpacing, tgtSpacing);
  const [nnx, nny, nnz] = newDims;
  const result = new Float32Array(nnx * nny * nnz);

  const scaleX = (nx - 1) / Math.max(nnx - 1, 1);
  const scaleY = (ny - 1) / Math.max(nny - 1, 1);
  const scaleZ = (nz - 1) / Math.max(nnz - 1, 1);

  for (let z = 0; z < nnz; z++) {
    const sz = z * scaleZ;
    const z0 = Math.floor(sz);
    const z1 = Math.min(z0 + 1, nz - 1);
    const wz = sz - z0;
    for (let y = 0; y < nny; y++) {
      const sy = y * scaleY;
      const y0 = Math.floor(sy);
      const y1 = Math.min(y0 + 1, ny - 1);
      const wy = sy - y0;
      for (let x = 0; x < nnx; x++) {
        const sx = x * scaleX;
        const x0 = Math.floor(sx);
        const x1 = Math.min(x0 + 1, nx - 1);
        const wx = sx - x0;

        const c000 = data[x0 + y0*nx + z0*nx*ny];
        const c100 = data[x1 + y0*nx + z0*nx*ny];
        const c010 = data[x0 + y1*nx + z0*nx*ny];
        const c110 = data[x1 + y1*nx + z0*nx*ny];
        const c001 = data[x0 + y0*nx + z1*nx*ny];
        const c101 = data[x1 + y0*nx + z1*nx*ny];
        const c011 = data[x0 + y1*nx + z1*nx*ny];
        const c111 = data[x1 + y1*nx + z1*nx*ny];

        const c00 = c000*(1-wx) + c100*wx;
        const c01 = c001*(1-wx) + c101*wx;
        const c10 = c010*(1-wx) + c110*wx;
        const c11 = c011*(1-wx) + c111*wx;
        const c0 = c00*(1-wy) + c10*wy;
        const c1 = c01*(1-wy) + c11*wy;

        result[x + y*nnx + z*nnx*nny] = c0*(1-wz) + c1*wz;
      }
    }
  }

  return { data: result, dims: newDims, spacing: tgtSpacing };
}

function extractSliceRange(data, dims, startZ, endZ, outputCtor = Float32Array) {
  const [nx, ny, nz] = dims;
  const clampedStart = Math.max(0, Math.min(nz, Math.floor(startZ)));
  const clampedEnd = Math.max(clampedStart, Math.min(nz, Math.floor(endZ)));
  const subsetNz = clampedEnd - clampedStart;
  const sliceSize = nx * ny;
  const result = new outputCtor(sliceSize * subsetNz);
  for (let z = 0; z < subsetNz; z++) {
    const srcOff = (clampedStart + z) * sliceSize;
    const dstOff = z * sliceSize;
    result.set(data.subarray(srcOff, srcOff + sliceSize), dstOff);
  }
  return { data: result, dims: [nx, ny, subsetNz] };
}

function embedSliceSubsection(data, subsectionDims, fullDims, startZ) {
  const [nx, ny, nz] = subsectionDims;
  const [fnx, fny, fnz] = fullDims;
  if (nx !== fnx || ny !== fny) {
    throw new Error('Subsection and full dimensions are incompatible for embedding');
  }
  if (startZ < 0 || startZ + nz > fnz) {
    throw new Error('Invalid subsection Z-range for embedding');
  }

  const result = new Uint8Array(fnx * fny * fnz);
  const sliceSize = nx * ny;
  for (let z = 0; z < nz; z++) {
    const srcOff = z * sliceSize;
    const dstOff = (startZ + z) * sliceSize;
    result.set(data.subarray(srcOff, srcOff + sliceSize), dstOff);
  }
  return result;
}

function zScoreNormalizeNonzero(data) {
  const n = data.length;
  let sum = 0, count = 0;
  for (let i = 0; i < n; i++) {
    if (data[i] !== 0) { sum += data[i]; count++; }
  }
  if (count === 0) return new Float32Array(n);
  const mean = sum / count;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    if (data[i] !== 0) { const d = data[i] - mean; sumSq += d * d; }
  }
  const std = Math.sqrt(sumSq / count) || 1;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (data[i] !== 0) result[i] = (data[i] - mean) / std;
  }
  return result;
}

function cropForeground(data, dims, margin) {
  const [nx, ny, nz] = dims;
  let minX = nx, maxX = 0, minY = ny, maxY = 0, minZ = nz, maxZ = 0;

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (data[x + y*nx + z*nx*ny] !== 0) {
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
      }
    }
  }

  if (maxX < minX) return { data: new Float32Array(0), dims: [0,0,0], origin: [0,0,0] };

  const ox = Math.max(0, minX - margin);
  const oy = Math.max(0, minY - margin);
  const oz = Math.max(0, minZ - margin);
  const ex = Math.min(nx, maxX + margin + 1);
  const ey = Math.min(ny, maxY + margin + 1);
  const ez = Math.min(nz, maxZ + margin + 1);
  const cnx = ex - ox, cny = ey - oy, cnz = ez - oz;

  const result = new Float32Array(cnx * cny * cnz);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      const srcOff = (z+oz)*nx*ny + (y+oy)*nx + ox;
      const dstOff = z*cnx*cny + y*cnx;
      result.set(data.subarray(srcOff, srcOff + cnx), dstOff);
    }
  }

  return { data: result, dims: [cnx, cny, cnz], origin: [ox, oy, oz] };
}

// ==================== 3D Sliding Window ====================

function computeGaussianWeightMap3D(dim0, dim1, dim2, sigma) {
  if (!sigma) sigma = Math.min(dim0, dim1, dim2) / 8;
  const weights = new Float32Array(dim0 * dim1 * dim2);
  const c0 = (dim0 - 1) / 2;
  const c1 = (dim1 - 1) / 2;
  const c2 = (dim2 - 1) / 2;
  const s2 = 2 * sigma * sigma;
  for (let i0 = 0; i0 < dim0; i0++) {
    const d0 = i0 - c0;
    for (let i1 = 0; i1 < dim1; i1++) {
      const d1 = i1 - c1;
      for (let i2 = 0; i2 < dim2; i2++) {
        const d2 = i2 - c2;
        weights[i0 * dim1 * dim2 + i1 * dim2 + i2] = Math.exp(-(d0*d0 + d1*d1 + d2*d2) / s2);
      }
    }
  }
  return weights;
}

function computePatchPositions3D(volumeDims, patchDims, overlap) {
  const positions = [];
  const seen = new Set();

  const steps = patchDims.map(p => Math.max(1, Math.round(p * (1 - overlap))));

  const counts = volumeDims.map((vd, i) => {
    if (vd <= patchDims[i]) return 1;
    return Math.max(1, Math.ceil((vd - patchDims[i]) / steps[i]) + 1);
  });

  for (let iz = 0; iz < counts[2]; iz++) {
    let z = iz * steps[2];
    if (z + patchDims[2] > volumeDims[2]) z = Math.max(0, volumeDims[2] - patchDims[2]);

    for (let iy = 0; iy < counts[1]; iy++) {
      let y = iy * steps[1];
      if (y + patchDims[1] > volumeDims[1]) y = Math.max(0, volumeDims[1] - patchDims[1]);

      for (let ix = 0; ix < counts[0]; ix++) {
        let x = ix * steps[0];
        if (x + patchDims[0] > volumeDims[0]) x = Math.max(0, volumeDims[0] - patchDims[0]);

        const key = `${x},${y},${z}`;
        if (!seen.has(key)) {
          seen.add(key);
          positions.push([x, y, z]);
        }
      }
    }
  }

  return positions;
}

function extractPatch3D(volume, volumeDims, position, patchDims) {
  const [v0, v1, v2] = volumeDims;
  const [p0, p1, p2] = patchDims;
  const [o0, o1, o2] = position;
  const patch = new Float32Array(p0 * p1 * p2);

  for (let i0 = 0; i0 < p0; i0++) {
    const g0 = o0 + i0;
    if (g0 < 0 || g0 >= v0) continue;
    for (let i1 = 0; i1 < p1; i1++) {
      const g1 = o1 + i1;
      if (g1 < 0 || g1 >= v1) continue;
      for (let i2 = 0; i2 < p2; i2++) {
        const g2 = o2 + i2;
        if (g2 < 0 || g2 >= v2) continue;

        const srcIdx = g0 + g1 * v0 + g2 * v0 * v1;
        const dstIdx = i0 * p1 * p2 + i1 * p2 + i2;
        patch[dstIdx] = volume[srcIdx];
      }
    }
  }

  return patch;
}

function accumulatePatch3D(probAccum, weightAccum, volumeDims, position, output, weights, patchDims) {
  const [v0, v1, v2] = volumeDims;
  const [p0, p1, p2] = patchDims;
  const [o0, o1, o2] = position;

  for (let i0 = 0; i0 < p0; i0++) {
    const g0 = o0 + i0;
    if (g0 < 0 || g0 >= v0) continue;
    for (let i1 = 0; i1 < p1; i1++) {
      const g1 = o1 + i1;
      if (g1 < 0 || g1 >= v1) continue;
      for (let i2 = 0; i2 < p2; i2++) {
        const g2 = o2 + i2;
        if (g2 < 0 || g2 >= v2) continue;

        const patchIdx = i0 * p1 * p2 + i1 * p2 + i2;
        const globalIdx = g0 + g1 * v0 + g2 * v0 * v1;
        const w = weights[patchIdx];
        probAccum[globalIdx] += output[patchIdx] * w;
        weightAccum[globalIdx] += w;
      }
    }
  }
}

// ==================== Postprocessing ====================

function connectedComponents3D(binaryMask, dims) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const labels = new Int32Array(n);
  let nextLabel = 1;
  const parent = [0];
  const rank = [0];

  function find(x) {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }

  function union(a, b) {
    a = find(a); b = find(b);
    if (a === b) return;
    if (rank[a] < rank[b]) { const t = a; a = b; b = t; }
    parent[b] = a;
    if (rank[a] === rank[b]) rank[a]++;
  }

  const neighborOffsets = [];
  for (let dz = -1; dz <= 0; dz++)
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        if (dz === 0 && dy === 0 && dx >= 0) continue;
        neighborOffsets.push([dx, dy, dz]);
      }

  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const idx = z*ny*nx + y*nx + x;
        if (!binaryMask[idx]) continue;
        const neighborLabels = [];
        for (let i = 0; i < neighborOffsets.length; i++) {
          const nx2 = x+neighborOffsets[i][0], ny2 = y+neighborOffsets[i][1], nz2 = z+neighborOffsets[i][2];
          if (nx2<0||nx2>=nx||ny2<0||ny2>=ny||nz2<0||nz2>=nz) continue;
          const nIdx = nz2*ny*nx + ny2*nx + nx2;
          if (labels[nIdx] > 0) neighborLabels.push(labels[nIdx]);
        }
        if (neighborLabels.length === 0) {
          labels[idx] = nextLabel;
          parent.push(nextLabel);
          rank.push(0);
          nextLabel++;
        } else {
          let minLabel = find(neighborLabels[0]);
          for (let i = 1; i < neighborLabels.length; i++) {
            const c = find(neighborLabels[i]);
            if (c < minLabel) minLabel = c;
          }
          labels[idx] = minLabel;
          for (let i = 0; i < neighborLabels.length; i++) union(minLabel, neighborLabels[i]);
        }
      }

  const canonicalMap = new Map();
  let finalLabel = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (!canonicalMap.has(root)) canonicalMap.set(root, ++finalLabel);
    labels[i] = canonicalMap.get(root);
  }
  return { labels, numComponents: finalLabel };
}

function removeSmallComponents(binaryMask, dims, minSize) {
  const n = dims[0] * dims[1] * dims[2];
  const { labels, numComponents } = connectedComponents3D(binaryMask, dims);

  if (numComponents === 0) return binaryMask;

  const sizes = new Int32Array(numComponents + 1);
  for (let i = 0; i < n; i++) {
    if (labels[i] > 0) sizes[labels[i]]++;
  }

  const result = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (labels[i] > 0 && sizes[labels[i]] >= minSize) {
      result[i] = 1;
    }
  }

  return result;
}

// ==================== Inverse Transform ====================

function uncrop(croppedData, croppedDims, fullDims, origin) {
  const [nx, ny, nz] = fullDims;
  const [cnx, cny, cnz] = croppedDims;
  const [ox, oy, oz] = origin;
  const result = new Uint8Array(nx * ny * nz);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      const srcOff = z*cnx*cny + y*cnx;
      const dstOff = (z+oz)*nx*ny + (y+oy)*nx + ox;
      result.set(croppedData.subarray(srcOff, srcOff + cnx), dstOff);
    }
  }
  return result;
}

function resampleLabelsNearest(data, dims, tgtDims) {
  const [nx, ny, nz] = dims;
  const [tnx, tny, tnz] = tgtDims;
  const result = new Uint8Array(tnx * tny * tnz);
  const scaleX = (nx - 1) / Math.max(tnx - 1, 1);
  const scaleY = (ny - 1) / Math.max(tny - 1, 1);
  const scaleZ = (nz - 1) / Math.max(tnz - 1, 1);
  for (let z = 0; z < tnz; z++) {
    const sz = Math.round(z * scaleZ);
    for (let y = 0; y < tny; y++) {
      const sy = Math.round(y * scaleY);
      for (let x = 0; x < tnx; x++) {
        const sx = Math.round(x * scaleX);
        result[x + y*tnx + z*tnx*tny] = data[sx + sy*nx + sz*nx*ny];
      }
    }
  }
  return result;
}

function inverseOrient(data, dims, perm, flip, origDims) {
  const [dx, dy, dz] = dims;
  const [nx, ny, nz] = origDims;
  const result = new Uint8Array(nx * ny * nz);
  for (let oz = 0; oz < dz; oz++) {
    for (let oy = 0; oy < dy; oy++) {
      for (let ox = 0; ox < dx; ox++) {
        const coords = [ox, oy, oz];
        const src = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
          src[perm[i]] = flip[i] ? (dims[i] - 1 - coords[i]) : coords[i];
        }
        const srcIdx = ox + oy*dx + oz*dx*dy;
        const dstIdx = src[0] + src[1]*nx + src[2]*nx*ny;
        result[dstIdx] = data[srcIdx];
      }
    }
  }
  return result;
}

// ==================== Model Loading ====================

async function fetchModel(url, modelName, progressBase, progressSpan) {
  const displayName = modelName || url.split('/').pop();
  const cacheKey = `${url}?v=${self._appVersion || ''}`;

  try {
    const cached = await localforage.getItem(cacheKey);
    if (cached && cached.byteLength > 100000) {
      postLog(`Model loaded from cache: ${displayName}`);
      postProgress(progressBase + progressSpan, `Cached: ${displayName}`);
      return cached;
    }
  } catch (e) { /* cache miss */ }

  postLog(`Downloading: ${displayName}...`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);

  const contentLength = parseInt(response.headers.get('content-length'), 10);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (contentLength) {
      const dlProgress = received / contentLength;
      const mb = (received / 1048576).toFixed(1);
      const totalMb = (contentLength / 1048576).toFixed(0);
      postProgress(progressBase + dlProgress * progressSpan, `Downloading ${displayName} (${mb}/${totalMb} MB)`);
    }
  }

  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }

  try {
    await localforage.setItem(cacheKey, data.buffer);
  } catch (e) {
    postLog('Warning: Could not cache model (storage full?)');
  }

  postLog(`Downloaded: ${displayName} (${(received / 1048576).toFixed(1)} MB)`);
  return data.buffer;
}

// ==================== WASM Preprocessing ====================

async function initWasmPreprocessing() {
  if (!wasmPreprocessingAvailable) return false;
  try {
    await wasm_bindgen('../preprocessing-wasm/preprocessing_bg.wasm');
    return true;
  } catch (e) {
    postLog('Warning: Could not initialize preprocessing WASM: ' + e.message);
    return false;
  }
}

// ==================== Utility ====================

function getOptimalWasmThreads() {
  return (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
}

// ==================== Step Functions ====================

function stepLoad(inputData) {
  postLog('Parsing input volume...');
  postProgress(0.02, 'Reading NIfTI...');
  const { imageData, dims, voxelSize, headerBytes, affine } = parseNiftiInput(inputData);
  const [nx, ny, nz] = dims;
  postLog(`Volume: ${nx}x${ny}x${nz}, spacing: ${voxelSize.map(v => v.toFixed(3)).join('x')}mm`);

  workerState.origDims = [...dims];
  workerState.affine = affine;
  workerState.headerBytes = headerBytes;

  // Orient to RAS
  postProgress(0.04, 'Orienting to RAS...');
  postLog('Orienting to RAS...');
  const { perm, flip } = getOrientationTransform(affine);
  const isIdentity = perm[0] === 0 && perm[1] === 1 && perm[2] === 2 && !flip[0] && !flip[1] && !flip[2];

  workerState.perm = perm;
  workerState.flip = flip;
  workerState.isIdentity = isIdentity;

  if (isIdentity) {
    workerState.rasData = imageData;
    workerState.rasDims = [...dims];
    workerState.rasSpacing = [...voxelSize];
  } else {
    const oriented = orientToRAS(imageData, dims, perm, flip);
    workerState.rasData = oriented.data;
    workerState.rasDims = oriented.dims;
    workerState.rasSpacing = [voxelSize[perm[0]], voxelSize[perm[1]], voxelSize[perm[2]]];

    // Rewrite headerBytes sform to match the RAS-reoriented data
    const srcVoxel = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      srcVoxel[perm[i]] = flip[i] ? (workerState.rasDims[i] - 1) : 0;
    }
    const origin = [0, 0, 0];
    for (let r = 0; r < 3; r++) {
      origin[r] = affine[r][0] * srcVoxel[0]
                + affine[r][1] * srcVoxel[1]
                + affine[r][2] * srcVoxel[2]
                + affine[r][3];
    }

    const hdrView = new DataView(headerBytes);
    hdrView.setInt16(254, 1, true);
    hdrView.setFloat32(280, workerState.rasSpacing[0], true);
    hdrView.setFloat32(284, 0, true);
    hdrView.setFloat32(288, 0, true);
    hdrView.setFloat32(292, origin[0], true);
    hdrView.setFloat32(296, 0, true);
    hdrView.setFloat32(300, workerState.rasSpacing[1], true);
    hdrView.setFloat32(304, 0, true);
    hdrView.setFloat32(308, origin[1], true);
    hdrView.setFloat32(312, 0, true);
    hdrView.setFloat32(316, 0, true);
    hdrView.setFloat32(320, workerState.rasSpacing[2], true);
    hdrView.setFloat32(324, origin[2], true);
    hdrView.setInt16(252, 0, true);
  }
  postLog(`RAS dims: ${workerState.rasDims.join('x')}`);

  // Clear downstream state
  workerState.brainMask = null;
  workerState.sliceStartZ = null;
  workerState.sliceEndZ = null;
  workerState.subsectionData = null;
  workerState.subsectionDims = null;
  workerState.denoisedData = null;

  // Post volume info for UI
  postVolumeInfo({
    rasDims: [...workerState.rasDims],
    rasSpacing: [...workerState.rasSpacing],
    totalSlices: workerState.rasDims[2]
  });

  postProgress(1.0, 'Volume loaded');
  postStepComplete('load');
}

function stepN4() {
  if (!workerState.rasData) {
    throw new Error('No volume loaded. Run Load first.');
  }
  if (!self._wasmReady) {
    throw new Error('Preprocessing WASM not available');
  }

  const { rasData, rasDims, rasSpacing, headerBytes } = workerState;
  const fullVoxels = rasDims[0] * rasDims[1] * rasDims[2];
  if (fullVoxels > MAX_PROCESSING_VOXELS) {
    postLog(
      `Warning: Full-volume N4 input is large (${(fullVoxels / 1e6).toFixed(0)}M voxels). `
      + 'Bias correction may run slowly or fail.'
    );
  }

  postProgress(0.1, 'Bias field correction (N4ITK)...');
  postLog('Running N4ITK bias field correction on full RAS volume...');

  // Log input stats for diagnostic comparison
  let inMin = Infinity, inMax = -Infinity, inSum = 0, inNonzero = 0;
  for (let i = 0; i < rasData.length; i++) {
    const v = rasData[i];
    if (v < inMin) inMin = v;
    if (v > inMax) inMax = v;
    if (v !== 0) { inSum += v; inNonzero++; }
  }
  postLog(`N4 input stats: min=${inMin.toFixed(2)}, max=${inMax.toFixed(2)}, mean_nz=${inNonzero ? (inSum/inNonzero).toFixed(2) : 'N/A'}, nonzero=${inNonzero}/${rasData.length}`);

  const corrected = wasm_bindgen.n4_bias_correct(
    rasData, rasDims[0], rasDims[1], rasDims[2],
    rasSpacing[0], rasSpacing[1], rasSpacing[2],
    4, 10, 0.005
  );

  // Log output stats
  let outMin = Infinity, outMax = -Infinity, outSum = 0, outNonzero = 0;
  for (let i = 0; i < corrected.length; i++) {
    const v = corrected[i];
    if (v < outMin) outMin = v;
    if (v > outMax) outMax = v;
    if (v !== 0) { outSum += v; outNonzero++; }
  }
  postLog(`N4 output stats: min=${outMin.toFixed(2)}, max=${outMax.toFixed(2)}, mean_nz=${outNonzero ? (outSum/outNonzero).toFixed(2) : 'N/A'}, nonzero=${outNonzero}/${corrected.length}`);

  workerState.rasData = corrected;
  postLog('Bias field correction complete');

  const n4Nifti = createFloat32Nifti(new Float32Array(corrected), headerBytes, rasDims, rasSpacing);
  postStageData('n4', n4Nifti, 'Bias field correction (N4ITK)');

  // Clear downstream state (BET + downstream invalidated)
  workerState.brainMask = null;
  workerState.sliceStartZ = null;
  workerState.sliceEndZ = null;
  workerState.subsectionData = null;
  workerState.subsectionDims = null;
  workerState.denoisedData = null;

  postProgress(1.0, 'N4 complete');
  postStepComplete('n4');
}

function stepBET(params) {
  if (!workerState.rasData) {
    throw new Error('No volume loaded. Run Load first.');
  }
  if (!self._wasmReady) {
    throw new Error('Preprocessing WASM not available');
  }

  const fractionalIntensity = params.fractionalIntensity ?? 0.5;
  const { rasData, rasDims, rasSpacing, headerBytes } = workerState;

  postProgress(0.1, 'Brain extraction (BET)...');
  postLog(`Running BET brain extraction (fi=${fractionalIntensity})...`);

  const progressCb = (current, total) => {
    const pct = Math.round((current / total) * 100);
    if (pct % 10 === 0) {
      postProgress(0.1 + 0.8 * (current / total), `BET: ${pct}%`);
    }
  };

  const brainMask = wasm_bindgen.bet_brain_extract(
    rasData,
    rasDims[0], rasDims[1], rasDims[2],
    rasSpacing[0], rasSpacing[1], rasSpacing[2],
    fractionalIntensity,
    progressCb
  );

  let maskCount = 0;
  for (let i = 0; i < brainMask.length; i++) {
    if (brainMask[i]) maskCount++;
  }
  const coverage = (100 * maskCount / rasData.length).toFixed(1);
  postLog(`Brain mask: ${maskCount} voxels (${coverage}% coverage)`);

  workerState.brainMask = brainMask;

  // BET does NOT modify rasData or invalidate downstream - mask is independent

  // Post masked preview
  const maskedPreview = new Float32Array(rasData.length);
  for (let i = 0; i < rasData.length; i++) {
    maskedPreview[i] = brainMask[i] ? rasData[i] : 0;
  }
  const betNifti = createFloat32Nifti(maskedPreview, headerBytes, rasDims, rasSpacing);
  postStageData('bet', betNifti, 'Brain extraction (BET)');

  postProgress(1.0, 'BET complete');
  postStepComplete('bet');
}

function stepSelectSlices(params) {
  if (!workerState.rasData) {
    throw new Error('No volume loaded. Run Load first.');
  }

  const { rasData, rasDims, rasSpacing, headerBytes } = workerState;
  const startZ = Math.max(0, Math.min(rasDims[2], Math.floor(params.startZ)));
  const endZ = Math.max(startZ, Math.min(rasDims[2], Math.floor(params.endZ)));

  postProgress(0.3, 'Selecting slice range...');
  postLog(`Selecting slices z=${startZ}-${endZ - 1} (${endZ - startZ}/${rasDims[2]} slices)`);

  const extracted = extractSliceRange(rasData, rasDims, startZ, endZ, Float32Array);
  workerState.subsectionData = extracted.data;
  workerState.subsectionDims = extracted.dims;
  workerState.sliceStartZ = startZ;
  workerState.sliceEndZ = endZ;

  // Clear downstream
  workerState.denoisedData = null;

  const subsectionNifti = createFloat32Nifti(
    new Float32Array(extracted.data),
    headerBytes,
    extracted.dims,
    rasSpacing
  );
  postStageData('subsection', subsectionNifti, `Slice subsection (z=${startZ}-${endZ - 1})`);

  postProgress(1.0, 'Slice selection complete');
  postStepComplete('slices');
}

function stepDenoise() {
  if (!workerState.subsectionData) {
    throw new Error('No slice range selected. Run Select Slices first.');
  }
  if (!self._wasmReady) {
    throw new Error('Preprocessing WASM not available');
  }

  const { subsectionData, subsectionDims, rasSpacing, headerBytes } = workerState;

  postProgress(0.1, 'Denoising (NLM)...');
  postLog('Running non-local means denoising on subsection...');

  const denoised = wasm_bindgen.nlm_denoise(
    subsectionData, subsectionDims[0], subsectionDims[1], subsectionDims[2],
    5, 1, 0.0
  );
  workerState.denoisedData = denoised;
  postLog('Denoising complete');

  const nlmNifti = createFloat32Nifti(
    new Float32Array(denoised),
    headerBytes,
    subsectionDims,
    rasSpacing
  );
  postStageData('nlm', nlmNifti, 'Denoising (NLM)');

  postProgress(1.0, 'Denoising complete');
  postStepComplete('denoise');
}

async function stepInference(params) {
  if (!workerState.subsectionData) {
    throw new Error('No slice range selected. Run Select Slices first.');
  }

  const {
    overlap = 0.5,
    threshold = 0.1,
    minComponentSize = 10,
    modelName = 'vesselboost.onnx',
    patchSize = [64, 64, 64],
    modelBaseUrl
  } = params;

  const [PATCH_DIM0, PATCH_DIM1, PATCH_DIM2] = patchSize;
  const CROP_MARGIN = 20;

  // Use denoised data if available, otherwise subsection data
  let currentData = workerState.denoisedData
    ? new Float32Array(workerState.denoisedData)
    : new Float32Array(workerState.subsectionData);
  let currentDims = [...workerState.subsectionDims];
  let currentSpacing = [...workerState.rasSpacing];
  const rasProcessingDims = [...currentDims];

  // Resample to fixed 0.3mm spacing
  postProgress(0.05, 'Resampling...');
  const targetSpacing = FIXED_TARGET_SPACING;
  const needsResample = currentSpacing.some((s, i) => Math.abs(s - targetSpacing[i]) > 0.01);

  if (needsResample) {
    const projectedDims = computeResampledDims(currentDims, currentSpacing, targetSpacing);
    const projectedVoxels = projectedDims[0] * projectedDims[1] * projectedDims[2];
    if (projectedVoxels > MAX_PROCESSING_VOXELS) {
      throw new Error(
        `Selected subsection is too large at 0.3mm (${projectedDims.join('x')}, ${(projectedVoxels / 1e6).toFixed(0)}M voxels). `
        + 'Reduce the slice range and try again.'
      );
    }

    postLog(`Resampling to ${targetSpacing.map(s => s.toFixed(2)).join('x')}mm...`);
    try {
      const resampled = resampleVolume(currentData, currentDims, currentSpacing, targetSpacing);
      currentData = resampled.data;
      currentDims = resampled.dims;
      currentSpacing = resampled.spacing;
      postLog(`Resampled: ${currentDims.join('x')}`);
    } catch (e) {
      if (e?.message && /Array buffer allocation failed|Invalid typed array length/i.test(e.message)) {
        throw new Error('Resampling ran out of memory. Reduce the slice range and try again.');
      }
      throw e;
    }
  }
  const processingDims = [...currentDims];

  // Normalize
  postProgress(0.10, 'Normalizing...');
  postLog('Z-score normalizing (nonzero voxels)...');
  currentData = zScoreNormalizeNonzero(currentData);

  // Crop foreground
  postProgress(0.12, 'Cropping foreground...');
  const cropped = cropForeground(currentData, currentDims, CROP_MARGIN);
  if (cropped.dims[0] === 0) {
    throw new Error('No foreground voxels found in volume');
  }
  currentData = cropped.data;
  currentDims = cropped.dims;
  const cropOrigin = cropped.origin;
  postLog(`Cropped: ${currentDims.join('x')} (origin: ${cropOrigin.join(',')})`);

  // Download and load model
  const modelUrl = `${modelBaseUrl}/${modelName}`;
  const modelData = await fetchModel(modelUrl, modelName, 0.12, 0.15);

  postProgress(0.27, 'Loading ONNX model...');
  const executionProviders = ['wasm'];
  postLog('Creating ONNX InferenceSession (wasm - 3D ops require WASM backend)...');
  const session = await ort.InferenceSession.create(modelData, {
    executionProviders,
    graphOptimizationLevel: 'all'
  });
  postLog(`Session created. Input: ${session.inputNames}, Output: ${session.outputNames}`);

  // 3D Sliding Window Inference
  const gaussianWeights = computeGaussianWeightMap3D(PATCH_DIM0, PATCH_DIM1, PATCH_DIM2, 8);
  const positions = computePatchPositions3D(currentDims, [PATCH_DIM0, PATCH_DIM1, PATCH_DIM2], overlap);
  const totalPatches = positions.length;
  postLog(`Starting 3D inference: ${totalPatches} patches (${PATCH_DIM0}x${PATCH_DIM1}x${PATCH_DIM2}), overlap=${overlap}, backend=wasm`);

  const totalVoxels = currentDims[0] * currentDims[1] * currentDims[2];
  const probAccum = new Float32Array(totalVoxels);
  const weightAccum = new Float32Array(totalVoxels);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const patchVoxels = PATCH_DIM0 * PATCH_DIM1 * PATCH_DIM2;

  const inferenceStartTime = performance.now();

  for (let pi = 0; pi < totalPatches; pi++) {
    const pos = positions[pi];
    const patch = extractPatch3D(currentData, currentDims, pos, [PATCH_DIM0, PATCH_DIM1, PATCH_DIM2]);

    const inputTensor = new ort.Tensor('float32', patch, [1, 1, PATCH_DIM0, PATCH_DIM1, PATCH_DIM2]);
    const results = await session.run({ [inputName]: inputTensor });
    const output = results[outputName].data;
    inputTensor.dispose();

    if (pi === 0) postLog(`First patch output range: [${output[0].toFixed(3)}, ${output[patchVoxels-1].toFixed(3)}]`);
    const probabilities = new Float32Array(patchVoxels);
    for (let i = 0; i < patchVoxels; i++) {
      probabilities[i] = 1.0 / (1.0 + Math.exp(-output[i]));
    }

    accumulatePatch3D(probAccum, weightAccum, currentDims, pos, probabilities, gaussianWeights, [PATCH_DIM0, PATCH_DIM1, PATCH_DIM2]);

    if (pi % 5 === 0 || pi === totalPatches - 1) {
      const elapsed = (performance.now() - inferenceStartTime) / 1000;
      const eta = (elapsed / (pi + 1)) * (totalPatches - pi - 1);
      postProgress(0.30 + 0.50 * ((pi + 1) / totalPatches), `Patch ${pi+1}/${totalPatches} (ETA: ${eta.toFixed(0)}s)`);
    }
  }

  const totalTime = ((performance.now() - inferenceStartTime) / 1000).toFixed(1);
  postLog(`Inference complete: ${totalPatches} patches in ${totalTime}s`);
  await session.release();

  // Threshold and binarize
  postProgress(0.82, 'Thresholding...');
  postLog(`Thresholding at p=${threshold}...`);
  const binaryMask = new Uint8Array(totalVoxels);
  for (let i = 0; i < totalVoxels; i++) {
    if (weightAccum[i] > 0) {
      const prob = probAccum[i] / weightAccum[i];
      if (prob >= threshold) {
        binaryMask[i] = 1;
      }
    }
  }

  // Remove small connected components
  postProgress(0.86, 'Removing small components...');
  postLog(`Removing components smaller than ${minComponentSize} voxels...`);
  const cleanedMask = removeSmallComponents(binaryMask, currentDims, minComponentSize);

  let totalSegmented = 0;
  for (let i = 0; i < totalVoxels; i++) {
    if (cleanedMask[i]) totalSegmented++;
  }
  postLog(`Segmented voxels: ${totalSegmented}`);

  // Inverse transform: uncrop
  postProgress(0.90, 'Inverse transform...');
  postLog('Applying inverse transforms...');
  let outputLabels = uncrop(cleanedMask, currentDims, processingDims, cropOrigin);

  // Inverse resample back to native RAS subsection dims
  if (needsResample) {
    outputLabels = resampleLabelsNearest(outputLabels, processingDims, rasProcessingDims);
  }

  // Apply brain mask (extract subset for slice range)
  if (workerState.brainMask) {
    const maskSubset = extractSliceRange(
      workerState.brainMask,
      workerState.rasDims,
      workerState.sliceStartZ,
      workerState.sliceEndZ,
      Uint8Array
    );
    let maskedOut = 0;
    for (let i = 0; i < outputLabels.length; i++) {
      if (outputLabels[i] && !maskSubset.data[i]) {
        outputLabels[i] = 0;
        maskedOut++;
      }
    }
    if (maskedOut > 0) {
      postLog(`Brain mask removed ${maskedOut} vessel voxels outside brain`);
    }
  }

  // Embed subsection back into full RAS grid
  const sliceApplied = workerState.sliceStartZ !== 0 || workerState.sliceEndZ !== workerState.rasDims[2];
  if (sliceApplied) {
    outputLabels = embedSliceSubsection(outputLabels, rasProcessingDims, workerState.rasDims, workerState.sliceStartZ);
  }

  // Inverse orient
  if (!workerState.isIdentity) {
    outputLabels = inverseOrient(outputLabels, workerState.rasDims, workerState.perm, workerState.flip, workerState.origDims);
  }

  // Create output NIfTI
  const outputNifti = createOutputNifti(outputLabels, workerState.headerBytes, workerState.origDims);
  postStageData('segmentation', outputNifti, 'Vessel segmentation');

  let finalVoxels = 0;
  for (let i = 0; i < outputLabels.length; i++) {
    if (outputLabels[i] > 0) finalVoxels++;
  }
  postLog(`Output: ${finalVoxels} vessel voxels`);

  postProgress(1.0, 'Complete');
  postStepComplete('inference');
  postComplete();
}

// ==================== Message Handler ====================

self.onmessage = async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case 'init':
      try {
        self._appVersion = e.data.version || '';
        ort.env.wasm.numThreads = getOptimalWasmThreads();
        ort.env.wasm.wasmPaths = '../wasm/';

        postLog(`Using WASM backend (${ort.env.wasm.numThreads} threads)`);

        self._wasmReady = await initWasmPreprocessing();
        if (self._wasmReady) {
          postLog('Preprocessing WASM ready (N4ITK + NLM + BET)');
        }

        localforage.config({
          name: 'VesselBoostModelCache',
          storeName: 'models'
        });

        self.postMessage({ type: 'initialized' });
      } catch (error) {
        postError(`Initialization failed: ${error.message}`);
      }
      break;

    case 'load':
      try {
        stepLoad(data.inputData);
      } catch (error) {
        console.error('Load error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'run-n4':
      try {
        stepN4();
      } catch (error) {
        console.error('N4 error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'run-bet':
      try {
        stepBET(data || {});
      } catch (error) {
        console.error('BET error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'select-slices':
      try {
        stepSelectSlices(data || {});
      } catch (error) {
        console.error('Slice selection error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'run-denoise':
      try {
        stepDenoise();
      } catch (error) {
        console.error('Denoise error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'run-inference':
      try {
        await stepInference(data || {});
      } catch (error) {
        console.error('Inference error:', error);
        postError(error?.message || String(error));
      }
      break;

    case 'reset-state':
      resetState();
      postLog('Worker state reset');
      break;

    // Legacy support for old 'run' message
    case 'run':
      try {
        // Decompose the old single-run into steps for backwards compat
        const { inputData, settings } = data;
        stepLoad(inputData);
        if (settings.biasCorrection && self._wasmReady) {
          try { stepN4(); } catch (e) { postLog(`Warning: N4 failed: ${e.message}`); }
        }
        if (self._wasmReady) {
          try { stepBET({ fractionalIntensity: settings.fractionalIntensity }); } catch (e) { postLog(`Warning: BET failed: ${e.message}`); }
        }
        const fraction = Math.max(0.01, Math.min(1, settings.sliceSubsectionFraction || 0.1));
        const totalZ = workerState.rasDims[2];
        const subsetNz = Math.max(1, Math.min(totalZ, Math.round(totalZ * fraction)));
        const startZ = Math.max(0, Math.floor((totalZ - subsetNz) / 2));
        const endZ = startZ + subsetNz;
        stepSelectSlices({ startZ, endZ });
        if (settings.denoising && self._wasmReady) {
          try { stepDenoise(); } catch (e) { postLog(`Warning: Denoising failed: ${e.message}`); }
        }
        await stepInference({
          overlap: settings.overlap,
          threshold: settings.probabilityThreshold,
          minComponentSize: settings.minComponentSize,
          modelName: settings.modelName,
          patchSize: settings.patchSize,
          modelBaseUrl: settings.modelBaseUrl
        });
      } catch (error) {
        console.error('Inference error:', error);
        postError(error?.message || String(error));
      }
      break;
  }
};
