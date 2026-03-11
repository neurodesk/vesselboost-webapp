#!/usr/bin/env node
/**
 * Test the fixed JS SynthStrip pipeline against Docker reference output.
 * This replicates the exact logic from inference-worker.js stepSynthStrip().
 */
import * as ort from 'onnxruntime-node';
import * as nifti from 'nifti-reader-js';
import { readFileSync, writeFileSync } from 'fs';

const TOF_PATH = '/Users/uqsbollm/Downloads/testdata/tof_sub004.nii';
const DOCKER_MASK_PATH = '/Users/uqsbollm/Downloads/testdata/synthstrip_docker_mask.nii';
const ONNX_PATH = '/Users/uqsbollm/github-repos/vesselboost-webapp/web/models/synthstrip.onnx';
const OUT_DIR = '/Users/uqsbollm/Downloads/testdata';

// ---- NIfTI helpers ----
function loadNifti(path) {
  const buf = readFileSync(path).buffer;
  let hdr, data;
  if (nifti.isCompressed(buf)) {
    const decomp = nifti.decompress(buf);
    hdr = nifti.readHeader(decomp);
    data = nifti.readImage(hdr, decomp);
  } else {
    hdr = nifti.readHeader(buf);
    data = nifti.readImage(hdr, buf);
  }
  const dims = [hdr.dims[1], hdr.dims[2], hdr.dims[3]];
  const spacing = [Math.abs(hdr.pixDims[1]), Math.abs(hdr.pixDims[2]), Math.abs(hdr.pixDims[3])];
  let typedData;
  switch (hdr.datatypeCode) {
    case 16: typedData = new Float32Array(data); break;
    case 64: typedData = new Float64Array(data); break;
    case 4:  typedData = new Int16Array(data); break;
    case 512: typedData = new Uint16Array(data); break;
    case 2:  typedData = new Uint8Array(data); break;
    default: throw new Error(`Unsupported datatype code: ${hdr.datatypeCode}`);
  }
  // Convert to Float32 and apply scl_slope/scl_inter scaling
  const slope = (hdr.scl_slope && isFinite(hdr.scl_slope) && hdr.scl_slope !== 0) ? hdr.scl_slope : 1;
  const inter = (hdr.scl_inter && isFinite(hdr.scl_inter)) ? hdr.scl_inter : 0;
  const f32 = new Float32Array(typedData.length);
  for (let i = 0; i < typedData.length; i++) f32[i] = typedData[i] * slope + inter;
  console.log(`  NIfTI: dtype=${hdr.datatypeCode}, slope=${slope}, inter=${inter}`);
  return { data: f32, dims, spacing, hdr };
}

function saveNiftiMask(mask, refPath, outPath) {
  // Read reference file to get header
  const buf = readFileSync(refPath).buffer;
  let hdr;
  if (nifti.isCompressed(buf)) {
    hdr = nifti.readHeader(nifti.decompress(buf));
  } else {
    hdr = nifti.readHeader(buf);
  }
  // Create output buffer: header + mask data as uint8
  const voxOffset = hdr.vox_offset;
  const headerBuf = buf.slice(0, voxOffset);
  const headerBytes = new Uint8Array(headerBuf);
  // Modify header for uint8 output
  const headerView = new DataView(headerBuf);
  const littleEndian = hdr.littleEndian;
  // datatype = 2 (uint8), bitpix = 8
  headerView.setInt16(70, 2, littleEndian);
  headerView.setInt16(72, 8, littleEndian);
  // scl_slope=1, scl_inter=0
  headerView.setFloat32(112, 1, littleEndian);
  headerView.setFloat32(116, 0, littleEndian);

  const outBuf = new Uint8Array(voxOffset + mask.length);
  outBuf.set(new Uint8Array(headerBuf), 0);
  outBuf.set(mask, voxOffset);
  writeFileSync(outPath, outBuf);
}

// ---- Replicate inference-worker.js helper functions ----
function resampleVolume(data, dims, srcSpacing, tgtSpacing) {
  const [nx, ny, nz] = dims;
  const newDims = [
    Math.max(1, Math.round(nx * srcSpacing[0] / tgtSpacing[0])),
    Math.max(1, Math.round(ny * srcSpacing[1] / tgtSpacing[1])),
    Math.max(1, Math.round(nz * srcSpacing[2] / tgtSpacing[2]))
  ];
  const [nnx, nny, nnz] = newDims;
  const result = new Float32Array(nnx * nny * nnz);
  const scaleX = (nx - 1) / Math.max(nnx - 1, 1);
  const scaleY = (ny - 1) / Math.max(nny - 1, 1);
  const scaleZ = (nz - 1) / Math.max(nnz - 1, 1);

  for (let z = 0; z < nnz; z++) {
    const sz = z * scaleZ;
    const z0 = Math.floor(sz); const z1 = Math.min(z0 + 1, nz - 1); const wz = sz - z0;
    for (let y = 0; y < nny; y++) {
      const sy = y * scaleY;
      const y0 = Math.floor(sy); const y1 = Math.min(y0 + 1, ny - 1); const wy = sy - y0;
      for (let x = 0; x < nnx; x++) {
        const sx = x * scaleX;
        const x0 = Math.floor(sx); const x1 = Math.min(x0 + 1, nx - 1); const wx = sx - x0;
        const c000 = data[x0 + y0*nx + z0*nx*ny], c100 = data[x1 + y0*nx + z0*nx*ny];
        const c010 = data[x0 + y1*nx + z0*nx*ny], c110 = data[x1 + y1*nx + z0*nx*ny];
        const c001 = data[x0 + y0*nx + z1*nx*ny], c101 = data[x1 + y0*nx + z1*nx*ny];
        const c011 = data[x0 + y1*nx + z1*nx*ny], c111 = data[x1 + y1*nx + z1*nx*ny];
        const c00 = c000*(1-wx) + c100*wx, c01 = c001*(1-wx) + c101*wx;
        const c10 = c010*(1-wx) + c110*wx, c11 = c011*(1-wx) + c111*wx;
        const c0 = c00*(1-wy) + c10*wy, c1 = c01*(1-wy) + c11*wy;
        result[x + y*nnx + z*nnx*nny] = c0*(1-wz) + c1*wz;
      }
    }
  }
  return { data: result, dims: newDims };
}

function connectedComponents3D(mask, dims) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const labels = new Int32Array(n);
  let numComponents = 0;
  const queue = [];
  for (let i = 0; i < n; i++) {
    if (mask[i] && labels[i] === 0) {
      numComponents++;
      labels[i] = numComponents;
      queue.push(i);
      while (queue.length > 0) {
        const idx = queue.pop();
        const x = idx % nx;
        const y = Math.floor(idx / nx) % ny;
        const z = Math.floor(idx / (nx * ny));
        const neighbors = [];
        if (x > 0) neighbors.push(idx - 1);
        if (x < nx - 1) neighbors.push(idx + 1);
        if (y > 0) neighbors.push(idx - nx);
        if (y < ny - 1) neighbors.push(idx + nx);
        if (z > 0) neighbors.push(idx - nx * ny);
        if (z < nz - 1) neighbors.push(idx + nx * ny);
        for (const ni of neighbors) {
          if (mask[ni] && labels[ni] === 0) {
            labels[ni] = numComponents;
            queue.push(ni);
          }
        }
      }
    }
  }
  return { labels, numComponents };
}

function keepLargestComponentAndFill(binaryMask, dims) {
  const n = dims[0] * dims[1] * dims[2];
  const { labels, numComponents } = connectedComponents3D(binaryMask, dims);
  if (numComponents <= 1) {
    // Still do fill even with single component
    if (numComponents === 0) return binaryMask;
  }
  const sizes = new Int32Array(numComponents + 1);
  for (let i = 0; i < n; i++) { if (labels[i] > 0) sizes[labels[i]]++; }
  let largestLabel = 1;
  for (let l = 2; l <= numComponents; l++) { if (sizes[l] > sizes[largestLabel]) largestLabel = l; }
  const result = new Uint8Array(n);
  for (let i = 0; i < n; i++) { if (labels[i] === largestLabel) result[i] = 1; }

  // Fill interior holes
  const inverted = new Uint8Array(n);
  for (let i = 0; i < n; i++) inverted[i] = result[i] ? 0 : 1;
  const bgCC = connectedComponents3D(inverted, dims);
  const [nx, ny, nz] = dims;
  const borderLabels = new Set();
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (x === 0 || x === nx-1 || y === 0 || y === ny-1 || z === 0 || z === nz-1) {
          const idx = z*ny*nx + y*nx + x;
          if (bgCC.labels[idx] > 0) borderLabels.add(bgCC.labels[idx]);
        }
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (bgCC.labels[i] > 0 && !borderLabels.has(bgCC.labels[i])) result[i] = 1;
  }
  return result;
}

function resampleLabelsNearest(data, dims, tgtDims) {
  const [nx, ny, nz] = dims;
  const [tnx, tny, tnz] = tgtDims;
  const result = new Uint8Array(tnx * tny * tnz);
  for (let z = 0; z < tnz; z++) {
    const sz = Math.min(Math.max(0, Math.floor((z + 0.5) * nz / tnz)), nz - 1);
    for (let y = 0; y < tny; y++) {
      const sy = Math.min(Math.max(0, Math.floor((y + 0.5) * ny / tny)), ny - 1);
      for (let x = 0; x < tnx; x++) {
        const sx = Math.min(Math.max(0, Math.floor((x + 0.5) * nx / tnx)), nx - 1);
        result[x + y*tnx + z*tnx*tny] = data[sx + sy*nx + sz*nx*ny];
      }
    }
  }
  return result;
}

// ---- Main: exact replica of stepSynthStrip from inference-worker.js ----
async function main() {
  console.log('Loading input...');
  const { data: inputData, dims: rasDims, spacing: rasSpacing } = loadNifti(TOF_PATH);
  console.log(`Input (native LAS): ${rasDims.join('x')}, spacing: ${rasSpacing.map(v=>v.toFixed(3)).join(',')}`);

  const TARGET_SPACING = [1.0, 1.0, 1.0];

  // 0a. Reorient native (LAS) → RAS (matching web app loading step)
  //     LAS→RAS: flip axis 0 (L→R), axes 1,2 unchanged (A,S)
  const rasData2 = new Float32Array(inputData.length);
  {
    const [nx, ny, nz] = rasDims;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const rx = nx - 1 - x; // flip L→R
          rasData2[rx + y * nx + z * nx * ny] = inputData[x + y * nx + z * nx * ny];
        }
      }
    }
  }
  console.log(`Reoriented LAS -> RAS: ${rasDims.join('x')}`);

  // 0b. Reorient RAS → LIA (SynthStrip model trained on LIA data)
  //     LIA[0]=flip(RAS[0]), LIA[1]=flip(RAS[2]), LIA[2]=RAS[1]
  const liaPerm = [0, 2, 1];
  const liaFlip = [true, true, false];
  const liaDims = [rasDims[liaPerm[0]], rasDims[liaPerm[1]], rasDims[liaPerm[2]]];
  const liaSpacing = [rasSpacing[liaPerm[0]], rasSpacing[liaPerm[1]], rasSpacing[liaPerm[2]]];
  const liaData = new Float32Array(inputData.length);
  {
    const [dx, dy, dz] = liaDims;
    for (let oz = 0; oz < dz; oz++) {
      for (let oy = 0; oy < dy; oy++) {
        for (let ox = 0; ox < dx; ox++) {
          const coords = [ox, oy, oz];
          const src = [0, 0, 0];
          for (let i = 0; i < 3; i++) {
            src[liaPerm[i]] = liaFlip[i] ? (liaDims[i] - 1 - coords[i]) : coords[i];
          }
          const srcIdx = src[0] + src[1] * rasDims[0] + src[2] * rasDims[0] * rasDims[1];
          liaData[ox + oy * dx + oz * dx * dy] = rasData2[srcIdx];
        }
      }
    }
  }
  console.log(`Reoriented RAS -> LIA: ${rasDims.join('x')} -> ${liaDims.join('x')}`);

  // 1. Resample to target spacing
  const needsResample = liaSpacing[0] !== TARGET_SPACING[0] || liaSpacing[1] !== TARGET_SPACING[1] || liaSpacing[2] !== TARGET_SPACING[2];
  let currentData, currentDims;
  if (needsResample) {
    const resampled = resampleVolume(liaData, liaDims, liaSpacing, TARGET_SPACING);
    currentData = resampled.data;
    currentDims = resampled.dims;
    console.log(`Resampled: ${liaDims.join('x')} -> ${currentDims.join('x')}`);
  } else {
    currentData = liaData;
    currentDims = [...liaDims];
  }

  // Save resampled LIA for comparison
  writeFileSync(`${OUT_DIR}/js_resampled_lia.bin`, Buffer.from(currentData.buffer));
  console.log(`Saved resampled LIA: ${currentDims.join('x')}, first 5 values: ${Array.from(currentData.slice(0,5)).map(v=>v.toFixed(4))}`);

  // 2. Crop to bounding box + center-pad (THE FIX)
  const resampledDims = [...currentDims];
  const resampledLen = currentDims[0] * currentDims[1] * currentDims[2];

  const [rnx, rny, rnz] = currentDims;
  let bboxMin = [rnx, rny, rnz], bboxMax = [0, 0, 0];
  for (let z = 0; z < rnz; z++) {
    for (let y = 0; y < rny; y++) {
      for (let x = 0; x < rnx; x++) {
        if (currentData[x + y * rnx + z * rnx * rny] > 0) {
          if (x < bboxMin[0]) bboxMin[0] = x;
          if (y < bboxMin[1]) bboxMin[1] = y;
          if (z < bboxMin[2]) bboxMin[2] = z;
          if (x > bboxMax[0]) bboxMax[0] = x;
          if (y > bboxMax[1]) bboxMax[1] = y;
          if (z > bboxMax[2]) bboxMax[2] = z;
        }
      }
    }
  }
  bboxMax = [bboxMax[0] + 1, bboxMax[1] + 1, bboxMax[2] + 1];
  const cropDims = [bboxMax[0] - bboxMin[0], bboxMax[1] - bboxMin[1], bboxMax[2] - bboxMin[2]];
  const croppedData = new Float32Array(cropDims[0] * cropDims[1] * cropDims[2]);
  for (let z = 0; z < cropDims[2]; z++) {
    for (let y = 0; y < cropDims[1]; y++) {
      for (let x = 0; x < cropDims[0]; x++) {
        const srcIdx = (bboxMin[0] + x) + (bboxMin[1] + y) * rnx + (bboxMin[2] + z) * rnx * rny;
        croppedData[x + y * cropDims[0] + z * cropDims[0] * cropDims[1]] = currentData[srcIdx];
      }
    }
  }
  console.log(`Cropped to bbox: ${currentDims.join('x')} -> ${cropDims.join('x')}`);

  const targetDims = cropDims.map(s => Math.min(320, Math.max(192, Math.ceil(s / 64) * 64)));
  const centerOffsets = targetDims.map((t, i) => Math.floor((t - cropDims[i]) / 2));
  const conformedData = new Float32Array(targetDims[0] * targetDims[1] * targetDims[2]);
  for (let z = 0; z < cropDims[2]; z++) {
    for (let y = 0; y < cropDims[1]; y++) {
      for (let x = 0; x < cropDims[0]; x++) {
        const dx = x + centerOffsets[0];
        const dy = y + centerOffsets[1];
        const dz = z + centerOffsets[2];
        conformedData[dx + dy * targetDims[0] + dz * targetDims[0] * targetDims[1]] =
          croppedData[x + y * cropDims[0] + z * cropDims[0] * cropDims[1]];
      }
    }
  }
  currentData = conformedData;
  currentDims = targetDims;
  console.log(`Conformed (center+pad): ${cropDims.join('x')} -> ${targetDims.join('x')} (offsets: ${centerOffsets.join(',')})`);

  // 3. Normalize AFTER conform (matches FreeSurfer pipeline)
  const totalConformed = currentDims[0] * currentDims[1] * currentDims[2];
  let vMin = Infinity;
  for (let i = 0; i < totalConformed; i++) { if (currentData[i] < vMin) vMin = currentData[i]; }
  for (let i = 0; i < totalConformed; i++) { currentData[i] -= vMin; }
  const sorted = Float32Array.from(currentData).sort();
  const p99 = sorted[Math.floor(totalConformed * 0.99)];
  const vRange = p99 || 1;
  for (let i = 0; i < totalConformed; i++) {
    currentData[i] = Math.min(1, Math.max(0, currentData[i] / vRange));
  }
  console.log(`Normalized: min=${vMin.toFixed(2)}, p99=${p99.toFixed(2)}`);

  // Save conformed for comparison with Python
  writeFileSync(`${OUT_DIR}/js_conformed_lia.bin`, Buffer.from(currentData.buffer));
  console.log(`Saved conformed: nonzero=${Array.from(currentData).filter(v=>v>0).length}/${totalConformed}`);

  // 4. ONNX inference
  console.log('Loading ONNX model...');
  const session = await ort.InferenceSession.create(ONNX_PATH, {
    executionProviders: ['cpu'],
  });
  const totalVoxels = currentDims[0] * currentDims[1] * currentDims[2];
  console.log(`Running inference on ${currentDims.join('x')} (${(totalVoxels/1e6).toFixed(1)}M voxels)...`);
  const [cnx, cny, cnz] = currentDims;

  // Transpose column-major → row-major (C-order) for ONNX
  const cOrderInput = new Float32Array(totalVoxels);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      for (let x = 0; x < cnx; x++) {
        cOrderInput[x * cny * cnz + y * cnz + z] = currentData[x + y * cnx + z * cnx * cny];
      }
    }
  }

  const inputTensor = new ort.Tensor('float32', cOrderInput, [1, 1, ...currentDims]);
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const sdtRaw = results[session.outputNames[0]].data;

  // Transpose output back to column-major
  const sdtData = new Float32Array(totalVoxels);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      for (let x = 0; x < cnx; x++) {
        sdtData[x + y * cnx + z * cnx * cny] = sdtRaw[x * cny * cnz + y * cnz + z];
      }
    }
  }

  let sdtMin = Infinity, sdtMax = -Infinity;
  for (let i = 0; i < totalVoxels; i++) {
    if (sdtData[i] < sdtMin) sdtMin = sdtData[i];
    if (sdtData[i] > sdtMax) sdtMax = sdtData[i];
  }
  console.log(`SDT range: [${sdtMin.toFixed(3)}, ${sdtMax.toFixed(3)}]`);

  // 5. Threshold
  const SDT_BORDER = 1;
  const paddedMask = new Uint8Array(totalVoxels);
  let maskCount = 0;
  for (let i = 0; i < totalVoxels; i++) {
    if (sdtData[i] < SDT_BORDER) { paddedMask[i] = 1; maskCount++; }
  }
  console.log(`Mask in conformed space: ${maskCount} (${(100*maskCount/totalVoxels).toFixed(1)}%)`);

  // 6. Reverse center+pad
  const croppedMask = new Uint8Array(cropDims[0] * cropDims[1] * cropDims[2]);
  for (let z = 0; z < cropDims[2]; z++) {
    for (let y = 0; y < cropDims[1]; y++) {
      for (let x = 0; x < cropDims[0]; x++) {
        const sx = x + centerOffsets[0];
        const sy = y + centerOffsets[1];
        const sz = z + centerOffsets[2];
        croppedMask[x + y * cropDims[0] + z * cropDims[0] * cropDims[1]] =
          paddedMask[sx + sy * targetDims[0] + sz * targetDims[0] * targetDims[1]];
      }
    }
  }

  // 7. Reverse crop
  let resampledMask = new Uint8Array(resampledLen);
  for (let z = 0; z < cropDims[2]; z++) {
    for (let y = 0; y < cropDims[1]; y++) {
      for (let x = 0; x < cropDims[0]; x++) {
        const dstIdx = (bboxMin[0] + x) + (bboxMin[1] + y) * rnx + (bboxMin[2] + z) * rnx * rny;
        resampledMask[dstIdx] = croppedMask[x + y * cropDims[0] + z * cropDims[0] * cropDims[1]];
      }
    }
  }

  // 8. Largest CC + fill holes
  console.log('Cleaning mask (largest CC + fill holes)...');
  resampledMask = keepLargestComponentAndFill(resampledMask, resampledDims);

  // 9. Resample back to LIA original dims, then reorient to native
  let liaMask;
  if (needsResample) {
    liaMask = resampleLabelsNearest(resampledMask, resampledDims, liaDims);
  } else {
    liaMask = resampledMask;
  }
  // Reorient LIA → RAS (inverse of RAS→LIA)
  const rasMask = new Uint8Array(inputData.length);
  {
    const [dx, dy, dz] = liaDims;
    for (let oz = 0; oz < dz; oz++) {
      for (let oy = 0; oy < dy; oy++) {
        for (let ox = 0; ox < dx; ox++) {
          if (!liaMask[ox + oy * dx + oz * dx * dy]) continue;
          const coords = [ox, oy, oz];
          const dst = [0, 0, 0];
          for (let i = 0; i < 3; i++) {
            dst[liaPerm[i]] = liaFlip[i] ? (liaDims[i] - 1 - coords[i]) : coords[i];
          }
          rasMask[dst[0] + dst[1] * rasDims[0] + dst[2] * rasDims[0] * rasDims[1]] = 1;
        }
      }
    }
  }
  // Reorient RAS → native (LAS): flip axis 0 back
  const finalMask = new Uint8Array(inputData.length);
  {
    const [nx, ny, nz] = rasDims;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          if (!rasMask[x + y * nx + z * nx * ny]) continue;
          const lx = nx - 1 - x; // flip R→L
          finalMask[lx + y * nx + z * nx * ny] = 1;
        }
      }
    }
  }

  let finalCount = 0;
  for (let i = 0; i < finalMask.length; i++) { if (finalMask[i]) finalCount++; }
  const coverage = (100 * finalCount / inputData.length).toFixed(1);
  console.log(`JS mask: ${finalCount} voxels (${coverage}% coverage)`);

  // 10. Save output
  saveNiftiMask(finalMask, TOF_PATH, `${OUT_DIR}/synthstrip_js_mask.nii`);
  console.log(`Saved: ${OUT_DIR}/synthstrip_js_mask.nii`);

  // 11. Compare to Docker reference
  console.log('\nComparing to Docker reference...');
  const { data: dockerData, dims: dockerDims } = loadNifti(DOCKER_MASK_PATH);
  console.log(`Docker mask dims: ${dockerDims.join('x')}`);

  let dockerCount = 0, overlap = 0, jsOnly = 0, dockerOnly = 0;
  for (let i = 0; i < finalMask.length; i++) {
    const js = finalMask[i] > 0;
    const dk = dockerData[i] > 0;
    if (dk) dockerCount++;
    if (js && dk) overlap++;
    if (js && !dk) jsOnly++;
    if (!js && dk) dockerOnly++;
  }
  const dice = 2 * overlap / (finalCount + dockerCount);
  const sensitivity = overlap / dockerCount;
  console.log(`Docker:      ${dockerCount} voxels (${(100*dockerCount/inputData.length).toFixed(1)}%)`);
  console.log(`JS (fixed):  ${finalCount} voxels (${coverage}%)`);
  console.log(`Overlap:     ${overlap}`);
  console.log(`JS only:     ${jsOnly}`);
  console.log(`Docker only: ${dockerOnly}`);
  console.log(`Dice:        ${dice.toFixed(4)}`);
  console.log(`Sensitivity: ${sensitivity.toFixed(4)}`);
}

main().catch(console.error);
